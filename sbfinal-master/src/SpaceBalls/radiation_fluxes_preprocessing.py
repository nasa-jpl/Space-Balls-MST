import os, sys
import numpy as np
from astropy.time import Time
import astropy.units as u
import multiprocessing
from multiprocessing import Pool
from scipy.interpolate import make_interp_spline
import time

from SpaceBalls.paths import CONFIG_DIR, MEDIA_DIR
sys.path.insert(0, str(CONFIG_DIR.parent))  # parent of 'config'
import SpaceBalls.radiation_settings as rad_settings
from SpaceBalls.sph_meshing import Grid, RegularLatLonGrid, QuadratureGrid, expand_sh, field_hist_rotation_multiproc

import config.constants as constants
from SpaceBalls.utils import get_norm_across_last_dim, progress_bar, jd_to_mmddyyyy
from SpaceBalls.plotter import Plotter

AU = constants.astronomical_unit(units='km')
RE = constants.earth_radius(units='km')
STEP_MINUTES = 1 # DO NOT CHANGE - must be equal to the one used for the files in solar_ephemerides

REQUIRED_FILES = {
    "toa": ['daily_hist_LW_toa', 'daily_hist_SW_toa', 'daily_hist_net_toa', 'daily_hist_net_toa_SFF_accurate',
            'daily_avg_net_toa', 'daily_hist_net_toa_surf_avg', # 'daily_hist_net_toa_lat_avg'
            ],
    "altitude": ['daily_hist_net_F_{alt_km}km', 'daily_hist_net_F_{alt_km}km_SFF_accurate',
                 'daily_avg_net_F_{alt_km}km', 'daily_avg_net_F_{alt_km}km_SFF_accurate',
                 'daily_avg_net_Fr_{alt_km}km', 'daily_avg_net_Fr_{alt_km}km_SFF_accurate',
                 'daily_surf_avg_net_{alt_km}km']

    # "altitude": ['daily_hist_net_Fr_{alt_km}km', 'daily_hist_net_Fr_{alt_km}km_SFF',
    #              'daily_hist_net_Fr_{alt_km}km_SFF_accurate',
    #              'daily_hist_net_Fr_{alt_km}km_surf_avg',
    #              'daily_avg_net_Fr_{alt_km}km', 'daily_avg_net_Fr_{alt_km}km_SFF',
    #              'daily_avg_net_Fr_{alt_km}km_SFF_accurate', 
    #              'daily_hist_net_F_{alt_km}km']
                 #, 'daily_hist_net_F_{alt_km}km_SFF', 'daily_avg_net_F_{alt_km}km', 'daily_avg_net_F_{alt_km}km_SFF']
}

def get_required_files(grid: Grid):

    key = 'toa' if grid.alt_km==0 else 'altitude'
    files = REQUIRED_FILES[key].copy()

    if grid.alt_km>0:
        files = [f.replace('{alt_km}', str(grid.alt_km)) for f in files]
    
    if grid.n_points>5810 and key=="altitude": # lebedev order 131 grid has 5810 points
        files = [f for f in files if "hist" not in f] # would be too much storage

    return files


def compute_radiation_maps(EEI_truth_name, grid: Grid, selected_days_idxs=None, n_cores=4):

    base_data_dir = os.path.join(MEDIA_DIR, 'true_EEI', EEI_truth_name)
    
    rad_config = rad_settings.radiation_settings_from_EEI_truth_name(EEI_truth_name)
    daily_jd_arrays = get_EEI_truth_daily_jd_arrays(rad_config["jd_interval"])
    mid_day_jd_array = mid_day_jd_array_from_jd_interval(rad_config["jd_interval"])
    n_days = len(mid_day_jd_array)

    if (grid.alt_km==0) or (selected_days_idxs is None) or (len(selected_days_idxs)==0):
        compute_all_days = True
        idxs_days_to_loop = range(n_days)
    else:
        compute_all_days = False
        daily_jd_arrays = [daily_jd_arrays[i] for i in selected_days_idxs]
        idxs_days_to_loop = selected_days_idxs

    for day_idx, jd_array in zip(idxs_days_to_loop, daily_jd_arrays):

        print(f"Now computing: daily hist for day {day_idx} at {grid.alt_km}km")
        n_steps = len(jd_array)
        mid_day_jd = mid_day_jd_array[day_idx]
        #TSI_1AU_day = rad_settings.get_TSI_1AU(mid_day_jd, rad_config["TSI_source"])
        R_ECEF_to_SunFrame_day_hist = get_R_SunFrame_hist(rad_config["ephemerides"], mid_day_jd)

        file_names, file_existences = get_file_names_and_existence(base_data_dir, day_idx, grid)
        #if not(file_existences[f"daily_hist_net_toa_SFF_accurate"]) and (day_idx in selected_days_idxs):
        #    compute_SFF = True # At TOA, the only reason we want the SFF map is for animations
        #else:
        #    compute_SFF = False
        net_toa_day_hist_SFF = None

        if grid.alt_km==0 and not(all(file_existences.values())):
            toa_LW_day_hist, toa_SW_day_hist, net_toa_day_hist = get_daily_hist_toa(base_data_dir, 
                                                                                    day_idx, 
                                                                                    mid_day_jd, 
                                                                                    jd_array, 
                                                                                    rad_config, 
                                                                                    grid)
            save_toa_files(toa_LW_day_hist, toa_SW_day_hist, net_toa_day_hist, net_toa_day_hist_SFF,
                   base_data_dir, day_idx, grid)

        elif grid.alt_km>0 and not(all(file_existences.values())):
            
            daily_hist_net_F_altitude, daily_hist_net_Fr_altitude = get_daily_hist_net_at_altitude(
                                    base_data_dir, day_idx, mid_day_jd, jd_array, rad_config, grid, 
                                    n_cores)


def get_daily_hist_toa(base_data_dir, day_idx, mid_day_jd, day_jd_array, rad_config, grid: Grid, load_emission=True, load_net=True):
    
    all_file_names, file_existences = get_file_names_and_existence(base_data_dir, day_idx, grid)
    #toa_emission_day_hist, net_toa_day_hist = None, None

    if not(file_existences["daily_hist_LW_toa"]) or not(file_existences["daily_hist_SW_toa"]) or not(file_existences["daily_hist_net_toa"]):  # requires recomputing
        print("Computing daily hist TOA...")
        toa_LW_day_hist, toa_SW_day_hist, net_toa_day_hist = compute_flux_hist_toa(day_jd_array, rad_config, grid)
        #toa_LW_day_hist, toa_SW_day_hist, net_toa_day_hist = compute_daily_hist_toa(mid_day_jd, rad_config, TSI_1AU_day, grid)
    
    else:
        print("Loading daily hist TOA...")
        toa_LW_day_hist, toa_SW_day_hist, net_toa_day_hist = None, None, None
        if load_emission:
            toa_LW_day_hist = np.load(all_file_names["daily_hist_LW_toa"]+'.npy')
            toa_SW_day_hist = np.load(all_file_names["daily_hist_SW_toa"]+'.npy')

        if load_net:
            net_toa_day_hist = np.load(all_file_names["daily_hist_net_toa"]+'.npy')

    return toa_LW_day_hist, toa_SW_day_hist, net_toa_day_hist


def get_hist_toa_new(EEI_truth_name, jd_array, grid: Grid, load_emission=True, load_net=True):

    base_data_dir = os.path.join(MEDIA_DIR, 'true_EEI', EEI_truth_name)
    rad_config = rad_settings.radiation_settings_from_EEI_truth_name(EEI_truth_name)
    day_idxs_jd_array = get_day_idxs_for_jd_array(jd_array, get_mid_day_jd_array(EEI_truth_name))

    # logic with these inputs so it can be called from inside compute_F_at_altitude and thus TOA hist is not computed again if it is stored
    # NOTE: it does not really make a difference because the TOA hists that we save are fast to compute anyway - worth it?

    full_mid_day_jd_array = np.round(jd_array)
    unique_mid_day_jd = np.unique(full_mid_day_jd_array)
    truth_jd_array = np.concatenate([get_day_jd_array(mid_day_jd) for mid_day_jd in unique_mid_day_jd])

    if (len(unique_mid_day_jd)==1) and np.array_equal(truth_jd_array, jd_array):
        # Possible extension: load multiple days at once - for mid_day_jd, day_idx in zip(unique_mid_day_jd, np.unique(day_idxs_jd_array)):
        day_idx = np.unique(day_idxs_jd_array)
        assert(len(day_idx)==1)
        toa_LW_day_hist, toa_SW_day_hist, net_toa_day_hist = get_daily_hist_toa(base_data_dir, 
                                                                                day_idx[0], 
                                                                                unique_mid_day_jd[0], 
                                                                                jd_array, 
                                                                                rad_config, 
                                                                                grid, 
                                                                                load_emission=load_emission, 
                                                                                load_net=load_net)
    else:
        toa_LW_day_hist, toa_SW_day_hist, net_toa_day_hist = compute_flux_hist_toa(jd_array, rad_config, grid)


    
    unique_day_idxs = np.unique(day_idxs_jd_array)
    
    



def get_daily_hist_net_at_altitude(base_data_dir, day_idx, mid_day_jd, day_jd_array, rad_config, grid: Grid, n_cores):
    
    all_file_names, file_existences = get_file_names_and_existence(base_data_dir, day_idx, grid)
    sff = True
    
    if file_existences.get('daily_hist_net_F_'+str(grid.alt_km)+'km', False):
        pass
        # load daily_hist_net_F_altitude and compute the rest (this option should not really ever happen if the preproc scripts have been run correctly)
    else:
        toa_grid = QuadratureGrid(alt_km=0, order=131) # integration to altitude MUST be done with the Lebedev TOA grid (orders of magnitude more accurate with less points)

        if sff:
            R_ECEF_to_SunFrame_day_hist = get_R_SunFrame_hist(rad_config["ephemerides"], mid_day_jd)
            stacked_r = apply_SFF_rotation(R_ECEF_to_SunFrame_day_hist, grid.stacked_grid_r, mode="SFF_to_ECEF")
            # we want the grid nodes to be static in the SFF, so the resulting hist is how they move in ECEF
        else:
            stacked_r = grid.stacked_grid_r[:,None,:]

        daily_hist_earth_F_altitude = compute_earth_F_at_altitude(day_jd_array, rad_config, stacked_r, toa_grid)
        if sff:
            daily_hist_earth_F_altitude = apply_SFF_rotation(R_ECEF_to_SunFrame_day_hist, daily_hist_earth_F_altitude, "ECEF_to_SFF")
        daily_hist_earth_Fr_altitude = F_vec_to_Fr(daily_hist_earth_F_altitude, grid.stacked_grid_u) # grid is already defined in SFF


        Plotter.plot_geo_data_new(daily_hist_earth_Fr_altitude[:,0], grid, 
                                  file_name='earth_Fr_ECEF_0', add_coastlines=False, make_symmetric_cmap=False)
        Plotter.plot_geo_data_new(daily_hist_earth_Fr_altitude[:,700], grid, 
                                  file_name='earth_Fr_ECEF_700', add_coastlines=False, make_symmetric_cmap=False)

        a-3


        r_sun_day = get_r_sun_day_hist(rad_config["ephemerides"], mid_day_jd)
        daily_hist_solar_F_altitude = get_solar_incoming_day_hist(r_sun_day, TSI_1AU_day, 
                                                                     stacked_r, grid, toa_grid)
        if sff:
            daily_hist_solar_F_altitude = apply_SFF_rotation(R_ECEF_to_SunFrame_day_hist, daily_hist_solar_F_altitude, "ECEF_to_SFF")
        daily_hist_solar_Fr_altitude = F_vec_to_Fr(daily_hist_solar_F_altitude, grid.stacked_grid_u)

        Plotter.plot_geo_data_new(daily_hist_solar_Fr_altitude[:,0], grid, 
                                  file_name='sun_Fr_SFF_0', add_coastlines=False, make_symmetric_cmap=True)
        Plotter.plot_geo_data_new(daily_hist_solar_Fr_altitude[:,700], grid, 
                                  file_name='sun_Fr_SFF_700', add_coastlines=False, make_symmetric_cmap=True)


    return daily_hist_net_F_altitude #, daily_hist_net_Fr_altitude


# deprecated function
"""
def compute_daily_hist_toa(mid_day_jd, rad_config, TSI_1AU_day, grid: Grid):
    
    r_sun_day = get_r_sun_day_hist(rad_config["ephemerides"], mid_day_jd)
    solar_incoming_day_hist_toa = get_solar_incoming_day_hist(r_sun_day, TSI_1AU_day, grid.stacked_grid_r, grid)
    # now solar_incoming_day_hist_toa is the full F vector
    solar_incoming_day_hist_toa = F_vec_to_Fr(solar_incoming_day_hist_toa, grid.stacked_grid_u)

    day_datestr = jd_to_mmddyyyy(mid_day_jd)
    d_sun_day = get_norm_across_last_dim(r_sun_day) # np.sqrt(np.einsum('ij,ij->i', r_sun_day, r_sun_day)) # same as np.linalg.norm(r_sun_day,2,1) but slightly faster
    TSI_Earth_day_vec = TSI_1AU_day * (AU/d_sun_day)**2

    grid_r_sun_hist, grid_d_sun_hist = get_grid_r_sun_hist(r_sun_day, grid.stacked_grid_r) #grid)
    grid_u_sun_hist = get_grid_u_sun_hist(grid_r_sun_hist, grid_d_sun_hist)
    zeroed_cos_theta_s_day_hist_toa = get_zeroed_cos_theta_s_hist(grid_u_sun_hist, grid)

    if not("time_interp" in rad_config):
        # what we have now: constant a/e thorughout a single day
        a_map, e_map = get_expanded_ae_maps(day_datestr, grid, rad_config) 
        LW_outgoing_day_hist_toa = e_map[:, None] / 4 * TSI_Earth_day_vec[None, :] # here we DO use the TSI at the center of the Earth (Knocke model and Monte docs)
        SW_outgoing_day_hist_toa = zeroed_cos_theta_s_day_hist_toa * a_map[:, None] * TSI_Earth_day_vec[None, :]
    else:
        # else: we want to interpolate the a/e maps to the 1-min time resolution
        print("Getting smooth daily ae hist...")
        a_hist, e_hist = get_smooth_daily_ae_hist(mid_day_jd, rad_config, grid)
        if rad_config["time_interp"]=="sh_interp":
            a_hist = expand_daily_map_hist(a_hist, grid, rad_config['sh_normalization'])
            e_hist = expand_daily_map_hist(e_hist, grid, rad_config['sh_normalization'])

        LW_outgoing_day_hist_toa = e_hist[:, :].T / 4 * TSI_Earth_day_vec[None, :] # here we DO use the TSI at the center of the Earth (Knocke model and Monte docs)
        SW_outgoing_day_hist_toa = zeroed_cos_theta_s_day_hist_toa * a_hist[:, :].T * TSI_Earth_day_vec[None, :]

    net_toa_day_hist = solar_incoming_day_hist_toa - (LW_outgoing_day_hist_toa + SW_outgoing_day_hist_toa)

    LW_outgoing_day_hist_toa = grid.reshape_if_needed(LW_outgoing_day_hist_toa)
    SW_outgoing_day_hist_toa = grid.reshape_if_needed(SW_outgoing_day_hist_toa)
    net_toa_day_hist = grid.reshape_if_needed(net_toa_day_hist) # TODO: this was originally for regular grids but is probably deprecated

    return LW_outgoing_day_hist_toa, SW_outgoing_day_hist_toa, net_toa_day_hist
"""

def compute_flux_hist_toa(jd_array, rad_config, grid: Grid, compute_net=True):
    
    
    TSI_1AU_jd_array = rad_settings.get_TSI_1AU(jd_array, rad_config['TSI_source'])

    r_sun_hist = get_r_sun_jd_hist(rad_config, jd_array)
    d_sun_day = get_norm_across_last_dim(r_sun_hist) # np.sqrt(np.einsum('ij,ij->i', r_sun_day, r_sun_day)) # same as np.linalg.norm(r_sun_day,2,1) but slightly faster
    TSI_Earth_day_vec = TSI_1AU_jd_array * (AU/d_sun_day)**2

    grid_r_sun_hist, grid_d_sun_hist = get_grid_r_sun_hist(r_sun_hist, grid.stacked_grid_r) #grid)
    grid_u_sun_hist = get_grid_u_sun_hist(grid_r_sun_hist, grid_d_sun_hist)
    zeroed_cos_theta_s_day_hist_toa = get_zeroed_cos_theta_s_hist(grid_u_sun_hist, grid)

    if not("time_interp" in rad_config):
        all_mid_day_jds = np.round(jd_array)
        e_hist, a_hist = np.zeros((grid.n_points, len(jd_array))), np.zeros((grid.n_points, len(jd_array)))

        for mid_day_jd in np.unique(all_mid_day_jds):
            day_datestr = jd_to_mmddyyyy(mid_day_jd)
            a_map, e_map = get_expanded_ae_maps(day_datestr, grid, rad_config)
            #a_map, e_map = np.zeros(grid.n_points), np.zeros(grid.n_points)
            a_hist[:, mid_day_jd==all_mid_day_jds] = a_map[:,None]
            e_hist[:, mid_day_jd==all_mid_day_jds] = e_map[:,None]

        #LW_outgoing_day_hist_toa = e_map[:, None] / 4 * TSI_Earth_day_vec[None, :] # here we DO use the TSI at the center of the Earth (Knocke model and Monte docs)
        #SW_outgoing_day_hist_toa = zeroed_cos_theta_s_day_hist_toa * a_map[:, None] * TSI_Earth_day_vec[None, :]
    else:
        # else: we want to interpolate the a/e maps to the 1-min time resolution
        print("Getting smooth daily ae hist...")
        a_hist, e_hist = get_smooth_daily_ae_hist(jd_array, rad_config, grid)


    LW_outgoing_day_hist_toa = e_hist / 4 * TSI_Earth_day_vec[None, :] # here we DO use the TSI at the center of the Earth (Knocke model and Monte docs)
    SW_outgoing_day_hist_toa = zeroed_cos_theta_s_day_hist_toa * a_hist * TSI_Earth_day_vec[None, :]

    if compute_net:
        solar_incoming_day_hist_toa = get_solar_incoming_day_hist(r_sun_hist, TSI_1AU_jd_array, grid.stacked_grid_r, grid)
        # now solar_incoming_day_hist_toa is the full F vector
        solar_incoming_day_hist_toa = F_vec_to_Fr(solar_incoming_day_hist_toa, grid.stacked_grid_u)
        net_toa_day_hist = solar_incoming_day_hist_toa - (LW_outgoing_day_hist_toa + SW_outgoing_day_hist_toa)
    else:
        net_toa_day_hist = None
    

    LW_outgoing_day_hist_toa = grid.reshape_if_needed(LW_outgoing_day_hist_toa)
    SW_outgoing_day_hist_toa = grid.reshape_if_needed(SW_outgoing_day_hist_toa)
    net_toa_day_hist = grid.reshape_if_needed(net_toa_day_hist) # TODO: this was originally for regular grids but is probably deprecated

    return LW_outgoing_day_hist_toa, SW_outgoing_day_hist_toa, net_toa_day_hist

"""
def compute_earth_F_at_altitude_old(jd_array, rad_config, stacked_r, toa_grid: Grid, store_dF=False): # , n_cores):

    # toa_grid = QuadratureGrid(alt_km=0, order=131) # integration to altitude MUST be done with the Lebedev TOA grid (orders of magnitude more accurate with less points)
    # toa_LW_day_hist, toa_SW_day_hist, _ = get_daily_hist_toa(base_data_dir, day_idx, mid_day_jd, TSI_1AU_day, 
    #                                                                 rad_config, toa_grid, load_net=False)

    toa_LW_hist, toa_SW_hist, _ = compute_flux_hist_toa(jd_array, rad_config, toa_grid)

    toa_L_day_hist_LW = irradiance_to_radiance(toa_LW_hist, rad_type="LW", r_alt=stacked_r, ADM_model=rad_config.get('ADM_model')) # will probably need a function get_u_sun_grid(day_idx, grid) to call with toa_grid right before this line and allow to use ADMs
    toa_L_day_hist_SW = irradiance_to_radiance(toa_SW_hist, rad_type="SW", r_alt=stacked_r, ADM_model=rad_config.get('ADM_model'))
    toa_L_day_hist = toa_L_day_hist_LW + toa_L_day_hist_SW

    n_steps = np.shape(toa_L_day_hist)[1]

    #TODO: below here will need to be a new function so that case b can call it from outside (loop if L hist is too large)
    if len(np.shape(stacked_r))==2:

        # cases a) stacked_r is sized np_alt_grid x 3 - r_rel is sized np_alt_grid x np_toa_grid x 3
        #   and b) stacked_r is sized n_steps x 3 (satellite hist) - same as case a - r_rel is sized n_steps x np_toa_grid x 3 
        
            # 1. Compute all relative vectors
        r_rel = stacked_r[:,None,:] - toa_grid.stacked_grid_r[None,:,:]
            # 2. Compute norm of relative vectors
        r_rel_norm_4 = np.einsum('ijk,ijk->ij', r_rel, r_rel)**2 # einsum method faster than np.linalg.norm
            # 3. Compute all geometric kernels
        geometric_kernel = np.einsum('ijk,jk->ij', r_rel, toa_grid.stacked_grid_u) / r_rel_norm_4 # this is cos(alpha)/(r_rel_norm**3)
        np.maximum(geometric_kernel, 0, out=geometric_kernel) # set negative cosines(alpha) to zero
            # 4. Compute flux vector integral
        emission_F_hist = np.einsum('it,ji,jik,i->jtk', 
                                        toa_L_day_hist,               # (np_toa x n_steps)
                                        geometric_kernel,             # (np x np_toa_grid)             (NOTE that np is either np_alt_grid or n_steps)
                                        r_rel,                        # (np x np_toa_grid x 3)
                                        toa_grid.integration_weights, # (np_toa_grid)
                                        optimize='optimal'
                                        )                             # output is sized (np x 3 x n_steps)

        if store_dF:  
            # if we want to keep the individual flux contributions (TODO: weighted by integration weight?) at each altitude point instead:
            # NOTE: this is only for case b, otherwise there is no use case and would take too much RAM
            # NOTE: STILL UNVERIFIED
            # TODO: we might want to save this for separate SW / LW contributions (easy when toa_L_day_hist is an input)
            emission_dF_day_hist = np.einsum('it,ti,tik,ti,i->tik', 
                                            toa_L_day_hist,               # (np_toa x n_steps)
                                            geometric_kernel,             # (n_steps x np_toa)          
                                            r_rel,                        # (n_steps x np_toa_grid x 3)
                                            toa_grid.integration_weights, # (np_toa_grid)
                                            optimize='optimal'
                                            )                             # output is sized (n_steps x np_toa x 3)

    elif len(np.shape(stacked_r))==3:
        # case c) stacked r is sized np_alt_grid x n_steps x 3 (rotating grid hist) - r_rel is sized np_alt_grid x np_toa_grid x n_steps x 3
        # r_rel will be sized np_toa x np_alt x n_steps x 3
        max_ram_GB = 10
        max_np_alt = 8e9 * max_ram_GB / (64 * toa_grid.n_points * n_steps * 3)
        n_chunks = np.max([1, int(np.floor(len(stacked_r) / max_np_alt))])
        stacked_r_chunks = np.array_split(stacked_r, n_chunks)
        emission_F_hist = [None] * n_chunks

        for i, stacked_r in enumerate(stacked_r_chunks):
            print(f"Computing chunk {i+1}/{n_chunks}...")
                # 1. Compute all relative vectors
            r_rel = stacked_r[:,None,:,:] - toa_grid.stacked_grid_r[None,:,None,:] 
                # 2. Compute norm of relative vectors
            r_rel_norm_4 = np.einsum('ijtk,ijtk->ijt', r_rel, r_rel)**2   # norm of all r_rel to the 4th power
                # 3. Compute all geometric kernels
            geometric_kernel = np.einsum('ijtk,jk->ijt', r_rel, toa_grid.stacked_grid_u) / r_rel_norm_4 # this is cos(alpha)/(r_rel_norm**3)
            np.maximum(geometric_kernel, 0, out=geometric_kernel) # set negative cosines(alpha) to zero
                # 4. Compute flux vector integral
            emission_F_hist[i] = np.einsum('it,jit,jitk,i->jtk', 
                                            toa_L_day_hist,               # (np_toa_grid x n_steps)
                                            geometric_kernel,             # (np_alt_grid x np_toa_grid x n_steps)            
                                            r_rel,                        # (np_alt_grid x np_toa_grid x n_steps x 3)
                                            toa_grid.integration_weights, # (np_toa_grid)
                                            optimize='optimal',           # passing optimal path from einsum_path(..., optimize='optimal') does not seem to help
                                            )                             # output is sized (np x 3 x n_steps)
        emission_F_hist = np.vstack(emission_F_hist)
        
    # # Finally, compute Fr (now in external function)
    return emission_F_hist
"""

def compute_earth_F_at_altitude(jd_array, rad_config: dict, stacked_r, toa_grid: Grid, store_dF=False, wavelength="sum"): # , n_cores):

    assert(len(np.shape(stacked_r))==3)
    n_p = np.shape(stacked_r)[0]
    n_steps_r = np.shape(stacked_r)[1]
    n_steps = len(jd_array)
    assert((n_steps_r==1) or (n_steps_r==n_steps))
    assert((wavelength=="sum") or (wavelength=="split") or (wavelength=="LW") or (wavelength=="SW"))

    ADM_model = rad_config.get('ADM_model')
    if ADM_model is None: nt = n_steps_r # nt is the most constraining one to build the arrays after this
    else: nt = n_steps

    max_ram_GB = 12
    max_nt = 8e9 * max_ram_GB / (64 * toa_grid.n_points * n_p * 3)
    n_chunks = np.max([1, int(np.floor(nt / max_nt))])
    print(f"Splitting day into {n_chunks} chunks")
    jd_chunks = np.array_split(jd_array, n_chunks)
    stacked_r_chunks = np.array_split(stacked_r, n_chunks, axis=1)
    if wavelength=="split":
        emission_F_LW_hist = [None] * n_chunks
        emission_F_SW_hist = [None] * n_chunks
    else:
        emission_F_hist = [None] * n_chunks

   
    for i, jd_array in enumerate(jd_chunks):
        print(f"Computing chunk {i+1}/{n_chunks}...")
        toa_LW_hist, toa_SW_hist, _ = compute_flux_hist_toa(jd_array, rad_config, toa_grid, compute_net=False)
        #toa_LW_hist, toa_SW_hist = np.zeros((toa_grid.n_points, len(jd_array))), np.zeros((toa_grid.n_points, len(jd_array)))

        print(f"Getting ADMs...")
        stacked_r_i = stacked_r if n_steps_r==1 else stacked_r_chunks[i]
        LW_I_to_L_map = get_irradiance_to_radiance_map(stacked_r_i, jd_array, toa_grid, rad_config, 
                                                    rad_type="LW", ADM_model=ADM_model) # shape (np, np_toa, n_steps)
        SW_I_to_L_map = get_irradiance_to_radiance_map(stacked_r_i, jd_array, toa_grid, rad_config, 
                                                        rad_type="SW", ADM_model=ADM_model) 

        print("Computing flux...")
            # 1. Compute all relative vectors
        r_rel = stacked_r_i[:,None,:,:] - toa_grid.stacked_grid_r[None,:,None,:] 
            # 2. Compute norm of relative vectors
        r_rel_norm_4 = np.einsum('ijtk,ijtk->ijt', r_rel, r_rel)**2   # norm of all r_rel to the 4th power
            # 3. Compute all geometric kernels
        geometric_kernel = np.einsum('ijtk,jk->ijt', r_rel, toa_grid.stacked_grid_u) / r_rel_norm_4 # this is cos(alpha)/(r_rel_norm**3)
        np.maximum(geometric_kernel, 0, out=geometric_kernel) # set negative cosines(alpha) to zero
            # 4. Compute flux vector integral
        if wavelength=="split":
            emission_F_LW_hist[i], emission_F_SW_hist[i] = (
                                 compute_earh_F_integral(toa_LW_hist, toa_SW_hist,
                                                        LW_I_to_L_map, SW_I_to_L_map,
                                                        geometric_kernel, r_rel, 
                                                        toa_grid.integration_weights,
                                                        wavelength=wl, store_dF=store_dF, 
                                                        ADM_model=ADM_model) for wl in ["LW", "SW"])
        else:
            emission_F_hist[i] = compute_earh_F_integral(toa_LW_hist, toa_SW_hist,
                                                        LW_I_to_L_map, SW_I_to_L_map,
                                                        geometric_kernel, r_rel, 
                                                        toa_grid.integration_weights,
                                                        wavelength=wavelength, store_dF=store_dF, 
                                                        ADM_model=ADM_model)

    if wavelength=="split":
        emission_F_LW_hist = np.vstack(emission_F_LW_hist)
        emission_F_SW_hist = np.vstack(emission_F_SW_hist)
        return emission_F_LW_hist, emission_F_SW_hist

    else:
        emission_F_hist = np.vstack(emission_F_hist)
        return emission_F_hist


def compute_earh_F_integral(toa_LW_hist, toa_SW_hist, LW_I_to_L_map, SW_I_to_L_map, 
                            geometric_kernel, r_rel, toa_weights,
                            wavelength="sum", store_dF=False, ADM_model=None):

    ein_out = 'jitk' if store_dF else 'jtk'
    n_steps_r = np.shape(r_rel)[2]
    assert(n_steps_r==np.shape(geometric_kernel)[2])
    
    if ADM_model is None:
        assert(np.ndim(LW_I_to_L_map) == 0) # LW_I_to_L_map is just a scalar
        assert(np.ndim(SW_I_to_L_map) == 0) # SW_I_to_L_map is just a scalar
        
        if wavelength=="sum":
            toa_L_hist = toa_LW_hist * LW_I_to_L_map + toa_SW_hist * SW_I_to_L_map
        elif wavelength=="LW":
            toa_L_hist = toa_LW_hist * LW_I_to_L_map
        elif wavelength=="SW":
            toa_L_hist = toa_SW_hist * SW_I_to_L_map

        if n_steps_r==1: 
            ein_tag = 'it,ji,jik,i->'+ein_out
            geometric_kernel = geometric_kernel[:,:,0]
            r_rel = r_rel[:,:,0,:]
        else:
            ein_tag = 'it,jit,jitk,i->'+ein_out

        F_hist = np.einsum(ein_tag, 
                           toa_L_hist,                  # (np_toa_grid x n_steps)
                           geometric_kernel,            # (np_alt_grid x np_toa_grid x n_steps)            
                           r_rel,                       # (np_alt_grid x np_toa_grid x n_steps x 3)
                           toa_weights,                 # (np_toa_grid)
                           optimize='optimal',          # passing optimal path from einsum_path(..., optimize='optimal') does not seem to help
                          )                             # output is sized (np x n_steps x 3) (store_dF=False) or
                                                        #                 (np x n_toa x n_steps x 3) (store_dF=True)
        return F_hist

    else:
        if n_steps_r==1: 
            ein_tag = 'it,jit,ji,jik,i->'+ein_out
            geometric_kernel = geometric_kernel[:,:,0]
            r_rel = r_rel[:,:,0,:]
        else:
            ein_tag = 'it,jit,jit,jitk,i->'+ein_out

        if (wavelength=="sum") or wavelength=="SW":
            F_hist_SW = np.einsum(ein_tag, 
                                  toa_SW_hist,                  # (np_toa_grid x n_steps)
                                  SW_I_to_L_map,                # (np_alt_grid x np_toa_grid x n_steps) or float?
                                  geometric_kernel,             # (np_alt_grid x np_toa_grid x n_steps)            
                                  r_rel,                        # (np_alt_grid x np_toa_grid x n_steps x 3)
                                  toa_weights,                  # (np_toa_grid)
                                  optimize='optimal',           # passing optimal path from einsum_path(..., optimize='optimal') does not seem to help
                                )                               # output is sized (np x n_steps x 3) (store_dF=False) or
                                                                #                 (np x n_toa x n_steps x 3) (store_dF=True)
        if (wavelength=="sum") or wavelength=="LW":
            F_hist_LW = np.einsum(ein_tag, 
                            toa_LW_hist,                   # (np_toa_grid x n_steps)
                            LW_I_to_L_map,                 # (np_alt_grid x np_toa_grid x n_steps) or float?
                            geometric_kernel,              # (np_alt_grid x np_toa_grid x n_steps)            
                            r_rel,                         # (np_alt_grid x np_toa_grid x n_steps x 3)
                            toa_weights,  # (np_toa_grid)
                            optimize='optimal',            # passing optimal path from einsum_path(..., optimize='optimal') does not seem to help
                            )                              # output is sized (np x n_steps x 3) (store_dF=False) or
                                                                #                 (np x n_toa x n_steps x 3) (store_dF=True)
        if wavelength=="sum":
            return (F_hist_SW + F_hist_LW)
        elif wavelength=="SW":
            return F_hist_SW
        elif wavelength=="LW":
            return F_hist_LW


def get_solar_incoming_day_hist(r_sun_day, TSI_1AU_day, stacked_r, grid: Grid, toa_grid:Grid=None): # toa_grid input for penumbra (TODO)
    # TODO: so this function can be called with a satellite r_hist, change grid input for stacked_r? shapes TBD...
    # NOTE: the same code now works with TSI_1AU sized (n_steps,) instead of single float
    grid_r_sun_hist, grid_d_sun_hist = get_grid_r_sun_hist(r_sun_day, stacked_r) #grid)
    TSI_grid_day_vec = TSI_1AU_day * (AU/grid_d_sun_hist)**2
    grid_u_sun_hist = get_grid_u_sun_hist(grid_r_sun_hist, grid_d_sun_hist)

    if grid.alt_km<=20: # TODO: what if we define TOA at 20 or other low altitudes? Make robust...
        # # Old version:
        zeroed_cos_theta_s_day_hist = get_zeroed_cos_theta_s_hist(grid_u_sun_hist, grid, earth_f=0)#toa_grid.flattening)
        grid_u_sun_hist_filtered = grid_u_sun_hist * (zeroed_cos_theta_s_day_hist[...,None]!=0)  # filter out eclipse

    else:
        # # instead: filter out eclipse for a general Earth shape and accounting for penumbra (Adhya et al 2003) 
        # #          so the function zeroed_cos_theta_s_day_hist is no longer needed, and grid input can then be removed

        # New version (Adhya et al 2003) # TODO: there are major issues with the paper but for now this seems good enough
        R_SUN = 695700
        if len(np.shape(stacked_r))==3: # np.shape(stacked_r)=(n_points,n_steps,3)
            #ri = np.cross(r_sun_day, stacked_r) # np.shape(r_sun_day)=(n_steps,3); 
            pass
        elif len(np.shape(stacked_r))==2: # np.shape(stacked_r)=(n_points,3)
            #stacked_r = stacked_r[:,None,:] # make a repmat instead until we fix the whole method
            stacked_r = np.repeat(stacked_r[:,None,:], len(r_sun_day), 1)

        ri = np.cross(r_sun_day, stacked_r)
        sp = np.cross(r_sun_day, ri)
        sp = sp / (get_norm_across_last_dim(sp)[...,None])
        rs1 = r_sun_day[None,:,:] + sp * R_SUN
        rs2 = r_sun_day[None,:,:] - sp * R_SUN
        b1, b2 = stacked_r - rs1, stacked_r - rs2
        b1_norm, b2_norm = get_norm_across_last_dim(b1), get_norm_across_last_dim(b2)
        det_1, dp1, dn1 = compute_adhya_intersection(stacked_r, b1, RE, RE*(1 - toa_grid.flattening))
        det_2, dp2, dn2 = compute_adhya_intersection(stacked_r, b2, RE, RE*(1 - toa_grid.flattening))
        assert(np.array_equal(np.isnan(det_1), np.isnan(det_2))) # should never happen

        shadow_f = np.ones_like(det_1)
        shadow_f[(det_1>=0) * (det_2<0) * (dp1 < b1_norm) * (dn1 < b1_norm)] = 0.5 # penumbra TODO: more rigorous 0-1 scale
        shadow_f[(det_1<0) * (det_2>=0) * (dp2 < b2_norm) * (dn2 < b2_norm)] = 0.5 # penumbra TODO: more rigorous 0-1 scale
        shadow_f[(det_1>0) * (det_2>0) * (dp1 < b1_norm) * (dp2 < b2_norm) * (dn1 < b1_norm) * (dn2 < b2_norm)] = 0 # umbra
        shadow_f[np.isnan(det_1) * (dp1 < b1_norm) * (dp2 < b2_norm) * (dn1 < b1_norm) * (dn2 < b2_norm)] = 0 # also umbra

        grid_u_sun_hist_filtered = grid_u_sun_hist * shadow_f[...,None]  

    solar_F_day_hist = -grid_u_sun_hist_filtered * TSI_grid_day_vec[...,None]  # # minus sign so that vecors are pointing AWAY from Sun; einsum proven to take the same

    return solar_F_day_hist 


def compute_adhya_intersection(a, b, p, q):

    a1, a2, a3 = a[...,0], a[...,1], a[...,2]
    b1, b2, b3 = b[...,0], b[...,1], b[...,2]

    #A = b1**2 * q**2 + b2**2 * q**2 + b3**2 * p**2
    #B = -2 * b2**2 * q**2 * a1 + 2 * b1 * q**2 * a2 - 2 * p**2 * b3**2 * a1 + 2 * p**2 * b1 * a3
    #C = q**2 * (b1**2 * a2**2 - b2**2 * a1**2 - 2 * b2 * b1 * a1 * a2) + p**2 * (b1**2 * a3**2 - b3**2 * a1**2 - 2 * b3 * b1 * a1 * a3) - b1**2 * p**2 * q**2
    
    # codex:
    A = b1**2*q**2 + b2**2*q**2 + b3**2*p**2

    B = (
        -2*b2**2*q**2*a1
        + 2*b1*b2*q**2*a2
        - 2*b3**2*p**2*a1
        + 2*b1*b3*p**2*a3
    )

    C = (
        q**2 * (b1*a2 - b2*a1)**2
        + p**2 * (b1*a3 - b3*a1)**2
        - b1**2*p**2*q**2
    )
    
    det = B**2 - 4*A*C
    intersec_idxs = (det>=0)

    xp, xn = np.zeros_like(det), np.zeros_like(det) # coordinates of the positive/negative solutions from the intersection eq.
    xp[intersec_idxs] = (-B[intersec_idxs] + np.sqrt(det[intersec_idxs])) / (2*A[intersec_idxs])
    xn[intersec_idxs] = (-B[intersec_idxs] - np.sqrt(det[intersec_idxs])) / (2*A[intersec_idxs])

    d_intersec_p = compute_intersec_d(xp, a, b, intersec_idxs)
    d_intersec_n = compute_intersec_d(xn, a, b, intersec_idxs)
    
    return det, d_intersec_p, d_intersec_n


def compute_intersec_d(x, a, b, idxs):

    a1, a2, a3 = a[...,0], a[...,1], a[...,2]
    b1, b2, b3 = b[...,0], b[...,1], b[...,2]

    y, z = np.zeros_like(x), np.zeros_like(x) # coordinates of the positive solution from the intersection eq.
    y[idxs] = a2[idxs] + b2[idxs]/b1[idxs] * (x[idxs] - a1[idxs])
    z[idxs] = a3[idxs] + b3[idxs]/b1[idxs] * (x[idxs] - a1[idxs])

    r_intersec = np.stack([x, y, z], axis=-1) # wrt Earth center
    r_intersec = r_intersec + b - a           # wrt to Sun tangence point

    return get_norm_across_last_dim(r_intersec)



def compute_F_hist_at_sat_r_hist(EEI_truth_name, sat_r_hist, sat_jd_hist, toa_grid: Grid):

    base_data_dir = os.path.join(MEDIA_DIR, 'true_EEI', EEI_truth_name)
    rad_config = rad_settings.radiation_settings_from_EEI_truth_name(EEI_truth_name)

    erp_F_LW_hist, erp_F_SW_hist = compute_earth_F_at_altitude(sat_jd_hist, rad_config, 
                                             sat_r_hist[None,...], toa_grid,
                                             store_dF=False, wavelength="split")
    print(np.shape(erp_F_LW_hist))
    #erp_F_LW_hist = np.sum(erp_F_LW_hist, axis=1)
    #erp_F_SW_hist = np.sum(erp_F_SW_hist, axis=1)

    print("done")

    return erp_F_LW_hist, erp_F_SW_hist 



def F_vec_to_Fr(F_hist, stacked_u):
    # F_hist shape:  (np x n_steps x 3)

    if len(np.shape(stacked_u))==2: 
        # case a)
        # stacked_u shape is (np x 3)
        ein_tag = 'itk,ik->it'

    elif len(np.shape(stacked_u))==3: 
        # case c) (and b?)
        # stacked_u shape is (np x n_steps x 3) (use for case b with np=1?)
        ein_tag = 'itk,itk->it'
    
    Fr_hist = -np.einsum(ein_tag, 
                         F_hist,           
                         stacked_u,       
                        )                 # 4x faster than np.sum + broadcast
    
    # IMPORTANT NOTE: the minus sign is because we define POSITIVE RADIAL flux as going IN.

    return Fr_hist


def get_r_sun_jd_hist(rad_config, jd_array):

    dir_r_sun = os.path.join(MEDIA_DIR, 'solar_ephemerides', rad_config["ephemerides"], 'daily_files')
    mid_day_jd_r_sun = np.load(os.path.join(dir_r_sun, 'jd_mid_day_array.npy'))

    day_idxs_jd_array = get_day_idxs_for_jd_array(jd_array, mid_day_jd_r_sun)
    unique_day_idxs = np.unique(day_idxs_jd_array)
    r_sun_ephem = np.vstack([get_r_sun_day_hist(rad_config["ephemerides"], mid_day_jd_r_sun[day_idx]) 
                            for day_idx in unique_day_idxs])

    # NOTE: the commented line retrieves the solar jd arrays from the MONTE-saved ephemerides, but times there are sliiiiiiightly different, presumably due to more nuanced time scale treatment
    #r_sun_ephem_jd = np.concatenate([get_jd_r_sun(ephem_tag, day_idx) for day_idx in unique_day_idxs])
    daily_jd_arrays = get_EEI_truth_daily_jd_arrays(rad_config["jd_interval"])
    r_sun_ephem_jd = np.concatenate([daily_jd_arrays[day_idx] for day_idx in unique_day_idxs])

    spline_sun = make_interp_spline(r_sun_ephem_jd, r_sun_ephem, k=3) # NOTE: k=3 seems to work better than k=1: for the fully circular case (constant d), resulting max-min is 0.02 instead of 1900 with k=1
    r_sun_at_jd_array = spline_sun(jd_array, extrapolate=True)

    return r_sun_at_jd_array


def get_grid_r_sun_hist(r_sun_day, stacked_r): 

    # r_sun_day: time series of vectors from the CoM of Earth to the CoM of Sun
    # returns: time hist of unit vectors pointint TO the Sun from all grid points

    # r_sun_day: time series of vectors from the CoM of Earth to the CoM of Sun
    # shape(r_sun_day) is (nt x 3)
    # shape(stacked_r) is (np x 3), where np
    
    if len(np.shape(stacked_r))==2:
        if len(stacked_r)!=len(r_sun_day): # not robust against the case where grid has 1440 points
            # case a)
            grid_r_sun_hist = (r_sun_day.T[None,:,:] - stacked_r[:,:,None]).transpose(0,2,1)
        else:
            # case b)
            grid_r_sun_hist = r_sun_day - stacked_r

    elif len(np.shape(stacked_r))==3:
        #case c)
        grid_r_sun_hist = (r_sun_day[None,:,:] - stacked_r)

    grid_d_sun_hist = get_norm_across_last_dim(grid_r_sun_hist) # np.sqrt(np.einsum('...j,...j->...', grid_r_sun_hist, grid_r_sun_hist))
    #grid_d_sun_hist = np.sqrt(np.einsum('ijk,ijk->ik',grid_r_sun_hist,grid_r_sun_hist)) # NOTE: equal to np.linalg.norm(grid_r_sun_hist, axis=1)

    return grid_r_sun_hist, grid_d_sun_hist


def get_grid_u_sun_hist(grid_r_sun_hist, grid_d_sun_hist):

    grid_u_sun_hist = grid_r_sun_hist / (grid_d_sun_hist[...,None]) # TODO: can einsum make it faster?

    return grid_u_sun_hist

def get_zeroed_cos_theta_s_hist(grid_u_sun_hist, grid: Grid, earth_f=None): # toa_grid: Grid
    # grid_u_sun_hist: unit vectors pointing FROM each grid point TO the Sun at every time step
    # earth_f: Earth flattening - only needed in case grid is at altitude (to properly compute shadows)
    if grid.alt_km > 20: assert(earth_f is not None)

    stacked_cos_theta_s_hist = np.einsum('itk,ik->it', grid_u_sun_hist, grid.stacked_grid_u) # same as: np.sum(grid_u_sun_hist * grid.stacked_grid_u[:,:,None], axis=1)

    if (grid.alt_km > 20) and (earth_f != 0): # TODO: will this be necessary at all? (Compute vector F instead)
        NotImplementedError()
    else:
        # Now valid just for TOA grids (general cos_lim breaks if TOA alt is <0, and for altitude we compute eclipse more accurately now anyway)
        # cos_lim = get_cos_theta_s_lim(grid.alt_km)
        cos_lim = 0
        stacked_cos_theta_s_hist[stacked_cos_theta_s_hist < cos_lim] = 0
    
    return stacked_cos_theta_s_hist


def get_cos_theta_s_lim(altitude_km):
    return -np.sqrt(1 - (RE/(RE + altitude_km))**2)     # this minus sign only if u_sun points towards the Sun
    # this formula is also valid for a non-infinitely-far-away Sun but not for a non-spherical Earth


def apply_SFF_rotation(R_ECEF_to_SunFrame_hist, stacked_vectors, mode="ECEF_to_SFF"):
    
    # stacked_vectors: list of vectors to be rotated at every epoch given by R_ECEF_to_SunFrame_hist
    # if stacked_vectors shape is (np x 3), same vectors are rotated at every epoch. 
    # if stacked_vectors shape is (np x 3 x n_steps), each set of vectors [:,:,i] is rotated at its corresponding time step i

    if mode=="ECEF_to_SFF":
        R_array = np.asarray(R_ECEF_to_SunFrame_hist)
    elif mode=="SFF_to_ECEF":
        R_array = np.asarray(R_ECEF_to_SunFrame_hist).transpose(0,2,1)
    else:
        print("mode has to be either ECEF_to_SFF or SFF_to_ECEF")

    if len(np.shape(stacked_vectors))==2:
        # stacked r shape: (np x 3)
        einsum_tag = 'tik,jk->jti'

    elif len(np.shape(stacked_vectors))==3:
        # stacked r shape: (np x 3 x n_steps)
        einsum_tag = 'tik,jtk->jti'

    return np.einsum(einsum_tag, R_array, stacked_vectors)

    """
    NOTE that the above is exactly the same as below, but faster and more compact

    rotated_v_hist = np.zeros((len(stacked_r), len(R_array), 3))
    if len(np.shape(stacked_vectors))==2:
        for i, R_mat in enumerate(R_array):
           rotated_v_hist[:,i,:] = ((R_mat) @ (stacked_r.T)).T
    
    elif len(np.shape(stacked_vectors))==3:
        for i, R_mat in enumerate(R_array):
           rotated_v_hist[:,i,:] = ((R_mat) @ (stacked_vectors[:,:,i].T)).T
    
    """


##############################################################################################
# helper functions: ##########################################################################
##############################################################################################

def get_EEI_truth_daily_jd_arrays(EEI_truth_jd_interval):

    # rad_config = rad_settings.radiation_settings_from_EEI_truth_name(EEI_truth_name)
    # mid_day_jd_array = mid_day_jd_array_from_jd_interval(rad_config["jd_interval"])
    mid_day_jd_array = mid_day_jd_array_from_jd_interval(EEI_truth_jd_interval)

    step_days = STEP_MINUTES / (60*24)
    daily_jd_arrays = [get_day_jd_array(mid_day_jd, step_days) for mid_day_jd in mid_day_jd_array]
    
    return daily_jd_arrays

def get_day_jd_array(mid_day_jd, step_days=STEP_MINUTES/(60*24)):
    return np.arange(mid_day_jd-0.5, mid_day_jd+0.5, step_days)


def mid_day_jd_array_from_jd_interval(jd_interval):

    edges_jd_array = np.arange(jd_interval[0], jd_interval[-1]+1, 1)
    mid_day_jd_array = (edges_jd_array[1:] + edges_jd_array[:-1]) / 2

    return mid_day_jd_array

def get_mid_day_jd_array(EEI_truth_name):
    rad_config = rad_settings.radiation_settings_from_EEI_truth_name(EEI_truth_name)
    return mid_day_jd_array_from_jd_interval(rad_config["jd_interval"])


def get_R_SunFrame_hist(ephem_tag, mid_day_jd): 

    dir_r_sun = os.path.join(MEDIA_DIR, 'solar_ephemerides', ephem_tag, 'daily_files')
    jd_r_sun = np.load(os.path.join(dir_r_sun, 'jd_mid_day_array.npy'))

    day_idx_r_sun = find_day_idx(jd_r_sun, mid_day_jd)
    R_hist_day = np.load(os.path.join(dir_r_sun, 'R_ECEF_to_SunFrame_day_'+str(day_idx_r_sun)+'.npy'))

    return R_hist_day


def find_day_idx(jd_r_sun, mid_day_jd):
    day_idx_r_sun = np.where(np.isin(jd_r_sun, mid_day_jd))[0][0]

    return day_idx_r_sun

def get_day_idxs_for_jd_array(jd_array, mid_day_jd_array):
    return np.floor(jd_array - mid_day_jd_array[0] + 0.5).astype(int)


def get_r_sun_day_hist(ephem_tag, mid_day_jd):
    dir_r_sun = os.path.join(MEDIA_DIR, 'solar_ephemerides', ephem_tag, 'daily_files')
    jd_r_sun = np.load(os.path.join(dir_r_sun, 'jd_mid_day_array.npy'))
    day_idx_r_sun = find_day_idx(jd_r_sun, mid_day_jd)
    r_sun_day = np.load(os.path.join(dir_r_sun, 'r_Sun_ECEF_day_'+str(day_idx_r_sun)+'.npy'))

    return r_sun_day

def get_jd_r_sun(ephem_tag, day_idx):
    dir_r_sun = os.path.join(MEDIA_DIR, 'solar_ephemerides', ephem_tag, 'daily_files')
    return np.load(os.path.join(dir_r_sun, 'jd_array_day_'+str(day_idx)+'.npy'))



def get_expanded_ae_maps(datestr, grid: Grid, rad_config_dict: dict):

    a_sh_map, e_sh_map = load_sh_maps(datestr, rad_config_dict)

    full_lat, full_lon = grid.stacked_grid_latlon.T
    normalization = rad_config_dict['sh_normalization']

    if 'Albedo' not in rad_config_dict["earth_components"]:
        a_map = np.zeros(grid.n_points)
    else:
        print(f"Expanding albedo map (normalization {normalization})...")
        a_map = expand_sh(a_sh_map, full_lon, full_lat, normalization)

    if 'Thermal' not in rad_config_dict["earth_components"]:
        e_map = np.zeros(grid.n_points)
    else:
        print(f"Expanding emissivity map (normalization {normalization})...")
        e_map = expand_sh(e_sh_map, full_lon, full_lat, normalization)

    return a_map, e_map


def load_sh_maps(datestr, rad_config_dict):

    mode = rad_config_dict['sh_mode']
    Nmax = rad_config_dict['Nmax']
    
    a_sh_map, e_sh_map = rad_settings.get_ae_sh_maps_numpy_new(mode, datestr)
    a_sh_map = a_sh_map[:, :(Nmax+1), :(Nmax+1)]
    e_sh_map = e_sh_map[:, :(Nmax+1), :(Nmax+1)]

    return a_sh_map, e_sh_map


def get_smooth_daily_ae_hist(out_jd_array, rad_config, grid:Grid=None):

    method=rad_config["time_interp"]
    if method=="map_interp":
        assert grid is not None
        full_lat, full_lon = grid.stacked_grid_latlon.T
    else:
        assert method=="sh_interp", "Method must be either 'map_interp' or 'sh_interp'"

    # step 1: load few previous and next days
    n = 3 # number of days to load to built interpolator

    all_mid_day_jds = np.unique(np.round(out_jd_array))
    #jd_array = mid_day_jd + np.arange(-n, n+1, 1)
    jd_array = np.arange(np.min(all_mid_day_jds) - n, np.max(all_mid_day_jds) + n + 1, 1)
    first_jd = rad_config["jd_interval"][0] + 0.5
    last_jd = rad_config["jd_interval"][1] - 0.5
    jd_array = np.delete(jd_array, jd_array<first_jd)
    jd_array = np.delete(jd_array, jd_array>last_jd)

    if first_jd in jd_array: 
        jd_array = np.insert(jd_array, 0, rad_config["jd_interval"][0])
    if last_jd in jd_array:
        jd_array = np.append(jd_array, rad_config["jd_interval"][1] - 1e-5)

    # step 2: load (and expand?) a&e maps
    all_a = [None] * len(jd_array)
    all_e = [None] * len(jd_array)

    for i, jd in enumerate(jd_array):
        datestr = jd_to_mmddyyyy(jd)
        a, e = load_sh_maps(datestr, rad_config)
        if method=="map_interp":
            # a = np.zeros(len(full_lat))
            # e = np.zeros(len(full_lat))
            a = expand_sh(a, full_lon, full_lat, rad_config['sh_normalization'])
            e = expand_sh(e, full_lon, full_lat, rad_config['sh_normalization'])
        all_a[i] = a
        all_e[i] = e

    # step 3: make splines and interpolate
    spline_a = make_interp_spline(jd_array, np.stack(all_a), k=3)
    spline_e = make_interp_spline(jd_array, np.stack(all_e), k=3)
    # TODO: check what happens with first half of first day and last half of last day

    a_hist = spline_a(out_jd_array, extrapolate=False)
    e_hist = spline_e(out_jd_array, extrapolate=False)

    if rad_config["time_interp"]=="sh_interp": # TODO: check shape
        a_hist = expand_daily_map_hist(a_hist, grid, rad_config['sh_normalization'])
        e_hist = expand_daily_map_hist(e_hist, grid, rad_config['sh_normalization'])

    return a_hist.T, e_hist.T


def expand_daily_map_hist(sh_hist, grid: Grid, normalization):

    full_lat, full_lon = grid.stacked_grid_latlon.T
    n_steps = np.shape(sh_hist)[0]
    map_hist = np.zeros((grid.n_points, n_steps))

    for i in range(n_steps):
        progress_bar(i, n_steps)
        map_hist[:,i] = expand_sh(sh_hist[i], full_lon, full_lat, normalization)

    return map_hist

# deprecated function
def irradiance_to_radiance(emission_hist, rad_type, r_alt=None, toa_grid_u_sun_hist=None, ADM_model=None):

    # emission_hist:        (np_toa x n_t)      [W/m^2] - integrated irradiance history of all toa grid points
    # toa_grid_u_sun_hist:  (np_toa x 3 x n_t)  []      - unit vectors from toa grid points to Sun
    # TODO: r_hist

    # RETURNS:
    # L_hist:               (np_toa x n_t)      [W/m^2sr] radiance history of all toa grid points

    if ADM_model is None: # Lambertian emission model
        return (1/np.pi) * emission_hist

def get_irradiance_to_radiance_map(stacked_r, jd_array, toa_grid: Grid, rad_config, rad_type, ADM_model=None):

    # INPUTS:
    # stacked_r:    shape (np, n_steps_r, 3) - n_steps_r can be either 1 or n_steps
    # jd_array:     shape (n_steps,)
    # toa_grid:     stacked_grid_r has shape (np_toa, 3)
    n_steps_r = np.shape(stacked_r)[1]
    n_steps = len(jd_array)
    assert((n_steps_r==1) or (n_steps_r==n_steps))

    if ADM_model is None:
        I_to_L = (1/np.pi)

    else:
        # # TODO: make sun stuff an external function to be called by the other methods that include duplication of this routine
        # r_sun_hist = get_r_sun_jd_hist(rad_config, jd_array)
        # grid_r_sun_hist, grid_d_sun_hist = get_grid_r_sun_hist(r_sun_hist, toa_grid.stacked_grid_r) #grid)
        # grid_u_sun_hist = get_grid_u_sun_hist(grid_r_sun_hist, grid_d_sun_hist)
        # zeroed_cos_theta_s_day_hist = get_zeroed_cos_theta_s_hist(grid_u_sun_hist, toa_grid)

        I_to_L = np.ones((len(stacked_r), toa_grid.n_points, n_steps)) * (1/np.pi)


    # OUTPUT:
    # general form:     shape (np, np_toa, n_steps)
    # if no ADM:        just a float? check whether it makes the later einsum break...

    return I_to_L


# to be revised:
def get_window_avg_maps(EEI_truth_name, jd_windows, map_name, grid: Grid):

    truth_jd_arrays = get_EEI_truth_daily_jd_arrays(EEI_truth_name)
    base_data_dir = os.path.join(MEDIA_DIR, 'true_EEI', EEI_truth_name)
    avg_map_array = [None] * len(jd_windows)

    for i, jd_window in enumerate(jd_windows):

        idxs = [(jd_vec_day[0]>=jd_window[0] and jd_vec_day[1]<=jd_window[1]) for jd_vec_day in truth_jd_arrays]
        idxs = np.squeeze(np.argwhere(idxs))
        all_maps = np.zeros((grid.n_points, len(idxs)))
        
        for j, day_idx in enumerate(idxs):
            file_names, _ = get_file_names_and_existence(base_data_dir, day_idx, grid, create_dirs=False)

            #true_avg = np.load(os.path.join(truth_dir, file_names['daily_avg_net_800km']+'.npy'))
            all_maps[:,j] = grid.vectorize_if_needed(np.load(os.path.join(base_data_dir, 
                                                                          file_names[map_name]+'.npy')))

        avg_map_array[i] = grid.compute_time_avg_map(all_maps) #np.nanmean(all_maps, axis=1)

    return avg_map_array

##############################################################################################
# file storage functions #####################################################################
##############################################################################################

def get_subdir_EEI_truth(EEI_truth, series_type, grid_name="grid_quad_n131"): # TODO: remove default 131
    
    subdir = os.path.join(MEDIA_DIR, 'true_EEI', EEI_truth, grid_name, 
                              series_type)
    return subdir

def get_file_names_and_existence(base_data_dir, day_idx, grid: Grid, create_dirs=True):
    
    all_out_file_types = get_required_files(grid)
    grid_data_dir = os.path.join(base_data_dir, grid.grid_name)
    
    if create_dirs:
        # create directories if they don't exist:
        for ftype in all_out_file_types:
            os.makedirs(os.path.join(grid_data_dir, ftype), exist_ok=True)

    all_file_names = {ftype: os.path.join(grid_data_dir, ftype, 'day_'+str(day_idx)) for ftype in all_out_file_types}
    file_existences = {ftype: (os.path.exists(fname_full+'.txt') or os.path.exists(fname_full+'.npy')) 
                        for ftype, fname_full in zip(all_out_file_types, all_file_names.values())}
    
    return all_file_names, file_existences

"""toa: ['daily_hist_LW_toa', 'daily_hist_SW_toa', 'daily_hist_net_toa', 'daily_hist_net_toa_SFF',
            'daily_avg_net_toa', 'daily_hist_net_toa_surf_avg',
            ], """

def save_toa_files(toa_LW_day_hist, toa_SW_day_hist, net_toa_day_hist, net_toa_day_hist_SFF,
                   base_data_dir, day_idx, grid: Grid,
                   R_ECEF_to_SunFrame_day_hist=None): #, n_cores=4, save_SFF_hist=True):

    all_file_names, file_existences = get_file_names_and_existence(base_data_dir, day_idx, grid)
    
    if not(file_existences['daily_hist_LW_toa']):
        print("Saving daily_hist_LW_toa file...")
        np.save(all_file_names['daily_hist_LW_toa'], toa_LW_day_hist)

    if not(file_existences['daily_hist_SW_toa']):
        print("Saving daily_hist_SW_toa file...")
        np.save(all_file_names['daily_hist_SW_toa'], toa_SW_day_hist)

    if (net_toa_day_hist_SFF is not None) and not(file_existences.get('daily_hist_net_toa_SFF_accurate')): #and save_SFF_hist:
        print("Saving daily_hist_net_toa file Sun-Fixed Frame...")
        #net_toa_day_hist_SFF = field_hist_rotation_multiproc(net_toa_day_hist, R_ECEF_to_SunFrame_day_hist, 
        #                                                     grid, n_cores=n_cores)
        np.save(all_file_names['daily_hist_net_toa_SFF'], net_toa_day_hist_SFF)
        # we're here - seems like interpolating will be easiest, but let's see later. For now lets go to altitude computation

    if not(file_existences.get('daily_avg_net_toa', True)): 
        print("Saving daily_avg_net_toa file...")
        net_toa_daily_avg = grid.compute_time_avg_map(net_toa_day_hist)
        np.save(all_file_names['daily_avg_net_toa'], net_toa_daily_avg)

    if not(file_existences['daily_hist_net_toa_surf_avg']):
        print("Saving daily_hist_net_toa_surf_avg file...")
        toa_surf_avg_day_hist = grid.compute_surf_integral(net_toa_day_hist, average=True)
        np.save(all_file_names['daily_hist_net_toa_surf_avg'], toa_surf_avg_day_hist)