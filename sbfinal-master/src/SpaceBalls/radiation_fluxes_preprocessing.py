import os, sys
import numpy as np
from astropy.time import Time
import astropy.units as u
import multiprocessing
from multiprocessing import Pool
from scipy.interpolate import make_interp_spline
from scipy.spatial.transform import Rotation
import time

from SpaceBalls.paths import CONFIG_DIR, MEDIA_DIR
#sys.path.insert(0, str(CONFIG_DIR.parent))  # parent of 'config'
import SpaceBalls.radiation_settings as rad_settings
from SpaceBalls.sph_meshing import Grid, RegularLatLonGrid, QuadratureGrid, expand_sh, field_hist_rotation_multiproc

import SpaceBalls.constants as constants
from SpaceBalls.utils import get_norm_across_last_dim, progress_bar, jd_to_mmddyyyy, get_jd_to_build_interp
from SpaceBalls.plotter import Plotter
from SpaceBalls.ADM_manager import load_erbe_sw_adm, get_erbe_scene_types, load_erbe_lw_adm

AU = constants.astronomical_unit(units='km')
RE = constants.earth_radius(units='km')
LIGHT_SPEED = constants.light_speed()
STEP_MINUTES = 1 # DO NOT CHANGE - must be equal to the one used for the files in solar_ephemerides
ERBE_SW_ADM = load_erbe_sw_adm(os.path.join(CONFIG_DIR, 'earth', 'ADMs', 'erbe_SW_ADM.dat'))
ERBE_LW_ADM = load_erbe_lw_adm(os.path.join(CONFIG_DIR, 'earth', 'ADMs', 'erbe_LW_ADM.dat'))
            
REQUIRED_FILES = {
    "toa": ['daily_hist_LW_toa', 'daily_hist_SW_toa', 'daily_hist_net_toa', 'daily_hist_net_toa_SFF_accurate',
            'daily_avg_net_toa', 'daily_hist_net_toa_surf_avg', # 'daily_hist_net_toa_lat_avg'
            ],
    "altitude": ['daily_hist_net_F_{alt_km}km', 'daily_hist_net_F_{alt_km}km_SFF_accurate', 'daily_avg_net_Fr_{alt_km}km_SFF_accurate',
                 'daily_avg_net_F_{alt_km}km', 'daily_avg_net_F_{alt_km}km_SFF_accurate',
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

    key = 'toa' if grid.is_TOA() else 'altitude'
    files = REQUIRED_FILES[key].copy()

    if grid.alt_km>0:
        files = [f.replace('{alt_km}', str(grid.alt_km)) for f in files]
    

    return files


def compute_radiation_maps(EEI_truth_name, grid: Grid, selected_days_idxs=None, compute_SFF=True):

    base_data_dir = os.path.join(MEDIA_DIR, 'true_EEI', EEI_truth_name)
    
    rad_config = rad_settings.radiation_settings_from_EEI_truth_name(EEI_truth_name)
    daily_jd_arrays = get_EEI_truth_daily_jd_arrays(rad_config["jd_interval"])
    mid_day_jd_array = mid_day_jd_array_from_jd_interval(rad_config["jd_interval"])
    n_days = len(mid_day_jd_array)

    if  (selected_days_idxs is None) or (len(selected_days_idxs)==0): # (grid.alt_km==0) or
        compute_all_days = True
        idxs_days_to_loop = range(n_days)
        rm_daily_hists = True if grid.n_points>3000 else False # 5810
            
    else:
        compute_all_days = False
        daily_jd_arrays = [daily_jd_arrays[i] for i in selected_days_idxs]
        idxs_days_to_loop = selected_days_idxs
        rm_daily_hists = False
    print(f"Compute all days: {compute_all_days}")

    for day_idx, jd_array in zip(idxs_days_to_loop, daily_jd_arrays):

        print(f"Now computing: daily hist for day {day_idx} at {grid.alt_km}km")
        n_steps = len(jd_array)
        mid_day_jd = mid_day_jd_array[day_idx]
        #TSI_1AU_day = rad_settings.get_TSI_1AU(mid_day_jd, rad_config["TSI_source"])
        R_ECEF_to_SunFrame_day_hist = get_R_SunFrame_hist(rad_config["ephemerides"], mid_day_jd)

        rm_SFF = False if compute_SFF else True
        file_names, file_existences = get_file_names_and_existence(base_data_dir, day_idx, grid,
                                                                   rm_daily_hists, rm_SFF)
        #if not(file_existences[f"daily_hist_net_toa_SFF_accurate"]) and (day_idx in selected_days_idxs):
        #    compute_SFF = True # At TOA, the only reason we want the SFF map is for animations
        #else:
        #    compute_SFF = False
        net_toa_day_hist_SFF = None # TODO: TOA SFF

        if grid.is_TOA() and not(all(file_existences.values())):
            toa_LW_day_hist, toa_SW_day_hist, net_toa_day_hist = get_daily_hist_toa(base_data_dir, 
                                                                                    day_idx, 
                                                                                    mid_day_jd, 
                                                                                    jd_array, 
                                                                                    rad_config, 
                                                                                    grid)
            save_toa_files(toa_LW_day_hist, toa_SW_day_hist, net_toa_day_hist, net_toa_day_hist_SFF,
                   file_names, file_existences, grid)

        elif grid.alt_km>0 and not(all(file_existences.values())):
            
            daily_hist_net_F_altitude = get_daily_hist_net_at_altitude(
                                    base_data_dir, day_idx, mid_day_jd, jd_array, rad_config, grid)
            
            if compute_SFF and not(file_existences[f"daily_hist_net_F_{grid.alt_km}km_SFF_accurate"]):
                daily_hist_net_F_altitude_SFF = get_daily_hist_net_at_altitude(
                                                    base_data_dir, day_idx, mid_day_jd, jd_array, rad_config, grid,
                                                    sff=True)
            else:
                daily_hist_net_F_altitude_SFF = None

            save_altitude_files(daily_hist_net_F_altitude, daily_hist_net_F_altitude_SFF,
                                file_names, file_existences, grid)


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
    
    



def get_daily_hist_net_at_altitude(base_data_dir, day_idx, mid_day_jd, day_jd_array, 
                                   rad_config: dict, grid: Grid, sff:bool=False):
    
    all_file_names, file_existences = get_file_names_and_existence(base_data_dir, day_idx, grid)
    
    if file_existences.get('daily_hist_net_F_'+str(grid.alt_km)+'km', False):
        pass
        # load daily_hist_net_F_altitude and compute the rest (this option should not really ever happen if the preproc scripts have been run correctly)
    else:
        toa_grid = get_toa_grid_from_rad_config(rad_config, quadrature_order=201)

        if sff:
            R_ECEF_to_SunFrame_day_hist = get_R_SunFrame_hist(rad_config["ephemerides"], mid_day_jd)
            stacked_r = apply_SFF_rotation(R_ECEF_to_SunFrame_day_hist, grid.stacked_grid_r, mode="SFF_to_ECEF")
            # we want the grid nodes to be static in the SFF, so the resulting hist is how they move in ECEF
        else:
            stacked_r = grid.stacked_grid_r[:,None,:]

        r_sun_day = get_r_sun_day_hist(rad_config["ephemerides"], mid_day_jd)
        TSI_1AU_day = rad_settings.get_TSI_1AU(day_jd_array, rad_config['TSI_source'])
        daily_hist_solar_F_altitude = get_solar_incoming_day_hist(r_sun_day, TSI_1AU_day, 
                                                                  stacked_r, grid, toa_grid,
                                                                  penumbra_method=rad_config.get('penumbra_method'))
        if sff:
            daily_hist_solar_F_altitude = apply_SFF_rotation(R_ECEF_to_SunFrame_day_hist, daily_hist_solar_F_altitude, "ECEF_to_SFF")
        # daily_hist_solar_Fr_altitude = F_vec_to_Fr(daily_hist_solar_F_altitude, grid.stacked_grid_u)
        # Plotter.plot_geo_data_new(daily_hist_solar_Fr_altitude[:,0], grid, 
        #                           file_name='sun_Fr_ECEF_0', add_coastlines=True, make_symmetric_cmap=True)
        # Plotter.plot_geo_data_new(daily_hist_solar_Fr_altitude[:,700], grid, 
        #                           file_name='sun_Fr_ECEF_700', add_coastlines=True, make_symmetric_cmap=True)

        
        daily_hist_earth_F_altitude = compute_earth_F_at_altitude(day_jd_array, rad_config, stacked_r, toa_grid)

        if sff:
            daily_hist_earth_F_altitude = apply_SFF_rotation(R_ECEF_to_SunFrame_day_hist, daily_hist_earth_F_altitude, "ECEF_to_SFF")
        # daily_hist_earth_Fr_altitude = F_vec_to_Fr(daily_hist_earth_F_altitude, grid.stacked_grid_u) # grid is already defined in SFF
        # Plotter.plot_geo_data_new(daily_hist_earth_Fr_altitude[:,700], grid, 
        #                           file_name='earth_Fr_ECEF_700', add_coastlines=True, make_symmetric_cmap=False)


        daily_hist_net_F_altitude = daily_hist_solar_F_altitude + daily_hist_earth_F_altitude


    return daily_hist_net_F_altitude


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

    _, _, grid_u_sun_hist = get_grid_r_sun_hist(r_sun_hist, grid.stacked_grid_r) 
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
    if ADM_model is None: nt = n_steps_r 
    else: nt = n_steps  # nt is the most constraining one to build the arrays after this

    max_ram_GB = 8
    max_nt = 8e9 * max_ram_GB / (64 * toa_grid.n_points * n_p * 3)
    n_chunks = np.max([1, int(np.floor(nt / max_nt))])
    if n_chunks>n_steps: n_chunks = n_steps # temporary patch if memory resists
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
        #toa_LW_hist, toa_SW_hist = np.random.rand(toa_grid.n_points, len(jd_array)), np.random.rand(toa_grid.n_points, len(jd_array))

        stacked_r_i = stacked_r if n_steps_r==1 else stacked_r_chunks[i]

        print("Computing flux...")
            # 1. Compute all relative vectors
        r_rel = stacked_r_i[:,None,:,:] - toa_grid.stacked_grid_r[None,:,None,:] 
            # 2. Compute norm of relative vectors
        r_rel_norm_2 = np.einsum('ijtk,ijtk->ijt', r_rel, r_rel)   # norm of all r_rel squared
            # 3. Compute all geometric kernels
        geometric_kernel = np.einsum('ijtk,jk->ijt', r_rel, toa_grid.stacked_grid_u) / (r_rel_norm_2**2) # this is cos(alpha)/(r_rel_norm**3)
        np.maximum(geometric_kernel, 0, out=geometric_kernel) # set negative cosines(alpha) to zero
            # 4. Compute flux vector integral
        print(f"Getting ADMs...")
        LW_I_to_L_map, SW_I_to_L_map = get_irradiance_to_radiance_map(
            r_rel, r_rel_norm_2, geometric_kernel, jd_array,
            toa_grid, rad_config, ADM_model=ADM_model, rad_type="both"
        )
        del r_rel_norm_2
        print(f"Computing integral with einsum...")
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
        time_axis = 2 if store_dF else 1
        emission_F_LW_hist = np.concatenate(emission_F_LW_hist, axis=time_axis) # np.vstack(emission_F_LW_hist)
        emission_F_SW_hist = np.concatenate(emission_F_SW_hist, axis=time_axis) # np.vstack(emission_F_SW_hist)
        return emission_F_LW_hist, emission_F_SW_hist

    else:
        emission_F_hist = np.concatenate(emission_F_hist, axis=1) # np.vstack(emission_F_hist) # we're here - vstack does not work if len of time axis is 1??
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


def get_solar_incoming_day_hist(r_sun_day, TSI_1AU_day, stacked_r, grid:Grid=None, 
                                toa_grid:Grid=None, penumbra_method=None, grid_bool=True): # toa_grid input for penumbra (TODO)
    # TODO: so this function can be called with a satellite r_hist, change grid input for stacked_r? shapes TBD...
    # NOTE: the same code now works with TSI_1AU sized (n_steps,) instead of single float
    # Here a 2-D input is always a static grid, even if n_points == n_steps.
    stacked_r = np.asarray(stacked_r)
    if stacked_r.ndim == 2:
        stacked_r = stacked_r[:, None, :]
    _, grid_d_sun_hist, grid_u_sun_hist = get_grid_r_sun_hist(r_sun_day, stacked_r)
    TSI_grid_day_vec = TSI_1AU_day * (AU/grid_d_sun_hist)**2

    if grid_bool and grid.is_TOA(): # TOA // TODO: is this robust to different TOA altitudes? Shouldn't be larger than 20...
        # # Old version:
        zeroed_cos_theta_s_day_hist = get_zeroed_cos_theta_s_hist(grid_u_sun_hist, grid, earth_f=0)#toa_grid.flattening)
        grid_u_sun_hist_filtered = grid_u_sun_hist * (zeroed_cos_theta_s_day_hist[...,None]!=0)  # filter out eclipse

    else:
        # # instead: filter out eclipse for a general Earth shape and accounting for penumbra (Adhya et al 2003) 
        # #          so the function zeroed_cos_theta_s_day_hist is no longer needed, and grid input can then be removed
        shadow_f = compute_shadow_f(stacked_r, r_sun_day, toa_grid, 
                                    penumbra_method=penumbra_method)
        grid_u_sun_hist_filtered = grid_u_sun_hist * shadow_f[...,None]  

    solar_F_day_hist = -grid_u_sun_hist_filtered * TSI_grid_day_vec[...,None]  # # minus sign so that vecors are pointing AWAY from Sun; einsum proven to take the same

    return solar_F_day_hist 


def compute_shadow_f(stacked_r, r_sun_day, toa_grid: Grid,
                     *, penumbra_method="0.5"):
    """Return illumination factors with shape (n_points, n_steps).

    Positions are geocentric, in km, in a common frame with Earth's polar
    axis along z (normally ECEF). ``stacked_r`` is (n_points, 3) or
    (n_points, n_steps, 3); a singleton time axis is also accepted. The Sun
    history is (n_steps, 3). The occulting ellipsoid has equatorial radius
    RE + toa_grid.alt_km and polar radius scaled by 1 - flattening.

    ``smooth_shadow=False`` uses a point Sun and returns 0 or 1. True uses
    two apparent solar limb rays in the Earth-observer-Sun plane, following
    the Adhya approach, and returns 0, 0.5, or 1. Despite its historical
    name, this switch does not produce a continuous penumbra transition.
    Two rays approximate an oblate Earth's full apparent silhouette; they
    are not an exact solar-disc/ellipse overlap calculation.

    Surface observers looking outward are illuminated; inward rays are
    occulted. A tangent contact away from the observer counts as occulted.
    Interior observers are occulted. Partial and annular eclipses both use
    the penumbra method (currently only "constant", giving 0.5).
    """
    smooth_shadow = False if penumbra_method is None else True

    if (penumbra_method is not None) and (penumbra_method != "0.5"):
        raise ValueError("Only penumbra_method='0.5' is implemented.")
    observers = np.asarray(stacked_r, dtype=float)
    sun = np.asarray(r_sun_day, dtype=float)
    if observers.ndim not in (2, 3) or observers.shape[-1] != 3:
        raise ValueError("stacked_r must have shape (n_points, 3) or (n_points, n_steps, 3).")
    if sun.ndim != 2 or sun.shape[-1] != 3:
        raise ValueError("r_sun_day must have shape (n_steps, 3).")
    if observers.ndim == 2:
        observers = observers[:, None, :]
    if observers.shape[1] not in (1, len(sun)):
        raise ValueError("The observer time axis must have length 1 or n_steps.")
    if not np.all(np.isfinite(observers)) or not np.all(np.isfinite(sun)):
        raise ValueError("Observer and Sun positions must be finite.")
    p = RE + toa_grid.alt_km
    flattening = toa_grid.flattening
    if not np.isfinite(p) or p <= 0 or not np.isfinite(flattening) or not 0 <= flattening < 1:
        raise ValueError("The occulting radius must be positive and 0 <= flattening < 1.")
    q = p * (1 - flattening)
    axes = np.array([p, p, q])
    if np.any(np.sum((sun / axes)**2, axis=-1) <= 1):
        raise ValueError("The Sun centre must be outside the occulting ellipsoid.")

    # Broadcasting avoids copying a static grid for every time step.
    observers, sun = np.broadcast_arrays(observers, sun[None, :, :])
    to_sun = sun - observers
    distance = get_norm_across_last_dim(to_sun)
    solar_radius = 695700.0 if smooth_shadow else 0.0
    if np.any(distance <= solar_radius):
        raise ValueError("Observers must be outside the Sun (and distinct from its centre).")
    if not smooth_shadow:
        return (~_shadow_segment_is_blocked(observers, to_sun, p, q)).astype(float)

    u = to_sun / distance[..., None]
    # Direction toward Earth's apparent centre, perpendicular to the Sun
    # sightline. Cross products avoid subtracting nearly parallel vectors.
    transverse = np.cross(u, np.cross(-observers, u))
    transverse_norm = get_norm_across_last_dim(transverse)
    earth_distance = get_norm_across_last_dim(observers)
    cos_alpha = np.sqrt(1 - (solar_radius / distance)**2)
    # A small apparent Earth can lie entirely within the solar disc while
    # both limb rays AND the Sun-centre ray miss it (off-axis antumbra).
    earth_centre_in_disc = ((transverse_norm <= earth_distance * solar_radius / distance)
                            & (np.einsum('...i,...i->...', observers, u) < 0)
                            & (earth_distance < distance * cos_alpha))
    aligned = transverse_norm <= 32 * np.finfo(float).eps * np.maximum(earth_distance, p)
    if np.any(aligned):
        # The plane is undefined on the shadow axis. Use the projected
        # polar direction (the smaller silhouette radius of an oblate
        # Earth); for a polar sightline any equatorial direction suffices.
        aligned_u = u[aligned]
        fallback = np.cross(aligned_u, np.cross([0., 0., 1.], aligned_u))
        fallback_norm = np.linalg.norm(fallback, axis=-1)
        polar = fallback_norm <= 32 * np.finfo(float).eps
        fallback[polar] = np.cross(aligned_u[polar], np.cross([1., 0., 0.], aligned_u[polar]))
        transverse[aligned] = fallback
        transverse_norm[aligned] = np.linalg.norm(fallback, axis=-1)
    transverse /= transverse_norm[..., None]

    # Exact tangent points on a spherical Sun, measured from the observer:
    # d*cos(alpha)^2*u +/- R_sun*cos(alpha)*transverse, sin(alpha)=R_sun/d.
    limb_centre = to_sun * cos_alpha[..., None]**2
    limb_offset = transverse * (solar_radius * cos_alpha)[..., None]
    near_blocked = _shadow_segment_is_blocked(observers, limb_centre + limb_offset, p, q)
    far_blocked = _shadow_segment_is_blocked(observers, limb_centre - limb_offset, p, q)
    umbra = near_blocked & far_blocked
    partial = (near_blocked | far_blocked | earth_centre_in_disc) & ~umbra
    shadow_f = np.ones(distance.shape)
    shadow_f[umbra] = 0.0
    if np.any(partial):
        shadow_f[partial] = _compute_penumbra_f(
            observers[partial], sun[partial], p, q, method=penumbra_method)
    return shadow_f


def _compute_penumbra_f(observers, sun, p, q, *, method):
    """Extension point for partial-eclipse flux, evaluated only in penumbra.

    Future methods can use these geometries for circular-disc overlap,
    a locally straight Earth limb, or solar-disc quadrature against the
    ellipsoid. Keep the eclipse classification independent of that choice.
    """
    if method == "0.5":
        return np.full(observers.shape[:-1], 0.5)
    raise ValueError(f"Unknown penumbra method: {method!r}")


def _shadow_segment_is_blocked(a, b, p, q):
    """Whether the segment a + t*b, 0 < t < 1, meets the ellipsoid.

    Test the closest point in ellipsoid-scaled coordinates. This avoids
    solving a quadratic for every ray and explicitly excludes intersections
    behind the observer or beyond the Sun. Zero-length rays are rejected by
    compute_shadow_f before this helper is called.
    """
    axes = np.array([p, p, q])
    origin, direction = a / axes, b / axes
    length_squared = np.einsum('...i,...i->...', direction, direction)
    t_closest = -np.einsum('...i,...i->...', origin, direction) / length_squared
    closest = origin + np.clip(t_closest, 0, 1)[..., None] * direction
    closest_squared = np.einsum('...i,...i->...', closest, closest)
    tolerance = 32 * np.finfo(float).eps
    blocked = (closest_squared < 1 - tolerance) | (
        (t_closest > 0) & (t_closest < 1) & (closest_squared <= 1 + tolerance))
    # A surface tangent may acquire a tiny positive t from roundoff.
    # Exclude that contact at the observer without excluding a distant
    # tangent, or any ray starting strictly inside the ellipsoid.
    origin_squared = np.einsum('...i,...i->...', origin, origin)
    surface_tangent = ((np.abs(origin_squared - 1) <= tolerance)
                       & (t_closest**2 * length_squared <= tolerance**2))
    return blocked & ~surface_tangent


def compute_adhya_intersection(a, b, p, q):
    """Intersect a + t*b with x²/p² + y²/p² + z²/q² = 1.

    Return (det, t_enter, t_exit), with ordered, signed line parameters.
    These replace the old unsigned distances from the solar limb: callers
    must now check the relevant parameter interval, e.g. 0 < t < 1 for a
    ray from a to a+b. Misses have negative det and NaN roots. Tangencies
    have det=0 and equal roots. A zero direction is invalid.

    In scaled coordinates A=b.b, B=2*a.b, C=a.a-1. The discriminant
    returned is (B²-4*A*C)/(4*A), evaluated as 1-|a cross unit(b)|² to
    avoid subtracting large, nearly equal quadratic terms near tangency.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim == 0 or b.ndim == 0 or a.shape[-1] != 3 or b.shape[-1] != 3:
        raise ValueError("a and b must have a final dimension of length 3.")
    if not np.isfinite(p) or not np.isfinite(q) or p <= 0 or q <= 0:
        raise ValueError("Ellipsoid semiaxes must be finite and positive.")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("a and b must be finite.")
    axes = np.array([p, p, q])
    origin, direction = np.broadcast_arrays(a / axes, b / axes)
    length = np.linalg.norm(direction, axis=-1)
    if np.any(length == 0):
        raise ValueError("The line direction must be nonzero.")
    unit = direction / length[..., None]
    half_B = np.sum(origin * unit, axis=-1)
    cross = np.cross(origin, unit)
    impact_squared = np.sum(cross * cross, axis=-1)
    det = 1 - impact_squared
    tolerance = 32 * np.finfo(float).eps * np.maximum(1, impact_squared)
    det = np.where(np.abs(det) <= tolerance, 0., det)
    sqrt_det = np.sqrt(np.maximum(det, 0))
    # Stable quadratic formula; recover the other root from their product.
    root = -half_B - np.copysign(sqrt_det, half_B)
    C = np.sum(origin * origin, axis=-1) - 1
    other = np.divide(C, root, out=np.zeros_like(root), where=root != 0)
    other = np.where(det == 0, root, other)
    t_enter = np.where(det >= 0, np.minimum(root, other) / length, np.nan)
    t_exit = np.where(det >= 0, np.maximum(root, other) / length, np.nan)
    return det, t_enter, t_exit


def compute_F_hist_at_sat_r_hist(EEI_truth_name, sat_r_hist, sat_jd_hist, toa_grid: Grid,
                                 erp_wl_split=True, store_dF=False):

    rad_config = rad_settings.radiation_settings_from_EEI_truth_name(EEI_truth_name)

    r_sun_hist = get_r_sun_jd_hist(rad_config, sat_jd_hist)
    TSI_1AU_day = rad_settings.get_TSI_1AU(sat_jd_hist, rad_config['TSI_source'])
    srp_F_hist = get_solar_incoming_day_hist(r_sun_hist, TSI_1AU_day, 
                                             sat_r_hist[None,...], grid_bool=False,
                                             toa_grid=toa_grid,
                                             penumbra_method=rad_config.get('penumbra_method'))
    if erp_wl_split:
        erp_F_LW_hist, erp_F_SW_hist = compute_earth_F_at_altitude(sat_jd_hist, rad_config, 
                                                sat_r_hist[None,...], toa_grid,
                                                store_dF=store_dF, wavelength="split")

        return erp_F_LW_hist, erp_F_SW_hist, srp_F_hist
    
    else:
        erp_F_hist = compute_earth_F_at_altitude(sat_jd_hist, rad_config, 
                                                sat_r_hist[None,...], toa_grid,
                                                store_dF=store_dF, wavelength="sum")

        return erp_F_hist, srp_F_hist

    # TODO: accelerations for plated satellites, which are not just a constant times F


def compute_a_hist_at_sat_r_hist(EEI_truth_name, sat_dict, sat_r_hist, sat_jd_hist, toa_grid: Grid,
                                 sat_R_body2ECEF_hist):

    # this function computes accelerations on a plated satellite due to ERP and SRP. 
    # first step is to call compute_F_hist_at_sat_r_hist to get the beam of fluxes (Earth element fluxes from all Earth surface nodes)

    n_steps = len(sat_r_hist)
    assert(len(sat_jd_hist) == len(sat_R_body2ECEF_hist) == n_steps)
    # sat_R_body2ECEF_hist = Rotation.random(n_steps).as_matrix()

    # all nets below still to be scaled by 1/(mc) to get acc. in ms-2

    # if different SW vs LW coefficients:
    if sat_dict['coeff_wl_split']:
        erp_dF_LW_hist, erp_dF_SW_hist, srp_F_hist = compute_F_hist_at_sat_r_hist(EEI_truth_name,
                                                            sat_r_hist, 
                                                            sat_jd_hist, 
                                                            toa_grid,
                                                            store_dF=True)
        # erp_dF_LW_hist is shaped (1, np_toa, nt, 3)
        # erp_dF_SW_hist is shaped (1, np_toa, nt, 3)
        # srp_dF_hist is shaped (1, nt, 3)

        # all nets below still to be scaled by 1/(mc) to get acc. in ms-2
        net_erp_LW = _integrate_net_vec_codex(erp_dF_LW_hist, sat_dict["plate_normals"], sat_dict["plate_areas"], 
                                            sat_dict["plate_ca_LW"], sat_dict["plate_cd_LW"], sat_dict["plate_cs_LW"], 
                                            sat_R_body2ECEF_hist)
        
        net_erp_SW = _integrate_net_vec_codex(erp_dF_SW_hist, sat_dict["plate_normals"], sat_dict["plate_areas"], 
                                            sat_dict["plate_ca_SW"], sat_dict["plate_cd_SW"], sat_dict["plate_cs_SW"],
                                            sat_R_body2ECEF_hist)
        net_erp = net_erp_LW + net_erp_SW

    else: # if same LW+SW coeffs:
        erp_dF_hist, srp_F_hist = compute_F_hist_at_sat_r_hist(EEI_truth_name,
                                                           sat_r_hist, 
                                                           sat_jd_hist, 
                                                           toa_grid,
                                                           store_dF=True,
                                                           erp_wl_split=False)
        net_erp = _integrate_net_vec_codex(erp_dF_hist, sat_dict["plate_normals"], sat_dict["plate_areas"], 
                                    sat_dict["plate_ca_SW"], sat_dict["plate_cd_SW"], sat_dict["plate_cs_SW"],
                                    sat_R_body2ECEF_hist)

    net_srp = _integrate_net_vec_codex(srp_F_hist[:,None,:,:], sat_dict["plate_normals"], sat_dict["plate_areas"], 
                                       sat_dict["plate_ca_SW"], sat_dict["plate_cd_SW"], sat_dict["plate_cs_SW"],
                                       sat_R_body2ECEF_hist)

    factor = 1 / (sat_dict['mass'] * LIGHT_SPEED)
    a_erp, a_srp = factor * net_erp, factor * net_srp

    return a_erp, a_srp


def _integrate_net_vec(dF_beam_hist, sat_plate_normals, sat_plate_areas, sat_plate_ca, sat_plate_cd, sat_plate_cs, sat_R_body2ECEF_hist):
    
    # the following computation does a the double surface integral (TOA surface + sat surface) at every time step without for loops, but is still 100x slower than what codex achieved from it

    sat_plate_normal_hist = np.einsum('tik,pk->pti', sat_R_body2ECEF_hist, sat_plate_normals)
    # checked to be the same as
    # sat_plate_normal_hist = np.zeros((n_plates, n_steps, 3))
    # for i, R_body2ECEF in enumerate(sat_R_body2ECEF_hist):
    #     sat_plate_normal_hist[:,i,:] = (R_body2ECEF @ sat_plate_normals.T).T

    norm_dF_beam_hist = get_norm_across_last_dim(dF_beam_hist)[...,None]
    u_dF_beam_hist = np.divide(
        dF_beam_hist,
        norm_dF_beam_hist,
        out=np.zeros_like(dF_beam_hist, dtype=float),
        where=norm_dF_beam_hist != 0,
    )
    cos = np.einsum('jitk,ntk->jint', -u_dF_beam_hist, sat_plate_normal_hist, optimize='optimal') # (1, np_toa, n_faces, nt)
    np.maximum(0, cos, out=cos) # this probably filters out 30-50% of previous non-zero elements

    s_term = np.einsum('p,jitk->jiptk', sat_plate_ca + sat_plate_cd, u_dF_beam_hist, optimize='optimal')
    n_term_interior = 2/3*sat_plate_cd[None,None,:,None] + 2*cos*sat_plate_cs[None,None,:,None]
    n_term = np.einsum('jipt,ptk->jiptk', n_term_interior, sat_plate_normal_hist, optimize='optimal')
    cR_tensor = s_term - n_term

    net_vec = np.einsum('jitk,p,jipt,jiptk->jtk', norm_dF_beam_hist, sat_plate_areas, cos, cR_tensor, optimize='optimal') # seems faster with no optimize
    #vec = area * cos * ((ca_i+cd_i) * s - (2/3*cd_i + 2*cos*cs_i) * n)

    return net_vec

import numpy as np


def _integrate_net_vec_codex(
    dF_beam_hist,
    sat_plate_normals,
    sat_plate_areas,
    sat_plate_ca,
    sat_plate_cd,
    sat_plate_cs,
    sat_R_body2ECEF_hist,
):
    """Return c * force in ECEF, shaped (n_observers, n_times, 3).

    Inputs:
      dF_beam_hist: (n_observers, n_sources, n_times, 3), in W/m^2.
      sat_plate_normals: (n_plates, 3), outward unit body-frame normals.
      areas and optical coefficients: (n_plates,), areas in m^2.
      sat_R_body2ECEF_hist: (n_times, 3, 3), orthonormal rotations
          shared across the observer axis.

    Assumes finite inputs and no mutual shadowing between plates.
    Divide by mass_kg * c_m_per_s to obtain acceleration in m/s^2.
    """
    F = np.asarray(dF_beam_hist)
    n = np.asarray(sat_plate_normals, dtype=np.float64)
    A = np.asarray(sat_plate_areas, dtype=np.float64)
    ca = np.asarray(sat_plate_ca, dtype=np.float64)
    cd = np.asarray(sat_plate_cd, dtype=np.float64)
    cs = np.asarray(sat_plate_cs, dtype=np.float64)
    R = np.asarray(sat_R_body2ECEF_hist, dtype=np.float64)

    n_observers, n_sources, n_times, _ = F.shape
    out = np.zeros((n_observers * n_times, 3), dtype=np.float64)

    if A.size == 0:
        return out.reshape(n_observers, n_times, 3)

    # Select nonzero vectors before normalization or plate calculations.
    observer, source, time = np.nonzero(np.any(F, axis=-1))

    if observer.size == 0:
        return out.reshape(n_observers, n_times, 3)

    beam = np.asarray(F[observer, source, time], dtype=np.float64)
    del source

    magnitude = np.sqrt(np.einsum("ki,ki->k", beam, beam))

    # R.T @ beam: transform photon propagation directions to body frame.
    direction = np.einsum("ki,kij->kj", beam, R[time])
    direction /= magnitude[:, None]
    del beam

    # The only large beam-by-plate array: (n_nonzero_beams, n_plates).
    mu = direction @ (-n.T)
    np.maximum(mu, 0.0, out=mu)

    # Absorption + incoming diffuse momentum.
    force_body = (mu @ (A * (ca + cd)))[:, None] * direction

    # Outgoing diffuse momentum.
    force_body -= mu @ (n * ((2.0 / 3.0) * A * cd)[:, None])

    # Specular momentum: reuse the cosine buffer for mu**2.
    np.square(mu, out=mu)
    force_body -= mu @ (n * (2.0 * A * cs)[:, None])
    del mu

    force_body *= magnitude[:, None]

    # Sum sources at each observer/time, preserving repeated indices.
    group = observer * n_times + time
    out[:, 0] = np.bincount(
        group, weights=force_body[:, 0], minlength=out.shape[0]
    )
    out[:, 1] = np.bincount(
        group, weights=force_body[:, 1], minlength=out.shape[0]
    )
    out[:, 2] = np.bincount(
        group, weights=force_body[:, 2], minlength=out.shape[0]
    )

    # Rotate only the final sums back to ECEF.
    return np.einsum(
        "tij,btj->bti", R, out.reshape(n_observers, n_times, 3)
    )




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


def get_grid_r_sun_hist(r_sun_day, stacked_r, compute_u=True): 

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
    if compute_u:
        grid_u_sun_hist = grid_r_sun_hist / (grid_d_sun_hist[...,None]) 
    else:
        grid_u_sun_hist = None

    return grid_r_sun_hist, grid_d_sun_hist, grid_u_sun_hist


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
    jd_interp = get_jd_to_build_interp(n, out_jd_array, rad_config["jd_interval"])

    # step 2: load (and expand?) a&e maps
    all_a = [None] * len(jd_interp)
    all_e = [None] * len(jd_interp)

    for i, jd in enumerate(jd_interp):
        datestr = jd_to_mmddyyyy(jd)
        a, e = load_sh_maps(datestr, rad_config)
        if method=="map_interp":
            #a = np.zeros(len(full_lat))
            #e = np.zeros(len(full_lat))
            a = expand_sh(a, full_lon, full_lat, rad_config['sh_normalization'])
            e = expand_sh(e, full_lon, full_lat, rad_config['sh_normalization'])
        all_a[i] = a
        all_e[i] = e

    # step 3: make splines and interpolate
    spline_a = make_interp_spline(jd_interp, np.stack(all_a), k=3)
    spline_e = make_interp_spline(jd_interp, np.stack(all_e), k=3)
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

def get_irradiance_to_radiance_map(r_rel, r_rel_norm_2, geometric_kernel, jd_array, toa_grid: Grid, rad_config: dict,
                                   rad_type: str = "both", ADM_model=None):
    """Return irradiance-to-radiance factors, sharing geometry and scenes by band.

    ``rad_type="both"`` (default) returns ``(LW_map, SW_map)``; ``"LW"`` or
    ``"SW"`` returns just that map. With no ADM model, each result is the
    Lambertian scalar 1/pi. Otherwise ERBE maps have shape (np, n_toa, nt),
    even for static geometry (r_rel's time axis may have length 1 or nt).
    Simplified ADM scenes have shape (n_toa,) and are shared across time
    without tiling; the factors still depend on observer and solar geometry.
    Invisible cells are zero in both maps; night-side cells are zero only
    in SW. ``rad_config['ADM_interp']`` selects nearest bins (default) or
    linear angular interpolation. LW seasons are DJF/MAM/JJA/SON, selected
    from each Julian date, and colatitude comes from the TOA grid latitude.
    """
    if not isinstance(rad_type, str) or rad_type.lower() not in ("both", "lw", "sw"):
        raise ValueError("rad_type must be 'both', 'LW', or 'SW'.")
    rad_type = rad_type.lower()
    use_lw, use_sw = rad_type != "sw", rad_type != "lw"
    n_steps_r = np.shape(r_rel)[2]
    n_steps = len(jd_array)
    assert (n_steps_r == 1) or (n_steps_r == n_steps)

    if ADM_model is None:
        return (1 / np.pi, 1 / np.pi) if rad_type == "both" else 1 / np.pi

    shape = (r_rel.shape[0], toa_grid.n_points, n_steps)
    lw_map = np.zeros(shape, dtype=float) if use_lw else None
    sw_map = np.zeros(shape, dtype=float) if use_sw else None
    cell_filter = np.broadcast_to(geometric_kernel != 0, shape)
    if use_sw and np.any(cell_filter):
        r_sun_hist = get_r_sun_jd_hist(rad_config, jd_array)
        _, _, toa_grid_u_sun_hist = get_grid_r_sun_hist(r_sun_hist, toa_grid.stacked_grid_r)
        zeroed_cos_theta_s_hist = get_zeroed_cos_theta_s_hist(toa_grid_u_sun_hist, toa_grid)
        if not use_lw:
            cell_filter = cell_filter & (zeroed_cos_theta_s_hist[None, :, :] != 0)

    observer_idx, toa_idx, time_idx = np.nonzero(cell_filter)
    del cell_filter
    if observer_idx.size:
        # Select visible cells once for both bands; static geometry is indexed
        # at time zero without allocating a tiled vector array.
        geometry_time_idx = 0 if n_steps_r == 1 else time_idx
        selected_r_rel = r_rel[observer_idx, toa_idx, geometry_time_idx]
        selected_grid_u = toa_grid.stacked_grid_u[toa_idx]
        r_rel_dot_grid_u = np.einsum('ik,ik->i', selected_r_rel, selected_grid_u)
        cos_theta_v = r_rel_dot_grid_u / np.sqrt(
            r_rel_norm_2[observer_idx, toa_idx, geometry_time_idx])
        viewing_zenith_angles = np.rad2deg(np.arccos(np.clip(cos_theta_v, -1, 1)))
        del cos_theta_v
        # Scene classification and cloud interpolation are shared by observers
        # and by SW/LW; only selected cells enter the ADM lookups.
        scene_types = get_erbe_scene_types(jd_array, toa_grid, rad_config)
        if scene_types.shape == (toa_grid.n_points,):
            selected_scene_types = scene_types[toa_idx]
        elif scene_types.shape == (toa_grid.n_points, n_steps):
            selected_scene_types = scene_types[toa_idx, time_idx]
        else:
            raise ValueError("ERBE scene types must have shape (n_toa,) or (n_toa, nt).")
        method = rad_config.get('ADM_interp', 'nearest')

        if use_lw:
            months = np.asarray(Time(jd_array, format='jd').ymdhms.month)
            seasons = (months % 12) // 3 + 1
            colatitudes = 90.0 - toa_grid.stacked_grid_latlon[:, 0]
            factors = ERBE_LW_ADM.factors(
                viewing_zenith_angles, colatitudes[toa_idx],
                selected_scene_types, seasons[time_idx], method=method)
            factors *= 1 / np.pi
            lw_map[observer_idx, toa_idx, time_idx] = factors
            del factors

        if use_sw:
            sunlit = zeroed_cos_theta_s_hist[toa_idx, time_idx] != 0
            selected_scene_types = selected_scene_types[sunlit]
            # Restrict solar/azimuth calculations to sunlit visible cells.
            observer_idx, toa_idx, time_idx = (idx[sunlit] for idx in (observer_idx, toa_idx, time_idx))
            selected_grid_u = selected_grid_u[sunlit]
            r_rel_proj = (selected_r_rel[sunlit]
                          - r_rel_dot_grid_u[sunlit, None] * selected_grid_u)
            viewing_zenith_angles = viewing_zenith_angles[sunlit]
            del selected_r_rel, r_rel_dot_grid_u, sunlit
            cos_theta_s = zeroed_cos_theta_s_hist[toa_idx, time_idx]
            u_sun_proj = (toa_grid_u_sun_hist[toa_idx, time_idx]
                          - cos_theta_s[:, None] * selected_grid_u)
            azimuth_denom = (get_norm_across_last_dim(r_rel_proj)
                             * get_norm_across_last_dim(u_sun_proj))
            # At exact solar/viewing zenith the azimuth is undefined; use 0 deg.
            cos_rel_az = np.ones_like(azimuth_denom)
            np.divide(np.einsum('ik,ik->i', r_rel_proj, u_sun_proj), azimuth_denom,
                      out=cos_rel_az, where=azimuth_denom > 0)
            del selected_grid_u, r_rel_proj, u_sun_proj, azimuth_denom
            factors = ERBE_SW_ADM.factors(
                solar_zenith_angles=np.rad2deg(np.arccos(np.clip(cos_theta_s, -1, 1))),
                viewing_zenith_angles=viewing_zenith_angles,
                relative_azimuth_angles=np.rad2deg(np.arccos(np.clip(cos_rel_az, -1, 1))),
                scene_types=selected_scene_types, method=method)
            factors *= 1 / np.pi
            sw_map[observer_idx, toa_idx, time_idx] = factors

    if rad_type == "both":
        return lw_map, sw_map
    return lw_map if use_lw else sw_map


def get_window_avg_maps(EEI_truth_name, jd_windows, map_name, grid: Grid):

    rad_config = rad_settings.radiation_settings_from_EEI_truth_name(EEI_truth_name)
    truth_jd_arrays = get_EEI_truth_daily_jd_arrays(rad_config["jd_interval"])
    base_data_dir = os.path.join(MEDIA_DIR, 'true_EEI', EEI_truth_name)
    avg_map_array = [None] * len(jd_windows)

    for i, jd_window in enumerate(jd_windows):
        print(f"Loading truth for window {i}")

        idxs = [(jd_vec_day[0]>=jd_window[0] and jd_vec_day[1]<=jd_window[1]) for jd_vec_day in truth_jd_arrays]
        idxs = np.squeeze(np.argwhere(idxs))
        all_maps = np.zeros((grid.n_points, len(idxs)))

        #print(f"Getting true avg maps...")
        for j, day_idx in enumerate(idxs):
            #progress_bar(i, len(idxs))
            file_names, _ = get_file_names_and_existence(base_data_dir, day_idx, grid, create_dirs=False)

            #true_avg = np.load(os.path.join(truth_dir, file_names['daily_avg_net_800km']+'.npy'))
            all_maps[:,j] = grid.vectorize_if_needed(np.load(os.path.join(base_data_dir, 
                                                                          file_names[map_name]+'.npy')))

        avg_map_array[i] = grid.compute_time_avg_map(all_maps) #np.nanmean(all_maps, axis=1)

    return avg_map_array


def get_toa_grid_from_rad_config(rad_config: dict, quadrature_order=None):

    if quadrature_order is None:
        if rad_config['Nmax']==2:
            quadrature_order = 131
        elif rad_config['Nmax']==45:
            quadrature_order = 151
            # NOTE: any intermediate cases to be added here (ideally based on convergence analysis)
        elif rad_config['Nmax']==179:
            quadrature_order = 251

    if rad_config['earth_shape']=="spherical":
        alt_km = 0
        flattening = 0
    elif rad_config['earth_shape']=="spherical_mean":
        alt_km = constants.earth_radius(model='mean') - constants.earth_radius(model='WGS84')
        flattening = 0
    elif rad_config['earth_shape']=='WGS84':
        alt_km = 0
        flattening = constants.earth_flattening(model='WGS84')
    elif rad_config['earth_shape']=='WGS84_TOA':
        alt_km = 20
        f_wgs84 = constants.earth_flattening(model='WGS84')
        R_wgs84 = constants.earth_radius(model='WGS84')
        flattening = 1 - ((R_wgs84 * (1-f_wgs84) + alt_km) / (R_wgs84 + alt_km))

    return QuadratureGrid(alt_km=alt_km, order=quadrature_order, flattening=flattening)
    



##############################################################################################
# file storage functions #####################################################################
##############################################################################################

def get_subdir_EEI_truth(EEI_truth, series_type, grid_name="grid_quad_n131"): # TODO: remove default 131
    
    subdir = os.path.join(MEDIA_DIR, 'true_EEI', EEI_truth, grid_name, 
                              series_type)
    return subdir

def get_file_names_and_existence(base_data_dir, day_idx, grid: Grid, 
                                 rm_daily_hists=False, rm_SFF=False, create_dirs=True):
    
    all_out_file_types = get_required_files(grid)
    if rm_SFF:
        all_out_file_types = [f for f in all_out_file_types if "SFF" not in f]

    # remove unfeasible storage options
    if rm_daily_hists:
        #if grid.n_points>5810: # lebedev order 131 grid has 5810 points
        all_out_file_types = [f for f in all_out_file_types if (("hist" not in f) or ("surf_avg" in f))] # would be too much storage

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
                   all_file_names, file_existences, grid: Grid,
                   R_ECEF_to_SunFrame_day_hist=None): #, n_cores=4, save_SFF_hist=True):
    # TODO: change base_data-Dir and day_idx for file dicts, sort out SFF (probably simplified)
    #all_file_names, file_existences = get_file_names_and_existence(base_data_dir, day_idx, grid)
    
    if not(file_existences.get('daily_hist_LW_toa', True)):
        print("Saving daily_hist_LW_toa file...")
        np.save(all_file_names['daily_hist_LW_toa'], toa_LW_day_hist)

    if not(file_existences.get('daily_hist_SW_toa', True)):
        print("Saving daily_hist_SW_toa file...")
        np.save(all_file_names['daily_hist_SW_toa'], toa_SW_day_hist)

    if not(file_existences.get('daily_hist_net_toa', True)):
        print("Saving daily_hist_net_toa file...")
        np.save(all_file_names['daily_hist_net_toa'], net_toa_day_hist)

    if (net_toa_day_hist_SFF is not None) and not(file_existences.get('daily_hist_net_toa_SFF_accurate')): #and save_SFF_hist:
        print("Saving daily_hist_net_toa file Sun-Fixed Frame...")
        #net_toa_day_hist_SFF = field_hist_rotation_multiproc(net_toa_day_hist, R_ECEF_to_SunFrame_day_hist, 
        #                                                     grid, n_cores=n_cores)
        np.save(all_file_names['daily_hist_net_toa_SFF'], net_toa_day_hist_SFF)
        # we're here - seems like interpolating will be easiest, but let's see later. For now lets go to altitude computation

    if not(file_existences.get('daily_avg_net_toa', True)): 
        print("Saving daily_avg_net_toa file...")
        net_toa_daily_avg = grid.compute_time_avg_map(net_toa_day_hist) # careful: is time still last dim?
        np.save(all_file_names['daily_avg_net_toa'], net_toa_daily_avg)

    if not(file_existences['daily_hist_net_toa_surf_avg']):
        print("Saving daily_hist_net_toa_surf_avg file...")
        toa_surf_avg_day_hist = grid.compute_surf_integral(net_toa_day_hist, average=True)
        np.save(all_file_names['daily_hist_net_toa_surf_avg'], toa_surf_avg_day_hist)

"""
"altitude": ['daily_hist_net_F_{alt_km}km', 'daily_hist_net_F_{alt_km}km_SFF_accurate',
             'daily_avg_net_F_{alt_km}km', 'daily_avg_net_F_{alt_km}km_SFF_accurate',
             'daily_surf_avg_net_{alt_km}km']"""

def save_altitude_files(daily_hist_net_F_altitude, daily_hist_net_F_altitude_SFF, 
                        file_names: dict, file_existences: dict, grid: Grid):

    if not(file_existences.get(f"daily_hist_net_F_{grid.alt_km}km", True)):
        print(f"Saving daily_hist_net_F_{grid.alt_km}km file...")
        np.save(file_names[f"daily_hist_net_F_{grid.alt_km}km"], daily_hist_net_F_altitude)

    if not(file_existences.get(f"daily_avg_net_F_{grid.alt_km}km", True)):
        print(f"Saving daily_avg_net_F_{grid.alt_km}km file...")
        daily_avg_net_F = np.mean(daily_hist_net_F_altitude, axis=1)
        np.save(file_names[f"daily_avg_net_F_{grid.alt_km}km"], daily_avg_net_F)

    if not(file_existences.get(f"daily_surf_avg_net_{grid.alt_km}km'", True)):
        print(f"Saving daily_surf_avg_net_{grid.alt_km}km' file...")
        daily_hist_net_Fr_altitude = F_vec_to_Fr(daily_hist_net_F_altitude, grid.stacked_grid_u)
        daily_hist_surf_avg = grid.compute_surf_integral(daily_hist_net_Fr_altitude, average=True)
        np.save(file_names[f"daily_surf_avg_net_{grid.alt_km}km"], daily_hist_surf_avg)

    if daily_hist_net_F_altitude_SFF is not None:
        if not(file_existences.get(f"daily_hist_net_F_{grid.alt_km}km_SFF_accurate", True)):
            print(f"Saving daily_hist_net_F_{grid.alt_km}km_SFF_accurate file...")
            np.save(file_names[f"daily_hist_net_F_{grid.alt_km}km_SFF_accurate"], daily_hist_net_F_altitude_SFF)

        if not(file_existences.get(f"daily_avg_net_F_{grid.alt_km}km_SFF_accurate", True)):
            print(f"Saving daily_avg_net_F_{grid.alt_km}km_SFF_accurate file...")
            daily_avg_net_F_SFF = np.mean(daily_hist_net_F_altitude_SFF, axis=1)
            np.save(file_names[f"daily_avg_net_F_{grid.alt_km}km_SFF_accurate"], daily_avg_net_F_SFF)
        
