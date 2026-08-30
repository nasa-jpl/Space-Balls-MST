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
from SpaceBalls.utils import get_all_r_rel, get_all_cos_alpha, get_r_rel_norm, progress_bar, jd_to_mmddyyyy
from SpaceBalls.plotter import Plotter

AU = constants.astronomical_unit(units='km')
RE = constants.earth_radius(units='km')

REQUIRED_FILES = {
    "toa": ['daily_hist_emission_toa', 'daily_hist_net_toa', 'daily_hist_net_toa_SFF',
            'daily_hist_net_toa_lat_avg', 'daily_hist_net_toa_surf_avg', 
            'daily_avg_net_toa'],

    "altitude": ['daily_hist_net_Fr_{alt_km}km', 'daily_hist_net_Fr_{alt_km}km_SFF',
                 'daily_hist_net_Fr_{alt_km}km_SFF_accurate',
                 'daily_hist_net_Fr_{alt_km}km_surf_avg',
                 'daily_avg_net_Fr_{alt_km}km', 'daily_avg_net_Fr_{alt_km}km_SFF',
                 'daily_avg_net_Fr_{alt_km}km_SFF_accurate', 
                 'daily_hist_net_F_{alt_km}km']
                 #, 'daily_hist_net_F_{alt_km}km_SFF', 'daily_avg_net_F_{alt_km}km', 'daily_avg_net_F_{alt_km}km_SFF']
}

def get_required_files(grid:Grid):

    key = 'toa' if grid.alt_km==0 else 'altitude'
    files = REQUIRED_FILES[key].copy()

    if grid.alt_km>0:
        files = [f.replace('{alt_km}', str(grid.alt_km)) for f in files]
    
    # filter unavailable/unfeasible options
    if grid.grid_type_name=="quadrature" and key=="toa":
        files.remove('daily_hist_net_toa_lat_avg')
    
    #if grid.n_points>6000 and key=="altitude": # lebedev order 131 grid has 5810 points
    #    files = [f for f in files if "hist" not in f]

    return files


"""
def required_files(alt_km, grid: Grid):
    if alt_km==0:           
        #if grid.grid_name=="regular_latlon":                                                        # SFF: sun-fixed frame
        return ['daily_hist_emission_toa', 'daily_hist_net_toa', 'daily_hist_net_toa_SFF',
                'daily_hist_net_toa_lat_avg', 'daily_hist_net_toa_surf_avg', 
                'daily_avg_net_toa']
        
        #elif grid.grid_name=="quadrature":
        #    return ['daily_hist_emission_toa', 'daily_hist_net_toa', 'daily_hist_net_toa_surf_avg']
    
    elif alt_km>0: 
        return ['daily_hist_net_Fr_'+str(alt_km)+'km', 'daily_hist_net_Fr_'+str(alt_km)+'km_SFF',
                'daily_avg_net_Fr_'+str(alt_km)+'km', 'daily_avg_net_Fr_'+str(alt_km)+'km_SFF',
                'daily_hist_net_F_'+str(alt_km)+'km']
                #'daily_avg_net_'+str(alt_km)+'km_simpl1', 'daily_avg_net_'+str(alt_km)+'km_simpl1_SFF']
"""


def compute_radiation_maps(EEI_truth_name, grid: Grid, selected_days_idxs=None, n_cores=4):
    
    daily_jd_arrays = get_EEI_truth_daily_jd_arrays(EEI_truth_name)
    
    rad_config = rad_settings.radiation_settings_from_EEI_truth_name(EEI_truth_name)
    mid_day_jd_array = mid_day_jd_array_from_jd_interval(rad_config["jd_interval"])
    n_days = len(mid_day_jd_array)

    if selected_days_idxs is None:
        assert grid.n_points <=8e3 # more than that would mean over a TB of stored data for all 5 years
        compute_all_days = True
    else:
        daily_jd_arrays_reduced = [daily_jd_arrays[i] for i in selected_days_idxs]
        compute_all_days = False

    #out_dir = os.path.join(MEDIA_DIR, 'true_EEI', EEI_truth_name, grid.grid_name)
    base_data_dir = os.path.join(MEDIA_DIR, 'true_EEI', EEI_truth_name)
    #out_dir = os.path.join(base_data_dir, grid.grid_name)

    #for altitude_km in altitude_array:

    #daily_jd_arrays_to_loop = daily_jd_arrays_reduced if (altitude_km>0 and not(compute_all_days)) else daily_jd_arrays
    daily_jd_arrays_to_loop = daily_jd_arrays if (compute_all_days or grid.alt_km==0) else daily_jd_arrays_reduced
    #idxs_days_to_loop = selected_days_idxs if altitude_km>0 else range(n_days)
    idxs_days_to_loop = range(n_days) if (compute_all_days or grid.alt_km==0) else selected_days_idxs

    for day_idx, jd_array in zip(idxs_days_to_loop, daily_jd_arrays_to_loop):

        print(f"Now computing: daily hist for day {day_idx} at {grid.alt_km}km")
        n_steps = len(jd_array)
        mid_day_jd = mid_day_jd_array[day_idx]
        TSI_1AU_day = rad_settings.get_TSI_1AU(mid_day_jd, rad_config["TSI_source"])
        R_ECEF_to_SunFrame_day_hist = get_R_SunFrame_hist(rad_config["ephemerides"], mid_day_jd)

        file_names, file_existences = get_file_names_and_existence(base_data_dir, day_idx, grid)
        
        if grid.alt_km==0 and not(all(file_existences.values())):
            toa_emission_day_hist, net_toa_day_hist = get_daily_hist_toa(base_data_dir, day_idx, mid_day_jd, TSI_1AU_day, 
                                                                         rad_config, grid)
            save_toa_files(toa_emission_day_hist, net_toa_day_hist, base_data_dir, day_idx, R_ECEF_to_SunFrame_day_hist, 
                           grid, n_cores=n_cores, save_SFF_hist=False) #(day_idx in selected_days_idxs))

        elif grid.alt_km>0 and not(all(file_existences.values())):

            if not(file_existences[f"daily_hist_net_Fr_{grid.alt_km}km_SFF_accurate"]):
                print("Computing SFF accurate maps")
                F_day_hist_SFF_acc, Fr_day_hist_SFF_acc = compute_daily_hist_net_at_altitude_SFF_accurate(
                                base_data_dir, day_idx, mid_day_jd, rad_config, TSI_1AU_day, 
                                R_ECEF_to_SunFrame_day_hist, grid, n_cores)
                np.save(file_names[f"daily_hist_net_Fr_{grid.alt_km}km_SFF_accurate"], Fr_day_hist_SFF_acc)

                Fr_day_hist_SFF_acc_daily_avg = grid.compute_time_avg_map(Fr_day_hist_SFF_acc)
                np.save(file_names[f"daily_avg_net_Fr_{grid.alt_km}km_SFF_accurate"], Fr_day_hist_SFF_acc_daily_avg)

            if (selected_days_idxs is not None) and (day_idx in selected_days_idxs):
                if not(file_existences[f"daily_hist_net_Fr_{grid.alt_km}km"]):
                    print("Computing all other maps")
                    daily_hist_net_F_altitude, daily_hist_net_Fr_altitude = get_daily_hist_net_at_altitude(
                        base_data_dir, day_idx, mid_day_jd, TSI_1AU_day, rad_config, grid, n_cores)
                    
                    save_altitude_files(daily_hist_net_F_altitude, daily_hist_net_Fr_altitude, base_data_dir, day_idx, 
                                        R_ECEF_to_SunFrame_day_hist, grid)


def get_daily_hist_toa(base_data_dir, day_idx, mid_day_jd, TSI_1AU_day, rad_config, grid: Grid, load_emission=True, load_net=True):
    
    all_file_names, file_existences = get_file_names_and_existence(base_data_dir, day_idx, grid)
    toa_emission_day_hist, net_toa_day_hist = None, None

    if not(file_existences["daily_hist_emission_toa"]) or not(file_existences["daily_hist_net_toa"]):  # requires recomputing
        
        print("Computing daily hist TOA...")
        toa_emission_day_hist, net_toa_day_hist = compute_daily_time_series_toa(mid_day_jd, rad_config, TSI_1AU_day, grid)
    
    else:
        print("Loading daily hist TOA...")
        if load_emission:
            try:
                toa_emission_day_hist = np.load(all_file_names["daily_hist_emission_toa"]+'.npy')
            except:
                #if grid.grid_type_name=="regular": n_lon, n_lat = len(lon_vec), len(lat_vec)
                toa_emission_day_hist = np.loadtxt(all_file_names["daily_hist_emission_toa"]+'.txt')
                assert(grid.grid_type_name=='regular_latlon') # old (txt) files were computed on regular grids only  # TODO: check file shape consistency
                n_steps = int(np.prod(np.shape(toa_emission_day_hist))) / (grid.n_lon * grid.n_lat)
                toa_emission_day_hist = toa_emission_day_hist.reshape((grid.n_lat, grid.n_lon, n_steps))
        
        if load_net:
            try:
                net_toa_day_hist = np.load(all_file_names["daily_hist_net_toa"]+'.npy')
            except:
                #if grid_type=="regular": n_lon, n_lat = len(lon_vec), len(lat_vec)
                net_toa_day_hist = np.loadtxt(all_file_names["daily_hist_net_toa"]+'.txt')
                assert(grid.grid_type_name=='regular_latlon') # old (txt) files were computed on regular grids only  # TODO: check file shape consistency
                n_steps = int(np.prod(np.shape(net_toa_day_hist)) / (grid.n_lon * grid.n_lat))
                net_toa_day_hist = net_toa_day_hist.reshape((grid.n_lat, grid.n_lon, n_steps))

    return toa_emission_day_hist, net_toa_day_hist


def get_daily_hist_net_at_altitude(base_data_dir, day_idx, mid_day_jd, TSI_1AU_day, rad_config, grid: Grid, n_cores):
    
    all_file_names, file_existences = get_file_names_and_existence(base_data_dir, day_idx, grid)
    
    # TODO: deal with daily hists in dense altitude grids
    #if not(file_existences['daily_hist_net_Fr_'+str(grid.alt_km)+'km']) or not(file_existences['daily_hist_net_F_'+str(grid.alt_km)+'km']):
    daily_hist_net_F_altitude, daily_hist_net_Fr_altitude = compute_daily_hist_net_at_altitude(
        base_data_dir, day_idx, mid_day_jd, rad_config, TSI_1AU_day, grid, n_cores)
    
    return daily_hist_net_F_altitude, daily_hist_net_Fr_altitude


def compute_daily_hist_net_at_altitude_SFF_accurate(base_data_dir, day_idx, mid_day_jd, rad_config, TSI_1AU_day, 
                                                    R_ECEF_to_SunFrame_day_hist, grid: Grid, n_cores):
    
    toa_grid = QuadratureGrid(alt_km=0, order=131) # integration to altitude MUST be done with the Lebedev TOA grid (orders of magnitude more accurate with less points)
    toa_emission_day_hist, _ = get_daily_hist_toa(base_data_dir, day_idx, mid_day_jd, TSI_1AU_day, 
                                                                    rad_config, toa_grid, load_net=False)
    toa_L_day_hist = irradiance_to_radiance(toa_emission_day_hist)
    n_steps = np.shape(toa_emission_day_hist)[1]
    r_sun_day = get_r_sun_day_hist(rad_config["ephemerides"], mid_day_jd)

    args_list = [(i, R_ECEF_to_SunFrame_day_hist[i,:,:], toa_L_day_hist[:,i], r_sun_day[i,:]) for i in range(n_steps)]
    with Pool(processes=n_cores, initializer=init_worker_SFF_accurate, initargs=(TSI_1AU_day, toa_grid, grid)) as pool:
        results = pool.starmap(compute_net_alt_SFF_accurate_at_step_worker, args_list)
    
    time_idxs = [res[0] for res in results]
    assert((np.unique(np.diff(time_idxs))==1).all() & len(np.unique(np.diff(time_idxs)))==1)

    emission_F_day_hist = np.stack([res[1] for res in results], axis=2)
    solar_F_day_hist = np.stack([res[2] for res in results], axis=2)
    
    total_F_day_hist = solar_F_day_hist + emission_F_day_hist
    total_Fr_day_hist = -np.einsum('ijk,ijk->ik', total_F_day_hist, grid.stacked_grid_u[:,:,None]) # NOTE the sign flip as we define POSITIVE radial flux as going IN

    return total_F_day_hist, total_Fr_day_hist

_pool_globals_2 = {}
def init_worker_SFF_accurate(TSI_1AU_day, toa_grid: Grid, altitude_grid: Grid):
    _pool_globals_2['TSI_1AU_day'] = TSI_1AU_day
    _pool_globals_2['toa_grid'] = toa_grid
    _pool_globals_2['altitude_grid'] = altitude_grid

def compute_net_alt_SFF_accurate_at_step_worker(i, R_ECEF_to_SunFrame_step, toa_L_step, r_sun_step):
    #print(f"Computing time step {i}...")  
    g = _pool_globals_2
    emission_Fi, solar_Fi = compute_net_alt_SFF_accurate_at_step(R_ECEF_to_SunFrame_step, 
                                                                 toa_L_step, r_sun_step,
                                                                 g['TSI_1AU_day'], g['toa_grid'],
                                                                 g['altitude_grid'])
    return i, emission_Fi, solar_Fi

    
def compute_net_alt_SFF_accurate_at_step(R_ECEF_to_SunFrame_step, toa_L_step, r_sun_step, TSI_1AU_day, toa_grid: Grid, altitude_grid: Grid):

    R_mat = (R_ECEF_to_SunFrame_step)
    stacked_grid_r_SFF = ((R_mat.T) @ (altitude_grid.stacked_grid_r.T)).T
    stacked_grid_U_SFF = stacked_grid_r_SFF / (np.sqrt(np.einsum('ij,ij->i', stacked_grid_r_SFF, stacked_grid_r_SFF))[:,None])

    # ERP block: some overlap with compute_daily_hist_net_at_altitude function

    # 1. Compute all relative vectors
    r_rel = get_all_r_rel(stacked_grid_r_SFF, toa_grid.stacked_grid_r)

    # 2. Compute norm of relative vectors
    r_rel_norm = get_r_rel_norm(r_rel, method="einsum")

    # 3. Compute all relative angles alpha
    cos_alpha = get_all_cos_alpha(r_rel, r_rel_norm, toa_grid.stacked_grid_u, method="einsum") # again tested to win wrt all other methods (at least in 65e3 CV x 5e3 TOA grid)

    emission_F = compute_all_fluxes_at_altitude(toa_L_step, cos_alpha, r_rel_norm, r_rel, 
                                                toa_grid.integration_weights)
    emission_F_step = (R_mat @ emission_F.T).T
    
    # SRP block: some overlap with get_solar_incoming_day_hist function

    #grid_r_sun = r_sun_step[None,:] - stacked_grid_r_SFF
    #grid_d_sun = np.sqrt(np.einsum('ij,ij->i',grid_r_sun,grid_r_sun))
    grid_r_sun, grid_d_sun = get_grid_r_sun_step(r_sun_step, stacked_grid_r_SFF)

    TSI_grid = TSI_1AU_day * (AU/grid_d_sun)**2
    grid_u_sun = grid_r_sun / grid_d_sun[:, None]

    ## modified get_zeroed_cos_theta_s_hist function:
    #stacked_cos_theta_s = np.einsum('ij,ij->i', grid_u_sun, stacked_grid_U_SFF) 
    #cos_lim = get_cos_theta_s_lim(altitude_grid.alt_km)
    #stacked_cos_theta_s[stacked_cos_theta_s < cos_lim] = 0 # no partial shadow, constant altitude
    stacked_cos_theta_s = get_zeroed_cos_theta_s_step(grid_u_sun, stacked_grid_U_SFF, altitude_grid.alt_km)

    grid_u_sun = grid_u_sun * (stacked_cos_theta_s[:,None]!=0) 
    solar_F = - grid_u_sun * TSI_grid[:,None]  # # minus sign so that vecors are pointing AWAY from Sun

    solar_F_step = (R_mat @ solar_F.T).T

    return emission_F_step, solar_F_step


def compute_daily_hist_net_at_altitude(base_data_dir, day_idx, mid_day_jd, rad_config, TSI_1AU_day, grid: Grid, n_cores):

    toa_grid = QuadratureGrid(alt_km=0, order=131) # integration to altitude MUST be done with the Lebedev TOA grid (orders of magnitude more accurate with less points)
    toa_emission_day_hist, _ = get_daily_hist_toa(base_data_dir, day_idx, mid_day_jd, TSI_1AU_day, 
                                                                    rad_config, toa_grid, load_net=False)
    toa_L_day_hist = irradiance_to_radiance(toa_emission_day_hist)
    n_steps = np.shape(toa_emission_day_hist)[1]

    # 1. Compute all relative vectors
    r_rel = get_all_r_rel(grid.stacked_grid_r, toa_grid.stacked_grid_r) # broadcasting is fastest. NOTE: if toa grd were not quadrature (e.g. regular 1deg x 1deg, there would not be enough RAM)

    # 2. Compute norm of relative vectors
    r_rel_norm = get_r_rel_norm(r_rel, method="einsum") # einsum method tested to win (30% faster regardless of the CV grid size)

    # 3. Compute all relative angles alpha
    cos_alpha = get_all_cos_alpha(r_rel, r_rel_norm, toa_grid.stacked_grid_u, method="einsum") # again tested to win wrt all other methods (at least in 65e3 CV x 5e3 TOA grid)
    
    args_list = [(i, toa_L_day_hist[:, i]) for i in range(n_steps)] # the indexing of toa_emission_day_hist only works with a quadrature toa grid (which has to be)

    # Approach 2: compute F and only at the end compute Fr

    with Pool(processes=n_cores, initializer=init_worker, 
              initargs=(cos_alpha, r_rel_norm, r_rel, toa_grid.integration_weights, grid.stacked_grid_u)) as pool:
        results = pool.starmap(compute_fluxes_worker2, args_list)

    time_idxs = [res[0] for res in results]
    assert((np.unique(np.diff(time_idxs))==1).all() & len(np.unique(np.diff(time_idxs)))==1)
    emission_F_day_hist = np.stack([res[1] for res in results], axis=2)

    r_sun_day = get_r_sun_day_hist(rad_config["ephemerides"], mid_day_jd)
    solar_F_day_hist, solar_Fr_day_hist, _ = get_solar_incoming_day_hist(r_sun_day, TSI_1AU_day, grid)
    
    total_F_day_hist = solar_F_day_hist + emission_F_day_hist
    total_Fr_day_hist = -np.einsum('ijk,ijk->ik', total_F_day_hist, grid.stacked_grid_u[:,:,None]) # NOTE the sign flip as we define POSITIVE radial flux as going IN

    # NOTE that total_Fr_day_hist gives the same result as if we pool compute_fluxes_worker to compute Fr maps concurrently, then do
    #           emission_Fr_day_hist = np.stack([res[2] for res in results], axis=1)
    #           total_Fr_day_hist = solar_Fr_day_hist - emission_Fr_day_hist
    # (checked within numerical precision) - the two approaches seem to have the same performance and cost

    total_F_day_hist = grid.reshape_if_needed(total_F_day_hist)
    total_Fr_day_hist = grid.reshape_if_needed(total_Fr_day_hist)

    return total_F_day_hist, total_Fr_day_hist


def compute_daily_hist_at_sat_r_hist(EEI_truth_name, sat_r_hist, sat_jd_hist, day_idx, 
                                     toa_grid = QuadratureGrid(alt_km=0, order=131)):

    # checked to work but review to avoid code duplication to be completed

    base_data_dir = os.path.join(MEDIA_DIR, 'true_EEI', EEI_truth_name)
    rad_config = rad_settings.radiation_settings_from_EEI_truth_name(EEI_truth_name)
    mid_day_jd = mid_day_jd_array_from_jd_interval(rad_config["jd_interval"])[day_idx]
    TSI_1AU_day = rad_settings.get_TSI_1AU(mid_day_jd, rad_config["TSI_source"])
    
    all_daily_jd_arrays = get_EEI_truth_daily_jd_arrays(EEI_truth_name)
    truth_jd_array = all_daily_jd_arrays[day_idx]

    r_sun_day = get_r_sun_day_hist(rad_config["ephemerides"], mid_day_jd)
    spline = make_interp_spline(truth_jd_array, r_sun_day, k=3) # NOTE: k=3 seems to work better than k=1: for the fully circular case (constant d), resulting max-min is 0.02 instead of 1900 with k=1
    r_sun_day_sat_times = spline(sat_jd_hist, extrapolate=True)
    d_sun_day_sat_times = np.sqrt(np.einsum('ij,ij->i', r_sun_day_sat_times, r_sun_day_sat_times))
    u_sun_day_sat_times = r_sun_day_sat_times / d_sun_day_sat_times[:,None]
    n_steps = len(sat_jd_hist) 

    if ((toa_grid.n_points * n_steps)<100e6) and ("knocke" not in toa_grid.grid_type_name):  # this boundary requires around 35GB of RAM:
        erp_Fr_hist, srp_Fr_hist = compute_daily_hist_at_sat_hist_all_at_once(base_data_dir, day_idx, mid_day_jd, 
                                                                            TSI_1AU_day, rad_config, toa_grid,
                                                                            truth_jd_array, sat_jd_hist, sat_r_hist, 
                                                                            r_sun_day_sat_times)
    else:
        print("Evaluating a/e SH maps on TOA grid...")
        day_datestr = Time(mid_day_jd, format='jd').to_datetime().strftime("%Y-%m-%d")
        a_map, e_map = get_expanded_ae_maps(day_datestr, toa_grid, rad_config) # TRUNCATION!!!
        
        #a_map, e_map = np.zeros(toa_grid.n_points), np.zeros(toa_grid.n_points)
        
        """
        # array split method:
        t1 = time.time()
        n_blocks = int(n_steps / 1000 )
        r_sun_day_blocks = np.array_split(r_sun_day_sat_times, n_blocks)
        d_sun_day_blocks = np.array_split(d_sun_day_sat_times, n_blocks)
        sat_r_hist_blocks = np.array_split(sat_r_hist, n_blocks)
        erp_Fr_hist_chunks = [None] * n_blocks
        srp_Fr_hist_chunks = [None] * n_blocks

        for i in range(n_blocks):
            print(f"Computing block {i+1}/{n_blocks}...")
            r_sun_chunk = r_sun_day_blocks[i]
            d_sun_chunk = d_sun_day_blocks[i]
            sat_r_chunk = sat_r_hist_blocks[i]

            toa_grid_r_sun, toa_grid_d_sun = get_grid_r_sun_hist(r_sun_chunk, toa_grid.stacked_grid_r)
            toa_grid_u_sun = toa_grid_r_sun / toa_grid_d_sun[:,None,:]
            #TSI_toa_grid = TSI_1AU_day * (AU/toa_grid_d_sun)**2
            TSI_earth = TSI_1AU_day * (AU/d_sun_chunk)**2
            zeroed_cos_theta_toa = get_zeroed_cos_theta_s_hist(toa_grid_u_sun, toa_grid)
            LW_outgoing_toa = e_map[:,None] / 4 * TSI_earth # here we DO use the TSI at the center of the Earth (Knocke model and Monte docs)
            #print(f"shape LW_outgoing_toa: {np.shape(LW_outgoing_toa)} ")
            #print(f"shape zeroed_cos_theta_toa: {np.shape(zeroed_cos_theta_toa)} ")
            #print(f"shape zeroed_cos_theta_toa: {np.shape(zeroed_cos_theta_toa)} ")

            SW_outgoing_toa = zeroed_cos_theta_toa * a_map[:,None] * TSI_earth[None,:]
            toa_emission_step = LW_outgoing_toa + SW_outgoing_toa
            toa_L = irradiance_to_radiance(toa_emission_step)

            erp_Fr_chunk, srp_Fr_chunk = flux_on_sat_computation(
                                                    sat_r_chunk,
                                                    toa_L,
                                                    toa_grid,
                                                    r_sun_chunk,
                                                    TSI_1AU_day)
            erp_Fr_hist_chunks[i] = erp_Fr_chunk
            srp_Fr_hist_chunks[i] = srp_Fr_chunk
        t2 = time.time()
        print(f"Time to loop over time chunks: {t2-t1}")
        """

        # full for loop method:
        t1 = time.time()
        erp_Fr_hist, srp_Fr_hist = np.zeros(n_steps), np.zeros(n_steps)
        for i in range(n_steps):
            #print(f"Computing step {i+1}/{n_steps}")
            progress_bar(i, n_steps)
            r_sun_step = r_sun_day_sat_times[i,:]
            d_sun_step = d_sun_day_sat_times[i]
            r_sat_step = sat_r_hist[i,:]
            # we're here - recompute knocke grid
            toa_grid.recompute_grid(r_sat_step)
            if ("knocke" in toa_grid.grid_type_name): 
                print(f"Recompute expanded a/e maps")
                a_map, e_map = get_expanded_ae_maps(day_datestr, toa_grid, rad_config) # TRUNCATION!!!
                    

            # 1. Get TOA emission at step i
            toa_grid_r_sun, toa_grid_d_sun = get_grid_r_sun_step(r_sun_step, toa_grid.stacked_grid_r)
            toa_grid_u_sun = toa_grid_r_sun / toa_grid_d_sun[:,None]
            #TSI_toa_grid = TSI_1AU_day * (AU/toa_grid_d_sun)**2
            TSI_earth = TSI_1AU_day * (AU/d_sun_step)**2
            zeroed_cos_theta_toa_step = get_zeroed_cos_theta_s_step(toa_grid_u_sun, toa_grid.stacked_grid_u, 
                                                                    alt_km=0)
            LW_outgoing_toa_step = e_map / 4 * TSI_earth # here we DO use the TSI at the center of the Earth (Knocke model and Monte docs)
            SW_outgoing_toa_step = zeroed_cos_theta_toa_step * a_map * TSI_earth
            toa_emission_step = LW_outgoing_toa_step + SW_outgoing_toa_step
            toa_L_step = irradiance_to_radiance(toa_emission_step)

            # 2. Compute flux on satellite at step i
            erp_Fr_step, srp_Fr_step = flux_on_sat_computation(r_sat_step[None,:],
                                                               toa_L_step[:,None],
                                                               toa_grid,
                                                               r_sun_step[None,:],
                                                               TSI_1AU_day)
            erp_Fr_hist[i] = erp_Fr_step[0]
            srp_Fr_hist[i] = srp_Fr_step[0]
        t2 = time.time()
        print(f"Time to loop each time step: {t2-t1}")

        """
        erp_Fr_hist_2 = np.concatenate(erp_Fr_hist_chunks)
        srp_Fr_hist_2 = np.concatenate(srp_Fr_hist_chunks)
        print(f"erp equal check: {(erp_Fr_hist_2==erp_Fr_hist).all()}")
        print(f"erp diff check: {np.max(np.abs(erp_Fr_hist_2-erp_Fr_hist))/np.mean(erp_Fr_hist)}")
        print(f"srp equal check: {(srp_Fr_hist_2==srp_Fr_hist).all()}")
        """
            
    return erp_Fr_hist, srp_Fr_hist

    #erp_F_hist = compute_all_fluxes_at_altitude(toa_L_day_hist_sat_times, cos_alpha, r_rel_norm, r_rel, toa_grid.integration_weights)
    
    # TBC...


def compute_daily_hist_at_sat_hist_all_at_once(base_data_dir, day_idx, mid_day_jd, TSI_1AU_day, rad_config, toa_grid: Grid,
                                               truth_jd_array, sat_jd_hist, sat_r_hist, r_sun_day_sat_times):
    
    # this is the function to be called if RAM memory allows it

    toa_emission_day_hist, _ = get_daily_hist_toa(base_data_dir, day_idx, mid_day_jd, TSI_1AU_day, 
                                                                    rad_config, toa_grid, load_net=False)

    toa_L_day_hist = irradiance_to_radiance(toa_emission_day_hist)
    toa_L_day_hist = toa_grid.vectorize_if_needed(toa_L_day_hist)
    # n_steps = np.shape(toa_emission_day_hist)[1]

    spline = make_interp_spline(truth_jd_array, toa_L_day_hist.T, k=1)
    toa_L_day_hist_sat_times = spline(sat_jd_hist, extrapolate=True).T # NOTE: np.vstack([np.interp(...)]) is 3x slower and does not apply extrapolation at the last minute. 
    # NOTE 2: The levels of extrapolation in this implemented scipy solution have been checked not to give any exploding results

    erp_Fr_hist, srp_Fr_hist = flux_on_sat_computation(sat_r_hist, toa_L_day_hist_sat_times, toa_grid, r_sun_day_sat_times, TSI_1AU_day)

    return erp_Fr_hist, srp_Fr_hist


def flux_on_sat_computation(sat_r_hist, toa_L_day_hist_sat_times, toa_grid: Grid, r_sun_day_sat_times, TSI_1AU_day):

    # 1. Compute all relative vectors
    r_rel = get_all_r_rel(sat_r_hist, toa_grid.stacked_grid_r) # broadcasting is fastest. NOTE: if toa grd were not quadrature (e.g. regular 1deg x 1deg, there would not be enough RAM)

    # 2. Compute norm of relative vectors
    r_rel_norm = get_r_rel_norm(r_rel, method="einsum") # einsum method tested to win (30% faster regardless of the CV grid size)

    # 3. Compute all relative angles alpha
    cos_alpha = get_all_cos_alpha(r_rel, r_rel_norm, toa_grid.stacked_grid_u, method="einsum") # again tested to win wrt all other methods (at least in 65e3 CV x 5e3 TOA grid)

    erp_F_hist = np.einsum(
        'tj,jt,t,jtk,jt->jk',
        toa_L_day_hist_sat_times,       # (n_toa, nt)
        cos_alpha,                      # (n_t, n_toa)
        toa_grid.integration_weights,   # (n_toa,)
        r_rel,                          # (n_t, n_toa, 3)
        1 / (r_rel_norm**3)             # (n_t, n_toa)
    )

    r_sat_norm = np.sqrt(np.einsum('ij,ij->i', sat_r_hist, sat_r_hist))
    sat_u_hist = sat_r_hist / r_sat_norm[:,None]
    erp_Fr_hist = - compute_radial_fluxes_at_altitude(erp_F_hist, sat_u_hist, method="einsum")

    # Solar here:

    r_sun_sat = r_sun_day_sat_times - sat_r_hist
    
    d_sun_sat = np.sqrt(np.einsum('ij,ij->i', r_sun_sat, r_sun_sat))
    u_sun_sat = r_sun_sat / d_sun_sat[:,None]
    TSI_sat_day_vec = TSI_1AU_day * (AU/d_sun_sat)**2

    stacked_cos_theta_s_hist = np.einsum('ij,ij->i', u_sun_sat, sat_u_hist) # same as: np.sum(grid_u_sun_hist * grid.stacked_grid_u[:,:,None], axis=1)

    alt_hist = r_sat_norm - RE
    cos_lim = get_cos_theta_s_lim(alt_hist)
    stacked_cos_theta_s_hist[stacked_cos_theta_s_hist < cos_lim] = 0 # no partial shadow, constant altitude
    
    # u_sun_sat = u_sun_sat * (stacked_cos_theta_s_hist[:,None]!=0)  # filter out eclipse
    # srp_F_hist = - u_sun_sat * TSI_sat_day_vec  # # minus sign so that vecors are pointing AWAY from Sun
    
    srp_Fr_hist = stacked_cos_theta_s_hist * TSI_sat_day_vec # eclipse is set to 0

    return erp_Fr_hist, srp_Fr_hist



_pool_globals = {}

def init_worker(cos_alpha_shared, r_rel_norm_shared, r_rel_shared, weights_shared, u_el_shared):
    """Initialize worker with shared constant data."""
    _pool_globals['cos_alpha'] = cos_alpha_shared
    _pool_globals['r_rel_norm'] = r_rel_norm_shared
    _pool_globals['r_rel'] = r_rel_shared
    _pool_globals['weights'] = weights_shared
    _pool_globals['stacked_grid_u_el'] = u_el_shared


def compute_fluxes_worker(i, toa_emission):
    """Worker that uses global initialized data."""
    g = _pool_globals
    emission_F = compute_all_fluxes_at_altitude(
        toa_emission, g['cos_alpha'], g['r_rel_norm'], g['r_rel'], g['weights'] #, method="einsum"
    )
    radial_emission_F = compute_radial_fluxes_at_altitude(emission_F, g['stacked_grid_u_el']) #, method="einsum")
    return i, emission_F, radial_emission_F


def compute_fluxes_worker2(i, toa_emission):
    """Worker that uses global initialized data."""
    print(f"Computing time step {i}...")
    g = _pool_globals
    emission_F = compute_all_fluxes_at_altitude(
        toa_emission, g['cos_alpha'], g['r_rel_norm'], g['r_rel'], g['weights'], method="einsum"
    )
    return i, emission_F


def compute_all_fluxes_at_altitude(toa_L, cos_alpha, r_rel_norm, r_rel, weights, mode="single_step"):

    # NOTE: inputs
    #       toa_L: luminance at Earth grid nodes
    #       cos_alpha: cosine of angle wrt Earth local normal, for all relative vectors between Earth grid nodes and altitude nodes
    #       r_rel: relative vectors between Earth grid nodes and altitude nodes
    #       r_rel_norm: norm of r_rel
    #       weights: integration weights of Earth grid 

    if mode=="single_step":
        ein_tag = 't,jt,t,jtk,jt->jk'

    elif mode=="all_steps":
        ein_tag = 'tj,jt,t,jtk,jt->jk'

    # NOTE: notation
    #       n_toa: number of nodes at the TOA (Earth) grid
    #       n_alt: number of nodes at the satellite altitude (one node would be a single satellite location, multiple nodes might be multiple altitude grid nodes or satellite locations)
    #       n_t: number of time steps if multiple time steps to be evaluated at once
    full_F = np.einsum(
        ein_tag,
        toa_L,      # (n_toa,) if single_step; (n_toa, n_t) if all_steps
        cos_alpha,  # (n_alt, n_toa)
        weights,    # (n_toa,)
        r_rel,      # (n_alt, n_toa, 3)
        1 / (r_rel_norm**3) # (n_alt, n_toa)
    )

    # faster than the broadcast option:         # field_to_integrate = (1/np.pi) * toa_emission[None,:,None] * cos_alpha[:,:,None] * (1/(r_rel_norm**3))[:,:,None] * r_rel
                                                # full_F = np.sum(field_to_integrate * weights[None,:,None], axis=1) * RE**2
    return full_F
    
def compute_radial_fluxes_at_altitude(full_F, stacked_altitude_grid_u, method="einsum"):

    full_radial_F = np.einsum('jk,jk->j', full_F, stacked_altitude_grid_u) # again faster than the broadcast option
    return full_radial_F
    
    # at this point I don't try the loops because einsum seems to always win!


def compute_daily_time_series_toa(mid_day_jd, rad_config, TSI_1AU_day, grid: Grid):
    
    r_sun_day = get_r_sun_day_hist(rad_config["ephemerides"], mid_day_jd)
    d_sun_day = np.sqrt(np.einsum('ij,ij->i', r_sun_day, r_sun_day)) # same as np.linalg.norm(r_sun_day,2,1) but slightly faster
    TSI_Earth_day_vec = TSI_1AU_day * (AU/d_sun_day)**2

    """ Old inaccurate mode:
    # NOTE: first block has a slight overlap with get_solar_incoming_day_hist, but we keep it like this because 
    # the SW outgoing needs zeroed_cos_theta_s_day_hist_toa and the 3D F vectors are not needed.
    t1 = time.time()
    u_sun_day = r_sun_day / (d_sun_day[:,None])
    #zeroed_cos_theta_s_day_hist_toa = get_zeroed_cos_theta_s_hist(r_sun_day, grid)
    zeroed_cos_theta_s_day_hist_toa = get_zeroed_cos_theta_s_hist_old(u_sun_day, grid)
    
    solar_incoming_day_hist_toa = zeroed_cos_theta_s_day_hist_toa * TSI_Earth_day_vec[None, :]
    t2 = time.time()
    print(f"Time to get solar_incoming_day_hist_toa inaccurate: {t2-t1}")
    """

    _, solar_incoming_day_hist_toa, zeroed_cos_theta_s_day_hist_toa = get_solar_incoming_day_hist(
                                                        r_sun_day, TSI_1AU_day, grid)

    
    day_datestr = Time(mid_day_jd, format='jd').to_datetime().strftime("%Y-%m-%d")

    #if not(rad_config.get('time_interp', False)):
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
    
    toa_emission_day_hist = LW_outgoing_day_hist_toa + SW_outgoing_day_hist_toa
    net_toa_day_hist = solar_incoming_day_hist_toa - toa_emission_day_hist

    toa_emission_day_hist = grid.reshape_if_needed(toa_emission_day_hist)
    net_toa_day_hist = grid.reshape_if_needed(net_toa_day_hist)

    return toa_emission_day_hist, net_toa_day_hist


def get_solar_incoming_day_hist(r_sun_day, TSI_1AU_day, grid: Grid):

    #r_sun_day = get_r_sun_day_hist(ephem_tag, mid_day_jd)

    grid_r_sun_hist, grid_d_sun_hist = get_grid_r_sun_hist(r_sun_day, grid.stacked_grid_r) #grid)
    TSI_grid_day_vec = TSI_1AU_day * (AU/grid_d_sun_hist)**2
    
    grid_u_sun_hist = get_grid_u_sun_hist(grid_r_sun_hist, grid_d_sun_hist)
    zeroed_cos_theta_s_day_hist = get_zeroed_cos_theta_s_hist(grid_u_sun_hist, grid)
    #zeroed_cos_theta_s_day_hist_toa_old = get_zeroed_cos_theta_s_hist_old(u_sun_day, grid) # if totally parallel sunrays are assumed

    grid_u_sun_hist = grid_u_sun_hist * (zeroed_cos_theta_s_day_hist[:,None,:]!=0)  # filter out eclipse
    
    solar_F_day_hist = - grid_u_sun_hist * TSI_grid_day_vec[:, None, :]  # # minus sign so that vecors are pointing AWAY from Sun
    solar_Fr_day_hist = zeroed_cos_theta_s_day_hist * TSI_grid_day_vec # eclipse is set to 0
    # IMPORTANT NOTE: this gives the same as: -np.sum(solar_F_day_hist * grid.stacked_grid_u[:,:,None], axis=1) (checked)
    # the minus sign is because we define POSITIVE RADIAL flux as going IN. We don't do this sign flip in the full F vectors (flux from the Sun points away from the Sun)!!!

    # NOTE: the above scales TSI at the Sun distance of each grid element instead of assuming the TSI at the center of the Earth for all of them.
    #       the latter approach would be done with the following lines. NOTE that the integrated difference seems to be non-negligible at the order of 0.02 W/m2
    # d_sun_day = np.sqrt(np.einsum('ij,ij->i', r_sun_day, r_sun_day)) # same as np.linalg.norm(r_sun_day,2,1) but slightly faster
    # TSI_Earth_day_vec = TSI_1AU_day * (AU/d_sun_day)**2
    # solar_F_day_hist = - grid_u_sun_hist * TSI_Earth_day_vec[None, None, :]  # # minus sign so that vecors are pointing AWAY from Sun
    # solar_Fr_day_hist = zeroed_cos_theta_s_day_hist * TSI_Earth_day_vec[None, :] # eclipse is set to 0
    
    return solar_F_day_hist, solar_Fr_day_hist, zeroed_cos_theta_s_day_hist


def irradiance_to_radiance(emission_hist, grid_u_sun_hist_toa=None, ADM_model=None):
    if ADM_model is None: # Lambertian emission model
        return (1/np.pi) * emission_hist


def get_cos_theta_s_lim(altitude_km):
    return -np.sqrt(1 - (RE/(RE + altitude_km))**2)     # this minus sign only if u_sun points towards the Sun
    # this formula is also valid for a non-infinitely-far-away Sun

def get_grid_u_sun_hist(grid_r_sun_hist, grid_d_sun_hist):
    # r_sun_day: time series of vectors from the CoM of Earth to the CoM of Sun
    # returns: time hist of unit vectors pointint TO the Sun from all grid points

    #grid_r_sun_hist, grid_d_sun_hist = get_grid_r_sun_hist(r_sun_day, grid)
    grid_u_sun_hist = grid_r_sun_hist / (grid_d_sun_hist[:,None,:])

    return grid_u_sun_hist
    
def get_grid_r_sun_hist(r_sun_day, stacked_r): # grid: Grid):
    # r_sun_day: time series of vectors from the CoM of Earth to the CoM of Sun
    grid_r_sun_hist = r_sun_day.T[None,:,:] - stacked_r[:,:,None] # grid.stacked_grid_r[:,:,None]
    grid_d_sun_hist = np.sqrt(np.einsum('ijk,ijk->ik',grid_r_sun_hist,grid_r_sun_hist)) # NOTE: equal to np.linalg.norm(grid_r_sun_hist, axis=1)

    return grid_r_sun_hist, grid_d_sun_hist

def get_grid_r_sun_step(r_sun, stacked_r):
    grid_r_sun = r_sun[None, :] - stacked_r
    grid_d_sun = np.sqrt(np.einsum('ij,ij->i',grid_r_sun,grid_r_sun))

    return grid_r_sun, grid_d_sun



def get_zeroed_cos_theta_s_hist(grid_u_sun_hist, grid: Grid):
    # grid_u_sun_hist: unit vectors pointing FROM each grid point TO the Sun at every time step
    #grid_u_sun_hist = get_grid_u_sun_hist(r_sun_day, grid)
    stacked_cos_theta_s_hist = np.einsum('ijk,ij->ik', grid_u_sun_hist, grid.stacked_grid_u)# same as: np.sum(grid_u_sun_hist * grid.stacked_grid_u[:,:,None], axis=1)

    cos_lim = get_cos_theta_s_lim(grid.alt_km)
    stacked_cos_theta_s_hist[stacked_cos_theta_s_hist < cos_lim] = 0 # no partial shadow, constant altitude
    
    return stacked_cos_theta_s_hist

def get_zeroed_cos_theta_s_step(grid_u_sun_step, stacked_grid_u, alt_km):

    stacked_cos_theta_s = np.einsum('ij,ij->i', grid_u_sun_step, stacked_grid_u) 
    cos_lim = get_cos_theta_s_lim(alt_km)
    stacked_cos_theta_s[stacked_cos_theta_s < cos_lim] = 0 # no partial shadow, constant altitude

    return stacked_cos_theta_s



def get_zeroed_cos_theta_s_hist_new(r_u_sun_hist, r_sat_hist): # TBC...
    # r_u_sun_hist: unit vectors pointing FROM each satellite point TO the Sun at every time step
    stacked_cos_theta_s_hist = np.einsum('ijk,ij->ik', grid_u_sun_hist, r_sat_hist) 
    # we're here - tbc...
    cos_lim = get_cos_theta_s_lim(grid.alt_km)
    stacked_cos_theta_s_hist[stacked_cos_theta_s_hist < cos_lim] = 0 # no partial shadow, constant altitude
    
    return stacked_cos_theta_s_hist


def get_zeroed_cos_theta_s_hist_old(u_sun_day, grid: Grid):
    
    stacked_cos_theta_s_day_hist = grid.stacked_grid_u @ np.transpose(u_sun_day)
    zeroed_cos_theta_s_day_hist = stacked_cos_theta_s_day_hist

    cos_lim = get_cos_theta_s_lim(grid.alt_km)
    zeroed_cos_theta_s_day_hist[stacked_cos_theta_s_day_hist < cos_lim] = 0     # this also changes cos_theta_s_day_hist but inside the function it doesn't really matter

    #if n_lat is not None and n_lon is not None:
    #    n_steps = np.shape(u_sun_day)[0]
    #    return np.reshape(zeroed_cos_theta_s_day_hist, (n_lat, n_lon, n_steps))
    #else:
    
    return zeroed_cos_theta_s_day_hist


def save_toa_files(toa_emission_day_hist, net_toa_day_hist, base_data_dir, day_idx, R_ECEF_to_SunFrame_day_hist, 
                   grid: Grid, n_cores=4, save_SFF_hist=True):
    
    all_file_names, file_existences = get_file_names_and_existence(base_data_dir, day_idx, grid)
    
    if not(file_existences['daily_hist_emission_toa']):
        print("Saving daily_hist_emission_toa file...")
        np.save(all_file_names['daily_hist_emission_toa'], toa_emission_day_hist)

    if not(file_existences['daily_hist_net_toa']):
        print("Saving daily_hist_net_toa file...")
        np.save(all_file_names['daily_hist_net_toa'], net_toa_day_hist)

    if not(file_existences.get('daily_hist_net_toa_SFF')) and save_SFF_hist:
        print("Saving daily_hist_net_toa file Sun-Fixed Frame...")
        net_toa_day_hist_SFF = field_hist_rotation_multiproc(net_toa_day_hist, R_ECEF_to_SunFrame_day_hist, 
                                                             grid, n_cores=n_cores)
        np.save(all_file_names['daily_hist_net_toa_SFF'], net_toa_day_hist_SFF)

    if not(file_existences.get('daily_hist_net_toa_lat_avg', True)): # quadrature grids don't have this so block is skipped
        print("Saving daily_hist_net_toa_lat_avg file...")
        assert(grid.grid_type_name=="regular_latlon")
        toa_lat_avg_day_hist = grid.compute_lat_avg_map(net_toa_day_hist)
        np.save(all_file_names['daily_hist_net_toa_lat_avg'], toa_lat_avg_day_hist)
    
    if not(file_existences['daily_hist_net_toa_surf_avg']):
        print("Saving daily_hist_net_toa_surf_avg file...")
        toa_surf_avg_day_hist = grid.compute_surf_integral(net_toa_day_hist, average=True)
        np.save(all_file_names['daily_hist_net_toa_surf_avg'], toa_surf_avg_day_hist)
    
    if not(file_existences.get('daily_avg_net_toa', True)): 
        print("Saving daily_avg_net_toa file...")
        net_toa_daily_avg = grid.compute_time_avg_map(net_toa_day_hist)
        np.save(all_file_names['daily_avg_net_toa'], net_toa_daily_avg)


def save_altitude_files(daily_hist_net_F, daily_hist_net_Fr, base_data_dir, day_idx, R_ECEF_to_SunFrame_day_hist, 
                        grid: Grid, n_cores=4, save_SFF_hist=True):
    
    all_file_names, file_existences = get_file_names_and_existence(base_data_dir, day_idx, grid)
    """
        "altitude": ['daily_hist_net_Fr_{alt_km}km', 'daily_hist_net_Fr_{alt_km}km_SFF',
                 'daily_hist_net_Fr_{alt_km}km_surf_avg',
                 'daily_avg_net_Fr_{alt_km}km', 'daily_avg_net_Fr_{alt_km}km_SFF',
                 'daily_hist_net_F_{alt_km}km', 'daily_hist_net_F_{alt_km}km_SFF',
                 'daily_avg_net_F_{alt_km}km', 'daily_avg_net_F_{alt_km}km_SFF']
    """
    if not(file_existences[f"daily_hist_net_Fr_{grid.alt_km}km"]):
        print("Saving: " + f"daily_hist_net_Fr_{grid.alt_km}km")
        np.save(all_file_names[f"daily_hist_net_Fr_{grid.alt_km}km"], daily_hist_net_Fr)

    if not(file_existences[f"daily_hist_net_Fr_{grid.alt_km}km_SFF"]):
        print("Saving: " + f"daily_hist_net_Fr_{grid.alt_km}km_SFF")
        daily_hist_net_Fr_SFF = field_hist_rotation_multiproc(daily_hist_net_Fr, R_ECEF_to_SunFrame_day_hist, 
                                                             grid, n_cores=n_cores)
        np.save(all_file_names[f"daily_hist_net_Fr_{grid.alt_km}km_SFF"], daily_hist_net_Fr_SFF)
    
    if not(file_existences[f"daily_hist_net_Fr_{grid.alt_km}km_surf_avg"]):
        scale = grid.total_area / (4 * np.pi * RE**2)
        daily_hist_net_Fr_surf_avg = grid.compute_surf_integral(daily_hist_net_Fr, average=True) * scale
        np.save(all_file_names[f"daily_hist_net_Fr_{grid.alt_km}km_surf_avg"], daily_hist_net_Fr_surf_avg)
    
    if not(file_existences[f"daily_avg_net_Fr_{grid.alt_km}km"]):
        daily_avg_net_Fr = grid.compute_time_avg_map(daily_hist_net_Fr)
        np.save(all_file_names[f"daily_avg_net_Fr_{grid.alt_km}km"], daily_avg_net_Fr)

    if not(file_existences[f"daily_avg_net_Fr_{grid.alt_km}km_SFF"]):
        if 'daily_hist_net_Fr_SFF' not in locals():
            daily_hist_net_Fr_SFF = np.load(all_file_names[f"daily_hist_net_Fr_{grid.alt_km}km_SFF"]+'.npy')

        daily_avg_net_Fr_SFF = grid.compute_time_avg_map(daily_hist_net_Fr_SFF)
        np.save(all_file_names[f"daily_avg_net_Fr_{grid.alt_km}km_SFF"], daily_avg_net_Fr_SFF)

    if not(file_existences[f"daily_hist_net_F_{grid.alt_km}km"]):
        print("Saving: " + f"daily_hist_net_F_{grid.alt_km}km")
        np.save(all_file_names[f"daily_hist_net_F_{grid.alt_km}km"], daily_hist_net_F)

        
def get_EEI_truth_daily_jd_arrays(EEI_truth_name):

    rad_config = rad_settings.radiation_settings_from_EEI_truth_name(EEI_truth_name)
    mid_day_jd_array = mid_day_jd_array_from_jd_interval(rad_config["jd_interval"])

    step_minutes = 1 # DO NOT CHANGE - must be equal to the one used for the files in solar_ephemerides
    step_days = step_minutes / (60*24)
    daily_jd_arrays = [get_day_jd_array(mid_day_jd, step_days) for mid_day_jd in mid_day_jd_array]
    
    return daily_jd_arrays

def get_day_jd_array(mid_day_jd, step_days=1/(60*24)):
    return np.arange(mid_day_jd-0.5, mid_day_jd+0.5, step_days)


def mid_day_jd_array_from_jd_interval(jd_interval):

    edges_jd_array = np.arange(jd_interval[0], jd_interval[-1]+1, 1)
    mid_day_jd_array = (edges_jd_array[1:] + edges_jd_array[:-1]) / 2

    return mid_day_jd_array

def get_mid_day_jd_array(EEI_truth_name):
    rad_config = rad_settings.radiation_settings_from_EEI_truth_name(EEI_truth_name)
    return mid_day_jd_array_from_jd_interval(rad_config["jd_interval"])


#def get_expanded_ae_maps(mode, datestr, Nmax, grid: Grid, normalization):
def get_expanded_ae_maps(datestr, grid: Grid, rad_config_dict: dict):

    #mode = rad_config_dict['sh_mode']
    #Nmax = rad_config_dict['Nmax']
    
    #a_sh_map, e_sh_map = rad_settings.get_ae_sh_maps_numpy_new(mode, datestr)
    #a_sh_map = a_sh_map[:, :(Nmax+1), :(Nmax+1)]
    #e_sh_map = e_sh_map[:, :(Nmax+1), :(Nmax+1)]

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


def get_smooth_daily_ae_hist(mid_day_jd, rad_config, grid:Grid=None):

    method=rad_config["time_interp"]
    if method=="map_interp":
        assert grid is not None
        full_lat, full_lon = grid.stacked_grid_latlon.T
    else:
        assert method=="sh_interp", "Method must be either 'map_interp' or 'sh_interp'"

    # step 1: load few previous and next days
    n = 3 # number of days to load to built interpolator
    jd_array = mid_day_jd + np.arange(-n, n+1, 1)
    first_jd = rad_config["jd_interval"][0] + 0.5
    last_jd = rad_config["jd_interval"][1] - 0.5
    jd_array = np.delete(jd_array, jd_array<first_jd)
    jd_array = np.delete(jd_array, jd_array>last_jd)

    # step 2: load (and expand?) a&e maps
    all_a = [None] * len(jd_array)
    all_e = [None] * len(jd_array)

    for i, jd in enumerate(jd_array):
        datestr = jd_to_mmddyyyy(jd)
        a, e = load_sh_maps(datestr, rad_config)
        if method=="map_interp":
            a = expand_sh(a, full_lon, full_lat, rad_config['sh_normalization'])
            e = expand_sh(e, full_lon, full_lat, rad_config['sh_normalization'])
        all_a[i] = a
        all_e[i] = e

    # step 3: make splines and interpolate
    spline_a = make_interp_spline(jd_array, np.stack(all_a), k=3)
    spline_e = make_interp_spline(jd_array, np.stack(all_e), k=3)

    day_jd_hist = get_day_jd_array(mid_day_jd)
    day_jd_hist[day_jd_hist < first_jd] = first_jd
    day_jd_hist[day_jd_hist > last_jd] = last_jd

    a_i_hist = spline_a(day_jd_hist, extrapolate=False)
    e_i_hist = spline_e(day_jd_hist, extrapolate=False)

    return a_i_hist, e_i_hist


def expand_daily_map_hist(sh_hist, grid: Grid, normalization):

    full_lat, full_lon = grid.stacked_grid_latlon.T
    n_steps = np.shape(sh_hist)[0]
    map_hist = np.zeros((grid.n_points, n_steps))

    for i in range(n_steps):
        progress_bar(i, n_steps)
        map_hist[:,i] = expand_sh(sh_hist[i], full_lon, full_lat, normalization)

    return map_hist




def get_r_sun_day_hist(ephem_tag, mid_day_jd):
    dir_r_sun = os.path.join(MEDIA_DIR, 'solar_ephemerides', ephem_tag, 'daily_files')
    jd_r_sun = np.load(os.path.join(dir_r_sun, 'jd_mid_day_array.npy'))
    day_idx_r_sun = find_day_idx_r_sun(jd_r_sun, mid_day_jd)
    r_sun_day = np.load(os.path.join(dir_r_sun, 'r_Sun_ECEF_day_'+str(day_idx_r_sun)+'.npy'))

    return r_sun_day


def get_rdu_sun_hist(ephem_tag, mid_day_jd): 

    r_sun_day = get_r_sun_day_hist(ephem_tag, mid_day_jd)
    
    d_sun_day = np.linalg.norm(r_sun_day,2,1)
    u_sun_day = r_sun_day / d_sun_day[:, None]

    return r_sun_day, d_sun_day, u_sun_day


def get_R_SunFrame_hist(ephem_tag, mid_day_jd): 

    dir_r_sun = os.path.join(MEDIA_DIR, 'solar_ephemerides', ephem_tag, 'daily_files')
    jd_r_sun = np.load(os.path.join(dir_r_sun, 'jd_mid_day_array.npy'))

    day_idx_r_sun = find_day_idx_r_sun(jd_r_sun, mid_day_jd)
    R_hist_day = np.load(os.path.join(dir_r_sun, 'R_ECEF_to_SunFrame_day_'+str(day_idx_r_sun)+'.npy'))

    return R_hist_day

def find_day_idx_r_sun(jd_r_sun, mid_day_jd):
    day_idx_r_sun = np.where(np.isin(jd_r_sun, mid_day_jd))[0][0]

    return day_idx_r_sun

#os.path.join(MEDIA_DIR, 'true_EEI', EEI_truth_name, grid.grid_name)

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


def get_subdir_EEI_truth(EEI_truth, series_type, grid_name="grid_quad_n131"):
    
    subdir = os.path.join(MEDIA_DIR, 'true_EEI', EEI_truth, grid_name, 
                              series_type)
    return subdir


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



if __name__=="__main__":
    EEI_truth_name = "EEI_truth_1"
    #grid = RegularLatLonGrid(alt_km=800, n_lat=18, n_lon=36)
    grid = QuadratureGrid(alt_km=800, order=131)
    compute_radiation_maps(EEI_truth_name, grid=grid, selected_days_idxs=[0,182], n_cores=6)