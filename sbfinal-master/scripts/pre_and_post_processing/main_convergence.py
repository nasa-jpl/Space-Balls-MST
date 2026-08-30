import numpy as np
import sys, os
from SpaceBalls.sph_meshing import RegularLatLonGrid, QuadratureGrid, KnockeGridMONTE
from SpaceBalls.radiation_fluxes_preprocessing import compute_daily_hist_at_sat_r_hist
import SpaceBalls.input_database_manager as input_manager
from SpaceBalls.plotter import Plotter
from SpaceBalls.postproc_EEI_estimation import get_acc_to_flux_factor
import config.constants as constants
import itertools
import random
from SpaceBalls.paths import CONFIG_DIR, MEDIA_DIR, INPUT_DIR, OUTPUT_DIR
lebedev_n_array = [41, 47, 53, 59, 65, 71, 77, 83,
                    89, 95, 101, 107, 113, 119, 125, 131]

def get_n_points_array(grid_array):
    
    n_points = np.zeros(len(grid_array))
    for i, grid in enumerate(grid_array):
        n_points[i] = grid.n_points
    
    return np.array(n_points)

EEI_name = "EEI_truth_15"
base_sb_tag = 'GRACE-FO_spherical' # 'Ajisai' # 'LAGEOS' # 
n_days_arc = 30
n_propagations = None
out_dir = os.path.join(MEDIA_DIR, 'convergence_analysis', base_sb_tag, EEI_name)
os.makedirs(out_dir, exist_ok=True)

# TODO: if EEI truth sets WGS84, function that retrieves f
flatt = 1/298.257223563

"""
true_r_hist = np.vstack([np.load(os.path.join(OUTPUT_DIR, base_sb_tag + '_' + str(100+n_propagations-1), 
                         'day_'+str(day_idx), 'xyz_ecef.npy')) for day_idx in range(n_days_arc)]) # finest propagation in MONTE 
jd_hist = np.vstack([np.load(os.path.join(OUTPUT_DIR, base_sb_tag + '_' + str(100+n_propagations-1), 
                         'day_'+str(day_idx), 'jd_vec.npy')) for day_idx in range(n_days_arc)]) # finest propagation in MONTE 
"""


#finest_regular_grid = RegularLatLonGrid(0, n_lat=180*25, n_lon=360*25, flattening=flatt)
finest_regular_grid = RegularLatLonGrid(0, lmax=2500, quad_type='GLQ', flattening=flatt)
#finest_regular_grid = QuadratureGrid(0, order=131)

regular_grids = [RegularLatLonGrid(0,  n_lat=int(np.round(180*f)), 
                                        n_lon=int(np.round(360*f)),
                                        flattening=flatt) for f in np.logspace(-1, 0.5, 15)]

lebedev_grids = [QuadratureGrid(0, order=o, flattening=flatt) for o in lebedev_n_array]
womersley_grids = [QuadratureGrid(0, order=o, default_womersley=True, flattening=flatt) for o in np.arange(33, 325, 4)]

n_points_glq = np.logspace(np.log10(regular_grids[0].n_points), np.log10(regular_grids[-1].n_points), 2*len(regular_grids))
l_glq = np.round(0.25 * (-3 + np.sqrt(1 + 8*n_points_glq)))
glq_grids = [RegularLatLonGrid(0, lmax=int(l), quad_type='GLQ',
                               flattening=flatt) for l in l_glq]

N_array = np.round(np.logspace(1.4, 5, 20))
nR_array = np.round(-0.5 + 0.5*np.sqrt((4*N_array - 1)/3))
nR_array = nR_array[:18]
knocke_monte_grids = [KnockeGridMONTE(int(nR), flattening=flatt) for nR in nR_array]

all_grid_arrays = [regular_grids, womersley_grids, lebedev_grids, glq_grids,
                   knocke_monte_grids]

grid_type_names = ['regular', 'womersley', 'lebedev', 'GLQ', 'Knocke MONTE']

np.save(os.path.join(out_dir, 'n_points_regular_grids'), get_n_points_array(regular_grids))
np.save(os.path.join(out_dir, 'n_points_womersley_grids'), get_n_points_array(womersley_grids))
np.save(os.path.join(out_dir, 'n_points_lebedev_grids'), get_n_points_array(lebedev_grids))
np.save(os.path.join(out_dir, 'n_points_glq_grids'), get_n_points_array(glq_grids))
np.save(os.path.join(out_dir, 'n_points_knocke_monte_grids'), get_n_points_array(knocke_monte_grids))


if n_propagations is not None: 
    monte_prop_names = [base_sb_tag + '_' + str(100+i) for i in range(n_propagations)]
    #all_Fr_error_time_series_monte =  [[] for _ in range(len(monte_prop_names))]



for day_idx in range(n_days_arc):
    print(f"Computing convergence day {day_idx}...")
    name_best_prop = base_sb_tag + '_' + str(100+n_propagations-1)if n_propagations is not None else base_sb_tag

    true_r_hist = np.load(os.path.join(OUTPUT_DIR, name_best_prop, 
                         'day_'+str(day_idx), 'xyz_ecef.npy'))
    jd_hist = np.load(os.path.join(OUTPUT_DIR, name_best_prop, 
                         'day_'+str(day_idx), 'jd_vec.npy')) 
    
    out_dir_day = os.path.join(out_dir, 'day_'+str(day_idx))
    os.makedirs(out_dir_day, exist_ok=True)
    errors_regular_grids_fname = os.path.join(out_dir_day, 'errors_Fr_regular_grids')
    errors_womersley_grids_fname = os.path.join(out_dir_day, 'errors_Fr_womersley_grids')
    errors_lebedev_grids_fname = os.path.join(out_dir_day, 'errors_Fr_lebedev_grids')
    errors_glq_grids_fname = os.path.join(out_dir_day, 'errors_Fr_glq_grids')
    errors_knocke_monte_grids_fname = os.path.join(out_dir_day, 'errors_Fr_knocke_monte_grids')
    all_errors_fnames = [errors_regular_grids_fname, errors_womersley_grids_fname,
                         errors_lebedev_grids_fname, errors_glq_grids_fname,
                         errors_knocke_monte_grids_fname]
    
    true_erp_Fr_fname = os.path.join(out_dir_day, 'true_erp_Fr.npy')
    if os.path.exists(true_erp_Fr_fname):
        print(f"Loading true Fr hist on day {day_idx}")
        true_erp_Fr = np.load(true_erp_Fr_fname)
    else:
        print(f"File {true_erp_Fr_fname} not found")
        print(f"Computing true Fr hist on day {day_idx}")
        true_erp_Fr, true_srp_Fr = compute_daily_hist_at_sat_r_hist(
            EEI_name, true_r_hist, jd_hist, day_idx, toa_grid=finest_regular_grid)
        np.save(true_erp_Fr_fname, true_erp_Fr) 
    
    if n_propagations is not None: 
        all_Fr_error_time_series_monte = [None] * len(monte_prop_names)
        for i, prop_name in enumerate(monte_prop_names):
            erp_acc = np.load(os.path.join(OUTPUT_DIR, prop_name,'day_' + str(day_idx), 'erp.npy')) * 1e3
            erp_Fr = erp_acc[:,0] * get_acc_to_flux_factor(prop_name)
            all_Fr_error_time_series_monte[i] = erp_Fr - true_erp_Fr
        print(f"Saving MONTE errors day {day_idx}")
        np.save(os.path.join(out_dir_day, 'errors_Fr_monte_grids'), np.array(all_Fr_error_time_series_monte))

            #all_Fr_error_time_series_monte[i].append(erp_Fr - true_erp_Fr)
            #print(erp_Fr - true_erp_Fr)

    for errors_fname, grid_array, grid_type_name in zip(all_errors_fnames, all_grid_arrays, grid_type_names):
        if not(os.path.exists(errors_fname+'.npy')):
            #print(f"grid_array: {grid_array}")
            print(f"Computing error time series for {grid_type_name} grid")
            all_Fr_error_time_series = [None] * len(grid_array)
            for i, grid in enumerate(grid_array):
                print(grid)
                print(f"n points: {grid.n_points}")
                erp_Fr, srp_Fr = compute_daily_hist_at_sat_r_hist(
                    EEI_name, true_r_hist, jd_hist, day_idx, toa_grid=grid)
                all_Fr_error_time_series[i] = erp_Fr - true_erp_Fr
            
            np.save(errors_fname, np.array(all_Fr_error_time_series))
            
    """
    if not(os.path.exists(errors_regular_grids_fname)):
        all_Fr_error_time_series_regular = [None] * len(regular_grids)
        for i, grid in enumerate(regular_grids):
            #print(i)
            erp_Fr, srp_Fr = compute_daily_hist_at_sat_r_hist(
                EEI_name, true_r_hist, jd_hist, day_idx, toa_grid=grid)
            all_Fr_error_time_series_regular[i] = erp_Fr - true_erp_Fr
        
        np.save(errors_regular_grids_fname, np.array(all_Fr_error_time_series_regular))
            
            #all_Fr_error_time_series_regular[i].append(erp_Fr - true_erp_Fr)

    if not(os.path.exists(errors_womersley_grids_fname)):
        all_Fr_error_time_series_womersley = [None] * len(womersley_grids)
        for i, grid in enumerate(womersley_grids):
            erp_Fr, srp_Fr = compute_daily_hist_at_sat_r_hist(
                EEI_name, true_r_hist, jd_hist, day_idx, toa_grid=grid)
            #all_Fr_error_time_series_womersley[i].append(erp_Fr - true_erp_Fr)
            all_Fr_error_time_series_womersley[i] = erp_Fr - true_erp_Fr
        np.save(errors_womersley_grids_fname, np.array(all_Fr_error_time_series_womersley))
    
    if not(os.path.exists(errors_lebedev_grids_fname)):
        all_Fr_error_time_series_lebedev = [None] * len(lebedev_grids)
        for i, grid in enumerate(lebedev_grids):
            erp_Fr, srp_Fr = compute_daily_hist_at_sat_r_hist(
                EEI_name, true_r_hist, jd_hist, day_idx, toa_grid=grid)
            all_Fr_error_time_series_lebedev[i] = erp_Fr - true_erp_Fr
        
        np.save(errors_lebedev_grids_fname, np.array(all_Fr_error_time_series_lebedev))

    if not(os.path.exists(errors_glq_grids_fname)):
        all_Fr_error_time_series_glq = [None] * len(glq_grids)
        for i, grid in enumerate(glq_grids):
            print(f"Computing GLQ grid errors grid {i}")
            erp_Fr, srp_Fr = compute_daily_hist_at_sat_r_hist(
                EEI_name, true_r_hist, jd_hist, day_idx, toa_grid=grid)
            all_Fr_error_time_series_glq[i] = erp_Fr - true_erp_Fr
    
        np.save(errors_glq_grids_fname, np.array(all_Fr_error_time_series_glq))

    if not(os.path.exists(errors_knocke_monte_grids_fname)):
        all_Fr_error_time_series_knocke_monte = [None] * len(knocke_monte_grids)
        for i, grid in enumerate(knocke_monte_grids):
            print(f"Computing Knocke MONTE grid errors grid {i}")
            erp_Fr, srp_Fr = compute_daily_hist_at_sat_r_hist(
                EEI_name, true_r_hist, jd_hist, day_idx, toa_grid=grid)
            all_Fr_error_time_series_knocke_monte[i] = erp_Fr - true_erp_Fr
    
        np.save(errors_knocke_monte_grids_fname, np.array(all_Fr_error_time_series_knocke_monte))
    """




#if n_propagations is not None: 
#    n_points_monte = np.zeros(len(monte_prop_names))
#    for i, prop_name in enumerate(monte_prop_names):
#        all_Fr_error_time_series_monte[i] = np.concatenate(all_Fr_error_time_series_monte[i])
#        n_rings = input_manager.get_dicts(prop_name)['integration_settings']['n_rings']
#        n_points_monte[i] = 1 + 3*n_rings * (1+n_rings)
#
#    #np.save(os.path.join(out_dir, 'errors_monte_grids'), np.array(all_Fr_error_time_series_monte))
#    np.save(os.path.join(out_dir, 'n_points_monte_grids'), n_points_monte)






