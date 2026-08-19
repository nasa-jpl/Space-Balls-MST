import sys, os
import numpy as np
import itertools
import time

from SpaceBalls.sph_meshing import Grid, QuadratureGrid, RegularLatLonGrid
from SpaceBalls.postproc_EEI_estimation import compute_rolling_avg_error_time_series, get_true_EEI_time_series #, compute_sample_quality_metrics
from SpaceBalls.radiation_fluxes_preprocessing import get_EEI_truth_daily_jd_arrays, get_mid_day_jd_array
from SpaceBalls.utils import nanrms, make_dict_hash_key, make_list_str_key
import config.constants as constants
from SpaceBalls.paths import CONFIG_DIR, MEDIA_DIR, OUTPUT_DIR
from SpaceBalls.plotter import Plotter

from multiprocessing import Pool

estimation_out_dir = os.path.join(MEDIA_DIR, 'EEI_estimations')

k = 3 # number of satellites in constellation
nproc = 12
smoothing_windows_days = [365, 182, 30]

grid = QuadratureGrid(alt_km=800, order=131)
frame = 'SFF'
method = 'binary_gridding'
fill_zeroes_method = 'theta_s_fit_simplified'
avg_method = 'numeric_integral'
meas_source = 'recomp_radial'  # 'recomp_radial' or 'monte'
# TODO: should be 'recomp_radial_131'

EEI_name = "EEI_truth_1" # for truth and recomputed measurements if needed
full_jd_array = np.concatenate(get_EEI_truth_daily_jd_arrays(EEI_name))
mid_day_jd_array = get_mid_day_jd_array(EEI_name)
EEI_time_series_truth = get_true_EEI_time_series(EEI_name)
# TODO: if EEI_name does not coincide with force_settings EEI, add it in meas_source!


def turn_tuple_list_into_list_list(tuple_list):
    return [[list(tup)] for tup in tuple_list]

_global = {}
def init_worker(window_days, col_name, error_time_series_dir, 
                grid: Grid, frame, method, fill_zeroes_method, avg_method, 
                meas_source):

    #_global['old_name'] = old_name
    _global['window_days'] = window_days
    _global['col_name'] = col_name
    _global['error_time_series_dir'] = error_time_series_dir
    _global["grid"] = grid
    _global["frame"] = frame
    _global["method"] = method
    _global["fill_zeroes_method"] = fill_zeroes_method
    _global["avg_method"] = avg_method
    _global["meas_source"] = meas_source

def store_results_to_database(SB_constellation):

    import SpaceBalls.output_database_manager as db_manager

    estimation_config_dict = {'EEI_truth_name': EEI_name, 
            'satellite_list': SB_constellation, # sort missing!
            'accelerometer_errors': {'drift': None,
                                    'scale': None,
                                    'noise': None},
            'model_errors': {},
            'stacking_frame': _global["frame"],
            'grid_name': _global["grid"].grid_name,
            'method': _global["method"],
            'fill_zeroes_method': _global["fill_zeroes_method"],
            'avg_method': _global["avg_method"],
            'meas_source': _global["meas_source"]}
    
    if estimation_config_dict['meas_source']=="monte": # legacy: keys were generated when this was assumed
        del(estimation_config_dict['meas_source'])
    config_hash_key = make_dict_hash_key(estimation_config_dict)
    
    # config table (including acc and model error tables)
    db_manager.insert_config_entry(estimation_config_dict, config_hash_key)

    # sample quality table
    constellation_key = make_list_str_key(SB_constellation)
    if not db_manager.check_if_entry_exists('sample_quality_table', 'constellation_key', constellation_key):
        # TODO: empty functions
        #compute_sample_quality_metrics(SB_constellation)
        db_manager.insert_sample_quality_entry()
    
    # results table
    row_exists = db_manager.check_if_entry_exists('EEI_postproc_results_table', 'estimation_key', config_hash_key)
    cell_exists = False if not(row_exists) else db_manager.check_if_entry_is_not_none('EEI_postproc_results_table',                 
                                                                                      'estimation_key', config_hash_key, _global['col_name'])
    if (not cell_exists):

        #constellation_name = '|'.join(sat_names)
        #error_time_series = compute_rolling_avg_error_time_series(SB_constellation, _global["window_days"], EEI_time_series_truth, 
        #                                                        full_jd_array, mid_day_jd_array)
        
        error_time_series = compute_rolling_avg_error_time_series(SB_constellation, _global["window_days"], _global["grid"], 
                                          _global["frame"], _global["method"],  _global["fill_zeroes_method"], 
                                          _global["avg_method"], _global["meas_source"],
                                          EEI_time_series_truth, full_jd_array, mid_day_jd_array)

        error_rms = nanrms(error_time_series)
        print(f"Error time series obtained for constellation {SB_constellation}. Writing rms to database...")

        db_manager.insert_results_entry(config_hash_key, 'error_RMS_'+str(_global["window_days"])+'_day_avg', error_rms)
        np.save(os.path.join(_global['error_time_series_dir'], config_hash_key+'.npy'), error_time_series)


    #db_manager.conn.close()


if __name__ == '__main__':

    #all_sc_tags = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'C1', 'C2', 'C3', 'D1', 'D2', 'D3', 
    #            'E1', 'E2', 'E3', 'F1', 'F2', 'F3', 'F11', 'F12', 'G1', 'G2', 'G3', 'H1', 'H2', 'H3', 'I1', 'I2', 'I3']
    import SpaceBalls.input_database_manager as input_manager
    """
    all_primary_keys = input_manager.get_primary_keys(filters={'force_settings': 2,
                                                               'integration_settings': 0},
                                                  table_name='sb_table',
                                                  pk_column='sb_hash')
    print(f"Total number of SB in pool: {len(all_primary_keys)}")
    completed_bool_array = [os.path.exists(os.path.join(OUTPUT_DIR, key, 'day_1825', 'erp.npy')) for key in all_primary_keys]
    all_primary_keys = [key for (key, comp) in zip(all_primary_keys, completed_bool_array) if comp]
    """
    orbits = ['R800_' + str(i) for i in range(36)] # ['G3', 'A1', 'F12'] # ['C1', 'I3', 'F2']
    all_primary_keys = np.concatenate([input_manager.get_primary_keys({'orbit_name': orb,
                                                        'force_settings': 2,
                                                        #'integration_settings': 0,
                                                        }) for orb in orbits])

    all_ksat_combinations = list(itertools.combinations((all_primary_keys), k))

    smoothing_windows_names = [str(days)+'_days' for days in smoothing_windows_days]

    for window_name, window_days in zip(smoothing_windows_names, smoothing_windows_days):
        error_time_series_dir = os.path.join(estimation_out_dir, str(window_days)+'-day_avg_error_time_series')
        os.makedirs(error_time_series_dir, exist_ok=True)

        col_name = 'error_RMS_'+str(window_days)+'_day_avg'

        with Pool(processes=nproc, initializer=init_worker, initargs=(window_days, col_name, error_time_series_dir,
                                                                      grid, frame, method, fill_zeroes_method, 
                                                                      avg_method, meas_source)) as pool:
            pool.starmap(store_results_to_database, turn_tuple_list_into_list_list(all_ksat_combinations))




# Plotter.plot_time_series(mid_day_jd_array,
#                     {'True EEI ('+window_name+' avg.)': EEI_smooth_true_coarse,
#                     'Estimated EEI('+window_name+' avg.)': EEI_smooth_estimated},
#                     f_width=2.5, f_height=0.5,
#                     ylabel='Net TOA flux (W/m$^2$)',
#                     y_scatter_dict={'$\phi(t)$ (W/m$^2$)': (full_jd_array, EEI_time_series_180x360)},
#                     scatter_alpha=0.007,
#                     out_dir=os.path.join(MEDIA_DIR, 'figures', constellation_name),
#                     file_name='EEI_estimate_'+window_name,
#                     title=constellation_name)

# Plotter.plot_time_series(mid_day_jd_array,
#                     {'Estimation error': error_time_series},
#                     f_width=2.5, f_height=0.5,
#                     ylabel='Error (W/m$^2$)',
#                     out_dir=os.path.join(MEDIA_DIR, 'figures', constellation_name),
#                     file_name='error_'+window_name)