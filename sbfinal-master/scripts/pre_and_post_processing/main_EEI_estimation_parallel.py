import sys, os
import numpy as np
import itertools
import time

from SpaceBalls.postproc_EEI_estimation import compute_rolling_avg_error_time_series, get_true_EEI_time_series, compute_sample_quality_metrics
from SpaceBalls.radiation_fluxes_preprocessing import get_EEI_truth_daily_jd_arrays, get_mid_day_jd_array
from SpaceBalls.utils import nanrms, make_dict_hash_key, make_list_str_key
import config.constants as constants
from SpaceBalls.paths import CONFIG_DIR, MEDIA_DIR
from SpaceBalls.plotter import Plotter

from multiprocessing import Pool

estimation_out_dir = os.path.join(MEDIA_DIR, 'EEI_estimations')

k = 3 # number of satellites in constellation
nproc = 1
smoothing_windows_days = [365, 182, 30]

EEI_name = "EEI_truth_1"
full_jd_array = np.concatenate(get_EEI_truth_daily_jd_arrays(EEI_name))
mid_day_jd_array = get_mid_day_jd_array(EEI_name)
EEI_time_series_180x360 = get_true_EEI_time_series(EEI_name, 360, 180)


def turn_tuple_list_into_list_list(tuple_list):
    return [[list(tup)] for tup in tuple_list]

_global = {}
def init_worker(old_name, window_days, col_name, error_time_series_dir):

    _global['old_name'] = old_name
    _global['window_days'] = window_days
    _global['col_name'] = col_name
    _global['error_time_series_dir'] = error_time_series_dir

def store_results_to_database(sat_names):

    import SpaceBalls.output_database_manager as db_manager

    SB_constellation = ['sc_' + sc_name for sc_name in sat_names]
    #SB_constellation = ['sc_H1', 'sc_H2', 'sc_H3']

    estimation_config_dict = {'EEI_truth_name': EEI_name, 
            'satellite_list': SB_constellation,
            'accelerometer_errors': {'drift': None,
                                    'scale': None,
                                    'noise': None},
            'method': {'SH_field_fit': {'lmax': 15}},    
            'model_errors': {}}
    config_hash_key = make_dict_hash_key(estimation_config_dict)
    
    # config table (including acc and model error tables)
    db_manager.insert_config_entry(estimation_config_dict, config_hash_key)

    # sample quality table
    constellation_key = make_list_str_key(SB_constellation)
    if not db_manager.check_if_entry_exists('sample_quality_table', 'constellation_key', constellation_key):
        # TODO: empty functions
        compute_sample_quality_metrics(SB_constellation)
        db_manager.insert_sample_quality_entry()
    
    # results table
    row_exists = db_manager.check_if_entry_exists('EEI_postproc_results_table', 'estimation_key', config_hash_key)
    cell_exists = False if not(row_exists) else db_manager.check_if_entry_is_not_none('EEI_postproc_results_table', 'estimation_key', 
                                                                                        config_hash_key, _global['col_name'])
    if (not row_exists) and (not cell_exists):

        # check if results already exist from deprecated routines:
        constellation_name = '|'.join(sat_names)
        old_folder = os.path.join(estimation_out_dir, constellation_name)
        old_file_name = os.path.join(old_folder, 'error_time_series_'+_global["old_name"]+'.npy')
        if os.path.exists(old_file_name):
            error_time_series = np.load(old_file_name)
        else:
            error_time_series = compute_rolling_avg_error_time_series(SB_constellation, _global["window_days"], EEI_time_series_180x360, 
                                                                    full_jd_array, mid_day_jd_array)
            
        error_rms = nanrms(error_time_series)
        print(f"Error time series obtained for constellation {SB_constellation}. Writing rms to database...")

        db_manager.insert_results_entry(config_hash_key, 'error_RMS_'+str(_global["window_days"])+'_day_avg', error_rms)
        np.save(os.path.join(_global['error_time_series_dir'], config_hash_key+'.npy'), error_time_series)

        if os.path.exists(old_file_name): os.remove(old_file_name)

    #db_manager.conn.close()


if __name__ == '__main__':

    all_sc_tags = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'C1', 'C2', 'C3', 'D1', 'D2', 'D3', 
                'E1', 'E2', 'E3', 'F1', 'F2', 'F3', 'F11', 'F12', 'G1', 'G2', 'G3', 'H1', 'H2', 'H3', 'I1', 'I2', 'I3']
    
    all_ksat_combinations = list(itertools.combinations((all_sc_tags), k))

    smoothing_windows_names = [str(days)+'_days' for days in smoothing_windows_days]

    for window_name, window_days in zip(smoothing_windows_names, smoothing_windows_days):
        error_time_series_dir = os.path.join(estimation_out_dir, str(window_days)+'-day_avg_error_time_series')
        os.makedirs(error_time_series_dir, exist_ok=True)

        if window_days==365: old_name = "1-year"
        if window_days==182: old_name = "6-month"
        if window_days==30:  old_name = "1-month"
        col_name = 'error_RMS_'+str(window_days)+'_day_avg'

        with Pool(processes=nproc, initializer=init_worker, initargs=(old_name,window_days,col_name, error_time_series_dir)) as pool:
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