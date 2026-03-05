from SpaceBalls.postproc_EEI_estimation import generate_constellation_day_animations, generate_constellation_stacking_animations

import sys, os
import numpy as np
import SpaceBalls.output_database_manager as db_manager
from SpaceBalls.plotter import Plotter
from SpaceBalls.postproc_EEI_estimation import generate_constellation_stacking_animations
from SpaceBalls.radiation_fluxes_preprocessing import get_EEI_truth_daily_jd_arrays, get_mid_day_jd_array
from SpaceBalls.postproc_EEI_estimation import get_true_EEI_time_series
from SpaceBalls.utils import normal_smoother
from SpaceBalls.paths import CONFIG_DIR, MEDIA_DIR
import time

RMS_lim = 0.15
window_days = 365
window_name = str(window_days) + '-day'

EEI_name = "EEI_truth_1"
#TODO: fix code duplication with compute_rolling_avg_error_time_series
full_jd_array = np.concatenate(get_EEI_truth_daily_jd_arrays(EEI_name))
mid_day_jd_array = get_mid_day_jd_array(EEI_name)
EEI_time_series_180x360 = get_true_EEI_time_series(EEI_name, 360, 180)

EEI_smooth_true = normal_smoother(EEI_time_series_180x360, full_jd_array, window_width_days=window_days, 
                                out_length_mode='same_with_nans')
EEI_smooth_true_coarse = np.interp(mid_day_jd_array, full_jd_array, EEI_smooth_true)
# end TODO

error_RMS_column_name = 'error_RMS_'+str(window_days)+'_day_avg'
results = db_manager.retrieve_column('EEI_postproc_results_table', error_RMS_column_name)

keys = db_manager.retrieve_column_from_condition('EEI_postproc_results_table', 'estimation_key',
                                                    error_RMS_column_name, 0.15, '<')
#satellites = db_manager.retrieve_column_from_condition('config_table', 'satellites', 'estimation_key')

results = db_manager.retrieve_columns_from_condition('EEI_postproc_results_table',
                                                     'config_table',
                                                     error_RMS_column_name,
                                                     'satellites',
                                                     error_RMS_column_name,
                                                     RMS_lim, 
                                                     'estimation_key',
                                                     '<')

db_manager.conn.close()

for row in results:
    key, rms, constellation_name = row
    generate_constellation_stacking_animations('case_5_years', sc_names=constellation_name.split("|"), 
                                            constellation_tag=constellation_name,
                                            n_days=365, n_lon=360, n_lat=180, frame='SFF')





#generate_constellation_stacking_animations('case_5_years', sc_names=['sc_C1', 'sc_C2', 'sc_C3'], 
#                                           constellation_tag='constellation_C',
#                                           n_days=365, n_lon=360, n_lat=180, frame='SFF')
#
#generate_constellation_day_animations('case_5_years', sc_names=['sc_C1', 'sc_C2', 'sc_C3'],
#                                        constellation_tag='constellation_C',
#                                        day_idxs=[0, 182], max_hours=12)

