from SpaceBalls.postproc_EEI_estimation import generate_constellation_day_animations, generate_constellation_stacking_animations

import sys, os
import numpy as np
import SpaceBalls.output_database_manager as db_manager
from SpaceBalls.plotter import Plotter
from SpaceBalls.radiation_settings import radiation_settings_from_EEI_truth_name
from SpaceBalls.postproc_EEI_estimation import generate_constellation_stacking_animations, generate_error_map_animations
from SpaceBalls.radiation_fluxes_preprocessing import get_EEI_truth_daily_jd_arrays, get_mid_day_jd_array
from SpaceBalls.postproc_EEI_estimation import get_true_EEI_time_series, check_EEI_consistency, get_all_jd_windows, mid_day_jd_array_from_jd_interval
from SpaceBalls.utils import normal_smoother
from SpaceBalls.paths import CONFIG_DIR, MEDIA_DIR
from SpaceBalls.sph_meshing import QuadratureGrid
import time


# main routine
#RMS_lim = #0.08
#window_days = #365
#window_name = str(window_days) + '-day'

EEI_name = "EEI_truth_1"
#TODO: fix code duplication with compute_rolling_avg_error_time_series
#full_jd_array = np.concatenate(get_EEI_truth_daily_jd_arrays(EEI_name))
#mid_day_jd_array = get_mid_day_jd_array(EEI_name)
#EEI_time_series_180x360 = get_true_EEI_time_series(EEI_name)

#EEI_smooth_true = normal_smoother(EEI_time_series_180x360, full_jd_array, window_width_days=window_days, 
#                                out_length_mode='same_with_nans')
#EEI_smooth_true_coarse = np.interp(mid_day_jd_array, full_jd_array, EEI_smooth_true)
# end TODO

#error_RMS_column_name = 'error_RMS_'+str(window_days)+'_day_avg'
##results = db_manager.retrieve_column('EEI_postproc_results_table', error_RMS_column_name)

#keys = db_manager.retrieve_column_from_condition('EEI_postproc_results_table', 'estimation_key',
#                                                    error_RMS_column_name, RMS_lim, '<') 
#satellites = db_manager.retrieve_column_from_condition('config_table', 'satellites', 'estimation_key')


results = db_manager.retrieve_best_rows(window_days=365, RMS_lim=2, n_best=3, 
                                        config_filters={'use_recomp_meas': 1,
                                                          'n_sb': 3})
db_manager.conn.close()

grid = QuadratureGrid(alt_km=800, order=131) # TODO: actual grid from estimation settings

for row in results:
    key, rms, constellation_name = row
    sb_names = constellation_name.split("|")

    for window_days in [30, 365]:
        EEI_name = check_EEI_consistency(sb_names)
        jd_windows = get_all_jd_windows(EEI_name, window_days)

        for frame in ['SFF', 'ECEF']:
            #print(f"Generating errormap animation for estimation key {key} in {frame}")
            #generate_error_map_animations(sb_names, key, jd_windows, grid, frame=frame)

            print(f"Generating stacking animation for estimation key {key} in {frame}")
            #rad_config = radiation_settings_from_EEI_truth_name(EEI_name)
            #EEI_full_window = rad_config["jd_interval"]
            #mid_day_jd = mid_day_jd_array_from_jd_interval(EEI_full_window)
            #fake_stacking_windows = [[EEI_full_window[0], jd] for jd in mid_day_jd]
            
            generate_constellation_stacking_animations(sb_names, key, jd_windows, grid)
            


    # generate_constellation_stacking_animations('case_5_years', sc_names=constellation_name.split("|"), 
    #                                         constellation_tag=constellation_name,
    #                                         n_days=365, n_lon=360, n_lat=180, frame='SFF')





#generate_constellation_stacking_animations('case_5_years', sc_names=['sc_C1', 'sc_C2', 'sc_C3'], 
#                                           constellation_tag='constellation_C',
#                                           n_days=365, n_lon=360, n_lat=180, frame='SFF')
#
#generate_constellation_day_animations('case_5_years', sc_names=['sc_C1', 'sc_C2', 'sc_C3'],
#                                        constellation_tag='constellation_C',
#                                        day_idxs=[0, 182], max_hours=12)

