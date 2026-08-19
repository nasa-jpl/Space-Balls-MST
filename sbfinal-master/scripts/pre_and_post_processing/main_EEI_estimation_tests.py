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
from SpaceBalls.sph_meshing import RegularLatLonGrid, QuadratureGrid

import numpy as np
import SpaceBalls.output_database_manager as db_manager
import SpaceBalls.input_database_manager as input_manager
from SpaceBalls.postproc_EEI_estimation import compute_rolling_avg_error_time_series, get_true_EEI_time_series #, compute_sample_quality_metrics

estimation_out_dir = os.path.join(MEDIA_DIR, 'EEI_estimations')
EEI_name = "EEI_truth_1"
full_jd_array = np.concatenate(get_EEI_truth_daily_jd_arrays(EEI_name))
mid_day_jd_array = get_mid_day_jd_array(EEI_name)
EEI_time_series_truth = get_true_EEI_time_series(EEI_name)

frame = 'SFF'
fill_zeroes_method = 'theta_s_fit_simplified'
avg_method = 'numeric_integral'
meas_source = 'recomp_radial'

windows = [365, 182, 30]
grids = [#RegularLatLonGrid(alt_km=800, n_lat=180, n_lon=360),
         QuadratureGrid(alt_km=800, order=131),
         QuadratureGrid(alt_km=800, order=325)]
methods = ['binary_gridding',
           'gaussian_1', 'gaussian_2', 'gaussian_3', 'gaussian_4',
           'hanning_2.5', 'hanning_5', 'hanning_7.5', 'hanning_10']

for window_days in windows:

    best_results = db_manager.retrieve_best_rows(window_days, 5, 5, 
                                                {'use_recomp_meas': 1,
                                                'n_sb': 3})
    constellations = [row[2].split('|') for row in best_results]
    error_time_series_dir = os.path.join(estimation_out_dir, str(window_days)+'-day_avg_error_time_series')
    os.makedirs(error_time_series_dir, exist_ok=True)

    for SB_constellation in constellations:
        for grid in grids:
            for method in methods:

                estimation_config_dict = {'EEI_truth_name': EEI_name, 
                    'satellite_list': SB_constellation, # sort missing!
                    'accelerometer_errors': {'drift': None,
                                            'scale': None,
                                            'noise': None},
                    'model_errors': {},
                    'stacking_frame': frame,
                    'grid_name': grid.grid_name,
                    'method': method,
                    'fill_zeroes_method': fill_zeroes_method,
                    'avg_method': avg_method,
                    'meas_source': meas_source
                }

                if estimation_config_dict['meas_source']=="monte": # legacy: keys were generated when this was assumed
                    del(estimation_config_dict['meas_source'])
                config_hash_key = make_dict_hash_key(estimation_config_dict)
                db_manager.insert_config_entry(estimation_config_dict, config_hash_key)

                col_name = 'error_RMS_'+str(window_days)+'_day_avg'
                row_exists = db_manager.check_if_entry_exists('EEI_postproc_results_table', 'estimation_key', config_hash_key)
                cell_exists = False if not(row_exists) else db_manager.check_if_entry_is_not_none('EEI_postproc_results_table',                 
                                                                                                'estimation_key', config_hash_key, col_name)
                if (not cell_exists):
                    try:
                        error_time_series = compute_rolling_avg_error_time_series(SB_constellation, window_days, grid, 
                                        frame, method,  fill_zeroes_method, 
                                        avg_method, meas_source,
                                        EEI_time_series_truth, full_jd_array, mid_day_jd_array)

                        error_rms = nanrms(error_time_series)
                        print(f"Error time series obtained for constellation {SB_constellation}. Writing rms to database...")

                        db_manager.insert_results_entry(config_hash_key, 'error_RMS_'+str(window_days)+'_day_avg', error_rms)
                        np.save(os.path.join(error_time_series_dir, config_hash_key+'.npy'), error_time_series)
                    except:
                        print(f"Computation with grid {grid.grid_name} and mathod {method} failed, likely due to insufficient RAM")

