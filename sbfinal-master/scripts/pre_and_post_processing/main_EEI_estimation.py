import sys, os
import numpy as np
import itertools

from SpaceBalls.postproc_EEI_estimation import get_true_EEI_time_series, estimate_EEI_avg, get_all_orbital_sma_0
from SpaceBalls.radiation_fluxes_preprocessing import get_EEI_truth_daily_jd_arrays, get_mid_day_jd_array
from SpaceBalls.utils import normal_smoother, progress_bar, get_rolling_jd_windows, nanrms
from SpaceBalls.plotter import Plotter
import config.constants as constants
from SpaceBalls.paths import CONFIG_DIR, MEDIA_DIR

EEI_name = "EEI_truth_1"
full_jd_array = np.concatenate(get_EEI_truth_daily_jd_arrays(EEI_name))
mid_day_jd_array = get_mid_day_jd_array(EEI_name)
EEI_time_series_180x360 = get_true_EEI_time_series(EEI_name, 360, 180)

all_sc_tags = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'C1', 'C2', 'C3', 'D1', 'D2', 'D3', 
               'E1', 'E2', 'E3', 'F1', 'F2', 'F3', 'F11', 'F22', 'G1', 'G2', 'G3', 'H1', 'H2', 'H3', 'I1', 'I2', 'I3']
all_sat_combinations_k3 = list(itertools.combinations((all_sc_tags), 3))

#constellation_letters = ['H', 'I']

#for letter in constellation_letters:

#    SB_constellation = ['sc_'+letter+str(i+1) for i in range(3)] #['sc_C1', 'sc_C2', 'sc_C3'] 



for sat_names in all_sat_combinations_k3:
    try:
        constellation_name = '|'.join(sat_names)
        SB_constellation = ['sc_' + sc_name for sc_name in sat_names]
        input_names = ['case_5_years_' + sc_name for sc_name in SB_constellation]
        all_altitudes = get_all_orbital_sma_0(input_names) - constants.earth_radius(units='km')
        assert(len(np.unique(all_altitudes))==1)     # for now we don't really know how to handle different altitudes simultaneously

        smoothing_windows_days = [30, 182, 365]
        smoothing_windows_names = ['1-month', '6-month', '1-year']

        for window_name, window_days in zip(smoothing_windows_names, smoothing_windows_days):
            print("")
            print(f"Running smooth estimate {window_days}-day window (constellation_"+constellation_name+")")
            
            EEI_smooth_true = normal_smoother(EEI_time_series_180x360, full_jd_array, window_width_days=window_days, 
                                            out_length_mode='same_with_nans')
            EEI_smooth_true_coarse = np.interp(mid_day_jd_array, full_jd_array, EEI_smooth_true)

            jd_windows = get_rolling_jd_windows(mid_day_jd_array, window_days)
            
            EEI_smooth_estimated = estimate_EEI_avg('case_5_years', sc_names=SB_constellation, 
                                            altitude=np.unique(all_altitudes), jd_windows=jd_windows, 
                                            n_lon=360, n_lat=180, frame="SFF", fill_method="theta_s_fit_simplified",
                                            make_plots=False)
            assert(np.shape(EEI_smooth_estimated)==np.shape(EEI_smooth_true_coarse))
            EEI_smooth_estimated[np.isnan(EEI_smooth_true_coarse)] = np.nan
                
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
            
            error_time_series = EEI_smooth_estimated-EEI_smooth_true_coarse
            # Plotter.plot_time_series(mid_day_jd_array,
            #                     {'Estimation error': error_time_series},
            #                     f_width=2.5, f_height=0.5,
            #                     ylabel='Error (W/m$^2$)',
            #                     out_dir=os.path.join(MEDIA_DIR, 'figures', constellation_name),
            #                     file_name='error_'+window_name)
            os.makedirs(os.path.join(MEDIA_DIR, 'EEI_estimations', constellation_name), exist_ok=True)
            np.save(os.path.join(MEDIA_DIR, 'EEI_estimations', constellation_name, 'error_time_series_'+window_name+'.npy'),
                    error_time_series)
    except:
        pass
