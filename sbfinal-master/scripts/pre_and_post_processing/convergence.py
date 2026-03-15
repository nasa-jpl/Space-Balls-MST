from SpaceBalls.utils import normal_smoother
from SpaceBalls.radiation_fluxes_preprocessing import get_EEI_truth_daily_jd_arrays, get_mid_day_jd_array
from SpaceBalls.postproc_EEI_estimation import get_true_EEI_time_series, estimate_EEI_avg_sh, estimate_EEI_avg
from SpaceBalls.plotter import Plotter
import numpy as np

frame = 'SFF'

# Load truth
EEI_name = "EEI_truth_1"
full_jd_array = np.concatenate(get_EEI_truth_daily_jd_arrays(EEI_name))
mid_day_jd_array = get_mid_day_jd_array(EEI_name)
EEI_time_series_180x360 = get_true_EEI_time_series(EEI_name, 360, 180)
EEI_smooth_true = normal_smoother(EEI_time_series_180x360, full_jd_array, window_width_days=365, 
                                      out_length_mode='same_with_nans')

yearly_day_shift = np.arange(0, 365*5, 365)
all_avg_errors = [None] * len(yearly_day_shift)

lmax_array = np.round(np.logspace(np.log10(10), np.log10(80), num=15)) #[10, 20, 30, 40, 50, 60, 70, 80]

for j, shift in enumerate(yearly_day_shift):
    jd_window = [2458119.5+shift, 2458119.5+shift+366]
    true_avg_year_1 = np.interp(np.mean(jd_window), full_jd_array, EEI_smooth_true)

    # SH mode    
    EEI_avg_errors_sh = np.zeros(len(lmax_array))

    for i, lmax in enumerate(lmax_array):

        avg_sh = estimate_EEI_avg_sh('case_5_years', sc_names=['sc_C101', 'sc_C201', 'sc_C301'], lmax=int(lmax),
                                altitude=800, jd_windows=np.array([jd_window]), frame=frame)
        EEI_avg_errors_sh[i] = avg_sh[0] - true_avg_year_1
    
    all_avg_errors[j] = EEI_avg_errors_sh

Plotter.plot_convergence(lmax_array, 
                         {'year '+str(j): np.abs(avg) for j, avg in enumerate(all_avg_errors)},
                         xlabel='l max ()', 
                         ylabel='EEI avg error (W/m$^2$)', f_height=0.8, f_width=0.78,
                         file_name='EEI_avg_convergence_SH_'+frame+'.png')
