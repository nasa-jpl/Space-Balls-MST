import numpy as np
import matplotlib.pyplot as plt
import sys, os
from SpaceBalls.postproc_EEI_estimation import get_true_EEI_time_series, estimate_EEI_avg, get_window_avg_maps, check_EEI_consistency, get_all_jd_windows #, get_all_orbital_sma_0, get_stacked_avg_maps_3D_mode
from SpaceBalls.radiation_fluxes_preprocessing import get_EEI_truth_daily_jd_arrays, get_mid_day_jd_array
from SpaceBalls.radiation_settings import radiation_settings_from_EEI_truth_name
from SpaceBalls.utils import normal_smoother, progress_bar, nanrms
from SpaceBalls.sph_meshing import Grid, RegularLatLonGrid, QuadratureGrid
import SpaceBalls.input_database_manager as input_manager
from SpaceBalls.plotter import Plotter
import config.constants as constants
RE = constants.earth_radius()
import itertools
import random
from SpaceBalls.paths import CONFIG_DIR, MEDIA_DIR, INPUT_DIR, OUTPUT_DIR

def get_EEI_smooth_true(EEI_name, window_days):

    #EEI_name = "EEI_truth_0"
    full_jd_array = np.concatenate(get_EEI_truth_daily_jd_arrays(EEI_name))
    mid_day_jd_array = get_mid_day_jd_array(EEI_name)
    EEI_true_time_series = get_true_EEI_time_series(EEI_name)

    EEI_smooth_true = normal_smoother(EEI_true_time_series, full_jd_array, window_width_days=window_days, 
                                    out_length_mode='same_with_nans')
    EEI_smooth_true_coarse = np.interp(mid_day_jd_array, full_jd_array, EEI_smooth_true)

    return EEI_smooth_true_coarse

def compute_EEI_estimated_time_series(SB_constellation, window_days, grid: Grid, frame, method, EEI_name=None, 
                                      accelerometer_dict=None, meas_mode='recomp_radial'):

    if EEI_name is None:
        EEI_name = check_EEI_consistency(SB_constellation)
    
    orbits_str = '|'.join([input_manager.get_dicts(sb)['orbit']['orbit_name'] for sb in SB_constellation])
    print(f"Running EEI estimatie for constellation {orbits_str}...")

    #for window_name, window_days in zip(smoothing_windows_names, smoothing_windows_days):
    print(f"Running smooth estimate {window_days}-day window")
    #rad_config = radiation_settings_from_EEI_truth_name(EEI_name)
    jd_windows = get_all_jd_windows(EEI_name, window_days)

    EEI_smooth_estimated = estimate_EEI_avg(
                                    sb_names=SB_constellation, 
                                    jd_windows=jd_windows, 
                                    grid=grid, method=method, frame=frame,
                                    meas_mode=meas_mode,
                                    return_avg_flux_maps=False,
                                    accelerometer_dict=accelerometer_dict
                                    )
    return EEI_smooth_estimated


def plot_estimation(estimations_dict, jd_array, constellation_name, add_zero_line=False):

    Plotter.plot_time_series(jd_array,
                    estimations_dict,
                    f_width=2.5, f_height=0.5,
                    ylabel='Net TOA flux (W/m$^2$)',
                    title=constellation_name,
                    zero_hline=add_zero_line,
                    file_name='estimation_2Hz')
    

    # single run

orbits = ['G3', 'A1', 'F12'] 
EEI_name = "EEI_truth_0" # check_EEI_consistency(SB_constellation)
window_days = 365
stacking_grid = QuadratureGrid(alt_km=800, order=131)
frame='SFF'
method='binary_gridding'
revs_per_T = 100
accelerometer_dict = {  'noise_asd': 'grattis',
                        'biases': np.array([[1e-6, 1e-6, 1e-6], 
                                            [1e-6, 1e-6, 1e-6], 
                                            [1e-6, 1e-6, 1e-6]]), # rows: sats; cols: RIC
                        'scale_matrices': [np.array([[1, 0, 0], 
                                                     [0, 1, 0], 
                                                     [0, 0, 1]]) for _ in range(3)],
                        'subsample_dt': 1.5,
                        'rotation_revs_per_T': np.array([1,1,1]) * revs_per_T,
                        'rotation_axes_RIC': np.array([[0,1,1], [0,1,1], [0,1,1]])
                    }

EEI_truth_smooth = get_EEI_smooth_true(EEI_name, window_days)
mid_day_jd_array = get_mid_day_jd_array(EEI_name)
SB_constellation = np.concatenate([input_manager.get_primary_keys({'orbit_name': orb,
                                                    'force_settings': 2,
                                                    'integration_settings': 0,
                                                    'delta_t_out': 0.5,
                                                    }) for orb in orbits])
orbits_str = '|'.join([input_manager.get_dicts(sb)['orbit']['orbit_name'] for sb in SB_constellation])

estimated_EEI = compute_EEI_estimated_time_series(SB_constellation, window_days, stacking_grid, frame, 
                                                  method, EEI_name, accelerometer_dict)


estimated_EEI_no_noise = compute_EEI_estimated_time_series(SB_constellation, window_days, stacking_grid, frame, 
                                                  method, EEI_name, {})

dict_to_plot = {'Estimated EEI ('+str(window_days)+'-day avg.; no acc. errors)': estimated_EEI_no_noise,
                'Estimated EEI ('+str(window_days)+'-day avg.)': estimated_EEI,
                'True EEI ('+str(window_days)+'-day avg.)': EEI_truth_smooth}
#plot_estimation(dict_to_plot, mid_day_jd_array, orbits_str)

rms = nanrms(estimated_EEI - EEI_truth_smooth)
#error_title = orbits_str + ' - error RMS: ' + f"{rms:.5f} W/m$^2$"
errors_dict = {}
for i, (label, estimated) in enumerate(dict_to_plot.items()):
    if i<(len(dict_to_plot.items()))-1:
        error = estimated - EEI_truth_smooth
        rms = nanrms(error)
        errors_dict[label + ' - RMS: ' + f"{rms:.3f}" + 'W/m$^2$'] = error
plot_estimation(errors_dict, mid_day_jd_array, orbits_str, add_zero_line=True)

"""
from SpaceBalls.postproc_EEI_estimation import get_radial_measurements_array, get_full_output_var_hist, get_orbital_periods
from SpaceBalls.utils import noise_asd_grattis, fft_to_power
import scipy.fft

cols = ['r', 'g', 'b']

n_days = 0.10
#acc_meas_radial = get_radial_measurements_array(SB_constellation, meas="acceleration")
jd_vecs =  get_full_output_var_hist(SB_constellation, 'jd_vec', downsample_dt=15)
periods_min = get_orbital_periods(SB_constellation) # min

fig, ax = plt.subplots(figsize=(8,2.5))
fig_long, ax_long = plt.subplots(figsize=(8,2.5))
fig_psd, ax_psd = plt.subplots(figsize=(6,4.5))
f_vec_grattis_paper = np.logspace(-4, 0, 300)
asd_grattis_paper = noise_asd_grattis(f_vec_grattis_paper)

for i, sb in enumerate(SB_constellation):
    step_days = np.mean(np.diff(jd_vecs[i]))
    dt = step_days*24*3600
    fs = 1/(dt)
    max_idx = int(np.round(n_days / step_days))
    errors_time_series_RIC = np.load(os.path.join(OUTPUT_DIR, sb, 'error_time_series.npy'))
    errors_radial = errors_time_series_RIC[:,0]
    #og_radial_meas = acc_meas_radial[i]
    
    #ax.plot(Plotter.jd_to_datetime(jd_vecs[i][:max_idx]), og_radial_meas[:max_idx] + errors_radial[:max_idx], 
    #        'x', label=f"Noised meas. SB {i}", color=cols[i])
    #ax.plot(Plotter.jd_to_datetime(jd_vecs[i][:max_idx]), og_radial_meas[:max_idx], 
    #        color='k', linestyle='--', lw=0.3)
    ax_long.plot(Plotter.jd_to_datetime(jd_vecs[i]), errors_radial, label=f"Noise SB {i}", 
                 color=cols[i], lw=0.2, alpha=0.5)
    
    X_noise = scipy.fft.rfft(errors_radial)                          # [m/s^2]
    freqs_noise = scipy.fft.rfftfreq(len(errors_radial), dt)   
    _, asd_noise = fft_to_power(X_noise, len(errors_radial), dt)
    ax_psd.loglog(freqs_noise, asd_noise, lw=0.2, alpha=0.5, color=cols[i], label=f"$a_R$ noise SB {i}")
    ax_psd.axvline(fs, lw=0.3, alpha=0.75, color=cols[i])
    f_w = 1/(periods_min[i]*60) * revs_per_T
    print(f"Revs per T: {revs_per_T}; fw={f_w}")
    ax_psd.axvline(f_w, lw=0.3, alpha=0.75, color=cols[i])
    
#ax.set_xlim([Plotter.jd_to_datetime(jd_vecs[0][0]), Plotter.jd_to_datetime(jd_vecs[0][max_idx])])
#ax.set_ylabel('Net radial acceleration (m/s$^2$)')
#Plotter.format_time_labels(ax)
#ax.legend()

ax_long.set_xlim([Plotter.jd_to_datetime(jd_vecs[0][0]), Plotter.jd_to_datetime(jd_vecs[0][-1])])
ax_long.set_ylabel('Net radial acceleration error (m/s$^2$)')
Plotter.format_time_labels(ax_long)

ax_psd.loglog(f_vec_grattis_paper, asd_grattis_paper, label='GRATTIS total (Conklin)', lw=5, color='k', alpha=0.4)
ax_psd.set_xlabel('Frequency (Hz)')
ax_psd.set_ylabel('ASD (m/s$^2$ Hz$^{-1/2}$)')
ax_top_1 = ax_psd.secondary_xaxis('top', functions=(lambda x: (1/x/(24*3600)), lambda x: (1/x/(24*3600))))
ax_top_1.set_xlabel('1/$f$ (days)')
ax_psd.legend()
ax_psd.set_ylim(bottom=1e-12)
fig_psd.savefig('psd_2Hz')

"""