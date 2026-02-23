import sys, os
import importlib.util
import numpy as np
import scipy.signal
import scipy.interpolate
from SpaceBalls.paths import CONFIG_DIR, MEDIA_DIR, INPUT_DIR, OUTPUT_DIR


def load_input_file(input_file_name):
    
    spec = importlib.util.spec_from_file_location(input_file_name, INPUT_DIR / f"{input_file_name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def get_two_perp_unit_vectors(u):

    i = np.argmin(np.abs(u))
    a = np.zeros(3)
    a[i] = 1.0

    v = a - np.dot(a, u) * u
    v /= np.linalg.norm(v)
    w = np.cross(u, v)

    return v, w

def progress_bar(i, n, length=40):
    """Efficient inline progress bar."""
    frac = i / n
    filled = int(length * frac)
    bar = '█' * filled + '-' * (length - filled)
    print(f'\r[{bar}] {100*frac:6.2f}%', end='', file=sys.stdout)
    if i == n:
        print()  # newline at the end


def get_clean_lonlat_vecs_for_plot(lon, lat):

        lon_diff = np.abs(np.diff(lon))
        lat_diff = np.abs(np.diff(lat))

        jump_indices = np.where((lon_diff > 30) | (lat_diff > 30))[0] + 1
        lon_clean = np.insert(lon, jump_indices, np.nan)
        lat_clean = np.insert(lat, jump_indices, np.nan)

        return lon_clean, lat_clean


def normal_smoother(full_vec, jd_vec, window_width_days, out_length_mode): # out_length_mode: 'same', 'same_with_nans', 'valid
    
    delta_t_days = np.mean(np.diff(jd_vec))
    n = len(full_vec)
    
    n_kernel = int(np.ceil(window_width_days/delta_t_days) // 2 * 2 + 1)  # round to next odd number
    half_n_kernel = int(np.floor(n_kernel/2))
    kernel = np.ones(n_kernel) / n_kernel
    
    if out_length_mode=="same_with_nans":
        smooth_x_vec = scipy.signal.fftconvolve(full_vec, kernel, 'valid')
        smooth_x_vec = np.concatenate((np.full(half_n_kernel, np.nan), smooth_x_vec, np.full(half_n_kernel, np.nan)))    
    else:
        smooth_x_vec = scipy.signal.fftconvolve(full_vec, kernel, out_length_mode)
    
    return smooth_x_vec



def fill_nans_in_data_array(data_array, method='nearest'): #methods: 'nearest', 'linear', 'cubic', 'zeroes'
    # deprecated function
    
    nan_lat_idxs, nan_lon_idxs = np.where(np.isnan(data_array))
    notnan_lat_idxs, notnan_lon_idxs = np.where(~np.isnan(data_array))
    assert(len(notnan_lat_idxs)==len(notnan_lon_idxs))

    notnan_radial_acc_vec = data_array[~np.isnan(data_array)].reshape(len(notnan_lat_idxs))
    
    if method=='zeroes':
        data_array[nan_lat_idxs, nan_lon_idxs] = 0.0
    else:
        P_i = np.array( [notnan_lat_idxs, notnan_lon_idxs] ).T
        Z_i = notnan_radial_acc_vec
        Pq = np.array( [nan_lat_idxs, nan_lon_idxs] ).T
        Z_interp = scipy.interpolate.griddata(P_i, Z_i, Pq, method=method)
        data_array[nan_lat_idxs, nan_lon_idxs] = Z_interp
    
    return data_array


def interp_zeroes_in_2D_data_array(data_array, method='nearest'): #methods: 'nearest', 'linear', 'cubic', 'zeroes'
    
    zero_lat_idxs, zero_lon_idxs = np.where(data_array==0)
    nonzero_lat_idxs, nonzero_lon_idxs = np.where(data_array!=0)
    assert(len(nonzero_lat_idxs)==len(nonzero_lon_idxs))

    notnan_radial_acc_vec = data_array[~np.isnan(data_array)].reshape(len(nonzero_lat_idxs))
    
    P_i = np.array( [nonzero_lat_idxs, nonzero_lon_idxs] ).T
    Z_i = notnan_radial_acc_vec
    Pq = np.array( [zero_lat_idxs, zero_lon_idxs] ).T
    Z_interp = scipy.interpolate.griddata(P_i, Z_i, Pq, method=method)
    data_array[zero_lat_idxs, zero_lon_idxs] = Z_interp
    
    return data_array


def compute_orbital_period(MU_km3s2, sma_km):
    T_min = 2*np.pi * sma_km**(3/2) / np.sqrt(MU_km3s2) / 60

    return T_min


def get_rolling_jd_windows(mid_day_jd_array, window_width_days):

    jd_windows = np.zeros((len(mid_day_jd_array), 2))
    for i, mid_day_jd in enumerate(mid_day_jd_array):
        jd_windows[i,:] = [
            max(mid_day_jd - window_width_days/2, mid_day_jd_array[0]  - 0.5),
            min(mid_day_jd + window_width_days/2, mid_day_jd_array[-1] + 0.5),
        ]

    return jd_windows


def nanrms(data, axis=None): # by Google AI
    """
    Calculate the Root Mean Square (RMS) of an array, ignoring NaN values.

    Parameters:
    data : array_like
        Input data.
    axis : {None, int, tuple[int]}, optional
        Axis or axes along which to compute the RMS. 
        If None, the RMS of the entire array is computed.

    Returns:
    float or ndarray
        The RMS value(s).
    """
    # Square the data, compute the mean while ignoring NaNs, then take the square root
    return np.sqrt(np.nanmean(data**2, axis=axis))