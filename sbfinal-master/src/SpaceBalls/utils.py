import sys, os
import importlib, importlib.util
import time
import numpy as np
import scipy.signal, scipy.interpolate, scipy.fft
import json
import hashlib
from SpaceBalls.paths import CONFIG_DIR, MEDIA_DIR, INPUT_DIR, OUTPUT_DIR


def load_input_file(input_file_name):
    
    #spec = importlib.util.spec_from_file_location(input_file_name, INPUT_DIR / f"{input_file_name}.py")
    spec = importlib.util.spec_from_file_location(input_file_name, os.path.join(INPUT_DIR, 'legacy_files', f"{input_file_name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module

def get_Earth_SH_gravity_model(grav_field_name):

    dir = os.path.join(CONFIG_DIR, 'earth', 'gravity')
    fname = grav_field_name + '_monte_grv'
    spec = importlib.util.spec_from_file_location(fname, os.path.join(dir, fname+'.py'))#dir / f"{fname}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module

    #module = importlib.import_module("config.earth.gravity")
    #return getattr(module, grav_field_name+"_monte_grv")

    # if grav_field_name=="GGM02C":
    #     from config.earth.gravity import GGM02c_monte_grv as GGM02C
    #     return GGM02C
    # elif grav_field_name=="GOCO06s":
    #     from config.earth.gravity import GOCO06s_monte_grv as GOCO06s
    #     return GOCO06s 
    # elif grav_field_name=="GOCO2025s":
    #     from config.earth.gravity import GOCO2025s_monte_grv as GOCO2025s
    #     return GOCO2025s
    # else:
    #     raise ValueError(f"Unrecognized gravity field {grav_field_name}")


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
    if n>1: 
        frac = i / (n-1)
        filled = int(length * frac)
        bar = '█' * filled + '-' * (length - filled)
        print(f'\r[{bar}] {100*frac:6.2f}%', end='', file=sys.stdout)
    if i == n-1:
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
    if len(np.shape(full_vec))==2:
        n_components = np.shape(full_vec)[0] # assume time along columns
        kernel = kernel[None,:] # NOTE: checked - scipy.signal.fftconvolve(full_vec, kernel[None,:], 'valid')[0,:]==scipy.signal.fftconvolve(full_vec[0,:], kernel, 'valid')
    
    if out_length_mode=="same_with_nans":
        smooth_x_vec = scipy.signal.fftconvolve(full_vec, kernel, 'valid')
        if len(np.shape(full_vec))==1:
            smooth_x_vec = np.concatenate((np.full(half_n_kernel, np.nan), smooth_x_vec, np.full(half_n_kernel, np.nan)))    
        elif len(np.shape(full_vec))==2: # assume time along columns
            nan_block = np.full((n_components, half_n_kernel), np.nan)
            smooth_x_vec = np.hstack((nan_block, smooth_x_vec, nan_block))
    else:
        smooth_x_vec = scipy.signal.fftconvolve(full_vec, kernel, out_length_mode)
    
    return smooth_x_vec

# ...existing code...
def normal_smoother_uneven(full_vec, jd_vec, window_width_days, out_length_mode="same"):
    """
    Boxcar smoother robust to uneven jd_vec spacing.

    Parameters
    ----------
    full_vec : ndarray
        1D array of length N (time) or 2D array with shape (n_components, N)
        where time runs along axis=1 (keeps original convention).
    jd_vec : 1D array_like
        Monotonic increasing time vector (same units as window_width_days).
    window_width_days : float
        Full width of the boxcar window (days).
    out_length_mode : {"same", "same_with_nans", "valid"}
        Behavior consistent with previous implementation.

    Returns
    -------
    smooth_x_vec : ndarray
        Smoothed array with shape depending on out_length_mode and input shape.
    """
    full_vec = np.asarray(full_vec)
    jd = np.asarray(jd_vec, dtype=float)

    if jd.ndim != 1:
        raise ValueError("jd_vec must be 1D")
    if not np.all(np.diff(jd) >= 0):
        raise ValueError("jd_vec must be non-decreasing / sorted ascending")

    if full_vec.ndim == 1:
        n_time = jd.size
        if full_vec.size != n_time:
            raise ValueError("full_vec and jd_vec must have same length")
        x = full_vec
        mask = np.isfinite(x)
        x_filled = np.where(mask, x, 0.0)

        # cumulative sums for O(n log n) windowed sums using searchsorted
        cs = np.concatenate(([0.0], np.cumsum(x_filled, dtype=float)))
        cc = np.concatenate(([0], np.cumsum(mask.astype(np.int64), dtype=np.int64)))

        half = float(window_width_days) / 2.0
        lefts = np.searchsorted(jd, jd - half, side="left")
        rights = np.searchsorted(jd, jd + half, side="right")

        sums = cs[rights] - cs[lefts]
        counts = cc[rights] - cc[lefts]

        with np.errstate(divide="ignore", invalid="ignore"):
            means = sums / counts
        means[counts == 0] = np.nan

        # determine incomplete windows (touching data edges)
        incomplete = (jd - half < jd[0]) | (jd + half > jd[-1])

        if out_length_mode == "same":
            smooth_x_vec = means
        elif out_length_mode == "same_with_nans":
            out = means.copy()
            out[incomplete] = np.nan
            smooth_x_vec = out
        elif out_length_mode == "valid":
            valid_idx = np.nonzero(~incomplete)[0]
            smooth_x_vec = means[valid_idx]
        else:
            raise ValueError("out_length_mode must be 'same', 'same_with_nans' or 'valid'")

        return smooth_x_vec

    elif full_vec.ndim == 2:
        # assume shape (n_components, n_time) as original implementation
        n_components, n_time = full_vec.shape
        if n_time != jd.size:
            raise ValueError("time dimension of full_vec must match jd_vec length")

        mask = np.isfinite(full_vec)
        x_filled = np.where(mask, full_vec, 0.0)

        # cumulative sums along time (axis=1); prepend zero column
        cs = np.concatenate((np.zeros((n_components, 1), dtype=float),
                             np.cumsum(x_filled, axis=1, dtype=float)), axis=1)
        cc = np.concatenate((np.zeros((n_components, 1), dtype=np.int64),
                             np.cumsum(mask.astype(np.int64), axis=1)), axis=1)

        half = float(window_width_days) / 2.0
        lefts = np.searchsorted(jd, jd - half, side="left")
        rights = np.searchsorted(jd, jd + half, side="right")

        # index cs/cc at columns given by rights and lefts -> shape (n_components, n_time)
        sums = cs[:, rights] - cs[:, lefts]
        counts = cc[:, rights] - cc[:, lefts]

        # safe division
        means = np.empty_like(sums, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            means = sums / counts
        means[counts == 0] = np.nan

        incomplete = (jd - half < jd[0]) | (jd + half > jd[-1])

        if out_length_mode == "same":
            smooth_x_vec = means
        elif out_length_mode == "same_with_nans":
            out = means.copy()
            out[:, incomplete] = np.nan
            smooth_x_vec = out
        elif out_length_mode == "valid":
            valid_idx = np.nonzero(~incomplete)[0]
            smooth_x_vec = means[:, valid_idx]
        else:
            raise ValueError("out_length_mode must be 'same', 'same_with_nans' or 'valid'")

        return smooth_x_vec

    else:
        raise ValueError("normal_smoother supports 1D or 2D full_vec only")
# ...existing code...

# ...existing code...
def normal_smoother_uneven_2(full_vec, jd_vec, window_width_days, out_length_mode="same"):
    """
    Boxcar (uniform weight) smoother robust to uneven jd_vec spacing.

    Parameters
    ----------
    full_vec : ndarray
        1D array of length N (time) or 2D array with shape (n_components, N)
        where time runs along axis=1.
    jd_vec : 1D array_like
        Monotonic increasing time vector (same units as window_width_days).
    window_width_days : float
        Full width of the boxcar window (days).
    out_length_mode : {"same", "same_with_nans", "valid"}
        Behavior of the output length.

    Returns
    -------
    smooth_x_vec : ndarray
    """
    full_vec = np.asarray(full_vec)
    jd = np.asarray(jd_vec, dtype=float)

    if jd.ndim != 1:
        raise ValueError("jd_vec must be 1D")
    if not np.all(np.diff(jd) >= 0):
        raise ValueError("jd_vec must be non-decreasing / sorted ascending")
    if window_width_days <= 0:
        return full_vec.copy()

    n = jd.size
    half = float(window_width_days) / 2.0

    # Build sample edges (length n+1)
    if n == 1:
        left_edge = jd[0] - 0.5
        right_edge = jd[0] + 0.5
        t_edges = np.array([left_edge, right_edge], dtype=float)
    else:
        mid = 0.5 * (jd[:-1] + jd[1:])
        left_edge = jd[0] - (mid[0] - jd[0])
        right_edge = jd[-1] + (jd[-1] - mid[-1])
        t_edges = np.concatenate(([left_edge], mid, [right_edge]))

    durations = t_edges[1:] - t_edges[:-1]   # length n

    def _S_of_w_1d(x_filled, cum_edges, w):
        # w: array
        w = np.asarray(w, dtype=float)
        idx = np.searchsorted(t_edges, w, side="right") - 1  # idx in [ -1 .. n-1 ]
        # masks
        before = w <= t_edges[0]
        after = w >= t_edges[-1]
        S = np.empty_like(w, dtype=float)
        S[before] = 0.0
        S[after] = cum_edges[-1]
        mid_mask = ~(before | after)
        if np.any(mid_mask):
            ii = idx[mid_mask]
            te = t_edges[ii]
            S[mid_mask] = cum_edges[ii] + x_filled[ii] * (w[mid_mask] - te)
        return S

    if full_vec.ndim == 1:
        if full_vec.size != n:
            raise ValueError("full_vec and jd_vec must have same length")
        mask = np.isfinite(full_vec)
        x_filled = np.where(mask, full_vec, 0.0)

        # cumulative integrals at edges length n+1
        cum_val_edges = np.concatenate(([0.0], np.cumsum(x_filled * durations, dtype=float)))
        cum_cnt_edges = np.concatenate(([0.0], np.cumsum(mask.astype(float) * durations, dtype=float)))

        wl = jd - half
        wr = jd + half

        S_wr = _S_of_w_1d(x_filled, cum_val_edges, wr)
        S_wl = _S_of_w_1d(x_filled, cum_val_edges, wl)
        D_wr = _S_of_w_1d(mask.astype(float), cum_cnt_edges, wr)
        D_wl = _S_of_w_1d(mask.astype(float), cum_cnt_edges, wl)

        numer = S_wr - S_wl
        denom = D_wr - D_wl

        with np.errstate(divide="ignore", invalid="ignore"):
            means = numer / denom
        means[denom == 0] = np.nan

        incomplete = (wl < jd[0]) | (wr > jd[-1])

        if out_length_mode == "same":
            return means
        elif out_length_mode == "same_with_nans":
            out = means.copy()
            out[incomplete] = np.nan
            return out
        elif out_length_mode == "valid":
            return means[~incomplete]
        else:
            raise ValueError("out_length_mode must be 'same', 'same_with_nans' or 'valid'")

    elif full_vec.ndim == 2:
        n_comp, n_time = full_vec.shape
        if n_time != n:
            raise ValueError("time dimension of full_vec must match jd_vec length")

        mask = np.isfinite(full_vec).astype(float)
        x_filled = np.where(np.isfinite(full_vec), full_vec, 0.0)

        # cumulative integrals at edges shape (n_comp, n+1)
        cum_val_edges = np.concatenate((np.zeros((n_comp, 1), dtype=float),
                                        np.cumsum(x_filled * durations, axis=1, dtype=float)), axis=1)
        cum_cnt_edges = np.concatenate((np.zeros((n_comp, 1), dtype=float),
                                        np.cumsum(mask * durations, axis=1, dtype=float)), axis=1)

        wl = jd - half
        wr = jd + half

        # vectorized S for 2D: use column indexing with idx arrays
        idx_wr = np.searchsorted(t_edges, wr, side="right") - 1
        idx_wl = np.searchsorted(t_edges, wl, side="right") - 1

        def S_2d(cum_edges, x_arr, idx_arr, w_arr):
            # cum_edges: (n_comp, n+1), x_arr: (n_comp, n), idx_arr: (n,), w_arr: (n,)
            before = w_arr <= t_edges[0]
            after = w_arr >= t_edges[-1]
            S = np.empty((n_comp, n), dtype=float)
            S[:, before] = 0.0
            S[:, after] = cum_edges[:, -1][:, None]
            mid_mask = ~(before | after)
            if np.any(mid_mask):
                ii = idx_arr[mid_mask]
                te = t_edges[ii]
                # cum_edges[:, ii] -> (n_comp, n_mid)
                S_mid = cum_edges[:, ii] + x_arr[:, ii] * (w_arr[mid_mask] - te)
                S[:, mid_mask] = S_mid
            return S

        S_wr = S_2d(cum_val_edges, x_filled, idx_wr, wr)
        S_wl = S_2d(cum_val_edges, x_filled, idx_wl, wl)
        D_wr = S_2d(cum_cnt_edges, mask, idx_wr, wr)
        D_wl = S_2d(cum_cnt_edges, mask, idx_wl, wl)

        numer = S_wr - S_wl
        denom = D_wr - D_wl

        with np.errstate(divide="ignore", invalid="ignore"):
            means = numer / denom
        means[denom == 0] = np.nan

        incomplete = (wl < jd[0]) | (wr > jd[-1])

        if out_length_mode == "same":
            return means
        elif out_length_mode == "same_with_nans":
            out = means.copy()
            out[:, incomplete] = np.nan
            return out
        elif out_length_mode == "valid":
            valid_idx = np.nonzero(~incomplete)[0]
            return means[:, valid_idx]
        else:
            raise ValueError("out_length_mode must be 'same', 'same_with_nans' or 'valid'")

    else:
        raise ValueError("normal_smoother_uneven supports 1D or 2D full_vec only")
# ...existing code...



def fill_nans_in_data_array(data_array, method='nearest'): #methods: 'nearest', 'linear', 'cubic', 'zeroes'
    # deprecated function
    
    nan_lat_idxs, nan_lon_idxs = np.where(np.isnan(data_array))
    notnan_lat_idxs, notnan_lon_idxs = np.where(~np.isnan(data_array))
    assert(len(notnan_lat_idxs)==len(notnan_lon_idxs))

    notnan_radial_acc_vec = data_array[~np.isnan(data_array)].reshape(len(notnan_lat_idxs))
    
    if method=='zeroes':
        data_array[nan_lat_idxs, nan_lon_idxs] = 0.0
    else: # TODO: unify with sphere_field_interp in sph_meshing?
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


def make_dict_hash_key(input_dict):

    input_dict_filtererd = {key:value for (key, value) in input_dict.items() 
                            if (not(isinstance(value, dict)) and value is not None) or 
                            (isinstance(value, dict) and value and not(all(v is None for v in value.values())))} 
                            # filter None entries, empty dicts and dicts that contain None for all fields
    key_string = json.dumps(input_dict_filtererd, sort_keys=True)
    
    #hash_str = hashlib.sha256(key_string.encode()).hexdigest()
    hash_str = hashlib.blake2b(key_string.encode(), digest_size=8).hexdigest()
    return hash_str

def make_list_str_key(input_list, sep='|'):
    return sep.join(sorted(input_list))


def get_2D_to_1D_idx(row_idx, col_idx, n_cols):
    # this function defines the idx transformation so that the result of the stretched matrix is the same as in np.reshape()
    return row_idx*n_cols + col_idx


def latlon_to_coord_name(latlon_pair):
    lat, lon = latlon_pair
    if lat>=0:
        lat_name = f"{lat:.3f} N"
    else:
        lat_name = f"{np.abs(lat):.3f} S" 

    if (lon>=0) & (lon<180):
        lon_name = f"{lon:.3f} E"
    else:
        lon_name = f"{np.abs(ensure_180_to_180(lon)):.3f} W"

    full_str = lat_name + '; ' + lon_name

    return full_str

def ensure_180_to_180(lon_vec):
    # lon_vec[lon_vec>180] = lon_vec[lon_vec>180]-360
    # return lon_vec
    
    a = np.asarray(lon_vec)
    return a - 360 * (a > 180)

def get_all_r_rel(stacked_r_el_1, stacked_r_el_2, dim_to_loop=None):

    # full broadcast method seems to be the fastest even for a 20+GB resulting array (factor 3-5)

    # full broadcast method:
    if dim_to_loop is None:
        all_r_rel = stacked_r_el_1[:,None,:] - stacked_r_el_2[None,:,:]  # sized n1 x n2 x 3
        return all_r_rel
    
    # loop over 1 method:
    elif dim_to_loop==0:
        n1 = len(stacked_r_el_1)
        all_r_rel = np.zeros((n1, len(stacked_r_el_2), 3))
        for i in range(n1):
            all_r_rel[i,:,:] = stacked_r_el_1[i,:] - stacked_r_el_2
        return all_r_rel

    # loop over 2 method:
    elif dim_to_loop==1:
        n2 = len(stacked_r_el_2)
        all_r_rel = np.zeros((len(stacked_r_el_1), n2, 3))
        for j in range(n2):
            all_r_rel[:,j,:] = stacked_r_el_1 - stacked_r_el_2[j,:]
        return all_r_rel
    

def get_all_u_rel(all_r_rel, dim_to_loop=None):

    # for the particular case where all_r_rel is sized 6.5e4 x 5e3 x 3, loop over 1 is slightly faster than boradcast but loop over 2 is much slower 

    # full broadcast method:
    if dim_to_loop is None:
        full_r_rel_norm = np.linalg.norm(all_r_rel, axis=2)
        full_u_rel = all_r_rel / full_r_rel_norm[:,:,None]
        return full_u_rel
    
    # loop over 1 method:
    elif dim_to_loop==0:
        all_u_rel = np.zeros_like(all_r_rel)
        for i in range(all_r_rel.shape[0]):
            all_u_rel[i,:,:] = all_r_rel[i,:,:] / np.linalg.norm(all_r_rel[i,:,:], axis=1)[:,None]
        
        return all_u_rel

    # loop over 2 method:
    elif dim_to_loop==1:
        all_u_rel = np.zeros_like(all_r_rel)
        for i in range(all_r_rel.shape[1]):
            all_u_rel[:,i,:] = all_r_rel[:,i,:] / np.linalg.norm(all_r_rel[:,i,:], axis=1)[:,None]
        
        return all_u_rel
    

def get_r_rel_norm(full_r_rel, method='einsum'):
    
    if method=="linalg":
        full_r_rel_norm = np.linalg.norm(full_r_rel, axis=2)
        return full_r_rel_norm
    
    elif method=="einsum":
        full_r_rel_norm = np.sqrt(np.einsum('ijk,ijk->ij', full_r_rel, full_r_rel))
        return full_r_rel_norm
    
    elif method=="einsum_opt": # seems slower than no optimize
        full_r_rel_norm = np.sqrt(np.einsum('ijk,ijk->ij', full_r_rel, full_r_rel, optimize=True))
        return full_r_rel_norm

    elif method=="loop_0":
        # loop over 0th dim
        n1, n2 = full_r_rel.shape[0], full_r_rel.shape[1]
        full_r_rel_norm = np.zeros((n1, n2))
        for i in range(n1):
            r_rel_i = full_r_rel[i,:,:] 
            full_r_rel_norm[i,:] = np.linalg.norm(r_rel_i, axis=1)
        return full_r_rel_norm
    
    elif method=="loop_1":
        n1, n2 = full_r_rel.shape[0], full_r_rel.shape[1]
        full_r_rel_norm = np.zeros((n1, n2))
        for i in range(n2):
            r_rel_i = full_r_rel[:,i,:] 
            full_r_rel_norm[:,i] = np.linalg.norm(r_rel_i, axis=1)
        return full_r_rel_norm


    

def get_all_cos_alpha(full_r_rel, r_rel_norm, stacked_u, negs_to_zero=True, method="einsum"): 
    # methods: 'broadcast', 'einsum', 'einsum_u', 'einsum_u0', 'einsum_u1', 'loop_0', 'loop_1'
    # 1st dim of full_r_rel must be equal to 0th dim of stacked_u

    if method=="broadcast":
        # brute-force broadcast method:
        full_u_rel = full_r_rel / r_rel_norm[:,:,None]
        full_cos_alpha = np.sum(full_u_rel * stacked_u[None,:,:], axis=2)
    
    elif method=="einsum":
        # einstein summation method:
        full_cos_alpha = np.einsum('ijk,jk->ij', full_r_rel, stacked_u) / r_rel_norm # no expansion
        
    elif method=="einsum_opt":
        # einstein summation method:
        full_cos_alpha = np.einsum('ijk,jk->ij', full_r_rel, stacked_u, optimize=True) / r_rel_norm # no expansion
        # this is faster (2.3 sec) than: full broadcast (3.6 sec); einsum with both unit vectors (with optimal computation of u_rel) (2.4 sec); loop over dim 0 (2.7 sec); and loop over dim 1 (13.7 sec). Computing times are for a 64800 x 5e3 x3 matrix

    elif method=="einsum_u": # NOTE: the following 3 are obsolete now that get_nrom is its own function
        u_rel = get_all_u_rel(full_r_rel) 
        full_cos_alpha = np.einsum('ijk,jk->ij', u_rel, stacked_u)

    elif method=="einsum_u0":
        u_rel = get_all_u_rel(full_r_rel, dim_to_loop=0) 
        full_cos_alpha = np.einsum('ijk,jk->ij', u_rel, stacked_u)

    elif method=="einsum_u1":
        u_rel = get_all_u_rel(full_r_rel, dim_to_loop=1) 
        full_cos_alpha = np.einsum('ijk,jk->ij', u_rel, stacked_u)

    elif method=="loop_0":
        # loop over 0th dim
        n1, n2 = full_r_rel.shape[0], full_r_rel.shape[1]
        full_cos_alpha = np.zeros((n1, n2))
        for i in range(n1):
            r_rel_i = full_r_rel[i,:,:] 
            full_cos_alpha[i,:]  = np.sum((r_rel_i / r_rel_norm[i,:,None]) * stacked_u, axis=1)
    
    elif method=="loop_0_einsum":
        # loop over 0th dim
        n1, n2 = full_r_rel.shape[0], full_r_rel.shape[1]
        full_cos_alpha = np.zeros((n1, n2))
        for i in range(n1):
            r_rel_i = full_r_rel[i,:,:] 
            full_cos_alpha[i,:]  = np.einsum('ij,ij->i', (r_rel_i / r_rel_norm[i,:,None]), stacked_u)
        
    elif method=="loop_1":
        n1, n2 = full_r_rel.shape[0], full_r_rel.shape[1]
        full_cos_alpha = np.zeros((n1, n2))
        for i in range(n2):
            r_rel_i = full_r_rel[:,i,:]
            full_cos_alpha[:,i]  = np.sum((r_rel_i / r_rel_norm[:,i,None]) * stacked_u[i,:], axis=1)
        
    if negs_to_zero:
        np.maximum(0, full_cos_alpha, out=full_cos_alpha) # slightly faster (15%) than full_cos_alpha[full_cos_alpha<0] = 0 

    return full_cos_alpha


def get_fft_spectrum(signal_in, t_array_in, one_sided=True):

    """
    Return FFT frequency axis plus amplitude, power, and power spectral density.

    Parameters
    ----------
    signal : array_like
        Real-valued time series.
    t_array : array_like
        Time stamps for the signal samples, in the same units used for frequency.
    one_sided : bool
        If True, return only non-negative frequencies for a real signal.

    Returns
    -------
    freq : ndarray
    amplitude : ndarray
        |X| / N, with one-sided scaling applied.
    power : ndarray
        bin power = |X|^2 / N^2, with one-sided scaling applied.
    psd : ndarray
        power spectral density = power / df, where df = 1 / (N * dt).
    """
    # first of all remove nans
    non_nan_filter = ~np.isnan(signal_in)
    signal = signal_in.copy()
    t_array = t_array_in.copy()
    signal = signal[non_nan_filter]
    t_array = t_array[non_nan_filter]

    x = np.asarray(signal, dtype=float)
    N = x.size
    dt = np.mean(np.diff(t_array))
    fs = 1.0 / dt
    df = 1.0 / (N * dt)

    X = scipy.fft.fft(x)

    if one_sided:
        freqs = scipy.fft.rfftfreq(N, dt)
        X = X[:freqs.size]
    else:
        freqs = scipy.fft.fftfreq(N, dt)

    amplitude = np.abs(X) / N
    power = np.abs(X)**2 / N**2
    psd = power / df

    if one_sided:
        if N % 2 == 0:
            # even N: keep DC and Nyquist unchanged, double the rest
            amplitude[1:-1] *= 2
            power[1:-1] *= 2
            psd[1:-1] *= 2
        else:
            # odd N: keep DC unchanged, double all positive-frequency bins
            amplitude[1:] *= 2
            power[1:] *= 2
            psd[1:] *= 2

    return freqs, amplitude, power, psd


def find_idxs(x, y): # for each element in x, find its index in y

    # 1. Standard O(N log N) sorting prep
    y_sorter = np.argsort(y)

    # 2. Find insertion points, forcing out-of-bounds indices to a safe index (0)
    # using a conditional array inline. 
    idx = np.searchsorted(y, x, sorter=y_sorter)
    safe_idx = np.where(idx < len(y), idx, 0)

    # 3. Map back and validate in a single step
    final_indices = np.where(y[y_sorter[safe_idx]] == x, y_sorter[safe_idx], -1)

    return final_indices


def noise_asd_grattis(f_vec):
    f_vec_abs = np.abs(f_vec)
    # noise sources:
    readout = 2/9*1e-7 * f_vec_abs**2
    actuation = 2*np.sqrt(2)*10**(-12.5) * f_vec_abs**(-0.5)
    actuation_stiffness = 1.05*1e-11 + 0 * f_vec_abs
    total_grattis = readout + actuation + actuation_stiffness

    return total_grattis

def noise_asd_grace(f_vec):
    # from that GRACE paper
    sqrt_PSD_x = np.sqrt(1 + (f_vec/0.5)**4 + (0.1/f_vec)) * 1e-9
    sqrt_PSD_yz = np.sqrt(1 + (f_vec/0.5)**4 + (0.005/f_vec)) * 1e-10

    return sqrt_PSD_x, sqrt_PSD_yz


def fft_to_power(X, N, dt):

    power = (np.abs(X))**2 / (N**2)             # [m^2/s^4]
    power[1:-1] *= 2.0   # one-sided            # [m^2/s^4]
    psd = power * N * dt                        # [m^2/s^3] = [m^2/(s^4*Hz)]
    asd = np.sqrt(psd)  

    return psd, asd


def generate_noise_time_series(N, dt, reference_asd="grattis"):

    #N = int(T_years * 365 * 24 * 60 * 60 / dt)  # []
    t_vec = np.arange(0, N*dt, dt)              # [sec]
    x = np.random.randn(N)                      # [m/s^2]

    X = scipy.fft.rfft(x)                       # [m/s^2]
    freqs = scipy.fft.rfftfreq(N, dt)   
    _, asd = fft_to_power(X, N, dt)

    if reference_asd=="grattis":
        asd_grattis= noise_asd_grattis(freqs)           # [m/(s^2*Hz^(-1/2))]
        asd_grattis[0] = asd_grattis[1]  # we set the ASD value for f=0 to the same as for the smallest nonzero freq., since the function returns ads(f=0)=inf

    X_filtered = X * (asd_grattis / asd)
    filtered_noise = scipy.fft.irfft(X_filtered, n=N)

    return t_vec, filtered_noise


def create_Rz_hist(theta_array): # unused
    # theta_array: 1D array of angles in radians, shape (N,)
    c = np.cos(theta_array)
    s = np.sin(theta_array)
    
    # Pre-allocate and fill the (N, 3, 3) rotation matrix history
    R = np.eye(3, dtype=theta_array.dtype)  # Base 3x3 identity
    R_history = np.tile(R, (len(theta_array), 1, 1))       # Shape: (N, 3, 3)
    
    # Assign the vectorized components via broadcasting
    R_history[:, 0, 0] = c
    R_history[:, 0, 1] = -s
    R_history[:, 1, 0] = s
    R_history[:, 1, 1] = c
    
    return R_history

def get_pairwise_midpoints(x):
    # Source - https://stackoverflow.com/a/23856065
    # Posted by Veedrac, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-06-12, License - CC BY-SA 3.0

    return ((x[1:] + x[:-1]) / 2)

    
def unique_with_tolerance(arr, tol):
    
    # Sort a copy of the array
    sorted_arr = np.sort(arr)
    
    # Find differences between consecutive elements
    diff = np.diff(sorted_arr)
    
    # Keep the first element, and any element larger than the previous by tol
    mask = np.append(True, diff > tol)
    
    return sorted_arr[mask]


def downsample_array(array, dt_orig, dt_new):
    assert (dt_new >= dt_orig), "Subsample dt can't be smaller than original dt!"

    downsample_fact = int(np.round(dt_new / dt_orig))
    shift = int(np.round(downsample_fact/2))
    if not((dt_new / dt_orig) % 1 == 0):
        print(f"Warning: downsample factor {dt_new / dt_orig} is not integer")

    if len(np.shape(array))==1:
        print(f"Subsampling time hist: orig. shape: {np.shape(array)}; downsample_fact={downsample_fact}")
        ret = array[shift::downsample_fact][:-1] # NOTE: rm last one because it can be nan coming from a smoothing
        print(f"Subsampling time hist: final shape: {np.shape(ret)}")
        return ret
    
    elif len(np.shape(array))==2:
        # assume time along rows
        print(f"Subsampling time hist: orig. shape: {np.shape(array)}; downsample_fact={downsample_fact}")
        ret = array[shift::downsample_fact, :][:-1,:]
        print(f"Subsampling time hist: final shape: {np.shape(ret)}")
        return ret

from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
import numpy as np

def smoother_ai(x, pos, width, out_length_mode="same"):
    """
    Gaussian smoothing with the same signature as before.

    For uniformly spaced samples this uses an optimized 1D Gaussian filter.
    For irregularly spaced samples it falls back to a KD-tree-based local
    Gaussian weighted average.
    """
    x = np.asarray(x, dtype=float)
    pos = np.asarray(pos, dtype=float)

    # Flatten common 1D-like inputs (e.g. column vectors)
    if x.ndim != 1:
        x = np.ravel(x)
    if pos.ndim != 1:
        pos = np.ravel(pos)

    if x.shape != pos.shape:
        raise ValueError("x and pos must have the same shape")
    if x.size == 0:
        return x.copy()
    if width <= 0:
        return x.copy()

    mask = np.isfinite(x)
    if not np.any(mask):
        return np.full_like(x, np.nan, dtype=float)

    x_filled = np.where(mask, x, 0.0)

    if x.size >= 2:
        dt = np.diff(pos)
        regular = np.allclose(dt, dt[0], rtol=1e-8, atol=1e-12)
    else:
        regular = True

    if regular:
        if x.size >= 2:
            dt0 = float(dt[0])
        else:
            dt0 = 1.0

        sigma = width / dt0
        y = gaussian_filter1d(
            x_filled,
            sigma=sigma,
            mode="constant",
            cval=0.0,
        )
        norm = gaussian_filter1d(
            mask.astype(float),
            sigma=sigma,
            mode="constant",
            cval=0.0,
        )
        y = np.divide(y, norm, out=np.full_like(y, np.nan), where=norm > 0)
    else:
        y = np.full(x.shape, np.nan, dtype=float)
        valid_idx = np.flatnonzero(mask)
        if valid_idx.size == 0:
            return y
        if valid_idx.size == 1:
            y[valid_idx[0]] = x[valid_idx[0]]
            return y

        pos_valid = pos[valid_idx]
        x_valid = x[valid_idx]

        # KD-tree expects shape (N, 1) for 1D points
        pos_valid_2d = pos_valid[:, None]
        tree = cKDTree(pos_valid_2d)

        max_dist = 5.0 * width
        for i, ii in enumerate(valid_idx):
            neigh = tree.query_ball_point(pos_valid_2d[i], r=max_dist)
            if not neigh:
                continue

            d = np.abs(pos_valid_2d[neigh, 0] - pos_valid_2d[i, 0])
            w = np.exp(-0.5 * (d / width) ** 2)
            if w.sum() > 0:
                y[ii] = np.dot(x_valid[neigh], w) / w.sum()

    if out_length_mode == "same_with_nans":
        y[~mask] = np.nan
    elif out_length_mode != "same":
        raise ValueError("out_length_mode must be 'same' or 'same_with_nans'")

    return y


def fit_cos_sin_fixed_freq(t, y, omega):
    """
    Fit y = A1*cos(2*pi*f*t) + A2*sin(2*pi*f*t) to a time series.

    Parameters
    ----------
    t : array-like
        Time samples.
    y : array-like
        Measured signal samples.
    f : float
        Fixed frequency (Hz).

    Returns
    -------
    A1 : float
        Cosine coefficient.
    A2 : float
        Sine coefficient.
    y_fit : ndarray
        Fitted signal.
    residual : ndarray
        y - y_fit.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    if t.ndim != 1 or y.ndim != 1:
        raise ValueError("t and y must be 1D arrays.")
    if len(t) != len(y):
        raise ValueError("t and y must have the same length.")

    X = np.column_stack([
        np.cos(omega * t),
        np.sin(omega * t),
    ])

    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    A1, A2 = coeffs

    y_fit = X @ coeffs
    residual = y - y_fit

    return A1, A2, y_fit, residual


import numpy as np
from scipy.signal import iirnotch, filtfilt

def notch_filter(t, y, f0, Q=30):
    """
    Remove a narrow band around a single frequency f0 from y(t).

    Parameters
    ----------
    t : array-like
        Time samples (uniformly spaced).
    y : array-like
        Input signal.
    f0 : float
        Frequency to reject (Hz).
    Q : float, optional
        Quality factor of the notch. Larger Q -> narrower rejection.

    Returns
    -------
    y_filtered : ndarray
        Filtered signal.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    if t.ndim != 1 or y.ndim != 1:
        raise ValueError("t and y must be 1D arrays.")
    if len(t) != len(y):
        raise ValueError("t and y must have the same length.")

    dt = np.median(np.diff(t))
    fs = 1.0 / dt

    if f0 <= 0 or f0 >= fs / 2:
        raise ValueError("f0 must satisfy 0 < f0 < Nyquist frequency.")

    b, a = iirnotch(f0, Q, fs=fs)
    y_filtered = filtfilt(b, a, y)

    return y_filtered