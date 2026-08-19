import os, sys
import time

import numpy as np
import scipy.sparse
import scipy.spatial
import astropy.coordinates as coord
from astropy.time import Time, TimeDelta
import astropy.units as u
import copy

from SpaceBalls.paths import CONFIG_DIR, MEDIA_DIR, INPUT_DIR, OUTPUT_DIR, PROJECT_ROOT
from SpaceBalls.radiation_settings import radiation_settings_from_EEI_truth_name, get_n_days_EEI_truth, get_TSI_1AU
from SpaceBalls.radiation_fluxes_preprocessing import get_R_SunFrame_hist, get_cos_theta_s_lim, get_subdir_EEI_truth, get_EEI_truth_daily_jd_arrays, mid_day_jd_array_from_jd_interval, get_window_avg_maps
from SpaceBalls.utils import load_input_file, progress_bar, normal_smoother, interp_zeroes_in_2D_data_array, compute_orbital_period, make_list_str_key, get_rolling_jd_windows, get_2D_to_1D_idx, find_idxs, fit_cos_sin_fixed_freq, notch_filter
from SpaceBalls.sph_meshing import Grid, QuadratureGrid, RegularLatLonGrid, fit_sh_field, interp_zeroes_in_grid_data
import SpaceBalls.input_database_manager as input_manager
import config.constants as constants
from SpaceBalls.plotter import Plotter
from SpaceBalls.utils import nanrms, generate_noise_time_series, downsample_array

AU = constants.astronomical_unit(units='km')
RE = constants.earth_radius(units='km')
TOA_S = 4*np.pi*RE**2      # TOA surface area in km2
LIGHT_SPEED = constants.light_speed(units='m/s')

OUTPUT_VARS = ['jd_vec', 'aero', 'erp', 'srp', 'total_grav', 'xyz_ecef', 'xyz_sun_frame',
               'erp_recomp_radial', 'srp_recomp_radial', 
               'erp_recomp_radial_131',
               'srp_recomp_radial_225', 'erp_recomp_radial_225',
               'srp_recomp_radial_2Hz', 'erp_recomp_radial_131_2Hz',
               'srp_recomp_radial_EEI_truth_0', 'erp_recomp_radial_131_EEI_truth_0',
               'error_time_series']
ANIMATIONS_DIR = os.path.join(MEDIA_DIR, 'animations')

def read_data(sb_names):

    # NOTE: the contents of this function are just for the sake of testing
    EEI_name = check_EEI_consistency(sb_names)
    EEI_settings = radiation_settings_from_EEI_truth_name(EEI_name)
    EEI_time_series = get_true_EEI_time_series(EEI_name, 360, 180)

    acc_to_flux_factors = [get_acc_to_flux_factor(sc) for sc in sb_names]

    jd_hist_array = get_full_output_var_hist(sb_names, 'jd_vec')
    h_lat_lon_hist_array = get_full_output_var_hist(sb_names, 'h_lat_lon')
    h_lat_lon_SFF_hist_array = get_full_output_var_hist(sb_names, 'h_lat_lon_SFF')
    r_ecef_hist_array = get_full_output_var_hist(sb_names, 'xyz_ecef')
    r_sff_hist_array = get_full_output_var_hist(sb_names, 'xyz_sun_frame')

    aero_acc_hist_array = get_full_output_var_hist(sb_names, 'aero')
    srp_acc_hist_array = get_full_output_var_hist(sb_names, 'srp')
    erp_acc_hist_array = get_full_output_var_hist(sb_names, 'erp')


def estimate_EEI_avg(sb_names, jd_windows, grid: Grid, frame, method='binary_gridding', 
                     fill_zeroes_method='theta_s_fit_simplified', avg_method='numeric_integral', 
                     meas_mode="monte", make_plots=False, return_avg_flux_maps=False, accelerometer_dict=dict()):
    
    if frame=="ECEF":
        print("ECEF stacking frame - changing fill_zeroes_method to nearest")
        fill_zeroes_method = 'nearest'

    # METHODS:
    #  - binary_gridding -> required fill_method (zeroes, theta_s_fit_simplified (default), theta_s_fit_accurate)
    #  - gaussian weighting # TBC

    n_batches = 1 if method=="binary_gridding" else 5  # TODO: estimate actual memory usage
    jd_windows_split = np.array_split(jd_windows, n_batches)

    estimated_EEI_series = [None] * n_batches
    if return_avg_flux_maps: estimated_maps = [None] * n_batches 

    for batch_i, jd_windows in enumerate(jd_windows_split):
        print(f"Computing batch {batch_i}")

        #if method=="binary_gridding": # every measurement is either inside a grid cell or not. No weights applied
        jd_hist_array = get_full_output_var_hist(sb_names, 'jd_vec', jd_windows=jd_windows, 
                                                 downsample_dt=accelerometer_dict.get('subsample_dt'))
        
        radial_flux_meas_arrays, meas_weight_arrays = get_radial_measurement_arrays(
            sb_names, grid, method, frame, meas='flux', jd_windows=jd_windows, meas_mode=meas_mode, accelerometer_dict=accelerometer_dict) # sparse arrays
        
        avg_flux_maps = get_stacked_avg_maps_new(jd_windows, jd_hist_array, radial_flux_meas_arrays, meas_weight_arrays) # streteched (column for all grid points)

        del(radial_flux_meas_arrays, meas_weight_arrays)
        
        # NOTE: the following version avoids looping over individual satellites but it's not more efficient (at least with <=10 sats.)
        # avg_flux_maps_2 = get_stacked_avg_maps_new_2(jd_windows, jd_hist_array, radial_flux_meas_arrays) # streteched (column for all grid points)
        
        # NOTE 2: the old function below has been checked to give the same result as the new one within numerical precision
        # avg_flux_maps_old = get_stacked_avg_maps(jd_windows, jd_hist_array, radial_flux_meas_arrays) # streteched (column for all grid points)
        
        if fill_zeroes_method=="theta_s_fit_simplified":
            avg_flux_maps = fill_unobserved_cells_map_array_with_theta_s_fit_simplified(avg_flux_maps, grid, 
                                                                                        [make_plots]*len(avg_flux_maps))
        elif fill_zeroes_method=="theta_s_fit_accurate":                                  
            avg_flux_maps = fill_unobserved_cells_map_array_with_theta_s_fit_accurate(sb_names, avg_flux_maps, jd_windows, 
                                                                                    grid, [make_plots]*len(avg_flux_maps))
        else:
            avg_flux_maps = fill_unobserved_cells_map_array(avg_flux_maps, grid, fill_zeroes_method)
        if return_avg_flux_maps: 
            estimated_maps[batch_i] = avg_flux_maps
        
        estimated_EEI_series[batch_i] = compute_avg_EEI_from_avg_maps(avg_flux_maps, grid, method=avg_method)
    
    if return_avg_flux_maps:
        return np.concatenate(estimated_EEI_series), np.vstack(estimated_maps)
    else:
        return np.concatenate(estimated_EEI_series)



def estimate_EEI_avg_single_batch(sb_names, jd_windows, grid: Grid, frame, method='binary_gridding', 
                     fill_zeroes_method='theta_s_fit_simplified', avg_method='numeric_integral', 
                     make_plots=False):
    # METHODS:
    #  - binary_gridding -> required fill_method (zeroes, theta_s_fit_simplified (default), theta_s_fit_accurate)
    #  - gaussian weighting # TBC

    jd_hist_array = get_full_output_var_hist(sb_names, 'jd_vec')

    #if method=="binary_gridding": # every measurement is either inside a grid cell or not. No weights applied

    radial_flux_meas_arrays = get_radial_measurement_arrays(sb_names, grid, method, frame, meas='flux') # sparse arrays
    
    avg_flux_maps = get_stacked_avg_maps_new(jd_windows, jd_hist_array, radial_flux_meas_arrays) # streteched (column for all grid points)
    
    # NOTE: the following version avoids looping over individual satellites but it's not more efficient (at least with <=10 sats.)
    # avg_flux_maps_2 = get_stacked_avg_maps_new_2(jd_windows, jd_hist_array, radial_flux_meas_arrays) # streteched (column for all grid points)
    
    # NOTE 2: the old function below has been checked to give the same result as the new one within numerical precision
    # avg_flux_maps_old = get_stacked_avg_maps(jd_windows, jd_hist_array, radial_flux_meas_arrays) # streteched (column for all grid points)
    
    if fill_zeroes_method=="theta_s_fit_simplified":
        avg_flux_maps = fill_unobserved_cells_map_array_with_theta_s_fit_simplified(avg_flux_maps, grid, 
                                                                                    [make_plots]*len(avg_flux_maps))
    elif fill_zeroes_method=="theta_s_fit_accurate":                                  
        avg_flux_maps = fill_unobserved_cells_map_array_with_theta_s_fit_accurate(sb_names, avg_flux_maps, jd_windows, 
                                                                                grid, [make_plots]*len(avg_flux_maps))
    else:
        avg_flux_maps = fill_unobserved_cells_map_array(avg_flux_maps, grid, fill_zeroes_method)
    
    estimated_EEI_series = compute_avg_EEI_from_avg_maps(avg_flux_maps, grid, method=avg_method)

    return estimated_EEI_series


def get_radial_measurement_arrays(input_names, grid: Grid, method, frame='ECEF', meas='flux', 
                                  jd_windows=None, meas_mode="monte", accelerometer_dict=dict()):  # frame: "ECEF" or "SFF"
    
    n_sb = len(input_names)
    _ = check_EEI_consistency(input_names)
    #h_lat_lon_hist_array = get_h_lat_lon_hist_array(input_names, frame)
    print(f"Getting r_hist_array...")
    r_hist_array = get_r_hist_array(input_names, frame, jd_windows, downsample_dt=accelerometer_dict.get('subsample_dt'))
    
    #if accelerometer_dict.get('subsample_dt') is not None:
    #    r_hist_array = [r_hist[:-1,:] for r_hist in r_hist_array] # awful patch! TODO: figure that out...

    # build 3D measurement arrays:
    all_radial_measurement_arrays = get_radial_measurements_array(input_names, meas, jd_windows, meas_mode, 
                                                                  accelerometer_dict=accelerometer_dict)
    #all_radial_measurement_arrays_og = get_radial_measurements_array(input_names, meas, jd_windows, meas_mode="monte")
    all_sparse_meas_arrays = [None] * n_sb
    all_sparse_weight_arrays = [None] * n_sb

    for sc_i in range(n_sb):
        print(f"Computing sparse arrays for sc {sc_i}")
        print(f"Shape all_radial_measurement_arrays[sc_i]: {np.shape(all_radial_measurement_arrays[sc_i])}")
        print(f"Shape r_hist_array[sc_i]: {np.shape(r_hist_array[sc_i])}")
        sparse_meas, sparse_weights = get_stacked_measurements_matrix(grid, all_radial_measurement_arrays[sc_i], 
                                                                       method, r_hist_array=r_hist_array[sc_i])
        all_sparse_meas_arrays[sc_i] = sparse_meas        
        all_sparse_weight_arrays[sc_i] = sparse_weights

    return all_sparse_meas_arrays, all_sparse_weight_arrays


def get_radial_measurements_array(input_names, meas, jd_windows=None, meas_mode="monte", accelerometer_dict=dict()):

    if meas=="flux":
        conversion_factors = [get_acc_to_flux_factor(sc) for sc in input_names]
    elif meas=="acceleration":
        conversion_factors = [1 for _ in range(len(input_names))]

    acc_meas_hist_array = get_simulated_accelerometer_measurements(input_names, jd_windows=jd_windows,
                                                                   meas_mode=meas_mode, accelerometer_dict=accelerometer_dict)

    all_radial_measurement_arrays = [None] * len(input_names)
    for sc_i in range(len(input_names)):
        all_radial_measurement_arrays[sc_i] = conversion_factors[sc_i] * acc_meas_hist_array[sc_i][:,0]
        #if "radial" in meas_mode:
        #    all_radial_measurement_arrays[sc_i] = conversion_factors[sc_i] * acc_meas_hist_array[sc_i]
        #else:
        #    all_radial_measurement_arrays[sc_i] = conversion_factors[sc_i] * acc_meas_hist_array[sc_i][:,0]
    
    return all_radial_measurement_arrays


def get_stacked_measurements_matrix(grid: Grid, meas_hist_vec, method, r_hist_array=None, latlon_hist_array=None):
    n_steps = np.shape(r_hist_array)[0]
    #assert(n_steps == len(lon_hist_vec) == len(meas_hist_vec))

    grid_idxs = get_grid_cell_idxs(grid, method, r_hist_array=r_hist_array)
    if method=="binary_gridding":
        full_sparse_meas_array = scipy.sparse.coo_array( 
                (meas_hist_vec, (grid_idxs, np.arange(n_steps))),
                shape=(grid.n_points, n_steps)
            )
        full_sparse_weight_array = full_sparse_meas_array.copy()
        full_sparse_weight_array.data[:] = 1

    else:
        
        # try to build csr instead (transposed):
        indices = np.concatenate(grid_idxs, dtype=np.uint32, casting='unsafe') # col idxs in csr, row idxs for us
        counts = np.fromiter((len(idxs) for idxs in grid_idxs), dtype=np.uint32)
        full_meas_hist = np.repeat(meas_hist_vec, counts)       # same as full_meas_hist = np.concatenate([meas*np.ones_like(idxs_i) for (meas, idxs_i) in zip(meas_hist_vec, grid_idxs)])
        
        indptr = np.concatenate([[0], np.cumsum(counts, dtype=np.uint32)])
        full_sparse_meas_array = scipy.sparse.csr_array(   # csr construction seems faster than coo (x3) and we need it in csr anyway for the convolve function
            (full_meas_hist, indices, indptr),
            shape=(n_steps, grid.n_points)
        ).T

        full_weight_hist = get_weights_sparse_array(grid, r_hist_array, counts, indices, method)

        full_sparse_weight_array = scipy.sparse.csr_array(   # csr construction seems faster than coo (x3) and we need it in csr anyway for the convolve function
            (full_weight_hist, indices, indptr),
            shape=(n_steps, grid.n_points)
        ).T
    
    return full_sparse_meas_array, full_sparse_weight_array


def get_weights_sparse_array(grid: Grid, r_hist_array, grid_cell_counts, grid_cell_idx_hist, method):
    
    full_cos_sph_angle_hist = get_all_cos_sph_angle_relations(grid, r_hist_array, grid_cell_counts, grid_cell_idx_hist)

    if "gaussian" in method:
        # Gaussian weighting:
        n_deg_1_sigma = float(method.split("_")[1])
        a = 1/(1 - np.cos(np.deg2rad(n_deg_1_sigma)))
        full_weight_hist = np.exp(-a * (1-full_cos_sph_angle_hist))
    
    elif "hanning" in method:
        lim_rad = np.deg2rad(float(method.split("_")[1]))
        full_sph_angle_hist = np.acos(full_cos_sph_angle_hist)
        full_weight_hist = 1 + np.cos(np.pi * full_sph_angle_hist / lim_rad)

        full_weight_hist[full_sph_angle_hist>lim_rad] = 0

    return full_weight_hist

def get_all_cos_sph_angle_relations(grid: Grid, r_hist_array, grid_cell_counts, grid_cell_idx_hist):

    u_hist_array = r_hist_array / (np.sqrt(np.einsum('ij,ij->i', r_hist_array, r_hist_array))[:,None])
    full_u_hist_array = np.repeat(u_hist_array, grid_cell_counts, axis=0)
    full_cos_sph_angle_hist = np.einsum('ij,ij->i', grid.stacked_grid_u[grid_cell_idx_hist,:], full_u_hist_array)

    return full_cos_sph_angle_hist




def get_grid_cell_idxs(grid: Grid, method, r_hist_array=None, latlon_hist_array=None):

    NN_tree = scipy.spatial.KDTree(grid.stacked_grid_r)

    if method=="binary_gridding": # every measurement is either inside a grid cell or not. No weights applied
        _, indices = NN_tree.query(r_hist_array)
        # # NOTE: for regular_latlon grids, the old routine that follows is faster and gives the same (except for a few edge cases)
        # lat_cell_numbers = np.searchsorted(-grid.lat_edges_vec, -latlon_hist_array[:,0]) - 1
        # lon_cell_numbers = np.searchsorted(grid.lon_edges_vec, latlon_hist_array[:,1]) - 1
        # global_idxs = get_2D_to_1D_idx(lat_cell_numbers, lon_cell_numbers, grid.n_lon)


    elif "gaussian" in method:
        n_deg_1_sigma = float(method.split("_")[1])
        lim_to_store = 2.5 * n_deg_1_sigma    # each measurement will have nonzero weight with all grid elements within a spherical cap 
                                            # of spherical angle radius equal to n times that of a 1-sigma (exp(-1)) weight
        straight_line_dist = (RE + grid.alt_km) * np.deg2rad(lim_to_store) # cartesian distance equivalent to such limit (sphere assumed)
        indices = NN_tree.query_ball_point(r_hist_array, straight_line_dist, workers=4)
    
    elif "hanning" in method:
        lim = float(method.split("_")[1])
        straight_line_dist = (RE + grid.alt_km) * np.deg2rad(lim*1.05) # cartesian distance equivalent to such limit (sphere assumed)
        indices = NN_tree.query_ball_point(r_hist_array, straight_line_dist, workers=4)
    
    # query_ball_point: Find all points within distance r of point(s) x. (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html#scipy.spatial.KDTree)


    return indices
    
    # grid_lats - lat_hist_vec[:,None] is impossible because it would need several hundered GBs of RAM
    # Haversine formula:
    # a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    # c = 2 * np.arcsin(np.sqrt(a))
    # km = 6378.137 * c


def get_stacked_avg_maps_new_2(jd_windows, jd_hist_array, all_2D_measurement_arrays) : #, fill_unobserved_cells_method='zeroes', altitude=None): # method: 'zeroes', 'nearest', 'linear', 'cubic', 'theta_s_interp'
    # each 2D measurement array is n_grid_elements x n_steps

    n_sc = len(all_2D_measurement_arrays)
    
    #first_last_idxs_array = [None] * n_sc
    t1 = time.time()
    first_last_idxs_array = [None] * n_sc

    for sc_i in range(n_sc):
        jd_vec = jd_hist_array[sc_i]
        first_last_idxs_array[sc_i] = get_jd_window_idxs(jd_vec, jd_windows)    # this is vectorized and single-time computed
    
    assert( are_all_arrays_equal(first_last_idxs_array) )

    #print("jd_vecs are exactly equal - stacking individual sc arrays")
    first_last_idxs = first_last_idxs_array[0]
    stacked_meas_2D_array = scipy.sparse.vstack(all_2D_measurement_arrays) # stacked here means place on top of one another

    all_summed_meas = convolve_A_AI_2(stacked_meas_2D_array, first_last_idxs)

    count_meas = stacked_meas_2D_array.copy()
    count_meas.data[:] = 1
    summed_counts = convolve_A_AI_2(count_meas, first_last_idxs)

    all_summed_meas = np.split(all_summed_meas, n_sc)
    
    summed_counts = np.split(summed_counts, n_sc)
    summed_counts = np.sum(summed_counts, axis=0)
    summed_counts[summed_counts==0]=1
    
    stacked_avg_maps = np.sum(all_summed_meas, axis=0) / summed_counts

    t2 = time.time()
    print(f"Time with AI suggested function: {t2-t1}")

    return stacked_avg_maps

def get_stacked_avg_maps_new(jd_windows, jd_hist_array, all_2D_measurement_arrays, all_2D_weight_arrays) : #, fill_unobserved_cells_method='zeroes', altitude=None): # method: 'zeroes', 'nearest', 'linear', 'cubic', 'theta_s_interp'
    # each 2D measurement array is n_grid_elements x n_steps

    n_sc = len(all_2D_measurement_arrays)

    all_summed_meas = [None] * n_sc
    all_summed_weights = [None] * n_sc
    
    t1 = time.time()

    for sc_i in range(n_sc):
        jd_vec = jd_hist_array[sc_i]
        first_last_idxs = get_jd_window_idxs(jd_vec, jd_windows)    # this is vectorized and single-time computed

        weighted_meas_array = all_2D_measurement_arrays[sc_i] * all_2D_weight_arrays[sc_i]
        all_summed_meas[sc_i] = convolve_A_AI_2(weighted_meas_array, first_last_idxs)
  
        #count_meas = sparse_meas_array.copy()
        #count_meas.data[:] = 1
        sparse_weights_array = all_2D_weight_arrays[sc_i]
        all_summed_weights[sc_i] = convolve_A_AI_2(sparse_weights_array, first_last_idxs)


    summed_weights = np.sum(all_summed_weights, axis=0)
    summed_weights[np.abs(summed_weights)<1e-8] = 1
    stacked_avg_maps = np.sum(all_summed_meas, axis=0) / summed_weights

    t2 = time.time()
    #print(f"Time with AI suggested function: {t2-t1}")

    return stacked_avg_maps.T


def convolve_A_AI_2(A_in, first_last_idxs):
    n_p, n_t = A_in.shape
    first_idx = np.asarray(first_last_idxs[:,0], dtype=np.int64)
    last_idx = np.asarray(first_last_idxs[:,1], dtype=np.int64)
    n_out = first_idx.shape[0]

    A_out = np.zeros((n_p, n_out), dtype=A_in.dtype)
    A_csr = A_in.tocsr()

    # require sorted windows
    assert np.all(first_idx[:-1] <= first_idx[1:])
    assert np.all(last_idx[:-1] <= last_idx[1:])

    for i in range(n_p):
        diff = np.zeros(n_out + 1, dtype=A_in.dtype)

        row_start = A_csr.indptr[i]
        row_end = A_csr.indptr[i+1]
        cols = A_csr.indices[row_start:row_end]
        vals = A_csr.data[row_start:row_end]

        if cols.size == 0:
            continue

        # windows j with first_idx[j] <= c < last_idx[j]
        j_max = np.searchsorted(first_idx, cols, side='right') - 1
        j_min = np.searchsorted(last_idx, cols, side='right')

        valid = j_min <= j_max
        if not np.any(valid):
            continue

        np.add.at(diff, j_min[valid], vals[valid])
        np.add.at(diff, j_max[valid] + 1, -vals[valid])

        A_out[i, :] = np.cumsum(diff[:-1])

    return A_out




def compute_avg_EEI_from_avg_maps(radial_flux_avg_maps, grid: Grid, method="c00_fit"):

    # old:
    # avg_EEI = np.sum(np.array(radial_flux_avg_maps) * grid_cell_areas_CV[None,:,:], axis=(1,2)) / Earth_S

    if method=="numeric_integral":
        # NOTE that compute_surf_integral includes weights scaled in km2, hence TOA_S must also be in km2
        avg_EEI_series = grid.compute_surf_integral(np.array(radial_flux_avg_maps).T) / TOA_S

    elif method=="c00_fit":
        avg_EEI_series = np.zeros(len(radial_flux_avg_maps))
        for i, map_i in enumerate(radial_flux_avg_maps):
            sh_set_i = fit_sh_field(map_i, grid.stacked_grid_latlon[:,1], grid.stacked_grid_latlon[:,0], 50)
            avg_EEI_series[i] = sh_set_i.coeffs[0, 0, 0] * grid.total_area / TOA_S

    return avg_EEI_series



def fill_unobserved_cells_map_array_with_theta_s_fit_simplified(avg_map_array, grid: Grid, make_plot_bool_array=None):
    
    if make_plot_bool_array is None: make_plot_bool_array = [False] * len(avg_map_array)    
    out_map_array = [None] * len(avg_map_array)
    #cos_theta_s_lim = get_cos_theta_s_lim(grid.alt_km)
    
    for i, avg_map in enumerate(avg_map_array):
        out_map_array[i] = fill_unobserved_cells_with_theta_s_interp(avg_map, grid, make_plots=make_plot_bool_array[i])

    return out_map_array


def fill_unobserved_cells_map_array_with_theta_s_fit_accurate(input_names, avg_map_array, jd_windows, grid: Grid, plots_bool_array):

    # radial_meas_array has been loaded before but this reloading is not looped and this is cleaner
    all_radial_measurement_arrays = get_radial_measurements_array(input_names, 'flux')
    cos_theta_s_hist_array = get_full_output_var_hist(input_names, 'cos_theta_s')
    jd_hist_array = get_full_output_var_hist(input_names, 'jd_vec')

    first_last_idxs_array = [None] * len(input_names)
    for sc_i in range(len(input_names)):
        jd_vec = jd_hist_array[sc_i]
        first_last_idxs_array[sc_i] = get_jd_window_idxs(jd_vec, jd_windows)
    

    out_map_array = [None] * len(avg_map_array)

    for i, avg_map in enumerate(avg_map_array):
        radial_meas_window = np.concatenate([all_radial_measurement_arrays[j][first_last_idxs_array[j][i][0]:first_last_idxs_array[j][i][1]] for j in range(len(input_names))])
        cos_theta_s_window = np.concatenate([cos_theta_s_hist_array[j][first_last_idxs_array[j][i][0]:first_last_idxs_array[j][i][1]] for j in range(len(input_names))])

        out_map_array[i] = fill_unobserved_cells_with_theta_s_interp(avg_map, grid, meas_vec_for_fit=radial_meas_window, 
                                                        cos_theta_s_vec_for_fit=cos_theta_s_window, make_plots=plots_bool_array[i])
        
    return out_map_array



def fill_unobserved_cells_with_theta_s_interp(avg_map, grid: Grid, meas_vec_for_fit=None, cos_theta_s_vec_for_fit=None, make_plots=False):
    
    cos_theta_s_lim = get_cos_theta_s_lim(grid.alt_km)

    # NOTE: map MUST be in the Sun-Fixed Frame!
    # NOTE 2: avg_map should be one-dimensional

    if make_plots: Plotter.plot_geo_data_new(avg_map, grid=grid)

    LON_vec, LAT_vec = grid.stacked_grid_latlon[:,1], grid.stacked_grid_latlon[:,0]
    map_cos_theta_s_vec = -np.cos(np.radians(LAT_vec)) * np.cos(np.radians(LON_vec))  

    #zero_idxs, nonzero_idxs = (avg_map==0), (avg_map!=0)
    zero_idxs = np.abs(avg_map)<1e-9
    nonzero_idxs = ~zero_idxs

    if meas_vec_for_fit is None and cos_theta_s_vec_for_fit is None:
        meas_vec_for_fit = avg_map[nonzero_idxs]
        cos_theta_s_vec_for_fit = map_cos_theta_s_vec[nonzero_idxs]

    c, m, n = compute_meas_vs_cos_theta_s_fit(meas_vec_for_fit, cos_theta_s_vec_for_fit, 
                                                cos_theta_s_lim, plot_bool=make_plots)
    avg_map[zero_idxs] = meas_vs_cos_theta_s_function(map_cos_theta_s_vec[zero_idxs], cos_theta_s_lim, c, m, n)

    if make_plots: Plotter.plot_geo_data_new(avg_map, grid=grid)

    return avg_map


def compute_meas_vs_cos_theta_s_fit(meas_array, cos_theta_s_array, cos_theta_s_lim, cos_theta_s_lim_buffer=0.025, plot_bool=False):

    assert(len(cos_theta_s_array)==len(meas_array))

    # linear part:
    idxs_linear = (cos_theta_s_array > (cos_theta_s_lim+cos_theta_s_lim_buffer))
    A_mat = np.column_stack((cos_theta_s_array[idxs_linear], np.ones_like(cos_theta_s_array[idxs_linear])))
    A_mat_T = np.transpose(A_mat)
    m, n = np.linalg.solve(A_mat_T @ A_mat, A_mat_T @ meas_array[idxs_linear])

    # constant part:
    c = np.mean(meas_array[cos_theta_s_array <= (cos_theta_s_lim-cos_theta_s_lim_buffer)])

    if plot_bool:
        x = np.sort(cos_theta_s_array)
        y = meas_vs_cos_theta_s_function(x, cos_theta_s_lim, c, m, n)
        Plotter.plot_data({'Fit': (x, y)}, {'Data': (cos_theta_s_array, meas_array)},
                          xlabel='cos(theta_s)', ylabel='Radial flux (W/m$^2$)', scatter_alpha=0.05)

    return c, m, n

def meas_vs_cos_theta_s_function(cos_theta_s, cos_theta_s_lim, c, m, n):

    idxs_ct = cos_theta_s <= cos_theta_s_lim
    idxs_linear = ~idxs_ct

    meas = np.zeros_like(cos_theta_s)
    meas[idxs_ct] = c
    meas[idxs_linear] = m * cos_theta_s[idxs_linear] + n
    
    return meas


def fill_unobserved_cells_map_array(avg_map_array, grid: Grid, method):
    
    out_map_array = [None] * len(avg_map_array)
    for i, avg_map in enumerate(avg_map_array):
        out_map_array[i] = fill_unobserved_cells(avg_map, grid, method)

    return out_map_array


def fill_unobserved_cells(avg_map, grid: Grid, method):

    if method=="zeroes":
        return avg_map
    
    elif method=='nearest' or method=='linear' or method=='cubic':
        #return interp_zeroes_in_2D_data_array(avg_map, method=method)
        return interp_zeroes_in_grid_data(avg_map, grid, method=method)

def load_ideal_RIC_acc_meas(input_names, add_drag=True, jd_windows=None, meas_mode="monte"):
    
    if meas_mode=="monte":
        srp_acc_tag_end = ''
        erp_acc_tag_end = ''
    else:
        #erp_acc_tag_end = '_recomp_radial_131_EEI_truth_0' #'_' + meas_mode
        #srp_acc_tag_end = '_recomp_radial_EEI_truth_0' #'_' + rm_toa_order(meas_mode) # TODO!

        erp_acc_tag_end = '_recomp_radial' #'_recomp_radial_131' #'_' + meas_mode
        srp_acc_tag_end = '_recomp_radial' #'_' + rm_toa_order(meas_mode) # TODO!

    #elif meas_mode=="recomp_radial":
    #    rp_acc_tag_end = '_recomp_radial'
    #    erp_acc_tag_end = '_recomp_radial'
    
    srp_acc_hist_array = get_full_output_var_hist(input_names, 'srp'+srp_acc_tag_end, jd_windows=jd_windows)
    erp_acc_hist_array = get_full_output_var_hist(input_names, 'erp'+erp_acc_tag_end, jd_windows=jd_windows)
    if add_drag:
        aero_acc_hist_array = get_full_output_var_hist(input_names, 'aero', jd_windows=jd_windows)
        if "radial" in meas_mode:
            aero_acc_hist_array = [aero[:,0] for aero in aero_acc_hist_array]
    else: # TODO: remove zeroes allocation to save RAM
        aero_acc_hist_array = [np.zeros_like(srp_hist) for srp_hist in srp_acc_hist_array]

    acc_meas_hist_array = [None] * len(input_names)
    for sc_i in range(len(input_names)):
        acc_meas_hist_array[sc_i] = (aero_acc_hist_array[sc_i] + srp_acc_hist_array[sc_i] + erp_acc_hist_array[sc_i]) * 1e3 # accelerations are stored in km/s2!


    if "radial" in meas_mode:  # PATCH: we need non-radial measurements for accelerometer errors, but we add them here systematically for consistency
        
        # acc_hist_monte = [np.zeros((len(erp), 3)) for erp in erp_acc_hist_array]
        acc_hist_monte = get_simulated_accelerometer_measurements(input_names, add_drag=True, 
                                                                    jd_windows=jd_windows, meas_mode="monte")
        acc_hist_RIC_array = [None] * len(acc_hist_monte)
        for i, acc_RIC_monte in enumerate(acc_hist_monte):
            acc_hist_RIC_array[i] = np.hstack((acc_meas_hist_array[i][:,None], acc_RIC_monte[:,1:]))
    else:
        acc_hist_RIC_array = acc_meas_hist_array

    return acc_hist_RIC_array


def get_simulated_accelerometer_measurements(input_names, add_drag=True, jd_windows=None, meas_mode="monte",
                                             accelerometer_dict=dict()): # add_noise=False

    acc_hist_RIC_array = load_ideal_RIC_acc_meas(input_names, add_drag, jd_windows, meas_mode)

    if accelerometer_dict: # accelerometer_dict is not empty - measurements are not perfect

        jd_arrays = get_full_output_var_hist(input_names, 'jd_vec', jd_windows=jd_windows) # pass as input will be more efficient?
        #assert(jd_arrays is not None)
        # noise_example = np.load(os.path.join(PROJECT_ROOT, 'notebooks', 'noise_example.npy'))
        # noise_example[np.isnan(noise_example)] = 0
        
        periods_sec = np.array(get_orbital_periods(input_names)) * 60
        w = np.array(accelerometer_dict['rotation_revs_per_T'])  * 2 * np.pi / periods_sec

        rot_axes_RIC = accelerometer_dict['rotation_axes_RIC']
        rot_axes_RIC = rot_axes_RIC / np.linalg.norm(rot_axes_RIC, axis=1)[:,None]


        for sc_i in range(len(input_names)):

            t_hist_sec = (jd_arrays[sc_i] - jd_arrays[sc_i][0]) * 24 * 3600
            theta_hist = w[sc_i]* t_hist_sec # TODO: time-variable w?
            
            acc_meas_with_errors = add_errors_to_acc_meas(acc_hist_RIC_array[sc_i], t_hist_sec, 
                                                          theta_hist, rot_axes_RIC[sc_i], 
                                                          accelerometer_dict['biases'][sc_i],
                                                          accelerometer_dict['scale_matrices'][sc_i], 
                                                          accelerometer_dict['noise_asd'])
            # bias rejection:
            for j, a_i in enumerate(acc_meas_with_errors.T):
                A1, A2, y_fit, residual = fit_cos_sin_fixed_freq(t_hist_sec, a_i, w[sc_i])
                acc_meas_with_errors[:,j] = residual

            # attempt of scale rejection:
            #for j, a_i in enumerate(acc_meas_with_errors.T):
            #    y_filtered = notch_filter(t_hist_sec, a_i, f0=w[sc_i]/(2*np.pi), Q=300)
            #    acc_meas_with_errors[:,j] = y_filtered

            new_meas_dt = accelerometer_dict['subsample_dt']
            if new_meas_dt is not None:
                orig_meas_t_array_sec = jd_arrays[sc_i] * 3600 * 24
                
                # we're here - check normal smoother for 3D arrays
                np.save(os.path.join(OUTPUT_DIR, input_names[sc_i], 'raw_acc_meas_with_errors'), acc_meas_with_errors)
                acc_meas_with_errors = normal_smoother(acc_meas_with_errors.T, orig_meas_t_array_sec, 
                                                        new_meas_dt, out_length_mode='same_with_nans')
                np.save(os.path.join(OUTPUT_DIR, input_names[sc_i], 'smooth_acc_meas_with_errors'), acc_meas_with_errors)
                
                acc_meas_with_errors = downsample_array(acc_meas_with_errors.T,
                                                           np.mean(np.diff(orig_meas_t_array_sec)),
                                                           new_meas_dt)
                assert(not(np.any(np.isnan(acc_meas_with_errors))))
                og_meas = downsample_array(acc_hist_RIC_array[sc_i],
                                          np.mean(np.diff(orig_meas_t_array_sec)),
                                          new_meas_dt)
            else:
                og_meas = acc_hist_RIC_array[sc_i]
            
            error_time_series = acc_meas_with_errors - og_meas
            np.save(os.path.join(OUTPUT_DIR, input_names[sc_i], 'error_time_series'), error_time_series)
            np.save(os.path.join(OUTPUT_DIR, input_names[sc_i], 'jd_error_time_series'), jd_arrays[sc_i])
            acc_hist_RIC_array[sc_i] = acc_meas_with_errors
    
    return acc_hist_RIC_array


def add_errors_to_acc_meas(acc_meas_hist, t_hist_sec, theta_hist, rot_axis_RIC, bias_vec, scale_mat, noise_asd_source):

    n_meas = len(t_hist_sec)
    dt_meas = np.mean(np.diff(t_hist_sec))
    
    # Euler angle method - tricky (i.e., not commutative hence no unambiguous definition of the pitch, yaw and roll angles)
    # theta_hist = wz_ABC_RIC[sc_i] * t_hist_sec
    # rot_hist_ABC2RIC = scipy.spatial.transform.Rotation.from_euler('z', theta_hist[:, None])

    # quaternion method: rotation axis and omega (both constant for now)
    q4 = np.cos(theta_hist/2)
    if len(np.shape(rot_axis_RIC))==1: # constant axis
        q1, q2, q3 = rot_axis_RIC[:,None] * np.sin(theta_hist/2)[None,:]

    elif len(np.shape(rot_axis_RIC))==2: # time history, shape 3 x n_steps
        q1, q2, q3 = rot_axis_RIC * np.sin(theta_hist/2)[None,:]

    rot_hist_ABC2RIC = scipy.spatial.transform.Rotation.from_quat(np.stack((q1, q2, q3, q4), axis=1), scalar_first=False)
    acc_meas_hist_ABC = rot_hist_ABC2RIC.inv().apply(acc_meas_hist).T

    _, noise_x = generate_noise_time_series(n_meas, dt_meas, reference_asd=noise_asd_source)
    _, noise_y = generate_noise_time_series(n_meas, dt_meas, reference_asd=noise_asd_source)
    _, noise_z = generate_noise_time_series(n_meas, dt_meas, reference_asd=noise_asd_source)
    noise_vec = np.vstack((noise_x, noise_y, noise_z))

    noised_acc_hist_ABC = acc_meas_hist_ABC + noise_vec + bias_vec[:, None]
    scaled_noised_acc_hist_ABC = scale_mat @ noised_acc_hist_ABC 
    scaled_noised_acc_hist_RIC = rot_hist_ABC2RIC.apply(scaled_noised_acc_hist_ABC.T)

    return scaled_noised_acc_hist_RIC



def get_jd_window_idxs(jd_vec, jd_windows):
    
    #first_idxs = np.searchsorted(jd_vec, jd_windows[:,0], side="left")
    #last_idxs =  np.searchsorted(jd_vec, jd_windows[:,1], side="right")

    first_idxs = np.searchsorted(jd_vec, jd_windows[:,0], side="left")
    last_idxs =  np.searchsorted(jd_vec, jd_windows[:,1], side="left")

    return np.column_stack((first_idxs, last_idxs))

def are_all_arrays_equal(array_list):
    stacked = np.stack(array_list)
    return np.all(stacked == stacked[0])


def get_h_lat_lon_hist_array(input_names, frame):  # TODO: add jd_windows and downsample dt arguments

    if frame=="ECEF":
        return get_full_output_var_hist(input_names, 'h_lat_lon')
    elif frame=="SFF":
        return get_full_output_var_hist(input_names, 'h_lat_lon_SFF')
    else:
        print("frame has to be either ECEF or SFF")

def get_r_hist_array(input_names, frame, jd_windows=None, downsample_dt=None):

    if frame=="ECEF":
        return get_full_output_var_hist(input_names, 'xyz_ecef', jd_windows=jd_windows, downsample_dt=downsample_dt)
    elif frame=="SFF":
        return get_full_output_var_hist(input_names, 'xyz_sun_frame', jd_windows=jd_windows, downsample_dt=downsample_dt)
    else:
        print("frame has to be either ECEF or SFF")


def get_full_output_var_hist(case_names, var_name, jd_windows=None, n_days=None, downsample_dt=None):

    # TODO: n_days should not really be an input
    if n_days is None:
        n_days, completed_bool = get_n_days_max(case_names)

    n_SC = len(case_names)
    
    all_var_hist = [None] * n_SC
    for i, case in enumerate(case_names):
        var_hist = get_output_variable_hist(case, n_days, var_name, 
                                            completed_bool, jd_windows=jd_windows,
                                            downsample_dt=downsample_dt) # completed_bool: only store stacked file if all days are done
        all_var_hist[i] = var_hist
    
    return all_var_hist


def get_output_variable_hist(sb_name, n_days, var_name, save_stacked_file_if_nonexistent=True, jd_windows=None, downsample_dt=None):   # for jd: var_name=jd_vec
    print(f"Getting output variable: {var_name}")
    #stacked_var_file_name = os.path.join(OUTPUT_DIR, case_name, 'data_output', 'concatenated_'+var_name+'.npy')
    stacked_var_file_name = os.path.join(OUTPUT_DIR, sb_name, 'concatenated_'+var_name+'.npy')
    if os.path.exists(stacked_var_file_name) and (n_days==1826 or (n_days is None)): # TODO: hard-coded n_days max
        stacked_var_array = np.load(stacked_var_file_name)
        if (jd_windows is not None) or (downsample_dt is not None): # otherwise, all measurements selected
            print(f"Entering get_global_meas_idxs for var {var_name}")
            idxs_to_keep = get_global_meas_idxs(sb_name, jd_windows=jd_windows, downsample_dt=downsample_dt)
            stacked_var_array = stacked_var_array[idxs_to_keep]
            #print(f"len(idxs_to_keep): {len(idxs_to_keep)}")
    else:
        # TODO: filter if jd_windows is not None (function get_global_meas_days to be written) -done?
        if var_name in OUTPUT_VARS:
            stacked_var_array = load_and_stack_output_var_hist(sb_name, var_name, n_days)
        else:
            stacked_var_array = compute_stacked_out_var_hist(sb_name, var_name, n_days)

        if save_stacked_file_if_nonexistent:
            assert(stacked_var_array is not None)
            np.save(stacked_var_file_name, stacked_var_array)
    
    return stacked_var_array

def get_global_meas_idxs(sb_name, jd_windows=None, downsample_dt=None):
    
    #input_dict = input_manager.get_dicts(sb_name)
    actual_full_jd_meas = get_output_variable_hist(
        sb_name, n_days=None, var_name='jd_vec', save_stacked_file_if_nonexistent=False)

    if jd_windows is not None:
        min_jd, max_jd = np.min(jd_windows), np.max(jd_windows)
        # TODO: if full concatenated file does not exist, use astropy. Also it's not very efficient to load it so many times
        idxs_to_keep_jd_windows = (actual_full_jd_meas >= min_jd) & (actual_full_jd_meas < max_jd)
    else:
        idxs_to_keep_jd_windows = np.full(len(actual_full_jd_meas), True)
    
    if downsample_dt is not None:
        if jd_windows is not None:
            jd_meas = actual_full_jd_meas[idxs_to_keep_jd_windows]
        dt_orig = np.mean(np.diff(jd_meas)) * 24 * 3600
        downsampled_jd_array = downsample_array(jd_meas, dt_orig, downsample_dt)
        idxs_to_keep_downsample = np.isin(actual_full_jd_meas, downsampled_jd_array)
    else:
        idxs_to_keep_downsample = np.full(len(actual_full_jd_meas), True)

    # return np.flatnonzero((actual_full_jd_meas >= min_jd) & (actual_full_jd_meas < max_jd))
    return (idxs_to_keep_downsample * idxs_to_keep_jd_windows)


def get_global_meas_idxs_tests(sb_name, jd_windows):
    input_dict = input_manager.get_dicts(sb_name)
    EEI_name = "EEI_truth_" + str(input_dict['force_settings']['EEI_truth'])
    sim_jd_full_window = radiation_settings_from_EEI_truth_name(EEI_name)['jd_interval']

    # delta_jd_meas_simpl = input_dict['delta_t_out'] / (24 * 3600)
    # full_jd_meas_simpl = np.arange(sim_jd_full_window[0], sim_jd_full_window[1], delta_jd_meas_simpl) 
    t1 = time.time()
    actual_full_jd_meas = get_output_variable_hist(sb_name, n_days=None, var_name='jd_vec', save_stacked_file_if_nonexistent=False)
    t2 = time.time()
    print(f"Time to load actual_full_jd_meas: {t2-t1}")
    
    t1 = time.time()
    jd0, jd1 = Time(sim_jd_full_window, format='jd', scale='utc')
    delta_t_meas = input_dict['delta_t_out'] * u.second
    n_steps = int((jd1 - jd0) / delta_t_meas)
    full_jd_meas = (jd0 + np.arange(n_steps)*delta_t_meas).jd   # this is basically the concatenated output jd_vec
    t2 = time.time()
    print(f"Time to compute full_jd_meas with astropy: {t2-t1}")
    # delta_jd_meas = np.mean(np.diff(full_jd_meas)) # NOTE: this can differ from delta_jd_meas_simpl due to the nuances of the UTC scale, 
                                                   # but it is the same (?) as np.mean(np.diff(actual_full_jd_meas))
    
    # trivial implementation:
        
    # NOTE: full_jd_meas_simpl and actual_full_jd_meas are slightly different due to the time scale inconsistency (86400 factor in TimeHandler (sbsim file) vs UTC scale in jd_vec (OutputManager init))
    idxs_to_keep_b = np.where((actual_full_jd_meas>=np.min(jd_windows)) * (actual_full_jd_meas<np.max(jd_windows)))[0]
    

    min_jd, max_jd = np.min(jd_windows), np.max(jd_windows)
    idxs_to_keep = np.where((full_jd_meas>=min_jd) * (full_jd_meas<max_jd))[0]

    # more efficient (factor 10 faster) implementation: tricky because dt is not always the same in UTC
    # # gap_to_first = min_jd - sim_jd_full_window[0]
    # # first_idx = int(np.ceil(gap_to_first / delta_jd_meas))
    # # n_meas_jd_window = int((max_jd - min_jd) / delta_jd_meas) + 1 # +1: chunks to edges
    # # idxs_to_keep = np.arange(first_idx, first_idx+n_meas_jd_window, dtype=int)

    return idxs_to_keep


def load_and_stack_output_var_hist(case_name, var_name, n_days):

    daily_var_hist_array = [None] * n_days
        
    print(f"Loading daily {var_name} files...")
    for i in range(n_days):
        progress_bar(i, n_days)
        day_subdir = get_day_subdir(case_name, i)

        npy_file = os.path.join(day_subdir, var_name+'.npy')
        if os.path.exists(npy_file):
            day_var_hist = np.load(npy_file)
        else:     
            csv_file = os.path.join(day_subdir, var_name+'.csv')
            if os.path.exists(csv_file):
                try:
                    day_var_hist = np.loadtxt(day_subdir + '/' + var_name + '.csv', 
                                    delimiter=",", skiprows=2)
                except:
                    day_var_hist = np.loadtxt(day_subdir + '/' + var_name + '.csv', 
                                    delimiter=" ", skiprows=1)
                #day_var_hist = np.loadtxt(csv_file) # TODO: half of acc outputs for case A have actually a different csv separation

        daily_var_hist_array[i] = day_var_hist
    
    assert(all([np.shape(hist)==np.shape(daily_var_hist_array[0]) for hist in daily_var_hist_array]))
    if len(np.shape(daily_var_hist_array[0]))==1: # each array is single-dimensional
        stacked_var_array = np.concatenate(daily_var_hist_array)
    elif len(np.shape(daily_var_hist_array[0]))==2:
        stacked_var_array = np.vstack(daily_var_hist_array)
    
    return stacked_var_array



def compute_stacked_out_var_hist(case_name, var_name, n_days):

    if var_name=="h_lat_lon":
        return compute_sph_coord_hist_array(case_name, n_days, frame='ecef')

    elif var_name=="h_lat_lon_SFF":
        return compute_sph_coord_hist_array(case_name, n_days, frame='sun_frame')

    elif var_name=="cos_theta_s":
        return compute_cos_theta_s_hist_array(case_name, n_days)
    
    elif var_name=="erp_recomp":
        return None


def recompute_all_erp_srp_acc_meas(sb_name):    # not completed and not necessary

    EEI_truth_name = "EEI_truth_1" # hard-coded; to be obtained from sb_name

    base_data_dir = os.path.join(MEDIA_DIR, 'true_EEI', EEI_truth_name)
    rad_config = radiation_settings_from_EEI_truth_name(EEI_truth_name)
    mid_day_jd_array = mid_day_jd_array_from_jd_interval(rad_config["jd_interval"])
    all_daily_jd_arrays = get_EEI_truth_daily_jd_arrays(EEI_truth_name)

    for day_idx, mid_day_jd in enumerate(mid_day_jd_array):
        truth_jd_array = all_daily_jd_arrays[day_idx]
        mid_day_jd = mid_day_jd_array[day_idx]
        TSI_1AU_day = get_TSI_1AU(mid_day_jd, rad_config["TSI_source"])




def compute_cos_theta_s_hist_array(input_names, n_days):

    h_lat_lon_hist_SFF = get_output_variable_hist(input_names, n_days, 'h_lat_lon_SFF')
    lat_hist, lon_hist = h_lat_lon_hist_SFF[:,1], h_lat_lon_hist_SFF[:,2]
    cos_theta_s_hist =  -np.cos(np.radians(lat_hist)) * np.cos(np.radians(lon_hist))
    
    return cos_theta_s_hist


def compute_sph_coord_hist_array(input_names, n_days, frame='ecef'):

    r_hist = get_output_variable_hist(input_names, n_days, 'xyz_'+frame)

    r, lat, lon = coord.cartesian_to_spherical(r_hist[:,0], r_hist[:, 1], r_hist[:,2])
    
    h_hist = r.value - constants.earth_radius(units='km')
    lat_hist = lat.to(u.deg).value
    lon_hist = lon.to(u.deg).value
    h_lat_lon_hist_array = np.transpose(np.vstack((h_hist, lat_hist, lon_hist)))

    return h_lat_lon_hist_array


def get_day_subdir(case_name, day_idx):
    #day_subdir = os.path.join(OUTPUT_DIR, case_name, 'data_output', 'day_'+str(day_idx))
    day_subdir = os.path.join(OUTPUT_DIR, case_name, 'day_'+str(day_idx))
    
    return day_subdir



def get_n_days_max(input_names):  # in case some of the sc have not finished running
    
    EEI_name = check_EEI_consistency(input_names)
    total_n_days = get_n_days_EEI_truth(EEI_name)

    n_days_array = [len(next(os.walk(os.path.join(OUTPUT_DIR, case)))[1]) for case in input_names] #only includes folders
    n_days = min(n_days_array)

    if n_days<total_n_days:
        n_days = n_days - 1

    return n_days, n_days==total_n_days


def check_EEI_consistency(input_names):

    all_dynamics = [None] * len(input_names)

    for i, input_name in enumerate(input_names):
        config_dicts = input_manager.get_dicts(input_name)
        all_dynamics[i] = config_dicts['force_settings']

    assert(all([settings==all_dynamics[0] for settings in all_dynamics]))
    
    EEI_name = 'EEI_truth_' + str(all_dynamics[0]['EEI_truth'])
    return EEI_name


def get_true_EEI_time_series(EEI_name, n_lon=None, n_lat=None):
    
    EEI_settings = radiation_settings_from_EEI_truth_name(EEI_name)
    n_days = get_n_days_EEI_truth(EEI_name)
    subdir = get_subdir_EEI_truth(EEI_name, 'daily_hist_net_toa_surf_avg')

    full_time_series_fname = os.path.join(subdir, 'concatenated_time_series.npy')
    if os.path.exists(full_time_series_fname):
        concatenated_net_toa_time_series = np.load(full_time_series_fname)
    else:
        all_net_toa_time_series = [None] * n_days
        for day_idx in range(n_days):
            day_fname = os.path.join(subdir, 'day_'+str(day_idx))
            if os.path.exists(day_fname+'.npy'):
                all_net_toa_time_series[day_idx] = np.load(day_fname+'.npy')
            elif os.path.exists(day_fname+'.txt'):
                all_net_toa_time_series[day_idx] = np.loadtxt(day_fname+'.txt')
        
        concatenated_net_toa_time_series = np.concatenate(all_net_toa_time_series)
        np.save(full_time_series_fname,concatenated_net_toa_time_series)

    return concatenated_net_toa_time_series

def compute_EEI_estimation_errors(SB_constellation, window_days, grid: Grid, frame,
                            method, fill_zeroes_method, avg_method, meas_source,
                            EEI_truth_time_series, jd_array_EEI_truth, true_avg_flux_maps,
                            evaluation_jd_array):

    print("")
    print(f"Running smooth estimate {window_days}-day window (constellation_"+make_list_str_key(SB_constellation)+")")
    
    EEI_smooth_true = normal_smoother(EEI_truth_time_series, jd_array_EEI_truth, window_width_days=window_days, 
                                    out_length_mode='same_with_nans')
    EEI_smooth_true_coarse = np.interp(evaluation_jd_array, jd_array_EEI_truth, EEI_smooth_true)

    jd_windows = get_rolling_jd_windows(evaluation_jd_array, window_days)
    
    EEI_smooth_estimated, estimated_maps = estimate_EEI_avg(sb_names=SB_constellation, 
                                    jd_windows=jd_windows, 
                                    grid=grid, frame=frame, 
                                    method=method,
                                    fill_zeroes_method=fill_zeroes_method,
                                    avg_method=avg_method,
                                    make_plots=False,
                                    meas_mode=meas_source,
                                    return_avg_flux_maps=True)
    
    
    assert(np.shape(EEI_smooth_estimated)==np.shape(EEI_smooth_true_coarse))
    EEI_smooth_estimated[np.isnan(EEI_smooth_true_coarse)] = np.nan
        
    error_time_series = EEI_smooth_estimated-EEI_smooth_true_coarse
    rms = nanrms(error_time_series)

    error_maps = np.asarray(estimated_maps) - np.asarray(true_avg_flux_maps)
    rms_error_map = nanrms(error_maps, axis=0)
    rms2 = grid.compute_surf_integral(rms_error_map) / (4*np.pi*RE**2)

    return error_time_series


# deprecated function (?):
def compute_rolling_avg_error_time_series(SB_constellation, window_days, grid: Grid, frame,
                                          method, fill_zeroes_method, avg_method, meas_source,
                                          EEI_truth_time_series, jd_array_EEI_truth, evaluation_jd_array):

    print("")
    print(f"Running smooth estimate {window_days}-day window (constellation_"+make_list_str_key(SB_constellation)+")")
    
    EEI_smooth_true = normal_smoother(EEI_truth_time_series, jd_array_EEI_truth, window_width_days=window_days, 
                                    out_length_mode='same_with_nans')
    EEI_smooth_true_coarse = np.interp(evaluation_jd_array, jd_array_EEI_truth, EEI_smooth_true)

    jd_windows = get_rolling_jd_windows(evaluation_jd_array, window_days)
    
    EEI_smooth_estimated = estimate_EEI_avg(sb_names=SB_constellation, 
                                    jd_windows=jd_windows, 
                                    grid=grid, frame=frame, 
                                    method=method,
                                    fill_zeroes_method=fill_zeroes_method,
                                    avg_method=avg_method,
                                    make_plots=False,
                                    meas_mode=meas_source)
    
    
    assert(np.shape(EEI_smooth_estimated)==np.shape(EEI_smooth_true_coarse))
    EEI_smooth_estimated[np.isnan(EEI_smooth_true_coarse)] = np.nan
        
    error_time_series = EEI_smooth_estimated-EEI_smooth_true_coarse

    return error_time_series


def get_acc_to_flux_factor(input_name):
    config_dicts = input_manager.get_dicts(input_name)
    
    if config_dicts['sc_params']['shape']=="sphere":
        # Cannonball model
        diff_reflect = config_dicts['sc_params']['diffReflect']
        diff_degrade = config_dicts['sc_params']['diffDegrade']
        K_fact = 1 + (4/3 * diff_degrade * diff_reflect)
        mass = config_dicts['sc_params']['mass']
        area = config_dicts['sc_params']['area']
        return - mass /area * LIGHT_SPEED / K_fact
    
    else:
        print("Not implemented!")



def generate_error_map_animations(sb_names, constellation_tag, jd_windows, grid: Grid, day_0_idx=0, frame='SFF'):
    
    window_lengths = np.squeeze(np.diff(jd_windows))
    win_length = np.max(window_lengths)
    jd_windows = jd_windows[window_lengths==win_length,:]

    out_dir = os.path.join(ANIMATIONS_DIR, constellation_tag)
    os.makedirs(out_dir, exist_ok=True)

    EEI_name = check_EEI_consistency(sb_names)
    rad_config = radiation_settings_from_EEI_truth_name(EEI_name)
    EEI_full_window = rad_config["jd_interval"]
    mid_day_jd = mid_day_jd_array_from_jd_interval(EEI_full_window)

    jd_0 = EEI_full_window[0] + day_0_idx # TODO: nonzero delta_jd_0 from the inputs database?
    daily_jd_arrays = get_EEI_truth_daily_jd_arrays(EEI_name)

    #assert(len(jd_windows)==len(mid_day_jd))
    
    jd_hist_array = get_full_output_var_hist(sb_names, 'jd_vec')

    if frame=="ECEF":
        h_lat_lon_hist_array = get_full_output_var_hist(sb_names, 'h_lat_lon')
        R_hist = np.stack([np.eye(3) for _ in range(len(mid_day_jd))], axis=0)
    elif frame=="SFF":
        h_lat_lon_hist_array = get_full_output_var_hist(sb_names, 'h_lat_lon_SFF')
        R_hist = load_R_mat_jd_array(mid_day_jd, EEI_name)
        #R_hist = np.zeros((len(mid_day_jd),3,3))
        #for day_idx in range(len(mid_day_jd)):
        #    R_day_hist = load_day_R_mat_hist(day_idx, EEI_name)
        #    R_hist[day_idx, :,:] = R_day_hist[720,:,:]  # only store the mid-day rotation matrix

    method="binary_gridding"

    frame_tag = '_SFF_accurate' if frame=="SFF" else ''

    # radial_flux_meas_arrays, weight_arrays = get_radial_measurement_arrays(sb_names, grid, method, frame, 
    #                                                                        meas='flux', meas_mode='recomp_radial') # sparse arrays
    # avg_flux_maps = get_stacked_avg_maps_new(jd_windows, jd_hist_array, radial_flux_meas_arrays, weight_arrays)
    EEI_smooth_estimated, estimated_maps = estimate_EEI_avg(
                                    sb_names=sb_names, 
                                    jd_windows=jd_windows, 
                                    grid=grid, method=method, frame=frame,
                                    meas_mode='recomp_radial',
                                    return_avg_flux_maps=True)
    
    true_avg_flux_maps = get_window_avg_maps(EEI_name, jd_windows, 'daily_avg_net_Fr_800km'+frame_tag, grid)
    error_maps = true_avg_flux_maps - estimated_maps
    
    maps_max_col = 70 if frame=="SFF" else np.max(np.abs(error_maps))
    
    inclinations = [input_manager.get_dicts(sb)["orbit"]["i_0"] for sb in sb_names]
    names = [input_manager.get_dicts(sb)["orbit"]["orbit_name"] for sb in sb_names]
    title_head = "Orbits " + ", ".join([f"{name} ({incl:.2f}$^o$)" for (name, incl) in zip(names, inclinations)])
    
    Plotter.make_map_animation_arbitrary_frame(
        map_hist=error_maps.T,
        jd_vec=mid_day_jd, #[window[1] for window in jd_windows],
        grid=grid,
        data_label='Stacked meas. radial flux (W/m$^2$)',
        out_dir=out_dir,
        filename='error_stacked_maps_'+str(int(win_length))+'-days_'+frame+'.mp4',
        sat_h_lat_lon_hist_array=h_lat_lon_hist_array,
        sat_hist_jd_array=jd_hist_array,
        orbital_periods_minutes=get_orbital_periods(sb_names),
        add_satellite_positions=False,
        add_groundtracks=False,
        add_coastlines=True,
        fade_coastlines=True,
        R_mat_hist=R_hist,
        groundtrack_linstyle='-',
        max_abs_colorscale=maps_max_col,
        title_header=title_head
    )


def generate_constellation_stacking_animations(sb_names, constellation_tag, jd_windows, grid: Grid, 
                                               day_0_idx=0, frame='SFF', mid_day_jd=None):
    window_lengths = np.squeeze(np.diff(jd_windows))
    win_length = np.max(window_lengths)
    min_length = np.min(window_lengths)

    #extra_first_block = [[jd_windows[0][0], jd_windows[0][0]+i+0.5] for i in range(int(np.floor(min_length)))]
    #jd_windows = np.vstack((extra_first_block, jd_windows))
    #jd_windows[window_lengths==win_length,:]

    out_dir = os.path.join(ANIMATIONS_DIR, constellation_tag)
    os.makedirs(out_dir, exist_ok=True)

    EEI_name = check_EEI_consistency(sb_names)
    rad_config = radiation_settings_from_EEI_truth_name(EEI_name)
    EEI_full_window = rad_config["jd_interval"]
    if mid_day_jd is None:
        mid_day_jd = mid_day_jd_array_from_jd_interval(EEI_full_window)

    jd_0 = EEI_full_window[0] + day_0_idx # TODO: nonzero delta_jd_0 from the inputs database?
    daily_jd_arrays = get_EEI_truth_daily_jd_arrays(EEI_name)

    #assert(len(jd_windows)==len(mid_day_jd))
    
    jd_hist_array = get_full_output_var_hist(sb_names, 'jd_vec')

    if frame=="ECEF":
        h_lat_lon_hist_array = get_full_output_var_hist(sb_names, 'h_lat_lon')
        R_hist = np.stack([np.eye(3) for _ in range(len(mid_day_jd))], axis=0)
    elif frame=="SFF":
        h_lat_lon_hist_array = get_full_output_var_hist(sb_names, 'h_lat_lon_SFF')
        R_hist = load_R_mat_jd_array(mid_day_jd, EEI_name)
        #R_hist = np.zeros((len(mid_day_jd),3,3))
        #for day_idx in range(len(mid_day_jd)):
        #    R_day_hist = load_day_R_mat_hist(day_idx, EEI_name)
        #    R_hist[day_idx, :,:] = R_day_hist[720,:,:]  # only store the mid-day rotation matrix

    method="binary_gridding"
    radial_flux_meas_arrays, weight_arrays = get_radial_measurement_arrays(sb_names, grid, method, frame, meas='flux') # sparse arrays
    avg_flux_maps = get_stacked_avg_maps_new(jd_windows, jd_hist_array, radial_flux_meas_arrays, weight_arrays)
    
    inclinations = [input_manager.get_dicts(sb)["orbit"]["i_0"] for sb in sb_names]
    names = [input_manager.get_dicts(sb)["orbit"]["orbit_name"] for sb in sb_names]
    title_head = "Orbits " + ", ".join([f"{name} ({incl:.2f}$^o$)" for (name, incl) in zip(names, inclinations)])
    
    Plotter.make_map_animation_arbitrary_frame(
        map_hist=avg_flux_maps.T,
        jd_vec=mid_day_jd, #[window[1] for window in jd_windows],
        grid=grid,
        data_label='Stacked meas. radial flux (W/m$^2$)',
        out_dir=out_dir,
        filename= 'stacking_all_days_'+str(int(win_length))+'-days_'+frame+'.mp4',
        sat_h_lat_lon_hist_array=h_lat_lon_hist_array,
        sat_hist_jd_array=jd_hist_array,
        orbital_periods_minutes=get_orbital_periods(sb_names),
        add_satellite_positions=False,
        add_coastlines=True,
        fade_coastlines=True,
        add_groundtracks=True,
        R_mat_hist=R_hist,
        groundtrack_linstyle='-',
        title_header=title_head,
        adaptative_colorbar=True
    )




def generate_constellation_day_animations(sb_names, constellation_tag, day_idxs, grid: Grid, max_hours=24):

    out_dir = os.path.join(ANIMATIONS_DIR, constellation_tag)
    os.makedirs(out_dir, exist_ok=True)

    EEI_name = check_EEI_consistency(sb_names)
    jd_hist_array = get_full_output_var_hist(sb_names, 'jd_vec')
    h_lat_lon_hist_array = get_full_output_var_hist(sb_names, 'h_lat_lon')
    h_lat_lon_SFF_hist_array = get_full_output_var_hist(sb_names, 'h_lat_lon_SFF')

    # TBC...

def get_orbital_periods(sb_names):
    MU = 398600
    all_T = [None] * len(sb_names)

    for i, sb in enumerate(sb_names):
        config_dicts = input_manager.get_dicts(sb)
        all_T[i] = compute_orbital_period(MU, config_dicts['orbit']['a_0'])

    return all_T


def load_R_mat_jd_array(jd_array, EEI_truth_name):
    
    rad_settings = radiation_settings_from_EEI_truth_name(EEI_truth_name)
    boa_fname = rad_settings['ephemerides']
    n_days = int(np.diff(rad_settings['jd_interval'])[0])

    path = os.path.join(MEDIA_DIR, 'solar_ephemerides', boa_fname, 'daily_files')
    concatenated_jd_array_file = os.path.join(path, 'concatenated_jd_array.npy')
    concatenated_day_idx_file = os.path.join(path, 'concatenated_day_idx.npy')
    if os.path.exists(concatenated_jd_array_file):
        concatenated_jd_array = np.load(concatenated_jd_array_file)
        concatenated_day_idx_array = np.load(concatenated_day_idx_file)
    else:
        all_jd_arrays = [None] * n_days
        all_day_idxs = [None] * n_days
        for i in range(n_days):
            progress_bar(i, n_days)
            all_jd_arrays[i] = np.load(os.path.join(path, 'jd_array_day_'+str(i)+'.npy'))
            all_day_idxs[i] = np.ones_like(all_jd_arrays[i]) * i
        
        concatenated_jd_array = np.concatenate(all_jd_arrays)
        concatenated_day_idx_array = np.concatenate(all_day_idxs)
        np.save(concatenated_jd_array_file, concatenated_jd_array)
        np.save(concatenated_day_idx_file, concatenated_day_idx_array)

    idxs = find_idxs(jd_array, concatenated_jd_array)
    day_idxs = np.asarray(concatenated_day_idx_array[idxs], dtype=int)
    
    R_hist = np.zeros((len(jd_array),3,3))
    day_idx_prev = np.nan
    for i, jd in enumerate(jd_array):
        progress_bar(i, len(jd_array))
        day_idx = day_idxs[i]
        if day_idx!=day_idx_prev:
            R_fname = os.path.join(path, 'R_ECEF_to_SunFrame_day_'+str(day_idx))
            day_R_array = np.load(R_fname+'.npy')
            day_jd_array = np.load(os.path.join(path, 'jd_array_day_'+str(day_idx)+'.npy'))
        day_idx_prev = day_idx

        step_idx = find_idxs(jd, day_jd_array)
        R_hist[i,:,:] = day_R_array[step_idx, :, :]

    return R_hist

            


def load_day_R_mat_hist(day_idx, EEI_truth_name):

    boa_fname = radiation_settings_from_EEI_truth_name(EEI_truth_name)['ephemerides']
    path = os.path.join(MEDIA_DIR, 'solar_ephemerides', boa_fname, 'daily_files')
    R_fname = os.path.join(path, 'R_ECEF_to_SunFrame_day_'+str(day_idx))
    day_R_array = np.load(R_fname+'.npy')

    return day_R_array


def get_all_jd_windows(EEI_name, window_days):
    
    rad_config = radiation_settings_from_EEI_truth_name(EEI_name)
    EEI_full_window = rad_config["jd_interval"]
    mid_day_jd = mid_day_jd_array_from_jd_interval(EEI_full_window)
    jd_windows = get_rolling_jd_windows(mid_day_jd, window_days)

    return jd_windows



# old functions


def get_stacked_avg_maps(jd_windows, jd_hist_array, all_2D_measurement_arrays) : #, fill_unobserved_cells_method='zeroes', altitude=None): # method: 'zeroes', 'nearest', 'linear', 'cubic', 'theta_s_interp'
    # each 2D measurement array is n_grid_elements x n_steps

    n_sc = len(all_2D_measurement_arrays)
    n_windows = len(jd_windows)
    
    first_last_idxs_array = [None] * n_sc

    for sc_i in range(n_sc):
        jd_vec = jd_hist_array[sc_i]
        first_last_idxs_array[sc_i] = get_jd_window_idxs(jd_vec, jd_windows)    # this is vectorized and single-time computed

    all_meas_avg_maps = [None] * n_windows

    if are_all_arrays_equal(first_last_idxs_array):
        print("jd_vecs are exactly equal - stacking individual sc arrays")
        first_last_idxs = first_last_idxs_array[0]
        stacked_meas_2D_array = scipy.sparse.vstack(all_2D_measurement_arrays) # stacked here means place on top of one another
        stacked_meas_2D_array = stacked_meas_2D_array.tocsr()

        t1 = time.time()
        for window_i, jd_window in enumerate(jd_windows):    # vectorizing this loop does not at all seem straightforward
            progress_bar(window_i, len(jd_windows))
            first_idx, last_idx = first_last_idxs[window_i,:]
            all_meas_counts = getnnz(stacked_meas_2D_array[:,first_idx:last_idx], axis=1)
            all_stacked_meas = stacked_meas_2D_array[:,first_idx:last_idx].sum(axis=1) #stacked as in compressed along time axis but they are still stacked as in the meaning before

            meas_counts_array = np.split(all_meas_counts, n_sc)
            stacked_measurements_array = np.split(all_stacked_meas, n_sc)
            all_meas_avg_maps[window_i] = get_meas_avg_map_array(stacked_measurements_array, meas_counts_array)
        t2 = time.time()
        print(f"Time to loop over jd_windows (all sattellites): {t2-t1}")

    else:
        print("jd_vecs are not equal - looping over individual sc arrays")
        for window_i, jd_window in enumerate(jd_windows):    # vectorizing this loop does not at all seem straightforward
            #progress_bar(window_i, len(jd_windows))
            stacked_measurements_array = [None] * n_sc
            meas_counts_array = [None] * n_sc

            for sc_i in range(n_sc):
                first_idx, last_idx = first_last_idxs_array[sc_i][window_i,:]

                meas_counts_array[sc_i] = getnnz(all_2D_measurement_arrays[sc_i][:,first_idx:last_idx], axis=1)
                stacked_measurements_array[sc_i] = all_2D_measurement_arrays[sc_i][:,first_idx:last_idx].sum(axis=1)

            all_meas_avg_maps[window_i] = get_meas_avg_map_array(stacked_measurements_array, meas_counts_array)
    
    return all_meas_avg_maps

def get_meas_avg_map_array(stacked_measurements_array, meas_counts_array):

    summed_meas_array = np.sum(stacked_measurements_array, axis=0)
    summed_counts_array = np.sum(meas_counts_array, axis=0)
    summed_counts_array[summed_counts_array==0] = 1

    return summed_meas_array / summed_counts_array



def getnnz(sparse_array_in, axis):

    sparse_array = copy.deepcopy(sparse_array_in)

    # it seems that creating a copy of the sparse array is not needed
    sparse_array.data[:] = 1
    nnz_array = sparse_array.sum(axis=axis)
    
    return nnz_array




if __name__ == "__main__":
    #sb_names = ['7b680b988948c2cd', '9e750b3b250f449b', 'a898e645c936d9db']
    #sb_names = ['2151699c6e13c489','bc67760fbec64597','ff616870cd0c6bdf']
    #jd_windows = 3*365+np.array([[2458119.5+i, 2458119.5+365+i] for i in range(100)])

    sb_names = ['32c08b39a7c5dce2', '7b54acc8701b0625', '82c0c7e77260bb76'] # best 3-sat, 1-yr
    
    EEI_name = check_EEI_consistency(sb_names)
    jd_windows = get_all_jd_windows(EEI_name, 365)[:10]

    grid = QuadratureGrid(alt_km=800, order=131)
    generate_error_map_animations(sb_names, 'test3', jd_windows, grid)
    #generate_constellation_stacking_animations(sb_names, 'test2', jd_windows, grid)

    """
    t1 = time.time()
    grid = QuadratureGrid(alt_km=800, order=131)
    estimated_EEI = estimate_EEI_avg(sb_names, jd_windows=jd_windows, grid=grid, frame='SFF')
    print(estimated_EEI)
    t2 = time.time()
    print(f"Time to compute EEI with quadrature grid (c00 fit): {t2-t1}")
    
    t1 = time.time()
    grid = QuadratureGrid(alt_km=800, order=131)
    estimated_EEI_2 = estimate_EEI_avg(sb_names, jd_windows=jd_windows, grid=grid, frame='SFF', avg_method='c00_fit')
    print(estimated_EEI_2)
    t2 = time.time()
    print(f"Time to compute EEI with quadrature grid (c00 fit): {t2-t1}")
    print(estimated_EEI_2 - estimated_EEI)

    t1 = time.time()
    grid = RegularLatLonGrid(alt_km=800, n_lat=180, n_lon=360)
    estimated_EEI_3 = estimate_EEI_avg(sb_names, jd_windows=jd_windows, grid=grid, frame='SFF')
    print(estimated_EEI_3)
    t2 = time.time()
    print(f"Time to compute EEI with regular grid: {t2-t1}")

    t1 = time.time()
    grid = RegularLatLonGrid(alt_km=800, n_lat=180, n_lon=360)
    estimated_EEI_4 = estimate_EEI_avg(sb_names, jd_windows=jd_windows, grid=grid, frame='SFF', avg_method='c00_fit')
    print(estimated_EEI_4)
    t2 = time.time()
    print(f"Time to compute EEI with regular grid: {t2-t1}")

    print(estimated_EEI_3 - estimated_EEI_4)

    read_data(sb_names)
    """



#############################################
###### DEPRECATED FUNCTION GRAVEYARD ########
#############################################

"""

def convolve_A_AI(A_in, w, dw):

    n_p, n_t = A_in.shape
    n_out = (n_t - w + 1 + dw - 1) // dw
    A_out = np.zeros((A_in.shape[0], n_out), dtype=A_in.dtype)

    A_csr = A_in.tocsr()
    for i in range(n_p):
        progress_bar(i, n_p)
        diff = np.zeros(n_out + 1, dtype=A_in.dtype)
        row_start = A_csr.indptr[i]
        row_end = A_csr.indptr[i+1]
        cols = A_csr.indices[row_start:row_end]
        vals = A_csr.data[row_start:row_end]

        for c, v in zip(cols, vals):
            j0 = (c - w + dw) // dw
            if j0 < 0:
                j0 = 0
            j1 = c // dw
            if j1 >= n_out:
                j1 = n_out - 1
            if j0 <= j1:
                diff[j0] += v
                diff[j1 + 1] -= v

        A_out[i, :] = np.cumsum(diff[:-1])
     
    return A_out


def get_stacked_avg_maps(jd_windows, jd_hist_array, all_2D_measurement_arrays) : #, fill_unobserved_cells_method='zeroes', altitude=None): # method: 'zeroes', 'nearest', 'linear', 'cubic', 'theta_s_interp'
    # each 2D measurement array is n_grid_elements x n_steps

    n_sc = len(all_2D_measurement_arrays)
    n_windows = len(jd_windows)
    
    first_last_idxs_array = [None] * n_sc

    for sc_i in range(n_sc):
        jd_vec = jd_hist_array[sc_i]
        first_last_idxs_array[sc_i] = get_jd_window_idxs(jd_vec, jd_windows)    # this is vectorized and single-time computed

    all_meas_avg_maps = [None] * n_windows

    if are_all_arrays_equal(first_last_idxs_array):
        print("jd_vecs are exactly equal - stacking individual sc arrays")
        first_last_idxs = first_last_idxs_array[0]
        stacked_meas_2D_array = scipy.sparse.vstack(all_2D_measurement_arrays) # stacked here means place on top of one another
        stacked_meas_2D_array = stacked_meas_2D_array.tocsr()

        t1 = time.time()
        for window_i, jd_window in enumerate(jd_windows):    # vectorizing this loop does not at all seem straightforward
            progress_bar(window_i, len(jd_windows))
            first_idx, last_idx = first_last_idxs[window_i,:]
            all_meas_counts = getnnz(stacked_meas_2D_array[:,first_idx:last_idx], axis=1)
            all_stacked_meas = stacked_meas_2D_array[:,first_idx:last_idx].sum(axis=1) #stacked as in compressed along time axis but they are still stacked as in the meaning before

            meas_counts_array = np.split(all_meas_counts, n_sc)
            stacked_measurements_array = np.split(all_stacked_meas, n_sc)
            all_meas_avg_maps[window_i] = get_meas_avg_map_array(stacked_measurements_array, meas_counts_array)
        t2 = time.time()
        print(f"Time to loop over jd_windows (all sattellites): {t2-t1}")

    else:
        print("jd_vecs are not equal - looping over individual sc arrays")
        for window_i, jd_window in enumerate(jd_windows):    # vectorizing this loop does not at all seem straightforward
            #progress_bar(window_i, len(jd_windows))
            stacked_measurements_array = [None] * n_sc
            meas_counts_array = [None] * n_sc

            for sc_i in range(n_sc):
                first_idx, last_idx = first_last_idxs_array[sc_i][window_i,:]

                meas_counts_array[sc_i] = getnnz(all_2D_measurement_arrays[sc_i][:,first_idx:last_idx], axis=1)
                stacked_measurements_array[sc_i] = all_2D_measurement_arrays[sc_i][:,first_idx:last_idx].sum(axis=1)

            all_meas_avg_maps[window_i] = get_meas_avg_map_array(stacked_measurements_array, meas_counts_array)
    
    return all_meas_avg_maps


def get_meas_avg_map_array(stacked_measurements_array, meas_counts_array):

    summed_meas_array = np.sum(stacked_measurements_array, axis=0)
    summed_counts_array = np.sum(meas_counts_array, axis=0)
    summed_counts_array[summed_counts_array==0] = 1

    return summed_meas_array / summed_counts_array



def getnnz(sparse_array_in, axis):

    sparse_array = copy.deepcopy(sparse_array_in)

    # it seems that creating a copy of the sparse array is not needed
    sparse_array.data[:] = 1
    nnz_array = sparse_array.sum(axis=axis)
    
    return nnz_array

"""