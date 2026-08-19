import os, sys
import importlib.util
import time

print("PYTHON:", sys.executable)
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))
print("CONDA_PREFIX:", os.environ.get("CONDA_PREFIX"))

# from importlib import spec_from_file_location
import numpy as np
import scipy.sparse
import astropy.coordinates as coord
import astropy.units as u

from SpaceBalls.paths import CONFIG_DIR, MEDIA_DIR, INPUT_DIR, OUTPUT_DIR
from SpaceBalls.radiation_settings import radiation_settings_from_EEI_truth_name
from SpaceBalls.radiation_fluxes_preprocessing import get_R_SunFrame_hist, get_cos_theta_s_lim
from SpaceBalls.utils import load_input_file, progress_bar, normal_smoother, interp_zeroes_in_2D_data_array, compute_orbital_period, make_list_str_key, get_rolling_jd_windows, get_2D_to_1D_idx
from SpaceBalls.sph_meshing import get_sphere_grid, get_spherical_grid_cell_areas, get_reshaped_grid, lonlat_to_r, fit_sh_field
import SpaceBalls.input_database_manager as input_manager
import config.constants as constants
from SpaceBalls.plotter import Plotter

AU = constants.astronomical_unit(units='km')
RE = constants.earth_radius(units='km')

OUTPUT_VARS = ['jd_vec', 'aero', 'erp', 'srp', 'total_grav', 'xyz_ecef', 'xyz_sun_frame']
ANIMATIONS_DIR = os.path.join(MEDIA_DIR, 'animations')

#sys.path.insert(0, str(INPUT_DIR.parent))  # parent of 'config'



def read_data(case, sc_names): # TODO: case is now "case_5_years", to be replaced with "case_EEI_1"

    # NOTE: the contents of this function are just for the sake of testing
    input_names = sc_names #[case + '_' + sc_name for sc_name in sc_names]
    EEI_name = check_EEI_consistency(input_names)
    EEI_settings = radiation_settings_from_EEI_truth_name(EEI_name)
    EEI_time_series = get_true_EEI_time_series(EEI_name, 360, 180)

    acc_to_flux_factors = [get_acc_to_flux_factor(sc) for sc in input_names]

    jd_hist_array = get_full_output_var_hist(input_names, 'jd_vec')
    h_lat_lon_hist_array = get_full_output_var_hist(input_names, 'h_lat_lon')
    h_lat_lon_SFF_hist_array = get_full_output_var_hist(input_names, 'h_lat_lon_SFF')
    r_ecef_hist_array = get_full_output_var_hist(input_names, 'xyz_ecef')
    r_sff_hist_array = get_full_output_var_hist(input_names, 'xyz_sun_frame')

    aero_acc_hist_array = get_full_output_var_hist(input_names, 'aero')
    srp_acc_hist_array = get_full_output_var_hist(input_names, 'srp')
    erp_acc_hist_array = get_full_output_var_hist(input_names, 'erp')


def estimate_EEI_avg_sh(case, sc_names, altitude, jd_windows, frame, lmax, make_plots=False, return_mode='EEI_avg'):

    Re = constants.earth_radius(units='km')
    input_names = sc_names # [case + '_' + sc_name for sc_name in sc_names]
    jd_hist_array = get_full_output_var_hist(input_names, 'jd_vec')
    first_last_idxs = [get_jd_window_idxs(jd_hist, jd_windows) for jd_hist in jd_hist_array]

    EEI_avg_array = np.zeros(len(jd_windows))
    sh_coeffs_array = [None] * len(jd_windows)

    for i, jd_window in enumerate(jd_windows):
        all_h_lat_lon_hist = np.concatenate([array[first_last_idxs[j][i][0]:first_last_idxs[j][i][1]] 
                                             for j, array in enumerate(get_h_lat_lon_hist_array(input_names, frame))])
        
        all_radial_measurements = np.concatenate([array[first_last_idxs[j][i][0]:first_last_idxs[j][i][1]] 
                                             for j, array in enumerate(get_radial_measurements_array(input_names, 'flux'))])
    
        print("Fitting SH map...")
        t1 = time.time()
        sh_coeffs = fit_sh_field(all_radial_measurements, all_h_lat_lon_hist[:,2], all_h_lat_lon_hist[:,1], lmax=lmax)
        t2 = time.time()
        print(f"Time to fit SH field: {t2-t1}")
        
        sh_coeffs_array[i] = sh_coeffs
        EEI_avg_array[i] = sh_coeffs.coeffs[0, 0, 0] * (Re + altitude)**2 / Re**2

    if return_mode=='EEI_avg':
        return EEI_avg_array
    elif return_mode=='sh_objects':
        return sh_coeffs_array
    

def estimate_EEI_avg(case, sc_names, altitude, jd_windows, n_lon, n_lat, frame, fill_method='zeroes', make_plots=False):

    if fill_method=="theta_s_fit": assert(frame=="SFF")
    
    input_names = sc_names # [case + '_' + sc_name for sc_name in sc_names]
    jd_hist_array = get_full_output_var_hist(input_names, 'jd_vec')

    ## old method:
    # radial_flux_meas_3D_arrays = get_radial_measurement_arrays(input_names, n_lon, n_lat, frame, meas='flux', mode='3D') # this step takes significantly longer
    # avg_flux_maps = get_stacked_avg_maps_3D_mode(jd_windows, jd_hist_array, radial_flux_meas_3D_arrays) # this comes wiht unobserved cells

    ## new method - verified to give the same result and 20-25% faster
    radial_flux_meas_2D_arrays = get_radial_measurement_arrays(input_names, n_lon, n_lat, frame, meas='flux', mode='2D') # this step takes significantly longer
    avg_flux_maps_stretched = get_stacked_avg_maps_2D_mode(jd_windows, jd_hist_array, radial_flux_meas_2D_arrays) # this comes wiht unobserved cells
    avg_flux_maps = [np.reshape(map_i, (n_lat,n_lon)) for map_i in avg_flux_maps_stretched]


    if fill_method=="theta_s_fit_simplified":
        avg_flux_maps = fill_unobserved_cells_map_array_with_theta_s_fit_simplified(avg_flux_maps, get_cos_theta_s_lim(altitude), 
                                                                                    [make_plots]*len(avg_flux_maps))
    elif fill_method=="theta_s_fit_accurate":                                  
        avg_flux_maps = fill_unobserved_cells_map_array_with_theta_s_fit_accurate(input_names, avg_flux_maps, jd_windows, 
                                                                                  get_cos_theta_s_lim(altitude), [make_plots]*len(avg_flux_maps))
    else:
        avg_flux_maps = fill_unobserved_cells_map_array(avg_flux_maps, fill_method)


    #avg_flux_maps = fill_unobserved_cells_map_array(input_names, avg_flux_maps, jd_windows, fill_method, altitude)

    EEI_avg = compute_avg_EEI_from_avg_maps(avg_flux_maps, altitude)

    return EEI_avg


def get_h_lat_lon_hist_array(input_names, frame):

    if frame=="ECEF":
        return get_full_output_var_hist(input_names, 'h_lat_lon')
    elif frame=="SFF":
        return get_full_output_var_hist(input_names, 'h_lat_lon_SFF')
    else:
        print("frame has to be either ECEF or SFF")



def get_radial_measurement_arrays(input_names, n_lon=360, n_lat=180, frame='ECEF', meas='flux', mode='3D'):  # frame: "ECEF" or "SFF"

    _ = check_EEI_consistency(input_names)

    #if frame=="ECEF":
    #    h_lat_lon_hist_array = get_full_output_var_hist(input_names, 'h_lat_lon')
    #elif frame=="SFF":
    #    h_lat_lon_hist_array = get_full_output_var_hist(input_names, 'h_lat_lon_SFF')
    h_lat_lon_hist_array = get_h_lat_lon_hist_array(input_names, frame)

    # build 3D measurement arrays:
    all_radial_measurement_arrays = get_radial_measurements_array(input_names, meas)

    for sc_i in range(len(input_names)): # we want to keep individual S/C arrays separate for windowing later (third axis is time axis and each satellite might have it different)
        
        radial_meas_hist = all_radial_measurement_arrays[sc_i]
        lat_hist, lon_hist = h_lat_lon_hist_array[sc_i][:,1], h_lat_lon_hist_array[sc_i][:,2]

        all_radial_measurement_arrays[sc_i] = get_stacked_measurements_matrix_regular_grid(
                                            n_lon, n_lat, lon_hist, lat_hist, radial_meas_hist, out_mode=mode
                                            )
    return all_radial_measurement_arrays



def get_radial_measurements_array(input_names, meas):

    if meas=="flux":
        conversion_factors = [get_acc_to_flux_factor(sc) for sc in input_names]
    elif meas=="acceleration":
        conversion_factors = [1 for _ in range(len(input_names))]

    acc_meas_hist_array = get_simulated_accelerometer_measurements(input_names)

    all_radial_measurement_arrays = [None] * len(input_names)
    for sc_i in range(len(input_names)):
        all_radial_measurement_arrays[sc_i] = conversion_factors[sc_i] * acc_meas_hist_array[sc_i][:,0]
    
    return all_radial_measurement_arrays



def are_all_arrays_equal(array_list):
    stacked = np.stack(array_list)
    return np.all(stacked == stacked[0])


# currently unused function
def get_stacked_avg_maps_2D_mode(jd_windows, jd_hist_array, all_2D_measurement_arrays) : #, fill_unobserved_cells_method='zeroes', altitude=None): # method: 'zeroes', 'nearest', 'linear', 'cubic', 'theta_s_interp'
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

        for window_i, jd_window in enumerate(jd_windows):    # vectorizing this loop does not at all seem straightforward
            #progress_bar(window_i, len(jd_windows))
            first_idx, last_idx = first_last_idxs[window_i,:]
            all_meas_counts = getnnz(stacked_meas_2D_array[:,first_idx:last_idx], axis=1)
            all_stacked_meas = stacked_meas_2D_array[:,first_idx:last_idx].sum(axis=1) #stacked as in compressed along time axis but they are still stacked as in the meaning before

            meas_counts_array = np.split(all_meas_counts, n_sc)
            stacked_measurements_array = np.split(all_stacked_meas, n_sc)
            all_meas_avg_maps[window_i] = get_meas_avg_map_array(stacked_measurements_array, meas_counts_array)

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



def get_stacked_avg_maps_3D_mode(jd_windows, jd_hist_array, all_3D_measurement_arrays) : #, fill_unobserved_cells_method='zeroes', altitude=None): # method: 'zeroes', 'nearest', 'linear', 'cubic', 'theta_s_interp'

    n_sc = len(all_3D_measurement_arrays)
    n_windows = len(jd_windows)
    #if fill_unobserved_cells_method=="theta_s_interp": assert(altitude is not None)
    
    first_last_idxs_array = [None] * n_sc

    for sc_i in range(n_sc):
        jd_vec = jd_hist_array[sc_i]
        first_last_idxs_array[sc_i] = get_jd_window_idxs(jd_vec, jd_windows)    # this is vectorized and single-time computed

    all_meas_avg_maps = [None] * n_windows

    for window_i, jd_window in enumerate(jd_windows):    # vectorizing this loop does not at all seem straightforward
        progress_bar(window_i, len(jd_windows))
        stacked_measurements_array = [None] * n_sc
        meas_counts_array = [None] * n_sc

        for sc_i in range(n_sc):
            first_idx, last_idx = first_last_idxs_array[sc_i][window_i,:]

            meas_counts_array[sc_i] = getnnz(all_3D_measurement_arrays[sc_i][:,:,first_idx:last_idx], axis=2)
            stacked_measurements_array[sc_i] = all_3D_measurement_arrays[sc_i][:,:,first_idx:last_idx].sum(axis=2)

        summed_meas_array = np.sum(stacked_measurements_array, axis=0)
        summed_counts_array = np.sum(meas_counts_array, axis=0)
        summed_counts_array[summed_counts_array==0] = 1

        all_meas_avg_maps[window_i] = summed_meas_array / summed_counts_array
    
    return all_meas_avg_maps


def fill_unobserved_cells_map_array(avg_map_array, method):
    
    out_map_array = [None] * len(avg_map_array)
    for i, avg_map in enumerate(avg_map_array):
        out_map_array[i] = fill_unobserved_cells(avg_map, method)

    return out_map_array


def fill_unobserved_cells(avg_map, method):

    if method=="zeroes":
        return avg_map
    
    elif method=='nearest' or method=='linear' or method=='cubic':
        return interp_zeroes_in_2D_data_array(avg_map, method=method)
    

# TODO: complete accurate method
def fill_unobserved_cells_map_array_with_theta_s_fit_accurate(input_names, avg_map_array, jd_windows, cos_theta_s_lim, plots_bool_array):

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

        out_map_array[i] = fill_unobserved_cells_with_theta_s_interp_new(avg_map, cos_theta_s_lim, meas_vec_for_fit=radial_meas_window, 
                                                        cos_theta_s_vec_for_fit=cos_theta_s_window, make_plots=plots_bool_array[i])
        
        #c, m, n = compute_meas_vs_cos_theta_s_fit(map_vec[nonzero_idxs], cos_theta_s_vec[nonzero_idxs], 
        #                                            cos_theta_s_lim, make_plot_bool_array[i])
        #out_map_array[i] = fill_unobserved_cells_with_theta_s_interp(avg_map, radial_meas_window, cos_theta_s_window, altitude_km)
        
    return out_map_array


def fill_unobserved_cells_with_theta_s_interp_new(avg_map, cos_theta_s_lim, meas_vec_for_fit=None, cos_theta_s_vec_for_fit=None, make_plots=False):
    
    # NOTE: map MUST be in the Sun-Fixed Frame!
    n_lat, n_lon = np.shape(avg_map)
    map_vec = np.reshape(avg_map, n_lat*n_lon)

    lon_vec, lat_vec, lon_edges_vec, lat_edges_vec = get_sphere_grid(n_lon, n_lat)
    if make_plots: Plotter.plot_geo_data(avg_map, lon_edges_vec, lat_edges_vec)

    lon_vec, lat_vec, lon_edges_vec, lat_edges_vec = get_sphere_grid(n_lon, n_lat)
    LON_vec, LAT_vec, _ = get_reshaped_grid(lon_vec, lat_vec)
    map_cos_theta_s_vec = -np.cos(np.radians(LAT_vec)) * np.cos(np.radians(LON_vec))  

    zero_idxs, nonzero_idxs = (map_vec==0), (map_vec!=0)
    if meas_vec_for_fit is None and cos_theta_s_vec_for_fit is None:
        meas_vec_for_fit = map_vec[nonzero_idxs] 
        cos_theta_s_vec_for_fit = map_cos_theta_s_vec[nonzero_idxs]

    c, m, n = compute_meas_vs_cos_theta_s_fit(meas_vec_for_fit, cos_theta_s_vec_for_fit, 
                                                cos_theta_s_lim, plot_bool=make_plots)
    map_vec[zero_idxs] = meas_vs_cos_theta_s_function(map_cos_theta_s_vec[zero_idxs], cos_theta_s_lim, c, m, n)
    filled_map = np.reshape(map_vec, (n_lat, n_lon))

    if make_plots: Plotter.plot_geo_data(filled_map, lon_edges_vec, lat_edges_vec)

    return filled_map


def fill_unobserved_cells_map_array_with_theta_s_fit_simplified(avg_map_array, cos_theta_s_lim, make_plot_bool_array=None):
    
    if make_plot_bool_array is None: make_plot_bool_array = [False] * len(avg_map_array)    
    out_map_array = [None] * len(avg_map_array)
    
    for i, avg_map in enumerate(avg_map_array):
        out_map_array[i] = fill_unobserved_cells_with_theta_s_interp_new(avg_map, cos_theta_s_lim, make_plots=make_plot_bool_array[i])

    return out_map_array


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


# deprecated function
def fill_unobserved_cells_with_theta_s_interp(map, obs_array, cos_theta_s_array, altitude_km): 
    # NOTE: map MUST be in the Sun-Fixed Frame!

    n_lat, n_lon = np.shape(map)
    lon_vec, lat_vec, lon_edges_vec, lat_edges_vec = get_sphere_grid(n_lon, n_lat)
    LON_vec, LAT_vec, shape = get_reshaped_grid(lon_vec, lat_vec)

    cos_theta_s_lim = get_cos_theta_s_lim(altitude_km)
    cos_theta_s_vec = -np.cos(np.radians(LAT_vec)) * np.cos(np.radians(LON_vec))  # proof somewhere in my notebook
    # the minus sign is to match the convention that u_sun points TOWARDS the Sun

    map_vec = np.reshape(map, np.shape(cos_theta_s_vec))
    idxs_to_keep = (map_vec!=0) & (cos_theta_s_vec>cos_theta_s_lim)

    # check if reshape gives the right stuff
    Plotter.plot_geo_data(cos_theta_s_vec.reshape((n_lat, n_lon)), lon_edges_vec, lat_edges_vec)
    Plotter.plot_geo_data(map, lon_edges_vec, lat_edges_vec)

    Plotter.plot_linear_regression(cos_theta_s_vec[idxs_to_keep], map_vec[idxs_to_keep], xlabel='cos theta_s', ylabel='meas. flux')
    
    idxs_to_keep = (cos_theta_s_array>cos_theta_s_lim)
    Plotter.plot_linear_regression(cos_theta_s_array[idxs_to_keep], obs_array[idxs_to_keep], xlabel='cos theta_s', ylabel='meas. flux')
    
    # WE'RE HERE - SPLIT BETWEEN FULL MEAS AND SIMPLIFIED FUNCTIONS, then run the rolling avg estimates

    print("checkpoint")




    



def get_jd_window_idxs(jd_vec, jd_windows):
    
    first_idxs = np.searchsorted(jd_vec, jd_windows[:,0], side="left")
    last_idxs =  np.searchsorted(jd_vec, jd_windows[:,1], side="right")

    return np.column_stack((first_idxs, last_idxs))


def compute_avg_EEI_from_avg_maps(radial_flux_avg_maps, altitude_km):

    n_lat, n_lon = np.unique([np.shape(map) for map in radial_flux_avg_maps])

    _, _, lon_edges_vec, lat_edges_vec = get_sphere_grid(n_lon, n_lat)
    grid_cell_areas_CV = get_spherical_grid_cell_areas(lat_edges_vec, lon_edges_vec,
                                                        R=(constants.earth_radius(units='km')+altitude_km)*1e3)
    Earth_S = 4*np.pi*(constants.earth_radius(units='m'))**2

    avg_EEI = np.sum(np.array(radial_flux_avg_maps) * grid_cell_areas_CV[None,:,:], axis=(1,2)) / Earth_S

    return avg_EEI
    

def compute_avg_EEI_from_avg_map(radial_flux_avg_map, altitude_km):

    n_lat, n_lon = np.shape(radial_flux_avg_map)
    _, _, lon_edges_vec, lat_edges_vec = get_sphere_grid(n_lon, n_lat)
    grid_cell_areas_CV = get_spherical_grid_cell_areas(lat_edges_vec, lon_edges_vec,
                                                        R=(constants.earth_radius(units='km')+altitude_km)*1e3)
    Earth_S = 4*np.pi*(constants.earth_radius(units='m'))**2

    avg_EEI = np.sum(radial_flux_avg_map * grid_cell_areas_CV) / Earth_S

    return avg_EEI
    


def get_simulated_accelerometer_measurements(input_names):

    aero_acc_hist_array = get_full_output_var_hist(input_names, 'aero')
    srp_acc_hist_array = get_full_output_var_hist(input_names, 'srp')
    erp_acc_hist_array = get_full_output_var_hist(input_names, 'erp')

    acc_meas_hist_array = [None] * len(input_names)
    for sc_i in range(len(input_names)):
        acc_meas_hist_array[sc_i] = (aero_acc_hist_array[sc_i] + srp_acc_hist_array[sc_i] + erp_acc_hist_array[sc_i]) * 1e3 # accelerations are stored in km/s2!

    return acc_meas_hist_array


def get_acc_to_flux_factor(input_name):
    config_dicts = input_manager.get_dicts(input_name)
    #input_file = load_input_file(input_name)
    #return - input_file.SC_INPUT['mass']/input_file.SC_INPUT['area'] * constants.light_speed(units='m/s')

    return -config_dicts['sc_params']['mass']/config_dicts['sc_params']['area'] * constants.light_speed(units='m/s')

        
def getnnz(sparse_array, axis):

    # it seems that creating a copy of the sparse array is not needed
    sparse_array.data[:] = 1
    nnz_array = sparse_array.sum(axis=axis)
    
    return nnz_array

def get_latlon_cell_idxs(lat_hist_vec, lon_hist_vec, n_lon, n_lat):

    _, _, lon_edges_vec, lat_edges_vec = get_sphere_grid(n_lon, n_lat)

    lat_cell_numbers = np.searchsorted(-lat_edges_vec, -lat_hist_vec) - 1
    lon_cell_numbers = np.searchsorted(lon_edges_vec, lon_hist_vec) - 1

    return lat_cell_numbers, lon_cell_numbers


def get_stacked_measurements_matrix_regular_grid(n_lon, n_lat, lon_hist_vec, lat_hist_vec, meas_hist_vec, out_mode='3D'):

    n_steps = len(lat_hist_vec)
    assert(n_steps == len(lon_hist_vec) == len(meas_hist_vec))
    # _, _, lon_edges_vec, lat_edges_vec = get_sphere_grid(n_lon, n_lat)
    # 
    # lat_cell_numbers = np.searchsorted(-lat_edges_vec, -lat_hist_vec) - 1
    # lon_cell_numbers = np.searchsorted(lon_edges_vec, lon_hist_vec) - 1
    lat_cell_numbers, lon_cell_numbers = get_latlon_cell_idxs(lat_hist_vec, lon_hist_vec, n_lon, n_lat)

    if out_mode=="3D":
        full_3D_array = scipy.sparse.coo_array( # omg that's hella fast
            (meas_hist_vec, (lat_cell_numbers, lon_cell_numbers, np.arange(n_steps))),
            shape=(n_lat, n_lon, n_steps)
        )
        return full_3D_array
    
    elif out_mode=="2D":
        global_idxs = get_2D_to_1D_idx(lat_cell_numbers, lon_cell_numbers, n_lon)
        full_2D_array = scipy.sparse.coo_array( 
            (meas_hist_vec, (global_idxs, np.arange(n_steps))),
            shape=(n_lat*n_lon, n_steps)
        )
        return full_2D_array
    
    else:
        print("out_mode has to be either 2D or 3D")


#def get_stacked_measurements_2D_matrix_regular_grid(n_lon, n_lat, lon_hist_vec, lat_hist_vec, meas_hist_vec):


def get_true_EEI_time_series(EEI_name, n_lon=None, n_lat=None):
    
    EEI_settings = radiation_settings_from_EEI_truth_name(EEI_name)
    n_days = get_n_days_EEI_truth(EEI_name)
    subdir = get_subdir_EEI_truth(EEI_name, 'daily_hist_net_toa_surf_avg', n_lon, n_lat)

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



def generate_constellation_stacking_animations(case, sc_names, constellation_tag, n_days, day_0_idx=0, n_lon=360, n_lat=180, frame='SFF'):
    
    out_dir = os.path.join(ANIMATIONS_DIR, constellation_tag)
    os.makedirs(out_dir, exist_ok=True)
    

    input_names = sc_names #[case + '_' + sc_name for sc_name in sc_names]
    EEI_name = check_EEI_consistency(input_names)
    EEI_full_window = radiation_settings_from_EEI_truth_name(EEI_name)['jd_interval']
    jd_0 = EEI_full_window[0] + day_0_idx

    jd_window_edges = np.arange(jd_0, jd_0+n_days, 1)
    jd_windows = np.array([[jd_0, jd+1] for jd in jd_window_edges])
    mid_day_jd = np.round(jd_window_edges)
    assert(len(jd_windows)==len(mid_day_jd))

    jd_hist_array = get_full_output_var_hist(input_names, 'jd_vec')
    T_array = get_orbital_periods(input_names)

    jd_hist_array = get_full_output_var_hist(input_names, 'jd_vec')
    if frame=="ECEF":
        h_lat_lon_hist_array = get_full_output_var_hist(input_names, 'h_lat_lon')
    elif frame=="SFF":
        h_lat_lon_hist_array = get_full_output_var_hist(input_names, 'h_lat_lon_SFF')
        R_hist = np.zeros((len(mid_day_jd),3,3))
        for day_idx in range(len(mid_day_jd)):
            R_day_hist = load_day_R_mat_hist(day_idx, EEI_name)
            R_hist[day_idx, :,:] = R_day_hist[720,:,:]


    radial_flux_meas_3D_arrays = get_radial_measurement_arrays(input_names, n_lon, n_lat, frame, meas='flux') # this step takes significantly longer
    avg_flux_maps = get_stacked_avg_maps_3D_mode(jd_windows, jd_hist_array, radial_flux_meas_3D_arrays) # this comes wiht unobserved cells
    reshaped_flux_map_array = np.zeros((n_lat, n_lon, len(avg_flux_maps)))
    for i, map_i in enumerate(avg_flux_maps):
        reshaped_flux_map_array[:,:,i] = map_i


    Plotter.make_map_animation_arbitrary_frame(
        map_grid_3D_rotated=reshaped_flux_map_array,
        jd_vec=[window[1] for window in jd_windows],
        n_lon=n_lon, n_lat=n_lat,
        data_label='Stacked meas. radial flux (W/m$^2$)',
        out_dir=out_dir,
        filename='Stacking_'+str(n_days)+'_days.mp4',
        sat_h_lat_lon_hist_array=h_lat_lon_hist_array,
        sat_hist_jd_array=jd_hist_array,
        orbital_periods_minutes=T_array,
        add_satellite_positions=False,
        add_coastlines=False,
        R_mat_hist=R_hist,
        groundtrack_linstyle='-'
    )



def generate_constellation_day_animations(case, sc_names, constellation_tag, day_idxs, n_lon=360, n_lat=180, max_hours=24):

    out_dir = os.path.join(ANIMATIONS_DIR, constellation_tag)
    os.makedirs(out_dir, exist_ok=True)

    input_names = sc_names #[case + '_' + sc_name for sc_name in sc_names]
    EEI_name = check_EEI_consistency(input_names)

    jd_hist_array = get_full_output_var_hist(input_names, 'jd_vec')
    h_lat_lon_hist_array = get_full_output_var_hist(input_names, 'h_lat_lon')
    h_lat_lon_SFF_hist_array = get_full_output_var_hist(input_names, 'h_lat_lon_SFF')
    
    T_array = get_orbital_periods(input_names)

    # plots to make:
    altitude_array = [0, 800]  # this 800 should be read from the sc input file (how about constellations wiht sats at different altitudes?)
    SFF_array = [True, False]
    avg_array = [False, True]


    for day_idx in day_idxs:
        for altitude_km in altitude_array:
            for SFF_bool in SFF_array:
                for avg_bool in avg_array:

                    if avg_bool and not(SFF_bool): continue # skip this map due to its small meaningfulness

                    altitude_tag = 'TOA' if altitude_km==0 else str(altitude_km)+'km'
                    SFF_tag = '_SFF' if SFF_bool else ''
                    avg_tag = '_wrt_avg' if avg_bool else ''
                    delta_str = 'Delta ' if avg_bool else ''
                    fname = 'day_' + str(day_idx) + '_' + altitude_tag + SFF_tag + avg_tag + '.mp4'

                    if not(os.path.exists(os.path.join(out_dir, fname))):
                        print(f"Generating {fname}")
                        map_day_hist, jd_map_day_hist = load_day_hist_map(
                                                                day_idx, EEI_name, n_lon, n_lat, altitude_km, 
                                                                SFF=SFF_bool, delta_avg=avg_bool)
                        max_steps = get_max_steps(jd_map_day_hist, max_hours)
                        fade_coastlines_bool = False if altitude_km==0 else True
                                    
                        if SFF_bool:
                            Plotter.make_map_animation_arbitrary_frame(
                                map_grid_3D_rotated=map_day_hist[:,:,:max_steps],
                                R_mat_hist=load_day_R_mat_hist(day_idx, EEI_name)[:max_steps,:,:],
                                jd_vec=jd_map_day_hist,
                                n_lon=n_lon, n_lat=n_lat,
                                data_label=delta_str + 'Net Radial ' + altitude_tag + 'Flux (W/m$^2$)',
                                out_dir=os.path.join(ANIMATIONS_DIR, constellation_tag),
                                filename=fname,
                                sat_h_lat_lon_hist_array=h_lat_lon_SFF_hist_array,
                                sat_hist_jd_array=jd_hist_array,
                                orbital_periods_minutes=T_array, 
                                fade_coastlines=fade_coastlines_bool
                            )

                        else:
                            Plotter.make_map_animation(map_grid_3D=map_day_hist[:,:,:max_steps],
                                            jd_vec=jd_map_day_hist[:max_steps],
                                            n_lon=n_lon, n_lat=n_lat,
                                            data_label=delta_str + 'Net Radial ' + altitude_tag + 'Flux (W/m$^2$)',
                                            out_dir=os.path.join(ANIMATIONS_DIR, constellation_tag),
                                            filename=fname,
                                            sat_h_lat_lon_hist_array=h_lat_lon_hist_array,
                                            sat_hist_jd_array=jd_hist_array,
                                            orbital_periods_minutes=T_array,
                                            fade_coastlines=fade_coastlines_bool)  

    print("All animations done!")


def get_max_steps(jd_vec, max_hours):
    assert(np.abs(jd_vec[-1] - jd_vec[0] - 1) < 1e-2 )

    return int(len(jd_vec) * max_hours/24)


def load_day_hist_map(day_idx, EEI_truth_name, n_lon, n_lat, altitude_km, SFF=False, delta_avg=False):

    # remember that day_idx will be the same for both the satellite propagation and the net daily hist maps because they are both subject to the same EEI_truth, which defines the timespan
    path = os.path.join(MEDIA_DIR, 'true_EEI', EEI_truth_name, 'grid_'+str(n_lat)+'x'+str(n_lon))
    altitude_tag = 'toa' if altitude_km==0 else str(altitude_km)+'km'
    SFF_tag = '_SFF' if SFF else ''
    fname = os.path.join(path, 'daily_hist_net_' + altitude_tag + SFF_tag, 'day_'+str(day_idx))

    print(f"Loading daily_hist at {altitude_tag} for day {day_idx}")
    try:
        net_map_day_hist = np.load(fname+'.npy')
    except:
        net_map_day_hist = np.loadtxt(fname+'.txt')
        n_steps = int(np.prod(np.shape(net_map_day_hist)) / (n_lon * n_lat))
        net_map_day_hist = net_map_day_hist.reshape((n_lat, n_lon, n_steps))
    
    if delta_avg:
        net_map_day_hist = net_map_day_hist - np.mean(net_map_day_hist, axis=2)[:,:,None]

    jd_day_hist = load_jd_day_hist_map(day_idx, EEI_truth_name)

    return net_map_day_hist, jd_day_hist

def load_day_R_mat_hist(day_idx, EEI_truth_name):

    boa_fname = radiation_settings_from_EEI_truth_name(EEI_truth_name)['ephemerides']
    path = os.path.join(MEDIA_DIR, 'solar_ephemerides', boa_fname, 'daily_files')
    R_fname = os.path.join(path, 'R_ECEF_to_SunFrame_day_'+str(day_idx))
    day_R_array = np.load(R_fname+'.npy')

    return day_R_array



def load_jd_day_hist_map(day_idx, EEI_truth_name):

    boa_fname = radiation_settings_from_EEI_truth_name(EEI_truth_name)['ephemerides']
    path = os.path.join(MEDIA_DIR, 'solar_ephemerides', boa_fname, 'daily_files')
    jd_fname = os.path.join(path, 'jd_array_day_'+str(day_idx))
    day_jd_array = np.load(jd_fname+'.npy')

    return day_jd_array




def get_n_days_max(input_names):  # in case some of the sc have not finished running
    
    EEI_name = check_EEI_consistency(input_names)
    total_n_days = get_n_days_EEI_truth(EEI_name)

    #n_days_array = [len(os.listdir(os.path.join(OUTPUT_DIR, case, 'data_output'))) for case in input_names] # erroneously included files
    #n_days_array = [len(next(os.walk(os.path.join(OUTPUT_DIR, case, 'data_output')))[1]) for case in input_names] #only includes folders
    n_days_array = [len(next(os.walk(os.path.join(OUTPUT_DIR, case)))[1]) for case in input_names] #only includes folders
    n_days = min(n_days_array)

    if n_days<total_n_days:
        n_days = n_days - 1

    return n_days, n_days==total_n_days


def get_n_days_EEI_truth(EEI_name):
    EEI_settings = radiation_settings_from_EEI_truth_name(EEI_name)
    total_n_days = int(np.diff(EEI_settings['jd_interval']).item())

    return total_n_days


def get_full_output_var_hist(case_names, var_name, n_days=None):

    # TODO: n_days should not really be an input
    if n_days is None:
        n_days, completed_bool = get_n_days_max(case_names)

    n_SC = len(case_names)
    
    all_var_hist = [None] * n_SC
    for i, case in enumerate(case_names):
        var_hist = get_output_variable_hist(case, n_days, var_name, 
                                            completed_bool) # only store stacked file if all days are done
        all_var_hist[i] = var_hist
    
    return all_var_hist




def get_output_variable_hist(case_name, n_days, var_name, save_stacked_file_if_nonexistent=True):   # for jd: var_name=jd_vec

    #stacked_var_file_name = os.path.join(OUTPUT_DIR, case_name, 'data_output', 'concatenated_'+var_name+'.npy')
    stacked_var_file_name = os.path.join(OUTPUT_DIR, case_name, 'concatenated_'+var_name+'.npy')
    if os.path.exists(stacked_var_file_name):
        stacked_var_array = np.load(stacked_var_file_name)
    else:
        
        if var_name in OUTPUT_VARS:
            stacked_var_array = load_and_stack_output_var_hist(case_name, var_name, n_days)
        else:
            stacked_var_array = compute_stacked_out_var_hist(case_name, var_name, n_days)

        if save_stacked_file_if_nonexistent:
            np.save(stacked_var_file_name, stacked_var_array)
    
    return stacked_var_array


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

def get_subdir_EEI_truth(EEI_truth, series_type, grid_name):
    
    subdir = os.path.join(MEDIA_DIR, 'true_EEI', EEI_truth, grid_name, 
                              series_type)
    return subdir


def get_all_orbital_sma_0(input_names):
    all_sma = np.zeros(len(input_names))

    for i, input_name in enumerate(input_names):    
        input_file = load_input_file(input_name)
        all_sma[i] = input_file.SC_KEP_0['a']
    
    return all_sma
    


def get_orbital_periods(input_names):

    MU = 398600 # TODO: make consistent with propagation gravity field
    all_sma_km = get_all_orbital_sma_0(input_names)
    all_periods = np.zeros_like(all_sma_km)

    for i, sma in enumerate(all_sma_km):
        all_periods[i] = compute_orbital_period(MU, sma)

    return all_periods


def check_EEI_consistency(input_names):

    all_dynamics = [None] * len(input_names)

    for i, input_name in enumerate(input_names):
        config_dicts = input_manager.get_dicts(input_name)
        all_dynamics[i] = config_dicts['force_settings']

    assert(all([settings==all_dynamics[0] for settings in all_dynamics]))
    
    EEI_name = 'EEI_truth_' + str(all_dynamics[0]['EEI_truth'])
    return EEI_name
        

def check_EEI_consistency_old(input_names):

    all_force_settings = [None] * len(input_names)

    for i, input_name in enumerate(input_names):    
        input_file = load_input_file(input_name)
        all_force_settings[i] = input_file.FORCE_SETTINGS # .get('EEI_case')

    assert(all([settings==all_force_settings[0] for settings in all_force_settings]))

    EEI_name = all_force_settings[0].get('EEI_name')
    if EEI_name is None:
        EEI_name = "EEI_truth_1"    # all simulations run before this was a field were consistent with EEI 1
    
    return EEI_name    


def compute_rolling_avg_error_time_series(sc_array, window_days, EEI_truth_time_series, jd_array_EEI_truth, evaluation_jd_array):

    input_names = ['case_5_years_' + sc_name for sc_name in sc_array]
    all_altitudes = get_all_orbital_sma_0(input_names) - constants.earth_radius(units='km')
    assert(len(np.unique(all_altitudes))==1)     # for now we don't really know how to handle different altitudes simultaneously

    print("")
    print(f"Running smooth estimate {window_days}-day window (constellation_"+make_list_str_key(sc_array)+")")
    
    EEI_smooth_true = normal_smoother(EEI_truth_time_series, jd_array_EEI_truth, window_width_days=window_days, 
                                    out_length_mode='same_with_nans')
    EEI_smooth_true_coarse = np.interp(evaluation_jd_array, jd_array_EEI_truth, EEI_smooth_true)

    jd_windows = get_rolling_jd_windows(evaluation_jd_array, window_days)
    
    EEI_smooth_estimated = estimate_EEI_avg('case_5_years', sc_names=sc_array, 
                                    altitude=np.unique(all_altitudes), jd_windows=jd_windows, 
                                    n_lon=360, n_lat=180, frame="SFF", fill_method="theta_s_fit_simplified",
                                    make_plots=False)
    assert(np.shape(EEI_smooth_estimated)==np.shape(EEI_smooth_true_coarse))
    EEI_smooth_estimated[np.isnan(EEI_smooth_true_coarse)] = np.nan
        
    error_time_series = EEI_smooth_estimated-EEI_smooth_true_coarse

    return error_time_series



def compute_sample_quality_metrics(constellation_list):

    input_names = ['case_5_years_' + sc_name for sc_name in constellation_list]

    grid_deg_spacing = 1  # TODO: make flexible
    n_lon = int(360/grid_deg_spacing)
    n_lat = int(180/grid_deg_spacing)

    lon_vec, lat_vec, lon_edges_vec, lat_edges_vec = get_sphere_grid(n_lon, n_lat)
    alt = 800 # TODO: read from sc input files
    r_grid = lonlat_to_r(lon_vec, lat_vec, alt, "spherical")


    h_lat_lon_hist_array = get_full_output_var_hist(input_names, 'h_lat_lon')
    lat_hist = np.concatenate([hist[:,1] for hist in h_lat_lon_hist_array])
    lon_hist = np.concatenate([hist[:,2] for hist in h_lat_lon_hist_array])
    n_steps = len(lat_hist)

    lat_idxs, lon_idxs = get_latlon_cell_idxs(lat_hist, lon_hist, n_lon, n_lat)

    full_r_ecef_hist_array = np.concatenate(get_full_output_var_hist(input_names, 'xyz_ecef'))

    r_offsets = full_r_ecef_hist_array - r_grid[lat_idxs, lon_idxs, :]

    x_offsets_avg_map = scipy.sparse.coo_array(
            (r_offsets[:,0], (lat_idxs, lon_idxs, np.arange(n_steps))),
            shape=(n_lat, n_lon, n_steps)
        ).mean(axis=2)
    
    y_offsets_avg_map = scipy.sparse.coo_array(
            (r_offsets[:,1], (lat_idxs, lon_idxs, np.arange(n_steps))),
            shape=(n_lat, n_lon, n_steps)
        ).mean(axis=2)
    
    z_offsets_avg_map = scipy.sparse.coo_array(
            (r_offsets[:,2], (lat_idxs, lon_idxs, np.arange(n_steps))),
            shape=(n_lat, n_lon, n_steps)
        ).mean(axis=2)
    
    r_offsets_avg_map = np.stack((x_offsets_avg_map, y_offsets_avg_map, z_offsets_avg_map), axis=2)
    offset_norms_avg_map = np.linalg.norm(r_offsets_avg_map, axis=2)

    print("done")



    h_lat_lon_SFF_hist_array = get_full_output_var_hist(input_names, 'h_lat_lon_SFF')
    r_sff_hist_array = get_full_output_var_hist(input_names, 'xyz_sun_frame')
    



if __name__ == "__main__":

    #read_data('case_5_years', sc_names=['sc_C1', 'sc_C2', 'sc_C3'])
    t1 = time.time()
    EEI_avg = estimate_EEI_avg('case_5_years', sc_names=['sc_C1', 'sc_C2', 'sc_C3'], 
                               altitude=800, jd_windows=[[2458119.5, 2458119.5+365]], 
                     n_lon=360, n_lat=180, frame="SFF")
    t2 = time.time()
    print(f"Full time to compute EEI: {t2-t1}")
    print(EEI_avg)

    #generate_constellation_day_animations('case_5_years', sc_names=['sc_A1', 'sc_A2', 'sc_A3'],
    #                                      constellation_tag='constellation_A',
    #                                      day_idxs=[0, 182], max_hours=12)






# # SFF sanity check:
# 
# day_0_jd = np.round(EEI_settings['jd_interval'][0])
# R_ECEF_2_SFF_hist_day_0 = get_R_SunFrame_hist(EEI_settings['ephemerides'], day_0_jd)
# 
# for i in range(3):
#     r_sff_0_monte = r_sff_hist_array[i][0]
#     r_sff_0_converted = R_ECEF_2_SFF_hist_day_0[0,:,:] @ r_ecef_hist_array[i][0]
#     diff = r_sff_0_monte - r_sff_0_converted
#     print(diff)
    





# def get_jd_hist(case_name, n_days):
#     
#     # day_subdirs = get_day_subdirs(case_name)
# 
#     try:
#         concatenated_jd_file_name = os.path.join(OUTPUT_DIR, case_name, 'concatenated_jd_vec.npy')
#         concatenated_jd_vec = np.load(concatenated_jd_file_name)
#     except:
#         daily_jd_vec_array = [None] * n_days
#         
#         print("Loading daily jd files...")
#         for i in range(n_days):
#             
#             progress_bar(i, n_days)
#             day_subdir = get_day_subdir(case_name, i)
#             try:
#                 day_jd_vec = np.load(os.path.join(day_subdir, 'jd_vec.npy'))
#             except:
#                 day_jd_vec = np.loadtxt(os.path.join(day_subdir, 'jd_vec.csv'))
# 
#             daily_jd_vec_array[i] = day_jd_vec
#         
#         concatenated_jd_vec = np.concatenate(daily_jd_vec_array)
#         np.save(concatenated_jd_file_name, concatenated_jd_vec)
#     
#     return concatenated_jd_vec




# def get_full_jd_hist(case_names, n_days=None):
# 
#     if n_days is None:
#         n_days = get_n_days_max(case_names)
# 
#     n_SC = len(case_names)
#     
#     all_jd_hist = [None] * n_SC
#     for i, case in enumerate(case_names):
#         jd_hist = get_variable_hist(case, n_days, 'jd_vec')
#         all_jd_hist[i] = jd_hist
#     
#     return all_jd_hist
# 
# def get_full_r_ecef_hist(case_names, n_days=None):
#     if n_days is None:
#         n_days = get_n_days_max(case_names)
# 
#     n_SC = len(case_names)
#     
#     all_r_hist = [None] * n_SC
#     for i, case in enumerate(case_names):
#         r_hist = get_variable_hist(case, n_days, 'xyz_ecef')
#         all_r_hist[i] = r_hist
    
#    return all_r_hist




# def get_stacked_avg_map_old(jd_window, jd_hist_array, all_3D_measurement_arrays):
# 
#     n_sc = len(all_3D_measurement_arrays)
#     
#     # get EEI_avg on a given time window
#     stacked_measurements_array = [None] * n_sc
#     meas_counts_array = [None] * n_sc
# 
#     for sc_i in range(n_sc):
# 
#         jd_vec = jd_hist_array[sc_i]
#         first_idx = np.searchsorted(jd_vec, jd_window[0], side="left")
#         last_idx  = np.searchsorted(jd_vec, jd_window[1], side="right") # robustness credit to ChatGPT
# 
#         meas_counts_array[sc_i] = getnnz(all_3D_measurement_arrays[sc_i][:,:,first_idx:last_idx], axis=2)
#         stacked_measurements_array[sc_i] = all_3D_measurement_arrays[sc_i][:,:,first_idx:last_idx].sum(axis=2)
#         
#     summed_meas_array = np.sum(stacked_measurements_array, axis=0)
#     summed_counts_array = np.sum(meas_counts_array, axis=0)
# 
#     no_meas_cell_idxs = summed_counts_array==0
#     summed_counts_array[no_meas_cell_idxs] = 1
#     summed_meas_array[no_meas_cell_idxs] = np.nan
# 
#     meas_avg_map = fill_nans_in_data_array(summed_meas_array, method='zeroes') / summed_counts_array
# 
#     return meas_avg_map