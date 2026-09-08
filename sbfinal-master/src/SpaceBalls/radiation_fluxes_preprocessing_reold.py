import os, sys
import numpy as np
import jax.numpy as jnp
from memory_profiler import profile, memory_usage

import time
from astropy.time import Time, TimeDelta
import astropy.units as u
import multiprocessing
from multiprocessing import Pool
from scipy.integrate import lebedev_rule
from astropy.coordinates import cartesian_to_spherical

from SpaceBalls.paths import CONFIG_DIR, MEDIA_DIR
sys.path.insert(0, str(CONFIG_DIR.parent))  # parent of 'config'
import SpaceBalls.radiation_settings as rad_settings
from SpaceBalls.sph_meshing import expand_sh, get_reshaped_grid, get_sphere_grid, get_spherical_grid_cell_areas, lonlat_to_r, get_stacked_spherical_grid_els, progress_bar, field_hist_rotation, field_hist_rotation_multiproc
import config.constants as constants
from SpaceBalls.utils import get_all_r_rel, get_all_cos_alpha, get_r_rel_norm
from SpaceBalls.plotter import Plotter

AU = constants.astronomical_unit(units='km')
RE = constants.earth_radius(units='km')

def required_files(alt_km, grid_type):
    if alt_km==0:           
        if grid_type=="regular":                                                        # SFF: sun-fixed frame
            return ['daily_hist_emission_toa', 'daily_hist_net_toa', 'daily_hist_net_toa_SFF',
                    'daily_hist_net_toa_lat_avg', 'daily_hist_net_toa_surf_avg',
                    'daily_avg_net_toa']
        elif grid_type=="lebedev":
            return ['daily_hist_emission_toa', 'daily_hist_net_toa', 'daily_hist_net_toa_surf_avg']
    
    elif alt_km>0: #TODO: lebedev grids for altitude.
        return ['daily_hist_net_Fr_'+str(alt_km)+'km', 'daily_hist_net_Fr_'+str(alt_km)+'km_SFF',
                'daily_avg_net_Fr_'+str(alt_km)+'km', 'daily_avg_net_Fr_'+str(alt_km)+'km_SFF']
                #'daily_avg_net_'+str(alt_km)+'km_simpl1', 'daily_avg_net_'+str(alt_km)+'km_simpl1_SFF']


def get_EEI_truth_daily_jd_arrays(EEI_truth_name):

    rad_config = rad_settings.radiation_settings_from_EEI_truth_name(EEI_truth_name)
    mid_day_jd_array = mid_day_jd_array_from_jd_interval(rad_config["jd_interval"])

    step_minutes = 1 # DO NOT CHANGE - must be equal to the one used for the files in solar_ephemerides
    step_days = step_minutes / (60*24)
    daily_jd_arrays = [np.arange(mid_day_jd-0.5, mid_day_jd+0.5, step_days) for mid_day_jd in mid_day_jd_array]
    
    return daily_jd_arrays


def compute_radiation_maps(EEI_truth_name, altitude_array = [0, 800, 1500], degrees_bins=None, 
                           selected_days_idxs = [0, 182, 1620, 1800], n_cores=12):
    
    daily_jd_arrays = get_EEI_truth_daily_jd_arrays(EEI_truth_name)
    
    rad_config = rad_settings.radiation_settings_from_EEI_truth_name(EEI_truth_name)
    mid_day_jd_array = mid_day_jd_array_from_jd_interval(rad_config["jd_interval"])
    n_days = len(mid_day_jd_array)
    
    daily_jd_arrays_reduced = [daily_jd_arrays[i] for i in selected_days_idxs]
    #for degrees_CV_bins in degrees_CV_bins_array:

    if degrees_bins is not None:
        grid_type = "regular"
        n_lon = int(360 // degrees_bins)
        n_lat = int(180 // degrees_bins)
        lon_vec, lat_vec, lon_edges_vec, lat_edges_vec = get_sphere_grid(n_lon, n_lat)
        grid_cell_areas_toa = get_spherical_grid_cell_areas(lat_edges_vec, lon_edges_vec, RE)
        out_dir = os.path.join(MEDIA_DIR, 'true_EEI', EEI_truth_name, 'grid_'+str(n_lat)+'x'+str(n_lon))
    else:   # TOA grid is Lebedev
        grid_type = "lebedev"
        n_lon, n_lat, lat_vec, lon_vec, grid_cell_areas_toa = None, None, None, None, None
        out_dir = os.path.join(MEDIA_DIR, 'true_EEI', EEI_truth_name, 'grid_lebedev')


    for altitude_km in altitude_array:
        daily_jd_arrays_to_loop = daily_jd_arrays_reduced if altitude_km>0 else daily_jd_arrays
        idxs_days_to_loop = selected_days_idxs if altitude_km>0 else range(n_days)
        # daily_jd_arrays_to_loop = daily_jd_arrays
        # idxs_days_to_loop = np.arange(n_days)
        
        for day_idx, jd_array in zip(idxs_days_to_loop, daily_jd_arrays_to_loop):
            
            print(f"Now computing: daily hist for day {day_idx} at {altitude_km}km")
            n_steps = len(jd_array)
            mid_day_jd = mid_day_jd_array[day_idx]
            TSI_1AU_day = rad_settings.get_TSI_1AU(mid_day_jd, rad_config["TSI_source"])
            R_ECEF_to_SunFrame_day_hist = get_R_SunFrame_hist(rad_config["ephemerides"], mid_day_jd)

            _, file_existences = get_file_names_and_existence(out_dir, altitude_km, day_idx, grid_type)
            
            if altitude_km==0 and not(all(file_existences.values())):

                toa_emission_day_hist, net_toa_day_hist = get_daily_hist_toa(out_dir, day_idx, mid_day_jd, TSI_1AU_day, rad_config,
                                                                                lon_vec, lat_vec, n_steps, grid_type)

                save_toa_files(toa_emission_day_hist, net_toa_day_hist, grid_cell_areas_toa, out_dir, 
                                day_idx, R_ECEF_to_SunFrame_day_hist, grid_type, save_SFF_hist=(day_idx in selected_days_idxs))

            elif altitude_km>0 and not(all(file_existences.values())):
                
                if day_idx in selected_days_idxs:
                    daily_hist_net_F_altitude, daily_hist_net_Fr_altitude = get_daily_hist_net_at_altitude(
                            out_dir, altitude_km, day_idx, mid_day_jd, TSI_1AU_day, rad_config, n_lon, n_lat, n_steps, n_cores, grid_type
                        )
                else:
                    daily_hist_net_altitude = None

                """
                if not(file_existences['daily_avg_net_'+str(altitude_km)+'km_simpl1']) or not(file_existences['daily_avg_net_'+str(altitude_km)+'km_simpl1_SFF']):
                    daily_avg_at_altitude_simpl1, daily_avg_at_altitude_simpl1_SFF = compute_daily_avg_net_at_altitude_simpl_1(
                            out_dir, altitude_km, day_idx, mid_day_jd, TSI_1AU_day, rad_config, n_lon, n_lat, n_steps, n_cores
                        )
                """
                # temporary patch:
                grid_r_el = lonlat_to_r(lon_vec, lat_vec, 0, "spherical")
                stacked_grid_r_el = np.reshape(grid_r_el, (n_lon*n_lat, 3))
                stacked_grid_u_el = stacked_grid_r_el / (constants.earth_radius(units='km') * np.ones((n_lon*n_lat,1)))
    
                save_altitude_files(daily_hist_net_F_altitude, daily_hist_net_Fr_altitude, #daily_avg_at_altitude_simpl1, daily_avg_at_altitude_simpl1_SFF, 
                                    altitude_km, day_idx, out_dir, R_ECEF_to_SunFrame_day_hist, stacked_grid_u_el)


            print("")

                    

def get_daily_hist_toa(data_dir, day_idx, mid_day_jd, TSI_1AU_day, rad_config, lon_vec, lat_vec, n_steps, grid_type):
    # PATCH: correct data_dir if grid_type is lebedev (data dir now includes the grid name of the target grid, but TOA needs to be loaded as lebedev when convolving to altitude regardless of the altitude grid)
    if grid_type == "lebedev" and lon_vec is not None and lat_vec is not None: 
        wrong_grid_name = str(len(lat_vec))+"x"+str(len(lon_vec))
        data_dir = data_dir.replace(wrong_grid_name, "lebedev")

    all_file_names, file_existences = get_file_names_and_existence(data_dir, 0, day_idx, grid_type)

    if not(file_existences["daily_hist_emission_toa"]) or not(file_existences["daily_hist_net_toa"]):  # requires recomputing
        
        print("Computing daily hist TOA...")
        if grid_type=="regular":
            n_lon, n_lat = len(lon_vec), len(lat_vec)
            toa_emission_day_hist, net_toa_day_hist = compute_daily_time_series_maps_toa(mid_day_jd, rad_config, lon_vec, lat_vec,
                                                                                        TSI_1AU_day)
        elif grid_type=="lebedev":
            toa_emission_day_hist, net_toa_day_hist = compute_daily_time_series_lebedev_toa(mid_day_jd, rad_config, TSI_1AU_day)
    
    else:
        print("Loading daily hist TOA...")
        try:
            toa_emission_day_hist = np.load(all_file_names["daily_hist_emission_toa"]+'.npy')
        except:
            if grid_type=="regular": n_lon, n_lat = len(lon_vec), len(lat_vec)
            toa_emission_day_hist = np.loadtxt(all_file_names["daily_hist_emission_toa"]+'.txt').reshape((n_lat, n_lon, n_steps))
        
        try:
            net_toa_day_hist = np.load(all_file_names["daily_hist_net_toa"]+'.npy')
        except:
            if grid_type=="regular": n_lon, n_lat = len(lon_vec), len(lat_vec)
            net_toa_day_hist = np.loadtxt(all_file_names["daily_hist_net_toa"]+'.txt').reshape((n_lat, n_lon, n_steps))


    return toa_emission_day_hist, net_toa_day_hist


def get_daily_hist_net_at_altitude(data_dir, altitude_km, day_idx, mid_day_jd, TSI_1AU_day, rad_config, n_lon, n_lat, n_steps, n_cores, grid_type):

    all_file_names, file_existences = get_file_names_and_existence(data_dir, altitude_km, day_idx, grid_type)
    
    if not(file_existences['daily_hist_net_Fr_'+str(altitude_km)+'km']):
        daily_hist_net_F_altitude, daily_hist_net_Fr_altitude = compute_daily_hist_net_at_altitude(data_dir, altitude_km, day_idx, mid_day_jd, TSI_1AU_day, 
                                                                     rad_config, n_lon, n_lat, n_steps, n_cores, grid_type)
        print("daily_hist_computed")
    else:
        print(f"Loading daily hist net at {altitude_km} km...")
        try:
            daily_hist_net_Fr_altitude = np.load(all_file_names['daily_hist_net_Fr_'+str(altitude_km)+'km']+'.npy')
            daily_hist_net_F_altitude = None
        except:
            daily_hist_net_Fr_altitude = np.loadtxt(all_file_names['daily_hist_net_'+str(altitude_km)+'km']+'.txt').reshape((n_lat, n_lon, n_steps))

    return daily_hist_net_F_altitude, daily_hist_net_Fr_altitude


def get_solar_incoming_day_hist(rad_config, mid_day_jd, altitude_km, stacked_grid_u_el, TSI_1AU_day):
    
    #lon_vec, lat_vec, _, _ = get_sphere_grid(n_lon, n_lat)
    #_, stacked_grid_u_el = get_stacked_spherical_grid_els(lon_vec, lat_vec, altitude_km, total_R=RE+altitude_km)     # stacked_grid_r_el is at altitude

    d_sun_day, u_sun_day = get_d_u_sun_hist(rad_config["ephemerides"], mid_day_jd)         # stacked_grid_u_el is independent of altitude
    zeroed_cos_theta_s_day_hist = get_zeroed_cos_theta_s_hist(stacked_grid_u_el, u_sun_day, altitude_km) #, n_lat=n_lat, n_lon=n_lon)

    u_sun_day_grid = np.ones_like(stacked_grid_u_el)[:,None,:] *  u_sun_day
    u_sun_day_grid[zeroed_cos_theta_s_day_hist==0] = 0
    
    TSI_Earth_day_vec = TSI_1AU_day * (AU/d_sun_day)**2
    #solar_incoming_day_hist = zeroed_cos_theta_s_day_hist * TSI_Earth_day_vec[None, :]   # TSI_Earth_day_vec[None, None, :]
    solar_incoming_day_hist = u_sun_day_grid * TSI_Earth_day_vec[None, :, None]   # TSI_Earth_day_vec[None, None, :]
    radial_solar_incoming_day_hist = zeroed_cos_theta_s_day_hist * TSI_Earth_day_vec[None, :]

    return solar_incoming_day_hist, radial_solar_incoming_day_hist


def compute_all_fluxes_at_altitude(toa_emission, cos_alpha, r_rel_norm, r_rel, weights, method="einsum"):

    if method=="broadcast":
        field_to_integrate = (1/np.pi) * toa_emission[None,:,None] * cos_alpha[:,:,None] * (1/(r_rel_norm**3))[:,:,None] * r_rel
        full_F = np.sum(field_to_integrate * weights[None,:,None], axis=1) * RE**2
        return full_F
    
    elif method=="einsum":
        full_F = np.einsum(
            't,jt,jtk->jk',
            (1/np.pi) * toa_emission,   # TODO: 1/π is to convert flux to per unit of solid angle. This conversion should be done externally and potentially account for ADMs
            cos_alpha * weights[None, :],
            r_rel / (r_rel_norm[:, :, None]**3)
        ) * RE**2
        return full_F
    
    elif method=="einsum_opt":
        full_F = np.einsum(
            't,jt,jtk->jk',
            (1/np.pi) * toa_emission,
            cos_alpha * weights[None, :],
            r_rel / (r_rel_norm[:, :, None]**3),
            optimize=True
        ) * RE**2
        return full_F
    
    elif method=="einsum_jax":
        full_F = jnp.einsum(
            't,jt,jtk->jk',
            (1/np.pi) * toa_emission,
            cos_alpha * weights[None, :],
            r_rel / (r_rel_norm[:, :, None]**3)
        ) * RE**2
        return full_F


    elif method=="loop_0_einsum":
        n_i, n_j, _ = r_rel.shape
        full_F = np.zeros((n_i, 3))
        for i in range(n_i):
            pass
            # at this point I don't try the loops because einsum seems to always win!


def compute_radial_fluxes_at_altitude(full_F, stacked_altitude_grid_u, method="einsum"):

    if method=="broadcast":
        full_radial_F = np.sum(full_F * stacked_altitude_grid_u, axis=1)
        return full_radial_F
    
    elif method=="einsum":
        full_radial_F = np.einsum('jk,jk->j', full_F, stacked_altitude_grid_u)
        return full_radial_F

    # at this point I don't try the loops because einsum seems to always win!

#@profile
def compute_daily_hist_net_at_altitude(data_dir, altitude_km, day_idx, mid_day_jd, TSI_1AU_day, rad_config, n_lon, n_lat, n_steps, n_cores, grid_type):
    

    # this is for regular grid only
    lon_vec, lat_vec, lon_edges_vec, lat_edges_vec = get_sphere_grid(n_lon, n_lat)
    stacked_grid_r_el, stacked_grid_u_el = get_stacked_spherical_grid_els(lon_vec, lat_vec, altitude_km, total_R=RE+altitude_km)     # stacked_grid_r_el is at altitude

    solar_incoming_day_hist, radial_solar_incoming_day_hist = get_solar_incoming_day_hist(rad_config, mid_day_jd, altitude_km, stacked_grid_u_el, TSI_1AU_day)
    
    
    # load TOA emission in Lebedev grid
    stacked_u_toa, weights = lebedev_rule(131)
    stacked_u_toa = stacked_u_toa.T
    stacked_r_toa = stacked_u_toa * RE
    toa_emission_day_hist, net_toa_day_hist = get_daily_hist_toa(data_dir, day_idx, mid_day_jd, TSI_1AU_day, rad_config, 
                                            lon_vec, lat_vec, n_steps, grid_type="lebedev") # integration to altitude MUST be done with the Lebedev TOA grid (orders of magnitude more accurate with less points)
    n_steps = np.shape(toa_emission_day_hist)[1]
    n_points = np.shape(stacked_grid_r_el)[0]

    # 1. Compute all relative vectors
    r_rel = get_all_r_rel(stacked_grid_r_el, stacked_r_toa) # broadcasting is fastest
    
    # 2. Compute norm of relative vectors
    r_rel_norm = get_r_rel_norm(r_rel, method="einsum") # einsum method tested to win (30% faster regardless of the CV grid size)

    # 3. Compute all relative angles alpha
    cos_alpha = get_all_cos_alpha(r_rel, r_rel_norm, stacked_u_toa, method="einsum") # again tested to win wrt all other methods (at least in 65e3 CV x 5e3 TOA grid)

    # 4. Compute flux map and radial flux map
    # n_steps = 10
    args_list = [(i, toa_emission_day_hist[:, i]) for i in range(n_steps)]
    print(f"multiproc start method: {multiprocessing.get_start_method()}") 

    with Pool(processes=n_cores, initializer=init_worker, initargs=(cos_alpha, r_rel_norm, r_rel, weights, stacked_grid_u_el)) as pool:
        
        results = pool.starmap(compute_fluxes_worker, args_list)

    #_pool_globals = {} checked not to free up any memory
    
    # Extract the arrays from results (skip the index 'i' since we assume order)
    emission_F_list = [res[1] for res in results]  # List of (n_points, 3) arrays
    radial_F_list = [res[2] for res in results]   # List of (n_points,) arrays

    # Stack into the final arrays
    emission_F_hist = np.stack(emission_F_list, axis=1)  # Shape: (n_points, n_steps, 3)
    radial_emission_F_hist = np.stack(radial_F_list, axis=1)  # Shape: (n_points, n_steps)
    

    """
    emission_F_hist = np.zeros((n_points, n_steps, 3))
    radial_emission_F_hist = np.zeros((n_points, n_steps))
    for i, emission_F, radial_F in results:
        emission_F_hist[:, i, :] = emission_F
        radial_emission_F_hist[:, i] = radial_F
    """

    # # non-parallel:
    # emission_F_hist = np.zeros((n_points, n_steps, 3))
    # radial_emission_F_hist = np.zeros((n_points, n_steps))
# 
    # for i, toa_emission in enumerate(toa_emission_day_hist[:,:n_steps].T):
    #     #progress_bar(i, n_steps)
# 
    #     # 4. Compute flux map
    #     emission_F_hist[:,i,:] = compute_all_fluxes_at_altitude(toa_emission, cos_alpha, r_rel_norm, r_rel, weights, method="einsum")
# 
    #     # 5. Get dot product for radial component of flux map
    #     radial_emission_F_hist[:,i] = compute_radial_fluxes_at_altitude(emission_F_hist[:,i,:], stacked_grid_u_el, method="einsum")

    daily_hist_net_F_hist =  solar_incoming_day_hist[:,:n_steps,:] - emission_F_hist
    daily_hist_net_radial_F_hist =  radial_solar_incoming_day_hist[:,:n_steps] - radial_emission_F_hist

    """
    for i in [0, 9]:

        reshaped_Fr = np.reshape(net_radial_F_hist[:,i], (n_lat, n_lon))
        Plotter.plot_geo_data(reshaped_Fr, lon_edges_vec, lat_edges_vec, file_name='testing_Fr_'+str(i),
                            make_symmetric_cmap=True, title=f"Altitude: {altitude_km} km", data_label='Net $F_R$ (W/m$^2$)')
        
        for j, x in zip([0,1,2], ['x', 'y', 'z']):
            reshaped_F = np.reshape(net_F_hist[:,i,j], (n_lat, n_lon))
            Plotter.plot_geo_data(reshaped_F, lon_edges_vec, lat_edges_vec, file_name='testing_F'+x+'_'+str(i),
                                make_symmetric_cmap=True, title=f"Altitude: {altitude_km} km", data_label='Net $F_'+x+'$ (W/m$^2$)')
            
    print("checks done")
    """



    # d_sun_day, u_sun_day = get_d_u_sun_hist(rad_config["ephemerides"], mid_day_jd)         # stacked_grid_u_el is independent of altitude
    # zeroed_cos_theta_s_day_hist = get_zeroed_cos_theta_s_hist(stacked_grid_u_el, u_sun_day, altitude_km, n_lat, n_lon)
    # 
    # TSI_Earth_day_vec = TSI_1AU_day * (AU/d_sun_day)**2
    # solar_incoming_day_hist = zeroed_cos_theta_s_day_hist * TSI_Earth_day_vec[None, None, :]



    # WE'RE HERE - the rest of the function is only for integrating a regular toa grid

    """
    grid_r_el_toa = lonlat_to_r(lon_vec, lat_vec, 0, "spherical")
    grid_u_el = grid_r_el_toa / RE
    grid_cell_areas_toa = get_spherical_grid_cell_areas(lat_edges_vec, lon_edges_vec, RE)

    toa_emission_mapped_to_altitude_hist = np.zeros((n_lat, n_lon, n_steps))

    print(f"Mapping TOA emission to {altitude_km}km...")
    for i in range(n_steps):

        progress_bar(i, n_steps)
        toa_emission_mapped_to_altitude_hist[:,:,i] = convolve_toa_emission_to_altitude(
            stacked_grid_r_el, stacked_grid_u_el,          # we are using the same grid for toa and altitude
            grid_r_el_toa, grid_u_el, grid_cell_areas_toa, 
            toa_emission_day_hist[:,:,i], n_lat, n_lon, n_workers=n_cores
        )
    print("")

    solar_incoming_day_hist = get_solar_incoming_day_hist(rad_config, mid_day_jd, altitude_km, n_lat, n_lon, TSI_1AU_day) # for regular grid: now missing reshape

    daily_hist_net_altitude = solar_incoming_day_hist - toa_emission_mapped_to_altitude_hist
    """

    return daily_hist_net_F_hist, daily_hist_net_radial_F_hist



def compute_daily_avg_net_at_altitude_simpl_1(data_dir, altitude_km, day_idx, mid_day_jd, TSI_1AU_day, rad_config, n_lon, n_lat, n_steps, n_cores):
    
    lon_vec, lat_vec, lon_edges_vec, lat_edges_vec = get_sphere_grid(n_lon, n_lat)
    stacked_grid_r_el, stacked_grid_u_el = get_stacked_spherical_grid_els(lon_vec, lat_vec, altitude_km, total_R=RE+altitude_km)     # stacked_grid_r_el is at altitude
    
    grid_r_el_toa = lonlat_to_r(lon_vec, lat_vec, 0, "spherical")
    grid_u_el = grid_r_el_toa / RE
    grid_cell_areas_toa = get_spherical_grid_cell_areas(lat_edges_vec, lon_edges_vec, RE)

    
    # ECEF:
    solar_incoming_day_hist = get_solar_incoming_day_hist(rad_config, mid_day_jd, altitude_km, n_lat, n_lon, TSI_1AU_day)
    solar_incoming_day_avg = np.mean(solar_incoming_day_hist, axis=2)
    
    toa_emission_day_hist, _ = get_daily_hist_toa(data_dir, day_idx, mid_day_jd, TSI_1AU_day, rad_config, 
                                            lon_vec, lat_vec, n_steps, create_dirs=False)

    daily_avg_emission_toa = np.mean(toa_emission_day_hist, axis=2)

    daily_avg_emission_toa_mapped_to_altitude = convolve_toa_emission_to_altitude(
        stacked_grid_r_el, stacked_grid_u_el,          # we are using the same grid for toa and altitude
        grid_r_el_toa, grid_u_el, grid_cell_areas_toa, 
        daily_avg_emission_toa, n_lat, n_lon, n_workers=n_cores
    )

    daily_avg_at_altitude_simpl1 = solar_incoming_day_avg - daily_avg_emission_toa_mapped_to_altitude


    # SFF:
    R_ECEF_to_SunFrame_day_hist = get_R_SunFrame_hist(rad_config["ephemerides"], mid_day_jd)

    #solar_incoming_day_hist_SFF = np.nan_to_num(field_hist_rotation(solar_incoming_day_hist, R_ECEF_to_SunFrame_day_hist))
    solar_incoming_day_hist_SFF = np.nan_to_num(field_hist_rotation_multiproc(solar_incoming_day_hist, R_ECEF_to_SunFrame_day_hist, n_cores=n_cores))

    daily_avg_incoming_solar_SFF = np.mean(solar_incoming_day_hist_SFF, axis=2)

    #toa_emission_day_hist_SFF = np.nan_to_num(field_hist_rotation(toa_emission_day_hist, R_ECEF_to_SunFrame_day_hist))
    toa_emission_day_hist_SFF = np.nan_to_num(field_hist_rotation_multiproc(toa_emission_day_hist, R_ECEF_to_SunFrame_day_hist, n_cores=n_cores))
    daily_avg_emission_toa_SFF = np.mean(toa_emission_day_hist_SFF, axis=2)

    daily_avg_emission_toa_mapped_to_altitude_SFF = convolve_toa_emission_to_altitude(
        stacked_grid_r_el, stacked_grid_u_el,          # we are using the same grid for toa and altitude
        grid_r_el_toa, grid_u_el, grid_cell_areas_toa, 
        daily_avg_emission_toa_SFF, n_lat, n_lon, n_workers=n_cores
    )

    daily_avg_at_altitude_simpl1_SFF = daily_avg_incoming_solar_SFF - daily_avg_emission_toa_mapped_to_altitude_SFF

    # back to ECEF now (should it give the same? My bet is no because the SFF gets rid of the diurnal cycle)
    #daily_avg_at_altitude_simpl1b = field_hist_rotation(daily_avg_at_altitude_simpl1_SFF, R_ECEF_to_SunFrame_day_hist, transpose_R=True)


    return daily_avg_at_altitude_simpl1, daily_avg_at_altitude_simpl1_SFF #, daily_avg_at_altitude_simpl1b


def save_toa_files(toa_emission_day_hist, net_toa_day_hist, grid_cell_areas_toa, data_dir, day_idx, R_ECEF_to_SunFrame_day_hist, grid_type, save_SFF_hist=True):
    

    all_file_names, file_existences = get_file_names_and_existence(data_dir, 0, day_idx, grid_type)
    
    if not(file_existences['daily_hist_emission_toa']):
        print("Saving daily_hist_emission_toa file...")
        np.save(all_file_names['daily_hist_emission_toa'], toa_emission_day_hist)

    if not(file_existences['daily_hist_net_toa']):
        print("Saving daily_hist_net_toa file...")
        np.save(all_file_names['daily_hist_net_toa'], net_toa_day_hist)
    
    if not(file_existences.get('daily_hist_net_toa_SFF'), True) and save_SFF_hist:  # missing key defaults to True so block is skipped
        print("Saving daily_hist_net_toa file Sun-Fixed Frame...")
        net_toa_day_hist_SFF = field_hist_rotation(net_toa_day_hist, R_ECEF_to_SunFrame_day_hist)
        np.save(all_file_names['daily_hist_net_toa_SFF'], net_toa_day_hist_SFF)

    if not(file_existences.get('daily_hist_net_toa_lat_avg', True)):
        print("Saving daily_hist_net_toa_lat_avg file...")
        toa_lat_avg_day_hist = np.mean(net_toa_day_hist, axis=1)    # mean works because element areas are constant at every latitude
        np.save(all_file_names['daily_hist_net_toa_lat_avg'], toa_lat_avg_day_hist)

    if not(file_existences['daily_hist_net_toa_surf_avg']):
        print("Saving daily_hist_net_toa_surf_avg file...")
        if grid_type=="regular":
            surface_area_toa = np.sum(grid_cell_areas_toa)
            toa_surf_avg_day_hist = np.sum(net_toa_day_hist * grid_cell_areas_toa[:,:,None], axis=(0,1)) / surface_area_toa
        elif grid_type=="lebedev":
            _, weights = lebedev_rule(131)  # HARDCODED - 131 is the highest available Levedev order
            toa_surf_avg_day_hist = np.sum(net_toa_day_hist * weights[:,None], axis=0) / (4*np.pi)
        np.save(all_file_names['daily_hist_net_toa_surf_avg'], toa_surf_avg_day_hist)

    if not(file_existences.get('daily_avg_net_toa', True)): 
        print("Saving daily_avg_net_toa file...")
        net_toa_daily_avg = np.mean(net_toa_day_hist, axis=2)
        np.save(all_file_names['daily_avg_net_toa'], net_toa_daily_avg)


def save_altitude_files(daily_hist_net_F_altitude, daily_hist_net_Fr_altitude, #daily_avg_at_altitude_simpl1, daily_avg_at_altitude_simpl1_SFF, 
                        altitude_km, day_idx, data_dir, R_ECEF_to_SunFrame_day_hist, stacked_grid_u_el):

    all_file_names, file_existences = get_file_names_and_existence(data_dir, altitude_km, day_idx, grid_type="regular")
    
    if (daily_hist_net_Fr_altitude is not None):
    
        if not(file_existences['daily_hist_net_Fr_'+str(altitude_km)+'km']):
            print(f"Saving daily_hist_net_{altitude_km}km file...")
            np.save(all_file_names['daily_hist_net_Fr_'+str(altitude_km)+'km'], daily_hist_net_Fr_altitude)

        if not(file_existences['daily_hist_net_Fr_'+str(altitude_km)+'km_SFF']):
            print(f"Saving daily_hist_net_Fr_{altitude_km}km file Sun-Fixed Frame...")
            daily_hist_net_altitude_SFF = field_hist_rotation(daily_hist_net_Fr_altitude, R_ECEF_to_SunFrame_day_hist,
                                                              stacked_grid_u_el=stacked_grid_u_el)
            np.save(all_file_names['daily_hist_net_Fr_'+str(altitude_km)+'km_SFF'], daily_hist_net_altitude_SFF)

        if not(file_existences['daily_avg_net_Fr_'+str(altitude_km)+'km']):
            print(f"Saving daily_avg_net_Fr_{altitude_km}km file...")
            #daily_avg_net_altitude = np.mean(daily_hist_net_Fr_altitude, axis=2)
            daily_avg_net_altitude = np.mean(daily_hist_net_Fr_altitude, axis=1)
            np.save(all_file_names['daily_avg_net_Fr_'+str(altitude_km)+'km'], daily_avg_net_altitude)
        
        if not(file_existences['daily_avg_net_Fr_'+str(altitude_km)+'km_SFF']):
            # if it didn't exist we just saved it above so we can always load it:
            print(f"Saving daily_avg_net_Fr_{altitude_km}km_SFF file...")
            daily_hist_net_altitude_SFF = np.load(all_file_names['daily_hist_net_Fr_'+str(altitude_km)+'km_SFF']+'.npy')
            #daily_avg_net_altitude_SFF = np.mean(daily_hist_net_altitude_SFF, axis=2)
            daily_avg_net_altitude_SFF = np.mean(daily_hist_net_altitude_SFF, axis=1)
            np.save(all_file_names['daily_avg_net_Fr_'+str(altitude_km)+'km_SFF'], daily_avg_net_altitude_SFF)

    """
    if not(file_existences['daily_avg_net_'+str(altitude_km)+'km_simpl1']):
        # simplification 1: we convolve the daily avg of the computed net emission toa to altitude and combine it with the actual mean of incoming solar
        print(f"Saving daily_avg_net_{altitude_km}km_simpl1 file...")
        np.save(all_file_names['daily_avg_net_'+str(altitude_km)+'km_simpl1'], daily_avg_at_altitude_simpl1)

    if not(file_existences['daily_avg_net_'+str(altitude_km)+'km_simpl1_SFF']):
        # simplification 1 SFF: simplification 1 but the process is done after rotating the separate fields to the SFF before computing the daily avg
        print(f"Saving daily_avg_net_{altitude_km}km_simpl1_SFF file...")
        np.save(all_file_names['daily_avg_net_'+str(altitude_km)+'km_simpl1_SFF'], daily_avg_at_altitude_simpl1_SFF)
    """
        

def compute_daily_time_series_lebedev_toa(mid_day_jd, rad_config, TSI_1AU_day, lebedev_n=131):

    day_datestr = Time(mid_day_jd, format='jd').to_datetime().strftime("%Y-%m-%d")
    d_sun_day, u_sun_day = get_d_u_sun_hist(rad_config["ephemerides"], mid_day_jd)
    TSI_Earth_day_vec = TSI_1AU_day * (AU/d_sun_day)**2
    
    u_el_leb, weights = lebedev_rule(lebedev_n)
    u_el_leb = np.transpose(u_el_leb)
    r_el_leb = u_el_leb * RE

    a_map, e_map = get_expanded_ae_maps(rad_config["sh_mode"], day_datestr, stacked_r_array=r_el_leb, grid_mode="arbitrary")
    zeroed_cos_theta_s_day_hist_toa = get_zeroed_cos_theta_s_hist(u_el_leb, u_sun_day, 0)

    solar_incoming_day_hist_toa = zeroed_cos_theta_s_day_hist_toa * TSI_Earth_day_vec[None, :]

    LW_outgoing_day_hist_toa = e_map[:, None] / 4 * TSI_Earth_day_vec[None, :]
    SW_outgoing_day_hist_toa = zeroed_cos_theta_s_day_hist_toa * a_map[:, None] * TSI_Earth_day_vec[None, :]

    toa_emission_day_hist = LW_outgoing_day_hist_toa + SW_outgoing_day_hist_toa
    net_toa_day_hist = solar_incoming_day_hist_toa - toa_emission_day_hist

    return toa_emission_day_hist, net_toa_day_hist



def compute_daily_time_series_maps_toa(mid_day_jd, rad_config, lon_vec, lat_vec, TSI_1AU_day):


    n_lon, n_lat = len(lon_vec), len(lat_vec)
    stacked_grid_r_el, stacked_grid_u_el = get_stacked_spherical_grid_els(lon_vec, lat_vec, 0, RE)     # stacked_grid_r_el is at altitude
    
    day_datestr = Time(mid_day_jd, format='jd').to_datetime().strftime("%Y-%m-%d")
    d_sun_day, u_sun_day = get_d_u_sun_hist(rad_config["ephemerides"], mid_day_jd)

    #TSI_1AU_day = rad_settings.get_TSI_1AU(mid_day_jd, rad_config["TSI_source"])
    TSI_Earth_day_vec = TSI_1AU_day * (AU/d_sun_day)**2

    a_map, e_map = get_expanded_ae_maps(rad_config["sh_mode"], day_datestr, lon_vec=lon_vec, lat_vec=lat_vec)
    zeroed_cos_theta_s_day_hist_toa = get_zeroed_cos_theta_s_hist(stacked_grid_u_el, u_sun_day, 0, n_lat=n_lat, n_lon=n_lon)
    
    solar_incoming_day_hist_toa = zeroed_cos_theta_s_day_hist_toa * TSI_Earth_day_vec[None, None, :]
    
    LW_outgoing_day_hist_toa = e_map[:, :, None] / 4 * TSI_Earth_day_vec[None, None, :]
    SW_outgoing_day_hist_toa = zeroed_cos_theta_s_day_hist_toa * a_map[:, :, None] * TSI_Earth_day_vec[None, None, :]
    
    toa_emission_day_hist = LW_outgoing_day_hist_toa + SW_outgoing_day_hist_toa
    net_toa_day_hist = solar_incoming_day_hist_toa - toa_emission_day_hist

    return toa_emission_day_hist, net_toa_day_hist

_pool_globals = {}

def init_worker(cos_alpha_shared, r_rel_norm_shared, r_rel_shared, weights_shared, u_el_shared):
    """Initialize worker with shared constant data."""
    _pool_globals['cos_alpha'] = cos_alpha_shared
    del(cos_alpha_shared)
    _pool_globals['r_rel_norm'] = r_rel_norm_shared
    del(r_rel_norm_shared)
    _pool_globals['r_rel'] = r_rel_shared
    del(r_rel_shared)
    _pool_globals['weights'] = weights_shared
    del(weights_shared)
    _pool_globals['stacked_grid_u_el'] = u_el_shared
    del(u_el_shared)


def compute_fluxes_worker(i, toa_emission):
    """Worker that uses global initialized data."""
    g = _pool_globals
    emission_F = compute_all_fluxes_at_altitude(
        toa_emission, g['cos_alpha'], g['r_rel_norm'], g['r_rel'], g['weights'], method="einsum"
    )
    radial_F = compute_radial_fluxes_at_altitude(emission_F, g['stacked_grid_u_el'], method="einsum")
    return i, emission_F, radial_F

"""
_global = {}

def init_worker(grid_r_el_toa, grid_u_el_toa, grid_cell_areas_toa, emission_toa_map):
    _global['grid_r_el_toa'] = grid_r_el_toa
    _global['grid_u_el_toa'] = grid_u_el_toa
    _global['grid_cell_areas_toa'] = grid_cell_areas_toa
    _global['net_toa_map'] = emission_toa_map

"""

def convolve_lebedev_toa_emission_to_regular_grid_altitude(stacked_grid_r_el_CV, stacked_grid_u_el_CV,
                                                           grid_r_el_toa, grid_u_el_toa, net_toa,
                                                           n_lat_CV, n_lon_CV):
    pass


def convolve_toa_emission_to_altitude(stacked_grid_r_el_CV, stacked_grid_u_el_CV,  # CV for control volume (sphere in orbit)
                    grid_r_el_toa, grid_u_el_toa, grid_cell_areas_toa, 
                    net_toa_map, n_lat_CV, n_lon_CV, n_workers=None):
    
    n_elements_CV = np.shape(stacked_grid_r_el_CV)[0] # should be equal to n_lat_CV * n_lon_CV
    #print("Entering get flux map function...")
    if n_workers is None:
        raise Exception("Single core integration not implemented! Use n_cores!=None")
    else:
        #print("Using multiprocessing...")
        args_list = [
            (stacked_grid_r_el_CV[j,:], stacked_grid_u_el_CV[j,:]) for j in range(n_elements_CV)
        ]
        #t1 = time.time()
        with Pool(processes=n_workers, initializer=init_worker, 
                    initargs=(grid_r_el_toa, grid_u_el_toa, grid_cell_areas_toa, net_toa_map)) as pool:
            results = pool.starmap(get_net_fluxes_at_CV_element_j, args_list)

        all_F_outgoing_radial_CV = results # np.array([res[0] for res in results])
        
        #t2 = time.time()
        #print(f"Time with multiprocessing ({n_workers} workers): {t2-t1}")
                        
    
    # print('')
    all_F_outgoing_radial_CV_map = np.reshape(all_F_outgoing_radial_CV, (n_lat_CV, n_lon_CV))
    
    return all_F_outgoing_radial_CV_map


def get_net_fluxes_at_CV_element_j(r_el_j_CV, u_el_j_CV): # , area_el_j_CV):
    
    g = _global
    grid_r_el = g['grid_r_el_toa']
    grid_u_el = g['grid_u_el_toa']
    grid_cell_areas_toa = g['grid_cell_areas_toa']
    net_toa_map = g['net_toa_map']
    
    all_r_rel = r_el_j_CV[None,None,:] - grid_r_el
    all_r_rel_norm = np.linalg.norm(all_r_rel,axis=2)
    all_u_rel_expanded = all_r_rel/all_r_rel_norm[:,:,None]
    
    all_cos_alpha = np.sum((all_u_rel_expanded)*grid_u_el, axis=2)
    all_cos_alpha[all_cos_alpha<0] = 0

    all_F_outgoing =  net_toa_map[:,:,None] * (1/np.pi) * grid_cell_areas_toa[:,:,None] * all_cos_alpha[:,:,None] * (1/(all_r_rel_norm**2))[:,:,None] * all_r_rel/(all_r_rel_norm[:,:,None])
    F_outgoing = np.sum(all_F_outgoing, axis=(0,1))
    F_radial_j = np.dot(F_outgoing, u_el_j_CV)
    
    
    return F_radial_j




def get_file_names_and_existence(data_dir, alt_km, day_idx, grid_type, create_dirs=True):
    
    all_out_file_types = required_files(alt_km, grid_type)
    
    if create_dirs:
        # create directories if they don't exist:
        for ftype in all_out_file_types:
            os.makedirs(os.path.join(data_dir, ftype), exist_ok=True)

    all_file_names = {ftype: os.path.join(data_dir, ftype, 'day_'+str(day_idx)) for ftype in all_out_file_types}
    file_existences = {ftype: (os.path.exists(fname_full+'.txt') or os.path.exists(fname_full+'.npy')) 
                        for ftype, fname_full in zip(all_out_file_types, all_file_names.values())}
    
    return all_file_names, file_existences

                

def get_expanded_ae_maps(mode, datestr, lon_vec=None, lat_vec=None, stacked_r_array=None, grid_mode="regular"):
    
    a_sh_map, e_sh_map = rad_settings.get_ae_sh_maps_numpy_new(mode, datestr)

    if grid_mode=="regular":
        assert (lon_vec is not None) and (lat_vec is not None), "For regular grid mode, lon_vec and lat_vec must be provided"
        lon_mesh_vectd, lat_mesh_vectd, mesh_shape = get_reshaped_grid(lon_vec,lat_vec)
        a_map = np.reshape(expand_sh(a_sh_map, lon_mesh_vectd, lat_mesh_vectd), mesh_shape)
        e_map = np.reshape(expand_sh(e_sh_map, lon_mesh_vectd, lat_mesh_vectd), mesh_shape)
    
    elif grid_mode=="arbitrary":
        assert (stacked_r_array is not None), "For arbitrary grid mode, stacked_r_array must be provided"
        (_, full_lat_vec, full_lon_vec) = cartesian_to_spherical(stacked_r_array[:,0], stacked_r_array[:,1], stacked_r_array[:,2])
        a_map = expand_sh(a_sh_map, full_lon_vec.deg, full_lat_vec.deg)
        e_map = expand_sh(e_sh_map, full_lon_vec.deg, full_lat_vec.deg)

    return a_map, e_map


def get_d_u_sun_hist(ephem_tag, mid_day_jd): # to save RAM memory

    dir_r_sun = os.path.join(MEDIA_DIR, 'solar_ephemerides', ephem_tag, 'daily_files')
    jd_r_sun = np.load(os.path.join(dir_r_sun, 'jd_mid_day_array.npy'))

    day_idx_r_sun = find_day_idx_r_sun(jd_r_sun, mid_day_jd)
    r_sun_day = np.load(os.path.join(dir_r_sun, 'r_Sun_ECEF_day_'+str(day_idx_r_sun)+'.npy'))
    d_sun_day = np.linalg.norm(r_sun_day,2,1)
    u_sun_day = r_sun_day / d_sun_day[:, None]

    return d_sun_day, u_sun_day


def get_R_SunFrame_hist(ephem_tag, mid_day_jd): # to save RAM memory

    dir_r_sun = os.path.join(MEDIA_DIR, 'solar_ephemerides', ephem_tag, 'daily_files')
    jd_r_sun = np.load(os.path.join(dir_r_sun, 'jd_mid_day_array.npy'))

    day_idx_r_sun = find_day_idx_r_sun(jd_r_sun, mid_day_jd)
    R_hist_day = np.load(os.path.join(dir_r_sun, 'R_ECEF_to_SunFrame_day_'+str(day_idx_r_sun)+'.npy'))

    return R_hist_day


def find_day_idx_r_sun(jd_r_sun, mid_day_jd):
    day_idx_r_sun = np.where(np.isin(jd_r_sun, mid_day_jd))[0][0]

    return day_idx_r_sun

def get_zeroed_cos_theta_s_hist(stacked_grid_u_el, u_sun_day, altitude_km, n_lat=None, n_lon=None):
    
    stacked_cos_theta_s_day_hist = stacked_grid_u_el @ np.transpose(u_sun_day)
    zeroed_cos_theta_s_day_hist = stacked_cos_theta_s_day_hist

    cos_lim = -np.sqrt(1 - (RE/(RE + altitude_km))**2)
    zeroed_cos_theta_s_day_hist[stacked_cos_theta_s_day_hist < cos_lim] = 0     # this also changes cos_theta_s_day_hist but inside the function it doesn't really matter

    if n_lat is not None and n_lon is not None:
        n_steps = np.shape(u_sun_day)[0]
        return np.reshape(zeroed_cos_theta_s_day_hist, (n_lat, n_lon, n_steps))
    else:
        return zeroed_cos_theta_s_day_hist


def get_cos_theta_s_lim(altitude_km):
    return -np.sqrt(1 - (RE/(RE + altitude_km))**2)     # this minus sign only if u_sun points towards the Sun


def mid_day_jd_array_from_jd_interval(jd_interval):

    edges_jd_array = np.arange(jd_interval[0], jd_interval[-1]+1, 1)
    mid_day_jd_array = (edges_jd_array[1:] + edges_jd_array[:-1]) / 2

    return mid_day_jd_array

def get_mid_day_jd_array(EEI_truth_name):
    rad_config = rad_settings.radiation_settings_from_EEI_truth_name(EEI_truth_name)
    return mid_day_jd_array_from_jd_interval(rad_config["jd_interval"])


if __name__ == "__main__":

    # this is just for thesting the functions above (but also everything an external main should do lol)

    compute_radiation_maps("EEI_truth_1", altitude_array = [0, 800], selected_days_idxs = [0, 182])