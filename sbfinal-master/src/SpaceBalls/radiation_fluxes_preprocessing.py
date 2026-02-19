import os, sys
import numpy as np
import time
from astropy.time import Time, TimeDelta
import astropy.units as u
import multiprocessing
from multiprocessing import Pool

from SpaceBalls.paths import CONFIG_DIR, MEDIA_DIR
sys.path.insert(0, str(CONFIG_DIR.parent))  # parent of 'config'
import SpaceBalls.radiation_settings as rad_settings
from SpaceBalls.sph_meshing import get_sphere_grid, get_spherical_grid_cell_areas, lonlat_to_r, expand_sh_grid, get_stacked_spherical_grid_els, progress_bar, field_hist_rotation
import config.constants as constants

AU = constants.astronomical_unit(units='km')
RE = constants.earth_radius(units='km')


def required_files(alt_km):
    if alt_km==0:                                                                   # SFF: sun-fixed frame
        return ['daily_hist_emission_toa', 'daily_hist_net_toa', 'daily_hist_net_toa_SFF', 
                'daily_hist_net_toa_lat_avg', 'daily_hist_net_toa_surf_avg',
                'daily_avg_net_toa']
    elif alt_km>0: 
        return ['daily_hist_net_'+str(alt_km)+'km', 'daily_hist_net_'+str(alt_km)+'km_SFF',
                'daily_avg_net_'+str(alt_km)+'km', 'daily_avg_net_'+str(alt_km)+'km_SFF']


def get_EEI_truth_daily_jd_arrays(EEI_truth_name):

    rad_config = rad_settings.radiation_settings_from_EEI_truth_name(EEI_truth_name)
    mid_day_jd_array = mid_day_jd_array_from_jd_interval(rad_config["jd_interval"])

    step_minutes = 1 # DO NOT CHANGE - must be equal to the one used for the files in solar_ephemerides
    step_days = step_minutes / (60*24)
    daily_jd_arrays = [np.arange(mid_day_jd-0.5, mid_day_jd+0.5, step_days) for mid_day_jd in mid_day_jd_array]
    
    return daily_jd_arrays


def compute_radiation_maps(EEI_truth_name, altitude_array = [0, 800, 1500], degrees_CV_bins_array = [1], 
                           selected_days_idxs = [0, 182, 1620, 1800], n_cores=12):
    
    daily_jd_arrays = get_EEI_truth_daily_jd_arrays(EEI_truth_name)
    
    rad_config = rad_settings.radiation_settings_from_EEI_truth_name(EEI_truth_name)
    mid_day_jd_array = mid_day_jd_array_from_jd_interval(rad_config["jd_interval"])
    n_days = len(mid_day_jd_array)
    
    daily_jd_arrays_reduced = [daily_jd_arrays[i] for i in selected_days_idxs]

    for degrees_CV_bins in degrees_CV_bins_array:

        n_lon = 360 // degrees_CV_bins
        n_lat = 180 // degrees_CV_bins
        lon_vec, lat_vec, lon_edges_vec, lat_edges_vec = get_sphere_grid(n_lon, n_lat)
        out_dir = os.path.join(MEDIA_DIR, 'true_EEI', EEI_truth_name, 'grid_'+str(n_lat)+'x'+str(n_lon))

        grid_cell_areas_toa = get_spherical_grid_cell_areas(lat_edges_vec, lon_edges_vec, RE)

        for altitude_km in altitude_array:
            daily_jd_arrays_to_loop = daily_jd_arrays_reduced if altitude_km>0 else daily_jd_arrays
            idxs_days_to_loop = selected_days_idxs if altitude_km>0 else range(n_days)
            
            for day_idx, jd_array in zip(idxs_days_to_loop, daily_jd_arrays_to_loop):
                
                print(f"Now computing: daily hist for day {day_idx} at {altitude_km}km")
                n_steps = len(jd_array)
                mid_day_jd = mid_day_jd_array[day_idx]
                TSI_1AU_day = rad_settings.get_TSI_1AU(mid_day_jd, rad_config["TSI_source"])
                R_ECEF_to_SunFrame_day_hist = get_R_SunFrame_hist(rad_config["ephemerides"], mid_day_jd)

                _, file_existences = get_file_names_and_existence(out_dir, altitude_km, day_idx)
                
                if altitude_km==0 and not(all(file_existences.values())):

                    toa_emission_day_hist, net_toa_day_hist = get_daily_hist_toa(out_dir, day_idx, mid_day_jd, TSI_1AU_day, rad_config,
                                                                                 lon_vec, lat_vec, n_steps)

                    save_toa_files(toa_emission_day_hist, net_toa_day_hist, grid_cell_areas_toa, out_dir, 
                                   day_idx, R_ECEF_to_SunFrame_day_hist, save_SFF_hist=(day_idx in selected_days_idxs))

                elif altitude_km>0 and not(all(file_existences.values())):

                    daily_hist_net_altitude = get_daily_hist_net_at_altitude(
                            out_dir, altitude_km, day_idx, mid_day_jd, TSI_1AU_day, rad_config, n_lon, n_lat, n_steps, n_cores
                        )
                    save_altitude_files(daily_hist_net_altitude, altitude_km, day_idx, out_dir, R_ECEF_to_SunFrame_day_hist)

                    # separate get vs compute daily function, work on SFF, check it all

                print("")

                    

def get_daily_hist_toa(data_dir, day_idx, mid_day_jd, TSI_1AU_day, rad_config, lon_vec, lat_vec, n_steps):

    all_file_names, file_existences = get_file_names_and_existence(data_dir, 0, day_idx)
    n_lon, n_lat = len(lon_vec), len(lat_vec)

    if not(file_existences["daily_hist_emission_toa"]) or not(file_existences["daily_hist_net_toa"]):  # requires recomputing
        
        print("Computing daily hist TOA...")
        toa_emission_day_hist, net_toa_day_hist = compute_daily_time_series_maps_toa(mid_day_jd, rad_config, lon_vec, lat_vec,
                                                    TSI_1AU_day)
    else:
        print("Loading daily hist TOA...")
        try:
            toa_emission_day_hist = np.load(all_file_names["daily_hist_emission_toa"]+'.npy')
        except:
            toa_emission_day_hist = np.loadtxt(all_file_names["daily_hist_emission_toa"]+'.txt').reshape((n_lat, n_lon, n_steps))
        
        try:
            net_toa_day_hist = np.load(all_file_names["daily_hist_net_toa"]+'.npy')
        except:
            net_toa_day_hist = np.loadtxt(all_file_names["daily_hist_net_toa"]+'.txt').reshape((n_lat, n_lon, n_steps))


    return toa_emission_day_hist, net_toa_day_hist


def get_daily_hist_net_at_altitude(data_dir, altitude_km, day_idx, mid_day_jd, TSI_1AU_day, rad_config, n_lon, n_lat, n_steps, n_cores):

    all_file_names, file_existences = get_file_names_and_existence(data_dir, altitude_km, day_idx)
    
    if not(file_existences['daily_hist_net_'+str(altitude_km)+'km']):
        daily_hist_net_altitude = compute_daily_hist_net_at_altitude(data_dir, altitude_km, day_idx, mid_day_jd, TSI_1AU_day, 
                                                                     rad_config, n_lon, n_lat, n_steps, n_cores)
    else:
        print(f"Loading daily hist net at {altitude_km} km...")
        try:
            daily_hist_net_altitude = np.load(all_file_names['daily_hist_net_'+str(altitude_km)+'km']+'.npy')
        except:
            daily_hist_net_altitude = np.loadtxt(all_file_names['daily_hist_net_'+str(altitude_km)+'km']+'.txt').reshape((n_lat, n_lon, n_steps))

    return daily_hist_net_altitude



def compute_daily_hist_net_at_altitude(data_dir, altitude_km, day_idx, mid_day_jd, TSI_1AU_day, rad_config, n_lon, n_lat, n_steps, n_cores):
    
    lon_vec, lat_vec, lon_edges_vec, lat_edges_vec = get_sphere_grid(n_lon, n_lat)
    stacked_grid_r_el, stacked_grid_u_el = get_stacked_spherical_grid_els(lon_vec, lat_vec, altitude_km, total_R=RE+altitude_km)     # stacked_grid_r_el is at altitude
    
    grid_r_el_toa = lonlat_to_r(lon_vec, lat_vec, 0, "spherical")
    grid_u_el = grid_r_el_toa / RE
    grid_cell_areas_toa = get_spherical_grid_cell_areas(lat_edges_vec, lon_edges_vec, RE)
    
    toa_emission_day_hist, _ = get_daily_hist_toa(data_dir, day_idx, mid_day_jd, TSI_1AU_day, rad_config, 
                                            lon_vec, lat_vec, n_steps)
    
    d_sun_day, u_sun_day = get_d_u_sun_hist(rad_config["ephemerides"], mid_day_jd)         # stacked_grid_u_el is independent of altitude
    zeroed_cos_theta_s_day_hist = get_zeroed_cos_theta_s_hist(stacked_grid_u_el, u_sun_day, altitude_km, n_lat, n_lon)
    
    TSI_Earth_day_vec = TSI_1AU_day * (AU/d_sun_day)**2
    solar_incoming_day_hist = zeroed_cos_theta_s_day_hist * TSI_Earth_day_vec[None, None, :]

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
    daily_hist_net_altitude = solar_incoming_day_hist - toa_emission_mapped_to_altitude_hist

    return daily_hist_net_altitude



def save_toa_files(toa_emission_day_hist, net_toa_day_hist, grid_cell_areas_toa, data_dir, day_idx, R_ECEF_to_SunFrame_day_hist, save_SFF_hist=True):
    
    surface_area_toa = np.sum(grid_cell_areas_toa)
    all_file_names, file_existences = get_file_names_and_existence(data_dir, 0, day_idx)
    
    if not(file_existences['daily_hist_emission_toa']):
        print("Saving daily_hist_emission_toa file...")
        np.save(all_file_names['daily_hist_emission_toa'], toa_emission_day_hist)

    if not(file_existences['daily_hist_net_toa']):
        print("Saving daily_hist_net_toa file...")
        np.save(all_file_names['daily_hist_net_toa'], net_toa_day_hist)
    
    if not(file_existences['daily_hist_net_toa_SFF']) and save_SFF_hist:
        print("Saving daily_hist_net_toa file Sun-Fixed Frame...")
        net_toa_day_hist_SFF = field_hist_rotation(net_toa_day_hist, R_ECEF_to_SunFrame_day_hist)
        np.save(all_file_names['daily_hist_net_toa_SFF'], net_toa_day_hist_SFF)

    if not(file_existences['daily_hist_net_toa_lat_avg']):
        print("Saving daily_hist_net_toa_lat_avg file...")
        toa_lat_avg_day_hist = np.mean(net_toa_day_hist, axis=1)    # mean works because element areas are constant at every latitude
        np.save(all_file_names['daily_hist_net_toa_lat_avg'], toa_lat_avg_day_hist)

    if not(file_existences['daily_hist_net_toa_surf_avg']):
        print("Saving daily_hist_net_toa_surf_avg file...")
        toa_surf_avg_day_hist = np.sum(net_toa_day_hist * grid_cell_areas_toa[:,:,None], axis=(0,1)) / surface_area_toa
        np.save(all_file_names['daily_hist_net_toa_surf_avg'], toa_surf_avg_day_hist)

    if not(file_existences['daily_avg_net_toa']):
        print("Saving daily_avg_net_toa file...")
        net_toa_daily_avg = np.mean(net_toa_day_hist, axis=2)
        np.save(all_file_names['daily_avg_net_toa'], net_toa_daily_avg)


def save_altitude_files(daily_hist_net_altitude, altitude_km, day_idx, data_dir, R_ECEF_to_SunFrame_day_hist):

    all_file_names, file_existences = get_file_names_and_existence(data_dir, altitude_km, day_idx)
    
    if not(file_existences['daily_hist_net_'+str(altitude_km)+'km']):
        print(f"Saving daily_hist_net_{altitude_km}km file...")
        np.save(all_file_names['daily_hist_net_'+str(altitude_km)+'km'], daily_hist_net_altitude)

    if not(file_existences['daily_hist_net_'+str(altitude_km)+'km_SFF']):
        print(f"Saving daily_hist_net_{altitude_km}km file Sun-Fixed Frame...")
        daily_hist_net_altitude_SFF = field_hist_rotation(daily_hist_net_altitude, R_ECEF_to_SunFrame_day_hist)
        np.save(all_file_names['daily_hist_net_'+str(altitude_km)+'km_SFF'], daily_hist_net_altitude_SFF)

    
    if not(file_existences['daily_avg_net_'+str(altitude_km)+'km']):  # TODO: exisitng daily_avg_net_800km files are wrong if memory doesn't fail
        print(f"Saving daily_avg_net_{altitude_km}km file...")
        daily_avg_net_altitude = np.mean(daily_hist_net_altitude, axis=2)
        np.save(all_file_names['daily_avg_net_'+str(altitude_km)+'km'], daily_avg_net_altitude)
    
    if not(file_existences['daily_avg_net_'+str(altitude_km)+'km_SFF']):
        # if it didn't exist we just saved it above so we can always load it:
        print(f"Saving daily_avg_net_{altitude_km}km_SFF file...")
        daily_hist_net_altitude_SFF = np.load(all_file_names['daily_hist_net_'+str(altitude_km)+'km_SFF']+'.npy')
        daily_avg_net_altitude_SFF = np.mean(daily_hist_net_altitude_SFF, axis=2)
        np.save(all_file_names['daily_avg_net_'+str(altitude_km)+'km_SFF'], daily_avg_net_altitude_SFF)



def compute_daily_time_series_maps_toa(mid_day_jd, rad_config, lon_vec, lat_vec,
                                       TSI_1AU_day):


    n_lon, n_lat = len(lon_vec), len(lat_vec)
    stacked_grid_r_el, stacked_grid_u_el = get_stacked_spherical_grid_els(lon_vec, lat_vec, 0, RE)     # stacked_grid_r_el is at altitude
    
    day_datestr = Time(mid_day_jd, format='jd').to_datetime().strftime("%Y-%m-%d")
    d_sun_day, u_sun_day = get_d_u_sun_hist(rad_config["ephemerides"], mid_day_jd)

    #TSI_1AU_day = rad_settings.get_TSI_1AU(mid_day_jd, rad_config["TSI_source"])
    TSI_Earth_day_vec = TSI_1AU_day * (AU/d_sun_day)**2

    a_map, e_map = get_expanded_ae_maps(rad_config["sh_mode"], day_datestr, lon_vec, lat_vec)
    zeroed_cos_theta_s_day_hist_toa = get_zeroed_cos_theta_s_hist(stacked_grid_u_el, u_sun_day, 0, n_lat, n_lon)
    
    solar_incoming_day_hist_toa = zeroed_cos_theta_s_day_hist_toa * TSI_Earth_day_vec[None, None, :]
    
    LW_outgoing_day_hist_toa = e_map[:, :, None] / 4 * TSI_Earth_day_vec[None, None, :]
    SW_outgoing_day_hist_toa = zeroed_cos_theta_s_day_hist_toa * a_map[:, :, None] * TSI_Earth_day_vec[None, None, :]
    
    toa_emission_day_hist = LW_outgoing_day_hist_toa + SW_outgoing_day_hist_toa
    net_toa_day_hist = solar_incoming_day_hist_toa - toa_emission_day_hist

    return toa_emission_day_hist, net_toa_day_hist


_global = {}

def init_worker(grid_r_el_toa, grid_u_el_toa, grid_cell_areas_toa, emission_toa_map):
    _global['grid_r_el_toa'] = grid_r_el_toa
    _global['grid_u_el_toa'] = grid_u_el_toa
    _global['grid_cell_areas_toa'] = grid_cell_areas_toa
    _global['net_toa_map'] = emission_toa_map
    


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




def get_file_names_and_existence(data_dir, alt_km, day_idx):
    
    all_out_file_types = required_files(alt_km)
    # create directories if they don't exist:
    for ftype in all_out_file_types:
        os.makedirs(os.path.join(data_dir, ftype), exist_ok=True)

    all_file_names = {ftype: os.path.join(data_dir, ftype, 'day_'+str(day_idx)) for ftype in all_out_file_types}
    file_existences = {ftype: (os.path.exists(fname_full+'.txt') or os.path.exists(fname_full+'.npy')) 
                        for ftype, fname_full in zip(all_out_file_types, all_file_names.values())}
    
    return all_file_names, file_existences

                

def get_expanded_ae_maps(mode, datestr, lon_vec, lat_vec):
                
    a_sh_map, e_sh_map = rad_settings.get_ae_sh_maps_numpy_new(mode, datestr)
    a_map = expand_sh_grid(a_sh_map, lon_vec, lat_vec)
    e_map = expand_sh_grid(e_sh_map, lon_vec, lat_vec)

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

def get_zeroed_cos_theta_s_hist(stacked_grid_u_el, u_sun_day, altitude_km, n_lat, n_lon):
    
    n_steps = np.shape(u_sun_day)[0]
    stacked_cos_theta_s_day_hist = stacked_grid_u_el @ np.transpose(u_sun_day)
    cos_theta_s_day_hist = np.reshape(stacked_cos_theta_s_day_hist, (n_lat, n_lon, n_steps))
    zeroed_cos_theta_s_day_hist = cos_theta_s_day_hist

    cos_lim = -np.sqrt(1 - (RE/(RE + altitude_km))**2)
    zeroed_cos_theta_s_day_hist[cos_theta_s_day_hist < cos_lim] = 0     # this also changes cos_theta_s_day_hist but inside the function it doesn't really matter

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