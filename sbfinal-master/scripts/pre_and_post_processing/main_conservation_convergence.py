import numpy as np
import sys, os
from SpaceBalls.postproc_EEI_estimation import get_true_EEI_time_series, get_subdir_EEI_truth
from SpaceBalls.radiation_fluxes_preprocessing import mid_day_jd_array_from_jd_interval, get_daily_hist_net_at_altitude, compute_flux_hist_toa, get_r_sun_jd_hist, get_solar_incoming_day_hist, get_EEI_truth_daily_jd_arrays, F_vec_to_Fr, get_toa_grid_from_rad_config
from SpaceBalls.radiation_settings import get_TSI_1AU, radiation_settings_from_EEI_truth_name
from SpaceBalls.sph_meshing import QuadratureGrid, RegularLatLonGrid, Grid
from SpaceBalls.utils import normal_smoother
from SpaceBalls.plotter import Plotter
from SpaceBalls.paths import OUTPUT_DIR, MEDIA_DIR

def glq_n_to_l(n):
    return np.array(np.round(0.25 * (-3 + np.sqrt(1 + 8*n))), dtype=int)

# EEI truths with a/e max. D/O 179
EEI_names = ['EEI_truth_35', 'EEI_truth_3', 'EEI_truth_4'] # 'EEI_truth_4'
altitude_km = 800

"""
######################### debug block
EEI_name = "EEI_truth_3"
rad_config = radiation_settings_from_EEI_truth_name(EEI_name)
toa_grid = get_toa_grid_from_rad_config(rad_config, quadrature_order=251)
altitude_grid = QuadratureGrid(800, 225)
all_daily_jd_arrays = get_EEI_truth_daily_jd_arrays(rad_config['jd_interval'])
mid_day_jd_array = mid_day_jd_array_from_jd_interval(rad_config["jd_interval"])
solar_F_alt_hist_ECEF, earth_F_alt_hist_ECEF = get_daily_hist_net_at_altitude(
                            mid_day_jd_array[0], all_daily_jd_arrays[0], rad_config, altitude_grid, sff=False,
                            toa_grid=toa_grid, return_separate=True, compute_solar=False)
np.save(os.path.join(MEDIA_DIR, 'test_earth_F_alt_hist_new.npy'), earth_F_alt_hist_ECEF)
test_hist_og_func = np.load(os.path.join(MEDIA_DIR, 'test_earth_F_alt_hist_og.npy'))
print(f"Max diff: {np.max(np.abs(test_hist_og_func - earth_F_alt_hist_ECEF))}")
a-3 # force code crash
#########################
"""

regular_altitude_grids = [RegularLatLonGrid(altitude_km, n_lat=int(np.round(180*f)), 
                                                         n_lon=int(np.round(360*f)),
                                                         flattening=0) for f in np.logspace(-0.7, 0.5, 15)]
regular_altitude_grids = regular_altitude_grids[:-5]

n_points_glq = np.logspace(np.log10(regular_altitude_grids[0].n_points), 
                           np.log10(regular_altitude_grids[-1].n_points), 
                           len(regular_altitude_grids))
l_glq = glq_n_to_l(n_points_glq)
glq_altitude_grids = [RegularLatLonGrid(altitude_km, lmax=int(l), quad_type='GLQ',
                                       flattening=0) for l in l_glq]

womersley_altitude_grids = [QuadratureGrid(altitude_km, order=o, 
                                           default_womersley=True, 
                                           flattening=0) for o in np.arange(33, 325, 16)]

n_days = 1

for EEI_name in EEI_names:

    rad_config = radiation_settings_from_EEI_truth_name(EEI_name)
    all_daily_jd_arrays = get_EEI_truth_daily_jd_arrays(rad_config['jd_interval'])
    full_jd_array = np.concatenate(all_daily_jd_arrays[:n_days])
    mid_day_jd_array = mid_day_jd_array_from_jd_interval(rad_config["jd_interval"])

    regular_toa_grids = [get_toa_grid_from_rad_config(rad_config, deg_bin_size=d) for d in (10, 5, 2.5, 1)]
    n_points_reg_toa_grids = np.array([grid.n_points for grid in regular_toa_grids])
    glq_toa_grids = [get_toa_grid_from_rad_config(rad_config, glq_order=l) for l in glq_n_to_l(n_points_reg_toa_grids)]
    womersley_orders = np.array([51, 101, 151, 225, 325]) # np.round(np.sqrt(2*n_points_reg_toa_grids))
    womersley_toa_grids = [get_toa_grid_from_rad_config(rad_config, quadrature_order=int(l), default_womersley=True) 
                           for l in womersley_orders]

    all_grid_packs = ((regular_toa_grids, regular_altitude_grids), 
                      (glq_toa_grids, glq_altitude_grids),
                      (womersley_toa_grids, womersley_altitude_grids))

    for grid_pack in all_grid_packs:  # regular, GLQ, Womersley
        toa_grids, altitude_grids = grid_pack

        for day_idx in range(n_days):
            day_jd_array = all_daily_jd_arrays[day_idx]
            mid_day_jd = mid_day_jd_array[day_idx]
            out_dir = os.path.join(MEDIA_DIR, 'conservation_convergence', EEI_name, 'day_' + str(day_idx))
            os.makedirs(out_dir, exist_ok=True)

            for toa_idx, toa_grid in enumerate(toa_grids):
                first_toa_idx = (toa_idx==0)
                toa_earth_fname = os.path.join(out_dir, 'integrated_earth_toa_hist_'+toa_grid.grid_name+'.npy')
                toa_solar_fname = os.path.join(out_dir, 'integrated_solar_toa_hist_'+toa_grid.grid_name+'.npy')

                if not(os.path.exists(toa_earth_fname)) or not(os.path.exists(toa_solar_fname)):
                    print(f"Now computing: TOA grid {toa_grid.grid_name}")
                    toa_LW_hist, toa_SW_hist, net_toa_hist = compute_flux_hist_toa(full_jd_array, rad_config, toa_grid)
                    #toa_LW_hist, toa_SW_hist, net_toa_hist = np.zeros((toa_grid.n_points, len(day_jd_array))), np.zeros((toa_grid.n_points, len(day_jd_array))), np.zeros((toa_grid.n_points, len(day_jd_array)))
                    toa_earth_hist = -(toa_LW_hist + toa_SW_hist)
                    toa_solar_hist = net_toa_hist - toa_earth_hist

                    integrated_earth_toa_hist = toa_grid.compute_surf_integral(toa_earth_hist)
                    integrated_solar_toa_hist = toa_grid.compute_surf_integral(toa_solar_hist)

                    np.save(toa_earth_fname, integrated_earth_toa_hist)
                    np.save(toa_solar_fname, integrated_solar_toa_hist)


                for altitude_grid in altitude_grids[:-3]:
                    fname_earth = os.path.join(out_dir, 'integrated_earth_'+str(altitude_km)+'_hist_'+altitude_grid.grid_name + 'from_toa_' + toa_grid.grid_name + '.npy')
                    fname_solar = os.path.join(out_dir, 'integrated_solar_'+str(altitude_km)+'_hist_'+altitude_grid.grid_name)

                    if (not(os.path.exists(fname_earth))): # and (not(os.path.exists(fname_solar))):
                        print(f"Now computing: altitude grid {altitude_grid.grid_name} from  TOA grid {toa_grid.grid_name}")
                        solar_F_alt_hist_ECEF, earth_F_alt_hist_ECEF = get_daily_hist_net_at_altitude(
                            mid_day_jd, day_jd_array, rad_config, altitude_grid, sff=False,
                            toa_grid=toa_grid, return_separate=True, compute_solar=first_toa_idx)
                        
                        earth_Fr_alt_hist_ECEF = F_vec_to_Fr(earth_F_alt_hist_ECEF, altitude_grid.stacked_grid_u)
                        integrated_earth_alt_hist = altitude_grid.compute_surf_integral(earth_Fr_alt_hist_ECEF)
                        np.save(fname_earth, integrated_earth_alt_hist)

                        if first_toa_idx:
                            solar_Fr_alt_hist_ECEF = F_vec_to_Fr(solar_F_alt_hist_ECEF, altitude_grid.stacked_grid_u)
                            integrated_solar_alt_hist = altitude_grid.compute_surf_integral(solar_Fr_alt_hist_ECEF)
                            np.save(fname_solar, integrated_solar_alt_hist)
                                                
                        