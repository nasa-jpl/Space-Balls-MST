import sys, os
import numpy as np
from scipy.interpolate import make_interp_spline
import SpaceBalls.output_database_manager as db_manager
import SpaceBalls.input_database_manager as input_manager
from SpaceBalls.plotter import Plotter
#from SpaceBalls.postproc_EEI_estimation import generate_constellation_stacking_animations
from SpaceBalls.radiation_fluxes_preprocessing import get_toa_grid_from_rad_config, get_EEI_truth_daily_jd_arrays, get_mid_day_jd_array, mid_day_jd_array_from_jd_interval, compute_F_hist_at_sat_r_hist
from SpaceBalls.radiation_settings import radiation_settings_from_EEI_truth_name
from SpaceBalls.postproc_EEI_estimation import get_true_EEI_time_series, get_acc_to_flux_factor
from SpaceBalls.sph_meshing import QuadratureGrid
from SpaceBalls.utils import normal_smoother
from SpaceBalls.paths import CONFIG_DIR, MEDIA_DIR, OUTPUT_DIR
from multiprocessing import Pool

# order_toa_grid = 131 # 131
# toa_grid = QuadratureGrid(alt_km=0, order=order_toa_grid)

recomp_EEI_truth_name = "EEI_truth_05"
toa_grid = get_toa_grid_from_rad_config(radiation_settings_from_EEI_truth_name(
    recomp_EEI_truth_name
    ))

new_step_sec = None

def recompute_sb(sb_tag):

    input_sb_dict = input_manager.get_dicts(sb_tag)
    EEI_truth_name = "EEI_truth_" + str(input_sb_dict['force_settings']['EEI_truth'])
    if (recomp_EEI_truth_name is not None) and (recomp_EEI_truth_name!=EEI_truth_name):
        # recomputation of acc to be made with a different EEI truth!
        EEI_tag = '_' + recomp_EEI_truth_name
        EEI_name_to_use = recomp_EEI_truth_name
    else:
        EEI_tag = ''

    rad_config = radiation_settings_from_EEI_truth_name(EEI_truth_name)
    mid_day_jd_array = mid_day_jd_array_from_jd_interval(rad_config["jd_interval"])

    delta_t_orig = input_sb_dict['delta_t_out']
    if (new_step_sec is not None) and (new_step_sec != delta_t_orig):
        # new sb entry!
        new_sb_bool = True
        new_sb_tag = input_manager.insert_sb(orbit_name=input_sb_dict['orbit']['orbit_name'],
                                             sc_name=input_sb_dict['sc_params']['sc_name'],
                                             force_settings_num=input_sb_dict['force_settings']['settings_num'],
                                             integration_settings_num=input_sb_dict['integration_settings']['settings_num'],
                                             delta_t_out=new_step_sec)
        print(f"New sb entry is {new_sb_tag}")
    else:
        new_sb_bool = False

    for day_idx, mid_day_jd in enumerate(mid_day_jd_array):
        print(f"SB {sb_tag}; EEI {EEI_name_to_use}: computing day {day_idx}/{len(mid_day_jd_array)}")

        r_hist_day = np.load(os.path.join(OUTPUT_DIR, sb_tag, 'day_'+str(day_idx), 'xyz_ecef.npy'))
        jd_hist_day = np.load(os.path.join(OUTPUT_DIR, sb_tag, 'day_'+str(day_idx), 'jd_vec.npy'))
        
        if new_sb_bool:
            spline = make_interp_spline(jd_hist_day, r_hist_day)
            new_step_days = new_step_sec / (24 * 3600)
            new_jd_hist_day = np.arange(jd_hist_day[0], jd_hist_day[-1], new_step_days)
            new_r_hist_day = spline(new_jd_hist_day, extrapolate=True)
            r_hist_day_to_use = new_r_hist_day
            jd_hist_day_to_use = new_jd_hist_day
        else:
            r_hist_day_to_use = r_hist_day
            jd_hist_day_to_use = jd_hist_day
        
        recomp_erp_F_hist, srp_F_hist = compute_F_hist_at_sat_r_hist(recomp_EEI_truth_name, #'EEI_truth_1', 
                                                     r_hist_day_to_use, 
                                                     jd_hist_day_to_use, 
                                                     toa_grid=toa_grid, erp_wl_split=False)
        factor = get_acc_to_flux_factor(sb_tag)
        if new_sb_bool: assert(factor==get_acc_to_flux_factor(new_sb_tag))

        acc_erp_hist = np.squeeze(recomp_erp_F_hist) / factor * 1e-3
        acc_srp_hist = np.squeeze(srp_F_hist) / factor * 1e-3

        erp_fname = 'erp_recomp_'+ toa_grid.grid_name + EEI_tag
        srp_fname = 'srp_recomp_' + EEI_tag

        if not(new_sb_bool):
            np.save(os.path.join(OUTPUT_DIR, sb_tag, 'day_'+str(day_idx), erp_fname), acc_erp_hist)
            np.save(os.path.join(OUTPUT_DIR, sb_tag, 'day_'+str(day_idx), srp_fname), acc_srp_hist)
        else:
            new_out_dir = os.path.join(OUTPUT_DIR, new_sb_tag, 'day_'+str(day_idx))
            os.makedirs(new_out_dir, exist_ok=True)
            np.save(os.path.join(new_out_dir, erp_fname), acc_erp_hist)
            np.save(os.path.join(new_out_dir, srp_fname), acc_srp_hist)
            np.save(os.path.join(new_out_dir, 'jd_vec.npy'), new_jd_hist_day)
            np.save(os.path.join(new_out_dir, 'xyz_ecef.npy'), new_r_hist_day)
            
            aero_hist_day = np.load(os.path.join(OUTPUT_DIR, sb_tag, 'day_'+str(day_idx), 'aero.npy'))
            spline = make_interp_spline(jd_hist_day, aero_hist_day)
            new_aero_hist_day = spline(new_jd_hist_day, extrapolate=True)
            np.save(os.path.join(new_out_dir, 'aero.npy'), new_aero_hist_day)

            r_hist_day_SFF = np.load(os.path.join(OUTPUT_DIR, sb_tag, 'day_'+str(day_idx), 'xyz_sun_frame.npy'))
            spline = make_interp_spline(jd_hist_day, r_hist_day_SFF)
            new_r_hist_day_SFF = spline(new_jd_hist_day, extrapolate=True)
            np.save(os.path.join(new_out_dir, 'xyz_sun_frame.npy'), new_r_hist_day_SFF)
        
        

#orbits = ['A1', 'C2', 'C3'] #['G3', 'A1', 'F12'] # ['C1', 'I3', 'F2']

# mix of best constellations:
orbits = ['I3', 'A1', 'F2', 'G3', 'F12', 'E1', 'E3', 'I2', 'C2'
          ]
#orbits = ['I3', 'A1', 'F2'] # best 1-yr window         ['7b54acc8701b0625', 'c1dd568b730da76c', '82c0c7e77260bb76']
#orbits = ['G3', 'A1', 'F12'] # 2nd best 1-yr window
#orbits = ['E3', 'E1', 'C2'] # best 1-month window
#orbits = ['E1', 'I2'] # best 1-yr 2 sats
#orbits = ['E1', 'E3'] # best 1-mo 2 sats

sb_to_recompute = np.concatenate([input_manager.get_primary_keys({'orbit_name': orb,
                                                    'force_settings': 2,
                                                    'SC_name': 'SC_0',
                                                    'force_settings': 2,
                                                    'integration_settings': 0,
                                                    'delta_t_out': 15
                                                    }) for orb in orbits])

if __name__ == "__main__":
    with Pool(processes=4) as pool:
        pool.map(recompute_sb, sb_to_recompute)




#for i, sb_tag in enumerate(ordered_sb):
#    recompute_sb(sb_tag)
