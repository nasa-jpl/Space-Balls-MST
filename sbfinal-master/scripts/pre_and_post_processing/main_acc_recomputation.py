import sys, os
import numpy as np
from scipy.interpolate import make_interp_spline
import SpaceBalls.output_database_manager as db_manager
import SpaceBalls.input_database_manager as input_manager
from SpaceBalls.plotter import Plotter
#from SpaceBalls.postproc_EEI_estimation import generate_constellation_stacking_animations
from SpaceBalls.radiation_fluxes_preprocessing import get_EEI_truth_daily_jd_arrays, get_mid_day_jd_array, mid_day_jd_array_from_jd_interval, compute_daily_hist_at_sat_r_hist
from SpaceBalls.radiation_settings import radiation_settings_from_EEI_truth_name
from SpaceBalls.postproc_EEI_estimation import get_true_EEI_time_series, get_acc_to_flux_factor
from SpaceBalls.sph_meshing import QuadratureGrid
from SpaceBalls.utils import normal_smoother
from SpaceBalls.paths import CONFIG_DIR, MEDIA_DIR, OUTPUT_DIR
from multiprocessing import Pool

order_toa_grid = 131 # 131
toa_grid = QuadratureGrid(alt_km=0, order=order_toa_grid)
new_step_sec = None
recomp_EEI_truth_name = "EEI_truth_0"

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
        
        recomp_erp_Fr_hist, recomp_srp_Fr_hist = compute_daily_hist_at_sat_r_hist(EEI_name_to_use, 
                                                            r_hist_day_to_use, jd_hist_day_to_use, 
                                                            day_idx, toa_grid=toa_grid)
        factor = get_acc_to_flux_factor(sb_tag)
        if new_sb_bool: assert(factor==get_acc_to_flux_factor(new_sb_tag))

        radial_acc_erp_hist = recomp_erp_Fr_hist / factor * 1e-3
        radial_acc_srp_hist = recomp_srp_Fr_hist / factor * 1e-3

        erp_fname = 'erp_recomp_radial_'+str(order_toa_grid) + EEI_tag
        srp_fname = 'srp_recomp_radial' + EEI_tag

        if not(new_sb_bool):
            np.save(os.path.join(OUTPUT_DIR, sb_tag, 'day_'+str(day_idx), erp_fname), radial_acc_erp_hist)
            np.save(os.path.join(OUTPUT_DIR, sb_tag, 'day_'+str(day_idx), srp_fname), radial_acc_srp_hist)
        else:
            new_out_dir = os.path.join(OUTPUT_DIR, new_sb_tag, 'day_'+str(day_idx))
            os.makedirs(new_out_dir, exist_ok=True)
            np.save(os.path.join(new_out_dir, erp_fname), radial_acc_erp_hist)
            np.save(os.path.join(new_out_dir, srp_fname), radial_acc_srp_hist)
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
        
        

orbits = ['A1', 'C2', 'C3'] #['G3', 'A1', 'F12'] # ['C1', 'I3', 'F2']
sb_to_recompute = np.concatenate([input_manager.get_primary_keys({'orbit_name': orb,
                                                    'force_settings': 2,
                                                    'integration_settings': 0,
                                                    'delta_t_out': 15
                                                    }) for orb in orbits])

if __name__ == "__main__":
    with Pool(processes=3) as pool:
        pool.map(recompute_sb, sb_to_recompute)

#for i, sb_tag in enumerate(ordered_sb):
#    recompute_sb(sb_tag)
