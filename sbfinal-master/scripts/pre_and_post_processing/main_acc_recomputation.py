import sys, os
import numpy as np
import pickle
from scipy.interpolate import make_interp_spline
import SpaceBalls.output_database_manager as db_manager
import SpaceBalls.input_database_manager as input_manager
from SpaceBalls.plotter import Plotter
#from SpaceBalls.postproc_EEI_estimation import generate_constellation_stacking_animations
from SpaceBalls.radiation_fluxes_preprocessing import DailyAEMapProvider, compute_a_hist_at_sat_r_hist, get_toa_grid_from_rad_config, get_EEI_truth_daily_jd_arrays, get_mid_day_jd_array, mid_day_jd_array_from_jd_interval, compute_F_hist_at_sat_r_hist
from SpaceBalls.radiation_settings import radiation_settings_from_EEI_truth_name
from SpaceBalls.postproc_EEI_estimation import get_true_EEI_time_series, get_acc_to_flux_factor, check_EEI_consistency
from SpaceBalls.sph_meshing import QuadratureGrid
from SpaceBalls.utils import normal_smoother, get_norm_across_last_dim, get_Rotation
from SpaceBalls.paths import CONFIG_DIR, MEDIA_DIR, OUTPUT_DIR
from multiprocessing import Pool

# order_toa_grid = 131 # 131
# toa_grid = QuadratureGrid(alt_km=0, order=order_toa_grid)

recomp_EEI_truth_name = "EEI_truth_0"
rad_config = radiation_settings_from_EEI_truth_name(recomp_EEI_truth_name)
toa_grid = get_toa_grid_from_rad_config(rad_config)
ae_map_provider = DailyAEMapProvider(toa_grid, rad_config)

new_step_sec = None



def recompute_multiple_sb(sb_tags):

    # EEI_truth_name = check_EEI_consistency(sb_tags)
    # if (recomp_EEI_truth_name is not None) and (recomp_EEI_truth_name!=EEI_truth_name):
    #     # recomputation of acc to be made with a different EEI truth!
    #     EEI_tag = '_' + recomp_EEI_truth_name
    #     EEI_name_to_use = recomp_EEI_truth_name
    # else:
    #     EEI_tag = ''
    EEI_tag = '_' + recomp_EEI_truth_name

    rad_config = radiation_settings_from_EEI_truth_name(recomp_EEI_truth_name)
    mid_day_jd_array = mid_day_jd_array_from_jd_interval(rad_config["jd_interval"])

    new_sb_array = [None] * len(sb_tags)
    sat_dicts = [None] * len(sb_tags)
    spin_axes = [None] * len(sb_tags)
    for i, sb_tag in enumerate(sb_tags):
        input_sb_dict = input_manager.get_dicts(sb_tag)
        delta_t_orig = input_sb_dict['delta_t_out']
        if (new_step_sec is not None) and (new_step_sec != delta_t_orig):
            # new sb entry!
            new_sb_array[i] = True
            new_sb_tag = input_manager.insert_sb(orbit_name=input_sb_dict['orbit']['orbit_name'],
                                                 sc_name=input_sb_dict['sc_params']['sc_name'],
                                                 force_settings_num=input_sb_dict['force_settings']['settings_num'],
                                                 integration_settings_num=input_sb_dict['integration_settings']['settings_num'],
                                                 delta_t_out=new_step_sec)
            print(f"New sb entry is {new_sb_tag}")

        else:
            new_sb_array[i] = False

        # load sat_dict
        sc_name = input_sb_dict['sc_params']['sc_name']
        sc_dict_fname = os.path.join(CONFIG_DIR, 'spacecraft', sc_name+'.pkl')
        if os.path.exists(sc_dict_fname):
            with open(sc_dict_fname, "rb") as f:
                sat_dict = pickle.load(f)
        else:
            print("sat dict is standard cannonball")
            sat_dict = {   
                'mass': input_sb_dict['sc_params']['mass'], # TODO: check that a single consistent SC dict is possible
                'area': input_sb_dict['sc_params']['area'],
                'coeff_wl_split': False,
                'cannonball': True,
                'ca_SW': 1, # TODO: make consistent with MONTE coeffs in the database
                'cs_SW': 0,
                'cd_SW': 0
            }
        sat_dicts[i] = sat_dict
        spin_axes[i] = input_sb_dict['spin_axis']

    if all(d == sat_dicts[0] for d in sat_dicts): # temporary patch?
        sat_dicts = sat_dicts[0]
        print("all sat dicts equal - select the first one")

    theta_0 = np.zeros(len(sb_tags))
    for day_idx, mid_day_jd in enumerate(mid_day_jd_array):
        print(f"SB {sb_tag}; EEI {recomp_EEI_truth_name}: computing day {day_idx}/{len(mid_day_jd_array)}")

        all_r_hist_day = [None] * len(sb_tags)
        all_jd_hist_day = [None] * len(sb_tags)
        all_h_hist_day = [None] * len(sb_tags)
        for i, sb_tag in enumerate(sb_tags):
            input_sb_dict = input_manager.get_dicts(sb_tag)
            r_hist_fname = os.path.join(OUTPUT_DIR, sb_tag, 'day_'+str(day_idx), 'xyz_ecef.npy')
            if os.path.exists(r_hist_fname):
                r_hist_day = np.load(r_hist_fname)
                jd_hist_day = np.load(os.path.join(OUTPUT_DIR, sb_tag, 'day_'+str(day_idx), 'jd_vec.npy'))
                if spin_axes[i] is not None:
                    unit_h_hist_day = np.load(os.path.join(OUTPUT_DIR, sb_tag, 'day_'+str(day_idx), 'hxhyhz_ecef.npy'))
                else:
                    unit_h_hist_day = None                                        
            else:
                # get r_hist for same orbit with SC_0
                alt_sb_key = input_manager.get_primary_keys({'orbit_name': input_sb_dict['orbit']['orbit_name'],
                                                            'force_settings': input_sb_dict['force_settings']['settings_num'],
                                                            'SC_name': 'SC_0',
                                                            'integration_settings': input_sb_dict['integration_settings']['settings_num'],
                                                            'delta_t_out': input_sb_dict['delta_t_out']
                                                            })
                assert(len(alt_sb_key)==1)
                alt_sb_key = alt_sb_key[0]
                r_hist_fname = os.path.join(OUTPUT_DIR, alt_sb_key, 'day_'+str(day_idx), 'xyz_ecef.npy')
                r_hist_day = np.load(r_hist_fname)
                jd_hist_day = np.load(os.path.join(OUTPUT_DIR, alt_sb_key, 'day_'+str(day_idx), 'jd_vec.npy'))
                if spin_axes[i] is not None:
                    unit_h_hist_day = np.load(os.path.join(OUTPUT_DIR, alt_sb_key, 'day_'+str(day_idx), 'hxhyhz_ecef.npy'))

            if new_sb_array[i]:
                spline = make_interp_spline(jd_hist_day, r_hist_day)
                new_step_days = new_step_sec / (24 * 3600)
                new_jd_hist_day = np.arange(jd_hist_day[0], jd_hist_day[-1], new_step_days)
                new_r_hist_day = spline(new_jd_hist_day, extrapolate=True)
                all_r_hist_day[i] = new_r_hist_day
                all_jd_hist_day[i] = new_jd_hist_day
                spline_h = make_interp_spline(jd_hist_day, unit_h_hist_day)
                new_unit_h_hist_day = spline_h(new_jd_hist_day, extrapolate=True)
                all_h_hist_day[i] = new_unit_h_hist_day
            else:
                all_r_hist_day[i] = r_hist_day
                all_jd_hist_day[i] = jd_hist_day
                all_h_hist_day[i] = unit_h_hist_day

        assert(all(np.array_equal(all_jd_hist_day[0], a) for a in all_jd_hist_day[1:]))
        jd_hist_day_to_use = all_jd_hist_day[0]
        r_hist_day_to_use = np.stack(all_r_hist_day, axis=0)
        

        """
        # old method: cannonball computation here
        print(f"Computing cannonball here...")
        all_recomp_erp_F_hist, all_recomp_srp_F_hist = compute_F_hist_at_sat_r_hist(recomp_EEI_truth_name, #'EEI_truth_1', 
                                                     r_hist_day_to_use, 
                                                     jd_hist_day_to_use, 
                                                     toa_grid=toa_grid, erp_wl_split=False)
        all_recomp_erp_a_hist = [None] * len(sb_tags)
        all_recomp_srp_a_hist = [None] * len(sb_tags)
        
        for i, (sb_tag, recomp_erp_F, recomp_srp_F) in enumerate(zip(sb_tags, all_recomp_erp_F_hist, all_recomp_srp_F_hist)):

            factor = get_acc_to_flux_factor(sb_tag)
            if new_sb_array[i]: assert(factor==get_acc_to_flux_factor(new_sb_tag))

            all_recomp_erp_a_hist[i] = np.squeeze(recomp_erp_F) / factor * 1e-3
            all_recomp_srp_a_hist[i] = np.squeeze(recomp_srp_F) / factor * 1e-3
        """

        # new method: external function computes accelerations with any sat model
        print(f"Computing accelerations...")
        if any(axis is not None for axis in spin_axes):
            sat_R_body2ECEF_hists = [None] * len(sb_tags)
            for i, sb_tag in enumerate(sb_tags):
                input_sb_dict = input_manager.get_dicts(sb_tag)
                if spin_axes[i]=='h':
                    rot_axis_hist = all_h_hist_day[i] / get_norm_across_last_dim(all_h_hist_day[i])[:,None]
                
                spin_w = 2 * np.pi * input_sb_dict['spin_f']
                theta = theta_0[i] + spin_w * (jd_hist_day_to_use - jd_hist_day_to_use[0]) * (24*3600)
                sat_R_body2ECEF_hists[i] = get_Rotation(rot_axis=rot_axis_hist.T, theta_hist=theta).as_matrix()
                theta_0[i] = theta[-1] + np.mean(np.diff(theta))
        else:
            sat_R_body2ECEF_hists = None

        all_recomp_erp_a_hist, all_recomp_srp_a_hist = compute_a_hist_at_sat_r_hist(
                                                            recomp_EEI_truth_name, #'EEI_truth_1', 
                                                            sat_dicts,
                                                            r_hist_day_to_use, 
                                                            jd_hist_day_to_use, 
                                                            toa_grid=toa_grid,
                                                            map_provider=ae_map_provider,
                                                            sat_R_body2ECEF_hists=sat_R_body2ECEF_hists)
        # 1e-3 is because need to store in kms-2; negative sign to preserve consistency with legacy bug
        all_recomp_erp_a_hist = - all_recomp_erp_a_hist * 1e-3
        all_recomp_srp_a_hist = - all_recomp_srp_a_hist * 1e-3

        for i, (sb_tag, acc_erp_hist, acc_srp_hist) in enumerate(zip(sb_tags, all_recomp_erp_a_hist, all_recomp_srp_a_hist)):

            erp_fname = 'erp_recomp_'+ toa_grid.grid_name + EEI_tag
            srp_fname = 'srp_recomp_' + EEI_tag

            if not(new_sb_array[i]):
                out_dir = os.path.join(OUTPUT_DIR, sb_tag, 'day_'+str(day_idx))
                os.makedirs(out_dir, exist_ok=True)
                np.save(os.path.join(out_dir, erp_fname), acc_erp_hist)
                np.save(os.path.join(out_dir, srp_fname), acc_srp_hist)
            
            else: # STILL TO BE REVISED IN THIS MULTISAT CASE
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
        print(acc_erp_hist[:4])
        a-3

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
#orbits = ['I3', 'A1', 'F2', 'G3', 'F12', 'E1', 'E3', 'I2', 'C2']
orbits = ['A1', 'A2', 'A3', 'D2']
#orbits = ['I3', 'A1', 'F2'] # best 1-yr window         ['7b54acc8701b0625', 'c1dd568b730da76c', '82c0c7e77260bb76']
#orbits = ['G3', 'A1', 'F12'] # 2nd best 1-yr window
#orbits = ['E3', 'E1', 'C2'] # best 1-month window
#orbits = ['E1', 'I2'] # best 1-yr 2 sats
#orbits = ['E1', 'E3'] # best 1-mo 2 sats


sb_to_recompute_kep = np.concatenate([input_manager.get_primary_keys({'orbit_name': orb,
                                                    'force_settings': 100,
                                                    'SC_name': 'SC_0',
                                                    'integration_settings': 0,
                                                    'delta_t_out': 15
                                                    }) for orb in orbits])

# sb_to_recompute_non_kep = input_manager.get_primary_keys({'orbit_name': 'D2',
#                                                     'force_settings': 2,
#                                                     'SC_name': 'SC_0',
#                                                     'integration_settings': 0,
#                                                     'delta_t_out': 15
#                                                     })
# sb_to_recompute = np.concatenate((sb_to_recompute_kep, sb_to_recompute_non_kep))


# sc_keys = input_manager.get_primary_keys({'accurate_coeff_source': 'GRACE_handbook'}, 
#                                       'sc_table', 'sc_name')
# sb_keys = [input_manager.get_primary_keys({'orbit_name': 'D2', 
#                                           'SC_name': sc_key 
#                                           }) for sc_key in sc_keys]
# sb_to_recompute = np.squeeze(sb_keys)

recompute_multiple_sb(sb_to_recompute_kep)

"""
if __name__ == "__main__":
    with Pool(processes=3) as pool:
        pool.map(recompute_sb, sb_to_recompute)
"""



#for i, sb_tag in enumerate(ordered_sb):
#    recompute_sb(sb_tag)

"""
[[-5.85973766e-12 -1.82258130e-12  2.39104080e-12]
 [-5.81950542e-12 -1.92831956e-12  2.40905496e-12]
 [-5.77357834e-12 -2.02979179e-12  2.42589906e-12]
 [-5.72354623e-12 -2.13188719e-12  2.43914543e-12]]
"""