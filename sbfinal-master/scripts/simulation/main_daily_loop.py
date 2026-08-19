import sys, os
import shutil # shutil.rmtree(dir) to remove subdir and all its contents
#from SpaceBalls.sbsim2 import SpaceBallsSim as SpaceBallsSim2
from SpaceBalls.sbsim import SpaceBallsSim as SpaceBallsSim
import SpaceBalls.input_database_manager as input_db_manager
from SpaceBalls.paths import MEDIA_DIR, OUTPUT_DIR
from multiprocessing import Pool

n_days = 2

def run_sb(sb_hash):
    print(f"Running SpaceBalls simulation for input hash: {sb_hash}")
    simulator = SpaceBallsSim(input_name=sb_hash, input_mode='hash')
    simulator.run_daily_loop(day_to_stop=n_days)

n_facets_vec = [6, 24, 48, 72, 96, 120, 180, 280, 450]
keys = []

for i, n_facets in enumerate(n_facets_vec):
    desired_sb = {
        'force_settings': 2,
        'integration_settings': 0,
        'delta_t_out': 15,
        'sc_name': 'SC_0_'+str(n_facets),
    }
    key_arr = input_db_manager.get_primary_keys(desired_sb)
    for key in key_arr:
        if not(os.path.exists(os.path.join(OUTPUT_DIR, key, 'day_0', 'erp.npy'))):
            keys += [key]

#run_sb('GRACE-FO_spherical')
run_sb('SB_R800_35_101')

#keys = input_db_manager.get_primary_keys(desired_sb)
#keys = [key for key in keys if 'LAGEOS' not in key]
#keys = ['LAGEOS_' + str(100+i) for i in range(20)]

##### UNCOMMENT THIS #####
#if __name__ == "__main__":
#    with Pool(processes=9) as pool:
#        pool.map(run_sb, keys)
##########################




#simulator = SpaceBallsSim(input_name='case_test_1')
#simulator = SpaceBallsSim2(input_names=['case_test_1', 'case_test_2', 'case_test_3', 'case_test_4'])
#simulator.run_daily_loop()
