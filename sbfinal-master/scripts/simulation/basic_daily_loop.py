import sys, os
import shutil # shutil.rmtree(dir) to remove subdir and all its contents
#from SpaceBalls.sbsim2 import SpaceBallsSim as SpaceBallsSim2
from SpaceBalls.sbsim import SpaceBallsSim as SpaceBallsSim
import SpaceBalls.input_database_manager as input_db_manager
from SpaceBalls.paths import MEDIA_DIR, OUTPUT_DIR
from multiprocessing import Pool

n_days = 1
simulator = SpaceBallsSim(input_name='example_input', input_mode='file', verbose=True)

simulator.run_daily_loop(day_to_stop=n_days-1)

