from sensitivity_basic_analysis import * # set_Am_factor_input_file, set_e_input_file, set_sma_extra_input_file, set_albedo_to_sw, set_albedo_to_lw
from mst import mst_old
from mst import mstloop
import time

AM_factor = 1
e = 0
extra_sma = 0

set_Am_factor_input_file(AM_factor)
set_e_input_file(e)
set_sma_extra_input_file(extra_sma)
set_albedo_to_both()

#mst_old.main('.sb_inputs')
mstloop.main('.sb_inputs')
lon_vec, lat_vec, t_vec_gt = get_lonlat()  # groundtracks
make_groundtrack_plot(lon_vec, lat_vec, '')

#run_CLI_command(["mpython", "mstloop.py", "sb_inputs"])