import Monte as M
import mpy.units as units

from mst import config
#from ...config_default import *

def set_grav(): 

    # Earth - gravity file
    #import config.earth.gravity.ggm02c_monte_grv as GGM02C
    from . import ggm02c_monte_grv as GGM02C

    return GGM02C
