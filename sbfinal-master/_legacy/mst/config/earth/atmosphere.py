import Monte as M
import mpy.units as units

#from mst import config
#from ..config_default import *
from ..files import AtmModel

def set_atmos(): #boa):

    # Earth - atmosphere file

    #AtmModel = '/home/reynerso/Python/CurF10_extended.txt'
    #AtmModel = '/home/reynerso/SphereX/CurF10_Apr2021.txt'
    #AtmModel = '../../../Users/reynerso/Documents/Monte/CurF10_Apr2021.txt'
    
    from ..files import AtmModel
    SolarFluxMag = 0.50

    return AtmModel, SolarFluxMag


