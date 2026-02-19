import Monte as M
import mpy.units as units
import math

from .gsdef import Groudstation

def set_gs(boa):

    gsname = "Altadena"
    lon =241.8687*units.deg
    lat = 34.1902*units.deg
    alt = 0.48768*units.km

    gs = Groudstation( boa, gsname, lon, lat, alt )
    gs.loadtraj(boa)

    return gs

