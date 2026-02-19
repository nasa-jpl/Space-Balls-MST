#from hvt.config import DEFAULT_INERTIAL_FRAME
import Monte as M
import mpy.units as units
#import math
#import warnings
#from pdb import set_trace as bp


class Groudstation(object):

    # Initialize ground station parameters - fixed long & lat
    # only name is needed and other params can be set later (or never for loaded trajectories)

    def __init__(self, boa, name, lon = 0.0 *units.deg, lat = 0.0 * units.deg, alt = 0.0 * units.km ):
        super().__init__()
        self.name = name
        self.lon = lon
        self.lat = lat
        self.alt = alt
        self.loadtraj(boa)

    def loadtraj(self, boa):

        # earth_shape = M.ShapeBoa.read( boa, "Earth" ) - shape already loaded in config.earth.shape.py
        '''
        gsname = "Altadena"
        gslong =241.8687*deg
        gslat = 34.1902*deg
        gsalt = 0.48768*km
        '''
        gs_state = M.State( boa, self.name,"Earth", M.Geodetic.longitude(self.lon), M.Geodetic.latitude(self.lat), M.Geodetic.height(self.alt))
        
        gstraj = M.EarthStnTraj( boa, self.name, gs_state )
        gstraj.setComplex("")
        
        '''
        now the boa is setup to do queries like these:
        gstqEME2000 = TrajQuery( boa , self.name, 'Earth', 'EME2000' )
        gsToSC = M.TrajQuery( boa, self.name, scname, 'EME2000' )
        '''