#from hvt.config import DEFAULT_INERTIAL_FRAME
import Monte as M
import mpy.units as units
import math
#import warnings
#from pdb import set_trace as bp


class Sensor(object):

    # Initialize sensor parameters
    # only name is needed and other params can be set later (or never for loaded trajectories)

    def __init__(self, name, frame, trajBody): # 
        super().__init__()
        self.name = name
        self.frame = frame
        self.trajBody = trajBody

    def init_circSensor(self, boa, fov = 30.0* units.deg, apRadius = 1.0 *units.m):
        #creates a default circular sensor
        self.fov = fov
        self.apRadius = apRadius

        sensor = M.CircularSensor ( boa, self.name, self.frame, self.trajBody, self.fov, self.apRadius )

        print("Sensor created:")
        print("sensor.name:", sensor.name() )
        print("sensor.frame:", sensor.frame() )
        print("sensor.trajBody:", sensor.trajBody() )       
        print("sensor.fov:", sensor.fov() )
        print("sensor.apRadius:", sensor.radius() )

        return sensor


