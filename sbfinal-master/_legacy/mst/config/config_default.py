# -*- coding: utf-8 -*-

import Monte as M
import mpy.units as units

import os, sys
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
#print('os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir):', os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
#print('os.path.dirname(__file__): ', os.path.dirname(__file__))
#print('os.pardir: ',os.pardir)
sys.path.insert(0, os.path.dirname(__file__)) # gives the path for the default boa file to be at the same level as this file

# default spacecraft name
DEFAULT_SC_NAME = "Spacecraft"

# Simulation timing defaults
SIM_START = M.Epoch('2021-Apr-1 00:00:00 UTC')
SIM_DURATION = 7.0 #days, 0.041666 = 1/24 so 1 hr

# default boa files
#Mac:'../../../Users/reynerso/Documents/Python/mst/mst/config/default.boa'   #"mst/config/default.boa"
#PC: r'/home/monte/Documents/Python/Spaceball/sb/mst/config/default.boa' 
# for location 'C:\Users\reynerso\Documents\Python\Spaceball\sb\mst\config\default.boa' 
DEFAULT_BOA = r'default.boa'

DEFAULT_BOA_OUT = DEFAULT_SC_NAME + '_out.boa'

# default ephemeris file
DEFAULT_EPHEMERIS_FILE = ""
DEFAULT_STK_OUT = DEFAULT_SC_NAME + '_stkout.e'

# date format (for output files)
DATE_FORMAT_FILENAMES = "$YYYY-$MM-$DD$UTC"
DATE_FORMAT_OUTPUTS = "$YYYY-$MM-$DD $HR:$MN:$SC(####)$UTC UTC"

# radius and flattening factor of the Earth
EARTH_RADIUS = 6378.137 * units.km
EARTH_FLATTENING = 0.00335281066475 # incorrect default? : 0.005886.  Also:  0.081819301 = earth eccentricity

# terminator (solar elevation)
TERMINATOR = 0 * units.deg

# minimum sun pointing angle default
SUN_POINTING_ANGLE_MIN = None

# default comm sites
COMM_SITES = ["Comm. Site 1"]

# default frames
DEFAULT_INERTIAL_FRAME = "Earth Space True Equator"
DEFAULT_EARTH_FIXED_FRAME = "Earth UT1 True Equator"
DEFAULT_GEODETIC_FRAME = "IAU Earth Fixed"

# default type of eclipse
# options: M.ShadowEvent.IN_UMBRA - umbra only
#          M.Shadow_Event.IN_PENUMBRA - penumbra only
#          M.ShadowEvent.IN_SHADOW - either umbra or penumbra
DEFAULT_ECLIPSE_TYPE = M.ShadowEvent.IN_UMBRA

# step size when searching EventSpecs
SEARCH_STEP_SIZE = 5 * units.minute

# preferred units for reports and plots
PREFERRED_UNITS = {"time": "sec",
                   "mass": "kg",
                   "length": "km",
                   "angle": "deg",
                   "duration": "hour"
                   }

# desired outputs
OUTPUTS = {"report": True, "pass_summary": True, "plots": True}

# specify grid for Earth polyhedron shape: # of segments in Latitude and Longitude
# ie LAT = 18 ; LONG = 36 implies 10 x 10 degree main grid
EARTH_POLY_GRID_LAT = 55   # 72  #60  #45 # 36  #18 # 90 #180
EARTH_POLY_GRID_LONG = 110 #144  #120 #90 # 72 # 36 # 180 #360

# Constants
LIGHT_SPEED = 2.998E8  #speed of light, units in m/s
AU = 149597870.691 #astronomical unit, units in km