#  User input file         
#
#
#from astroalgs9p3 import *
#from quaternion3p3 import *
#from tkinter import filedialog #tkFileDialog
import Monte as M
#import mpy.io.data as defaultData
#import mpy.traj.force.grav.basic as basicGrav
from mpy.units import *
#from mst import config
from mst.config.time_handling import covert_Time_to_Epoch as convertTime


TIMESTART = {'year':2022,
        'month':3,
        'day':21,
        'hour':0,
        'minute':0,
        'second':0.0,
        }

TIMEFINISH = {'year':2022,
        'month':3,
        'day':23,
        'hour':0,
        'minute':0,
        'second':0.0,
        }

INPUT = { 
        'name': "Lageos",        
        'mass': 100.0, #kg
        'area': 0.07,#0.43205,
        'cd': 2.0,
        'diffReflect': 0.0,  # diffuse reflectivity coeff
        'diffDegrade': 1.0,  # degradation of reflectivity coeff

          # Shapefactors
          
          #'thrustlvl': 644.099, #N
          #'isp': 3000.0, #230.9,  #s
          #'tcolor': "blue"
          #'start_time': 0.0,
          #'duration': 6000.0,
        }

# Initial state vector - osculating
# 1:
STATE = { 'a':  12271.14, #7378.14, #1000 km mean alt  #semi-maj axis in km
          'e': 0.004, #0.001, # eccentricity
          'i': 109.7, #99.48, #58.5064, #inclination
          'o': 180.0, #not RAAN here but long. of node: 180.0 deg = 12 pm MLTAN, 90.0 = 6am
          'u': 68.0, #Arg of perigee
          'theta': 292.0, #true anomaly
        }


SETTINGS = { 
    'gravforce': True,
    'srpforce' : True,
    'albedoforce' : True,  # set only one of the next 3 albedo force flags to True
    'albedothermalforce' : False,
    'albedoonlyforce' : False, 
    'aeroforce' : True,
    'sat1_propagate': True,
    'output_sample_time': 360.0,         #the time spacing at which output is sent to the file in sec.
		  }

# Plotting parameters go in here:

PLOT = {'ploton': True,
        'albedo_plot': True,
        'srp_plot': True,
        'aero_plot': True,
        'grav_plot': True
        }


''' Other Settings - not used
        
        'sat2_propagate': False,
        'step_size': 0.1,#0.001,          #for integrator step, in sec
        'sample_frequency': 20,
        'velocity_tolerance': 0.5,
        'rev_number':1,
        'output':1,                       #used in runprog to vary printing file outputs
        'input':1,                        #for the load_init function for propagation start: 1=elements 2=posvel
        'start_time':0.0,
        'current_time':0.0,               #keeps track of current relative time
        'end_time':60.0,            # 60.0 (1 min), #84600.0 (1 day), #5600.0 (1 LEO orbit)
        'run':0,                          #used for running Bouncer program
        'display_factor':10,              #frequency for display dots
        'display_count':0,                #for initial count for display refresh
        'working_dir':'/Users/reynerso/Documents/Python/Orbint/Orbint23a/', #'C:\\Python25\\',   #default directory for working files
        'display': False,                 #to disable display to speed calcs
        'suninfo': False,                 #will perform sun info calcs
        'sattosat': False,                #will perform sat to sat calcs
        'satview': False,                 #will perform gs to sat visibility calcs
        'apearth': True,                  #accounts for Earth gravitation
        'apdrag': True,                   #accounts for Earth atmospheric drag
        'aplift': True,                   #accounts for Earth atmospheric lift
        'burns': True,                   #accounts for Thruster burns in THRUST
'''


# not used currently:
RVSTATE = { 'x': -30183073.4469, # Position Vector: x-component
            'y': 10808943.3046, # y-component
            'z': -19.2539264169, #z-component
            'vx': -1204.0803561, #Velocity Vector: x-component
            'vy': -3313.37065984, #y-component
            'vz': 0.00222878692134, #z-component
        }    

'''
#G = 6.673E-11 #gravitational constant, m^3/(kg*s^2)
#mu = G*mplanet

EARTH = {'grav_const': 3.98600441E14, # mu , units of m^3 per s^2
          'grav_accel': 9.80, # in m/s^2
          'body_radius': 6378140.0, # in m
          'eccentricity': 0.08182,
          'flat_factor': 1.0067395,
          'name': 'Earth'
        }

# Plotting parameters go in here:

PLOTSET = { 'maxx': 2000.0,
         'minx': 0.0,
         'xint': 200.0,
         'maxy': 1000.0,
         'miny': 0.0,
         'yint': 100.0,
         }
'''

#class definitions

class Outputs:  # contains lists of outputs
    def __init__(self):
        self.x = []
        self.y = []
        self.z = []
        self.vx = []
        self.vy = []
        self.vz = []
        self.r = []
        self.v = []
        self.gamma = []
        self.t = []
        self.long = []
        self.lat = []
        self.eclipse = []
        self.sunRA = []
        self.sundec = []
        self.sunux = []
        self.sunuy = []
        self.sunuz = []
        self.mooneclipse = []
        self.moonux = []
        self.moonuy = []
        self.moonuz = []

class Satellite:
    def __init__(self,index,input,state,rvstate): #,thrust,thresh,color,sensors,quat):
        self.mass = input['mass']
        self.area = input['area']
        self.cd = input['cd']
        #self.thrustlvl = input['thrustlvl']
        #self.isp = input['isp']
        self.name = input['name']
        self.diffReflect = input['diffReflect']
        self.diffDegrade = input['diffDegrade']
        self.a = state['a']
        self.e = state['e']
        self.i = state['i']
        self.o = state['o']
        self.u = state['u']
        self.theta = state['theta']
        #self.thrust = thrust
        #self.thresh = thresh
        self.outputs = Outputs()
        self.x = rvstate['x']
        self.y = rvstate['y']
        self.z = rvstate['z']
        self.vx = rvstate['vx']
        self.vy = rvstate['vy']
        self.vz = rvstate['vz']
        self.index = index
        #self.trackcolor = input['tcolor']
        self.eclipse = 0
        #self.sunpos = vector()
        self.sunRA = 0.0
        self.sundec = 0.0
        #self.attitude = quat #quaternion()
        #self.sensors = sensors
        #self.color = color
        self.input = input
        self.state = state
        self.rvstate = rvstate
        self.plot2D = []
        self.lat = 0.0
        self.long = 0.0

class Body:
    def __init__(self,body):
        self.grav_const = body['grav_const']
        self.grav_accel = body['grav_accel']
        self.body_radius = body['body_radius']
        self.e = body['eccentricity']
        self.f = body['flat_factor']
        self.name = body['name']

class Time:
    def __init__(self,time):
        self.year = time['year']
        self.month = time['month']
        self.day = time['day']
        self.hour = time['hour']
        self.minute = time['minute']
        self.second = time['second']
        #self.julian = jultime(self.hour,self.minute,self.second,self.day,self.month,self.year)
        #self.gstdeg = gstdeg(self.julian)
    def displaytime(self,time):
        yr = str(time.year)
        mo = str(time.month)
        dy = str(time.day)
        hr = str(time.hour)
        mn = str(time.minute)
        sc = str(time.second)

        if time.minute < 10:
            minzero = "0"
        else:
            minzero = "" # use to add extra leading zeros if necessary
        if time.second < 10:
            seczero = "0"
        else:
            seczero = "" # use to add extra leading zeros if necessary

        self.stringout = hr+":"+minzero+mn+":"+seczero+sc+"  "+mo+"/"+dy+"/"+yr

#-----------------------------------------------------------------------------

# Displays an ASCII string given a Time class parameter

def displaytime(time):
    yr = str(time.year)
    mo = str(time.month)
    dy = str(time.day)
    hr = str(time.hour)
    mn = str(time.minute)
    sc = str(time.second)
    if time.minute < 10:
        minzero = "0"
    else:
        minzero = "" # use to add extra leading zeros if necessary
    if time.second < 10:
        seczero = "0"
    else:
        seczero = "" # use to add extra leading zeros if necessary
    stringout = hr+":"+minzero+mn+":"+seczero+sc+"  "+mo+"/"+dy+"/"+yr
    return stringout

#

#body=Body(EARTH)

sat1=Satellite(1,INPUT,STATE,RVSTATE) #,THRUST,THRESH,'yellow',sensors1,q1)

satlist = []

if SETTINGS['sat1_propagate']:
    satlist.append(sat1)

#if SETTINGS['sat2_propagate']:
# satlist.append(sat2)

#satlist=[sat1,sat2]

#stationlist=[station1, station2]

start=Time(TIMESTART)
finish=Time(TIMEFINISH)

startepoch = convertTime(start)
endepoch = convertTime(finish)

SETTINGS['startepoch']= startepoch
SETTINGS['endepoch']= endepoch

