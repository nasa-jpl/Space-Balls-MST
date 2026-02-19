
##Space Balls simulation environment - aka MONTE Spacecraft Tool – MST 

Copyright (c) 2023-24 California Institute of Technology (“Caltech”). U.S. Government sponsorship acknowledged. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 


## Mac setup notes:

I copied/cloned repo to a directory: ../Spaceball/sb_<date>/

Mac dev/execution path is: ../Spaceball/sb_<date>/sb/mst/
        Executatble files: mst.py or mstloop.py
        Include file name after space with no .py extension
        
Typical execution statement: (execute from local monte install directory; again, don't include the file extension for the input file) 
        
        monte-160 % ./mpython /Users/reynerso/Documents/Python/Spaceball/sb_7Jul22/sb/mst/mst.py sb_inputs
        
        or
        
        monte-160 % ./mpython /Users/reynerso/Documents/Python/Spaceball/sb_7Jul22/sb/mst/mstloop.py sb_inputs

The file default.boa should be found and copied to: sb/mst/config/ if it is not in the repo already.


## Input files:  
        
        (ie. sb_inputs.py has been the primary file used in development and has all added parameters for new features)

An input file is used to change parameters more easily in one file rather than searching through many.  
The inputs are paramters in a Python dictionary form.  Just change the values as desired.  6 Input dictionaries with keys and sample values are shown as an example as follows:

        
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

INPUT = { 'name': "Spaceball",
        'mass': 100.0, #kg
        'shape': "cube",  #"cylinder", #"plate", #"sphere", #name of shape type:  sphere, plate, or cylinder
        'area': 0.7854,#m^2  radius = 0.5 m, a = 0.785 m^2
        'radius': 0.5, #m radius for cylinder Shape def
        'length': 1.0, #m length for cylinder Shape def
        'cd': 2.0,
        'diffReflect': 0.0,  # diffuse reflectivity coeff
        'diffDegrade': 1.0,  # degradation of reflectivity coeff
        
         # albedofile = path & name of input file for albedo - coverage type only
        'albedofile': '~/Documents/Python/Spaceball/data/CERES_SYN1deg-Day_Terra-Aqua-MODIS_Ed4.1_Subset_20210314_SWreflect.csv',
        'use_albedofile':  False,
         # iralbedofile = path & name of input file for IR albedo - coverage type only
        'iralbedofile': '~/Documents/Python/Spaceball/data/CERES_SYN1deg-Day_Terra-Aqua-MODIS_Ed4.1_Subset_20210326_SWreflect.csv',
        'use_iralbedofile':  False,
         # set for using folder of files (one for sw one for lw);  then enter proper path for files
        'use_albedofolder': True,
        'albedofolder': '/home/monte/Documents/Python/Spaceball/data/hourly/sw/', #'~/Documents/Python/Spaceball/data/hourly/sw/',
        'use_iralbedofolder': True,
        'iralbedofolder': '/home/monte/Documents/Python/Spaceball/data/hourly/lw/',  #'~/Documents/Python/Spaceball/data/hourly/lw/',
        
        # Direction choosing - use for sc shapes requiring a direction
        'dirmode': 'nadir',  #chose mode type:  fixed, velocity, zenith, nadir, ground, sun, moon, (FUTURE:crosstrack for monte 165 and higher) 
        'attmode': 'sun',  #chose mode type:  fixed, velocity, zenith, nadir, ground, sun, moon, (FUTURE:crosstrack for monte 165 and higher) 
            #attmode uses the dirmode as the first direction and attmode to find the second direction that helps define the direction frame

        # for each direction or attitude mode, include a list of required inputs for that mode.
        # fixed:    [ x, y ,z ]
        # velocity: []
        # zenith:   []
        # nadir:    []
        # crosstrack: [ time/epoch? ]
        # ground:   [ gsname? ]
        # sun:      []
        # moon:     []
        'dirinputs': [[ 0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
        }

### Initial state vector - osculating

STATE = { 'a': 7678.14, #7378.14, #1000 km mean alt  #semi-maj axis in km
          'e': 0.004, #0.001, # eccentricity
          'i': 100.66, #99.48, #58.5064, #inclination
          'o': 180.0, #not RAAN here but long. of node: 180.0 deg = 12 pm MLTAN, 90.0 = 6am
          'u': 68.0, #Arg of perigee
          'theta': 292.0, #true anomaly
        }

SETTINGS = { 
    'gravforce': True,
    'srpforce' : False,
    'albedoforce' : False,  # set only one of the next 3 albedo force flags to True
    'albedothermalforce' : False,
    'albedoonlyforce' : False, 
    'aeroforce' : True,
    'covalbedoforce' : True,
    'myforce' : False,
    'lowthrustforce' : False,
    'sat1_propagate': True,
    'output_sample_time': 60,         #in dynrun, the times at which output is sent to the file in sec.

### Plotting parameters go in here:

PLOT = {'ploton': True,
        'albedo_plot': False,
        'srp_plot': False,
        'aero_plot': False,
        'grav_plot': False,
        'albedo_cov_plot': True 
        'iralbedo_cov_plot': True,  
        'totalbedo_cov_plot': True,
        'position_plot': False,
        'velocity_plot': False,
        }

Example files are provided in the ../mst/ directory: user_inputs.py, sb_inputs.py, and lageos_inputs.py
The sb_inputs.py file is the most recent so use that format to prevent errors (as this one has updated fields).

 
## Notes on parameters in the users input file:
        
        INPUT:
                The following parameters are used if the 'covalbedoforce' flag is set to True in SETTINGS   
                        'albedofile' and 'iralbedofile': these are single files, usually for a particular day - so averaged over that day
                        'use_*' flags indicate the use of those files;  if they are not used, an internal alorithm will compute the values (needs verification)
        STATE:
                Use osculating elements.
                a = semimajor axis
                e = eccentricity
                i = inclination in deg.
                o = longitude of node in deg.
                u = argument of periapsis in deg.
                theta = true anomaly in deg.
        SETTINGS:
                'gravforce': gravity; always set to True
                'srpforce': Solar Radiation Pressure
                'albedoforce', 'albedothermalforce', 'albedoonlyforce':  Set only one of these albedo force flags to True. 'abedoforce' is the sum of SW and LW
                'aeroforce': Aerodynamic force 
                'covalbedoforce': This flag uses the Coverage Albedo models as opposed to the built-in Monte albedo algorithms.
                Note: Set either the 'covalbedoforce' or the 'albedoforce' flags to True (but not both). Last 3 SETTINGS flags are currently unused.
        PLOT:
                Chooses which plots to display. If the force flags are not set to True, then the corresponding plots will not be created. Forces in the 
                SETTINGS area should be set to True to ensure the data is generated.  The 'ploton' setting should be set to True or no plots are
                created.
        

## Command line run example:

monte-160 % ./mpython /Users/reynerso/Documents/Python/Spaceball/sb/mst/mst.py sb_inputs (filename with no py extension)

in general:  monte-160 % ./mpython /<repo path>/sb/mst/mst.py <user file with no .py extension>
        
note: substitute your own installation directory for /Users/reynerso/Documents/Python/ above. The input file uses the format in ../sb/mst/sb_inputs.py.  You can copy this python file to the same directory, modify, and rename.
