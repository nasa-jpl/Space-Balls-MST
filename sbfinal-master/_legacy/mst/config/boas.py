import Monte as M
import mpy.units as units

from mst import config
from .config_default import *

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

def set_boa(): #boa):
    
    print('config.DEFAULT_BOA: ' , os.path.dirname(__file__)+'/'+config.DEFAULT_BOA)

    boa = M.BoaLoad(os.path.dirname(__file__) +'/'+ config.DEFAULT_BOA)

    traj = M.TrajSetBoa.read( boa )
    trajobjs = traj.getAll()
    print('Objects in the boa: ', trajobjs)
    
    ''' # Use this section to add more BOAs and BSP trajectories
    boa2name = 'de421.boa' #'../../../Users/reynerso/Documents/Monte/TrajectoryFiles/de421.boa'
    boa.load(os.path.dirname(__file__) +'/'+ boa2name)
    #boa.load('/nav/common/import/ephem/de421.boa')

    #Query what is in the boa:

    traj = M.TrajSetBoa.read( boa )
    trajobjs = traj.getAll()
    print('Objects after the 2nd boa: ', trajobjs)

        #print('loading RSO...')
        #boa.load('/nav/nisar/dev/traj/sjmaclel/RefOrbit/18-PhaseDv2/TrajFiles/refTraj_BOA_PhaseD_160.boa')

    boa3name = 'ON_ORBIT_BOL_to_10May2021.bsp' #'../../../Users/reynerso/Documents/OCO2/PassChecks/ON_ORBIT_BOL_to_10May2021.bsp'
    boa.load( os.path.dirname(__file__) +'/'+ boa3name )

    traj = M.TrajSetBoa.read( boa )
    trajobjs = traj.getAll()
    print('Objects after the 3rd boa: ', trajobjs)
    '''

    earthBody = M.BodyDataBoa.read(boa, "Earth")
    print('Earth data from BodyDataBoa: ', earthBody)


    boaControl = M.ErrorControlBoa.read(boa)
    boaControl.setAction('Missing EarthAngle data' , M.ErrorAction.IGNORE )

    return boa
