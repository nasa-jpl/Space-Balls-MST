import Monte as M
import mpy.units as units
from ...config.earth.gravity.grav import set_grav
from ...config.config_default import DEFAULT_BOA_OUT as boaout
#from .MyTestForce import MyForceModel, create
from .MySimpleSrp import MySimpleSrp0 as MSS
from .LowThrustForce import LowThrustForce


# integrate trajectory

def run_integrator(boa, sc, th, initState, forces, frameName, userfile):

    myForce = M.PySimpleForce( MSS(boa, sc.name) )  # instantiate the PySimpleForce using MySimpleSRP
    lowthrustForce = M.PyForce( LowThrustForce(boa, sc.name, 0.001 , eclipse = False))  #thrusts[sc_name], eclipse=eclipse ) )

    query = M.TrajQuery(boa, sc.name, 'Earth', 'IAU Earth Pole')

    GravModel = set_grav()

    initTime = th.start_epoch
    endTime = th.stop_epoch

    integState = M.IntegrationState(boa)
    integState.addState(th.start_epoch, sc.name, "Earth", frameName, initState) # ie frameName = "Earth Inertial at Launch"
    integState.addMass(th.start_epoch, sc.name, sc.mass)
    prop = M.DivaPropagator(boa, "DIVA", integState)
    prop.setMinStep(1e-9*units.sec)
    prop.setMaxStep(1e9*units.sec)

    print(' ')
    print("Listing forces: ", forces)
    print(' ')

    for f in forces:
        prop.addForce(f(boa, sc.name))
    
    if userfile.SETTINGS['myforce']:
        prop.addForce( myForce )  # addind myForce to the propagator

    if userfile.SETTINGS['lowthrustforce']:
        prop.addForce( lowthrustForce ) 

    print("prop.forces() - ForceModel list: ", prop.forces() )

    M.DivaTraj(boa, sc.name, prop)
    M.DivaMass(boa, sc.name, prop)

    StopEvents = [ M.CoordinateEvent(query, M.Conic.semiMajorAxis(),  GravModel.meanr + 140*units.km , M.CoordinateEvent.DECREASING ) ]
    prop.setStopEvents( StopEvents )

    integTimeStart = M.Epoch.now()

    print (integTimeStart)
    # prop.create( boa ,initTime ,endTime )

    outputName = boaout #...config.config_default.DEFAULT_BOA_OUT  #config.files.outputName #'OCO2_out.boa'
    boa2 = M.BoaFile(outputName, M.BoaFile.TRUNCATE)  #creates a new Boa file, if named file exists then erase the old file

    prop.create( boa2, th.start_epoch, th.stop_epoch )  # main propagation method. 
    boa.mergeTree(boa2)

    integTimeEnd = M.Epoch.now()
    print( (integTimeEnd - integTimeStart).hours(), 'hours to integrate', (prop.info().endTime() - initTime).days(), 'days of trajectory' )
    endTime  = prop.info().endTime()
    print (endTime)
    trajLifetime = (endTime-th.start_epoch).convert('days')
    print ('Trajectory Lifetime is ',trajLifetime,' days.')

    print (' ')

    return endTime
