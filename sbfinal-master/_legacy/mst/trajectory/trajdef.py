#from hvt.config import DEFAULT_INERTIAL_FRAME
import Monte as M
import mpy.units as units
import math
#import warnings
#from pdb import set_trace as bp

# Use to set a spacecraft state using orbial elements

def set_State( boa, name, epoch, frameName ):
    
    # Set initial state prior to propagation

    initArgPeri = 68.0
    initState = M.State (
                    M.Conic.semiMajorAxis( 7378.14 *units.km )  # 7037.34 is osc, mean is 7028.14; mean alt. 650km  
                    ,M.Conic.eccentricity( 0.001 )          # 0.0012550 is for frozen orbit
                    ,M.Conic.inclination( 99.48 *units.deg )  # 97.99 is inclination for meab alt 650 km    
                    ,M.Conic.longitudeOfNode( 180.0 *units.deg) #MLTAN 6AM for 90 deg; 12PM for 180    
                    ,M.Conic.argumentOfPeriapsis( initArgPeri *units.deg)  
                    ,M.Conic.trueAnomaly( (359.9-initArgPeri) *units.deg)           
                    ,M.StateInfo ( boa, epoch, name, "Earth", frameName ) # frameName = "Earth Inertial at Launch" )
                    )    
   
    return initState


def set_State_UserConfig( boa, sc, epoch, frameName ):
    
    # Set initial state prior to propagation

    initArgPeri = sc.u #68.0
    initState = M.State (
                    M.Conic.semiMajorAxis( sc.a *units.km )  # 7378.14 mean alt 1000km, 7037.34 is osc, mean is 7028.14; mean alt. 650km  
                    ,M.Conic.eccentricity( sc.e )          # 0.0012550 is for frozen orbit
                    ,M.Conic.inclination( sc.i *units.deg )  # 99.48 deg for ss orbit at altitude 1000km, 97.99 is inclination for meab alt 650 km    
                    ,M.Conic.longitudeOfNode( sc.o *units.deg) #MLTAN 6AM for 90 deg; 12PM for 180 deg   
                    ,M.Conic.argumentOfPeriapsis( initArgPeri *units.deg)  
                    ,M.Conic.trueAnomaly( sc.theta *units.deg)    # just choosing a true anomaly: use (359.9-initArgPeri)        
                    ,M.StateInfo ( boa, epoch, sc.name, "Earth", frameName ) # frameName = "Earth Inertial at Launch" )
                    )    
   
    return initState

