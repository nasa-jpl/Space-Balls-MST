import Monte as M
import mpy.units as units

from ...analysis.solarflux import find_solar_flux, find_solar_force_atbody



def load_srp(boa, sc, body, forces, th):
    
    forces.append(M.SolarPressureForce)
    #srp model

    #set the flux level:
    flux=.1020506244e+09*units.kg*units.km**3/(units.m**2*units.sec**2)
    #print("initial input flux is: ", flux)
    #Note this is the solar equiv force
    #ref: https://monte.jpl.nasa.gov/monte/doc/161/source/Monte/SolarPressure.html#monte-solarpressure-solarflux
    # if this is not set here, the default is: 1.019794376000000e+17 *N   ; this corresponds with: 1366.1 W/m^2 for the solar constant

    
    # method to set the flux:  needs a ref epoch/time:
    time = th.start_epoch 

    force = find_solar_force_atbody( boa, body, time)
    print("Input flux is changed to a value for a particular epoch. Epoch is: ",time )
    print( " flux is: ", force )

    """
    flux=find_solar_flux(boa, body, time)
    print("Input flux is changed to a value for a particular epoch. Epoch is: ",time )
    print( " flux is: ", flux )
    """


    #create the object:
    solarRadPress=M.SolarPressure(boa, sc.name, sc.name + ' Shape', force)

    #enable shadowing:
    solarRadPress.addShadowBody(M.BodyName.Earth)

    print( "Solar Radiation Pressure loaded" )

    return solarRadPress



    

    

