import Monte as M
import mpy.units as units

from mst.config.config_default import *
#from ..config.config_default import *


def find_solar_dist( boa, sc, time ):  # finds the distance of a sc to the sun at a particular epoch
    query = M.TrajQuery( boa, sc.name, 'Sun', 'EMO2000' ) 
    oscStateSun = query.state( time, 'EMO2000' ) # ie frameFixedName = 'IAU Earth Fixed' 
    sundist = oscStateSun.posMag()
    #print("")
    #print("The distance from the SC to the Sun is: ", sundist)
    return sundist


def find_solar_dist_tobody( boa, body, time):   # finds the distance of a body to the sun at a particular epoch
    query = M.TrajQuery( boa, body, 'Sun', 'EMO2000' ) 
    oscStateSun = query.state( time, 'EMO2000' ) # ie frameFixedName = 'IAU Earth Fixed' 
    sundist = oscStateSun.posMag()
    #print("")
    #print("The distance from the body: ", body," is: ", sundist)
    return sundist        


def find_solar_flux( boa, sc, time ): # finds the solar energy of a SC at a particular epoch in W/m^2 - returns Float value
    sundist = find_solar_dist( boa, sc, time )
    solarenergy =  3.0605E19/sundist**2 * units.km * units.km # removes Monte units dimensions; converts to W/m^2
    # Value basis: Radiation from sun: S = 3.846 x 10 ^ 26 Watts ; Spreading factor is  S//(4*pi*R^2); So value above, 3.0605E15, is S/(4*pi)
    #print("")
    #print("The solar energy is: ", solarenergy, " in W/km^2")  
    return solarenergy  


def find_solar_flux_atbody( boa, body, time ): # finds the solar energy of a body at a particular epoch in W/m^2 - returns Float value
    sundist = find_solar_dist_tobody( boa, body, time )
    solarenergy =  3.0605E19/sundist**2 * units.km * units.km # removes Monte units dimensions; converts to W/m^2
    # Value basis: Radiation from sun: S = 3.846 x 10 ^ 26 Watts ; Spreading factor is  S//(4*pi*R^2); So value above, 3.0605E15, is S/(4*pi)
    #print("")
    #print("The solar energy at the body", body ," is: ", solarenergy, " in W/km^2")  
    return solarenergy  


def find_solar_force( boa, sc, time):  # finds the solar force on a SC at a particular epoch
    # use this as solar flux input for a M.SolarPressure object
    # To convert use  C = K/c * AU^2 , where K = solarenergy in W/m^2
    # The default value is 1.0197943760e+14 km*kg / s^2. This corresponds to 1366.1 W/m^2 for the solar constant. See the class docs for details.
    solarenergy = find_solar_flux( boa, sc, time )
    c = LIGHT_SPEED  #units m/s^2
    au = AU * 1000 * 1000 # AU converted to units of meters
    solar_force = solarenergy / c * au**2
    solar_force = solar_force * units.N/ 1000000
    #print("")
    #print("The solar force is: ", solar_force, " ( units: m*kg / s^2 = N ) ")  
    return solar_force


def find_solar_force_atbody( boa, body, time):  # finds the solar force at a body at a particular epoch
    # use this as solar flux input for a M.SolarPressure object
    # To convert use  C = K/c * AU^2 , where K = solarenergy in W/m^2
    # The default value is 1.0197943760e+14 km*kg / s^2. This corresponds to 1366.1 W/m^2 for the solar constant. See the class docs for details.
    solarenergy = find_solar_flux_atbody( boa, body, time )
    c = LIGHT_SPEED  #units m/s^2
    au = AU * 1000 * 1000 # AU converted to units of meters
    solar_force = solarenergy / c * au**2
    solar_force = solar_force * units.N/ 1000000
    #print("")
    #print("The solar force is: ", solar_force, " ( units: m*kg / s^2 = N ) ")  
    return solar_force


def find_sun_dir( boa, body, time, frame):  #create a direction to the Sun from the body at a particular time and in a particular frame
    query = M.TrajQuery( boa, 'Sun', body, frame ) 
    sundir = M.PositionDir( query )
    sundir_at_time = sundir.unit( time, frame )
    #print("")
    #print("The direction to the Sun from the body: ", body," is: ", sundir_at_time)
    return sundir_at_time       




'''
if __name__ == "__main__":
    #used for testing functions

    #testing find_sun_dir:

    from ..config.boas import set_boa
    boa = set_boa()

    body = 'Earth'

    #Epoch format: M.Epoch('2021-Apr-1 00:00:00 UTC')
    epochstring = '2021-Apr-1 00:00:00 UTC'  #yr+'-'+mo+'-'+dy+' '+hr+':'+mn+':'+sc+' UTC'   
    time = M.Epoch(epochstring)
    print('epoch defined:',epochstring)
    print(time)

    frame =  'IAU Earth Fixed' 

    sundir_at_time = find_sun_dir( boa, body, time, frame)
'''





    