# import force functions:

#from .albedo import load_albedo

from .gravity import load_gravity
from .drag import load_drag
from .solarrad import load_srp
from .albedo import load_albedo, load_albedo_thermalonly, load_albedo_only
from .covalbedo import load_fakesrp

#from ...user_inputs import SETTINGS as settings
from mst import config

# Propagation Section

# Set Forces

def set_intforces(boa, sc, body, forces, userfile, th):
    
    settings = userfile.SETTINGS
    pressforces = []
    
    print('\n\nsettings: ', settings , '\n\n')

    if settings[ 'gravforce' ]:
        grav = load_gravity(boa, sc.name, forces)
        pressforces.append(grav)

    if settings[ 'aeroforce' ]:    
        aeroPress = load_drag(boa, sc, forces)
        pressforces.append(aeroPress)
        print('aeroPress.__class__: ', aeroPress.__class__ )    

    if settings[ 'srpforce' ]:
        #th = config.time_handling.TimeHandler(userfile.SETTINGS['startepoch']) # not needed as it messed up the loop
        #th.dur_fm_stop(userfile.SETTINGS['endepoch'])
        srpPress = load_srp(boa, sc, body, forces, th)
        pressforces.append(srpPress)
        print('srpPress.__class__: ', srpPress.__class__ )

    if settings[ 'albedoforce' ]:
        earthAlbedoPress = load_albedo(boa, sc, forces, th, userfile)
        pressforces.append(earthAlbedoPress)
        print('earthAlbedoPress.__class__: ', earthAlbedoPress.__class__ )

    if settings[ 'albedothermalforce' ]:
        earthAlbedoPress2 = load_albedo_thermalonly(boa, sc, forces)
        pressforces.append(earthAlbedoPress2)

    if settings[ 'albedoonlyforce' ]:
        earthAlbedoPress3 = load_albedo_only(boa, sc, forces)
        pressforces.append(earthAlbedoPress3)

    # placeholder for Coverage albedo force
    if settings[ 'covalbedoforce' ]:
        th = config.time_handling.TimeHandler(userfile.SETTINGS['startepoch'])
        th.dur_fm_stop(userfile.SETTINGS['endepoch'])        
        earthCovAlbedoPress = load_fakesrp( boa, sc, body, forces, th)
        #pressforces.append( earthCovAlbedoPress )
        #print('earthCovAlbedoPress.__class__: ', earthCovAlbedoPress.__class__ )
    

    return pressforces 



