import Monte as M
import mpy.units as units
import math

from .scdef import Spacecraft
from ..config import directions as dirs


def set_sc( name, boa, sc, gs, th, SETTINGS ):  #sets parameters for the spacecraft, sc is defined in the userfile, Satellite class object
    # currently run from toplevel mst.py script

    scnameprop = sc.name  #'OCO2' # sc name used for propacation
    A = sc.area  #1.0                       # for shape and drag calc
    area = A * units.m * units.m	# for drag calc 
    mass = sc.mass * units.kg   #100.0 # for drag calc  
    rad = sc.rad * units.m # for cyl shape
    len = sc.len * units.m # for cyl shape
    shape = sc.shape # name of shape: sphere, plate, or cylinder
    cd = sc.cd #2.2                  # for drag calc  
    diffReflect = sc.diffReflect # Diffuse Reflectivity of shape
    diffDegrade = sc.diffDegrade # Degradation of shape diffuse reflectivity
    print('name, area, mass, cd: ', scnameprop, area, mass, cd, diffReflect, diffDegrade)
    dirmode = sc.dirmode
    dirinputs = sc.dirinputs
    attmode = sc.attmode
    #start = SETTINGS['startepoch']
    #end = SETTINGS['endepoch']

    space = Spacecraft(scnameprop, shape, area, rad, len, mass, cd, diffReflect, diffDegrade )


    # direction mode used to set direction:

    if dirmode == 'fixed':
        direct = dirs.set_FixedDirECI(boa, dirinputs[0][0], dirinputs[0][1], dirinputs[0][2] )# 1.0, 1.0, 1.0)  #initial placeholder for a direction - using ECI fixed
    elif dirmode == 'velocity':
        direct = dirs.set_VelDirECI(boa, sc.name)
    elif dirmode == 'sun':
        direct = dirs.set_SunDirECI(boa) 
    elif dirmode == 'moon':
        direct = dirs.set_MoonDirECI(boa)
    elif dirmode == 'nadir':
        direct = dirs.set_NadirECI(boa, sc.name)
    elif dirmode == 'zenith':
        direct = dirs.set_ZenithECI(boa, sc.name)
    #elif dirmode == 'crosstrack':  # Can't use with monte ver 160, but it should for versions >= 165
        #direct = dirs.set_CrossTrackECI(boa, sc.name)
    elif dirmode == 'ground':
        direct = dirs.set_SctoGsDirECI(boa, sc.name, gs.name)     
    else:
        print("Direction Mode error - userfile did not use a proper mode name")


    # Attitude mode used to set direction #2:  (used in cube shape)

    if attmode == 'fixed':
        direct2 = dirs.set_FixedDirECI(boa, dirinputs[1][0], dirinputs[1][1], dirinputs[1][2] )# uses 2nd vector in dirinputs to set fixed dir
    elif attmode == 'velocity':
        direct2 = dirs.set_VelDirECI(boa, sc.name)
    elif attmode == 'sun':
        direct2 = dirs.set_SunDirECI(boa) 
    elif attmode == 'moon':
        direct2 = dirs.set_MoonDirECI(boa)
    elif attmode == 'nadir':
        direct2 = dirs.set_NadirECI(boa, sc.name)
    elif attmode == 'zenith':
        direct2 = dirs.set_ZenithECI(boa, sc.name)
    #elif attmode == 'crosstrack':  # Can't use with monte ver 160, but it should for versions >= 165
        #direct2 = dirs.set_CrossTrackECI(boa, sc.name)
    elif attmode == 'ground':
        direct2 = dirs.set_SctoGsDirECI(boa, sc.name, gs.name)     
    else:
        print("Attitude Mode error - userfile did not use a proper mode name")


    # shape name used to set the shape

    if shape == 'sphere':
        space.init_shape(boa)

    elif shape == 'plate':
        space.init_plate_shape(boa, direct)

    elif shape == 'cylinder':
        space.init_cyl_shape(boa, direct)
    
    elif shape == 'cube':
        space.init_cube_shape(boa, th, direct, direct2)

    elif shape == 'sphere128':  # 128 facets
        frameName = "sphere 128 frame"
        csvfilename = 'sphere128-1m.csv'
        space.init_rough_sphere_shape(boa, th, direct, direct2, frameName, csvfilename)

    elif shape == 'buckyball': # aka soccer ball or truncated icosahedron
        frameName = "buckball frame"
        csvfilename = 'sphere32-truncIcos-1m-difRef.csv'
        space.init_rough_sphere_shape(boa, th, direct, direct2, frameName, csvfilename)    

    elif shape == 'icosahedron': # aka soccer ball or truncated icosahedron
        frameName = "icosahedron frame"
        csvfilename = 'icosahedron1m.csv'
        space.init_rough_sphere_shape(boa, th, direct, direct2, frameName, csvfilename)    

    else:
        print("Shape not defined in sc_user1.set_sc - needs to be a sphere, plate, or cylinder.  Setting to a default sphere shape...")
        space.init_shape(boa)


    # sc shape model: sphere
    '''
    #sent to scdef.py init_shape method:
    SphElem = M.ScSphere("Sphere", math.sqrt( A/math.pi ) * units.m )
    shape = M.ScShape( boa, name + " Shape" )
    shape.insert( SphElem )
    '''
    return space




