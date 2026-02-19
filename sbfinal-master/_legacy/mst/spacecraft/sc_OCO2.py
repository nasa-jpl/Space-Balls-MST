import Monte as M
import mpy.units as units
import math

from .scdef import Spacecraft



def set_sc(name,boa):

    scnameprop = name #'OCO2' # sc name used for propacation
    A = 5.547                       # for shape and drag calc
    area = A * units.m * units.m	# for drag calc 
    mass = 428.1 * units.kg		    # for drag calc  
    cd = 2.2                  # for drag calc  
    diffReflect = 0.0 #diffReflect # Reflectance of shape reflectance
    diffDegrade = 1.0 #diffDegrade # Degradation of shape reflectance
    print('name, area, mass, cd: ', scnameprop, area, mass, cd, diffReflect, diffDegrade)
    sc = Spacecraft(scnameprop, area, mass, cd, diffReflect, diffDegrade)

    sc.init_shape(boa)

    # sc shape model: sphere
    '''
    #sent to scdef.py init_shape method:
    SphElem = M.ScSphere("Sphere", math.sqrt( A/math.pi ) * units.m )
    shape = M.ScShape( boa, name + " Shape" )
    shape.insert( SphElem )
    '''
    return sc

    
