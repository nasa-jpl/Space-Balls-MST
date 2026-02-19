import Monte as M
import mpy.units as units
from ...config.earth.gravity.grav import set_grav

GravModel = set_grav()

# forces: gravity

def load_gravity(boa, scname, forces):
    
    forces.append(M.GravityForce)
    grav = M.Gravity(boa,scname)
    EarthSOI = 9.24e5*units.km
    ##EarthHarmonics = SphHarmonics(boa,"Earth",CoordName.IauEarthFixed,GGM02C.meanr,
    ##                                GGM02C.jCof[:16],15,GGM02C.cCof,GGM02C.sCof,GGM02C.normalized)
    #EarthHarmonics = SphHarmonics(boa,"Earth",CoordName.IauEarthFixed,GGM02C.meanr,
    #                                GGM02C.jCof[:37],36,GGM02C.cCof,GGM02C.sCof,GGM02C.normalized)
    EarthHarmonics = M.SphHarmonics(boa,"Earth",M.CoordName.IauEarthFixed,GravModel.meanr,
                                    GravModel.jCof[:7],6,GravModel.cCof,GravModel.sCof,GravModel.normalized)
    grav.insert( M.GravityNode("","","Earth",EarthSOI,M.GravityNode.SPHERICAL) )
    grav.insert( M.GravityNode("","","Moon",0*units.km))    #, GravityNode.NEWTONIAN) )
    grav.insert( M.GravityNode("","","Sun",0*units.km))#, GravityNode.NEWTONIAN) )

    print( "grav loaded")

    return grav