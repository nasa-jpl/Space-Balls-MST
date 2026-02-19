from numpy import deg2rad
import Monte as M
import mpy.units as units
#from .frames import make_Dir_Frame

#from mst import config
#from .config_default import *
from time_handling import TimeHandler


def set_FixedDir(boa, coordFrame, x, y, z): # creates a fixed direction M.Direction object relative to some input frame
    dirDbl3Vec = M.Dbl3Vec( x, y, z )
    fixedDir = M.FixedDir(boa, coordFrame, dirDbl3Vec)
    return fixedDir

def set_FixedDirECI(boa, x, y, z): # creates a fixed direction M.Direction object relative to ECI (EME2000) - ie Inertial Hold
    dirDbl3Vec = M.Dbl3Vec( x, y, z )
    coordFrame = "EME2000"
    fixedDir = M.FixedDir(boa, coordFrame, dirDbl3Vec)
    return fixedDir   

def set_FixedDirToSun(boa, time):  # sets a fixed direction in ECI (EME2000) from Earth to the Sun at a particular time (epoch) - but remains fixed
    coordFrame = "EME2000"
    suntqEME2000 = M.TrajQuery(boa ,'Sun' ,'Earth' , coordFrame)
    oscStateSun = suntqEME2000.state(time, coordFrame)
    x = M.Cartesian.x(oscStateSun).convert('m')
    y = M.Cartesian.y(oscStateSun).convert('m')
    z = M.Cartesian.z(oscStateSun).convert('m')
    dirDbl3Vec = M.Dbl3Vec( x, y, z )
    fixedDir = M.FixedDir(boa, coordFrame, dirDbl3Vec)
    return fixedDir    

def set_SunDirECI(boa): # finds the sun direction and creates a direction object relative to ECI (EME2000) - moves with time
    coordFrame = "EME2000"
    suntqEME2000 = M.TrajQuery(boa ,'Sun' ,'Earth' , coordFrame)
    sundir = M.PositionDir( suntqEME2000 )
    return sundir     

def set_MoonDirECI(boa):  # finds the moon direction and creates a direction object relative to ECI (EME2000) - moves with time
    coordFrame = "EME2000"
    moontqEME2000 = M.TrajQuery(boa ,'Moon' ,'Earth' , coordFrame)
    moondir = M.PositionDir( moontqEME2000 )
    return moondir       

def set_NadirECI(boa, sc):  # sets the direction from the SC to Earth nadir direction
    coordFrame = "EME2000"
    sctqEME2000 = M.TrajQuery(boa , sc, 'Earth' , coordFrame)
    nadirdir = M.NadirDir( sctqEME2000 )
    return nadirdir     

def set_ZenithECI(boa, sc):  # sets the direction from the SC to Earth zenith direction
    coordFrame = "EME2000"
    sctqEME2000 = M.TrajQuery(boa , sc, 'Earth' , coordFrame)
    nadirdir = M.NadirDir( sctqEME2000 )
    zenithdir = -nadirdir
    return zenithdir     

def set_VelDirECI(boa, sc): # sets the direction from the SC to orbital velocity direction
    coordFrame = "EME2000"
    sctqEME2000 = M.TrajQuery(boa , sc, 'Earth' , coordFrame)
    veldir = M.VelocityDir( sctqEME2000 )
    return veldir    

def set_SctoGsDirECI(boa, sc, gs):  # sets the direction from the SC to a GS position on the Earth,  GS is predefined 
    coordFrame = "EME2000"
    gstqEME2000 = M.TrajQuery(boa , sc, gs , coordFrame)
    gsdir = M.PositionDir( gstqEME2000 )
    return gsdir     

def set_GstoScDirECI(boa, sc, gs):  # sets the direction from the GS to a SC position in ECI,  GS is predefined 
    coordFrame = "EME2000"
    gstqEME2000 = M.TrajQuery(boa , sc, gs , coordFrame)
    gsdir = M.PositionDir( gstqEME2000 )
    scdir = -gsdir
    return scdir     

def set_CrossTrackECI(boa, sc): # sets the direction from the SC to the orbital angular momentum direciton, r x v
    coordFrame = "EME2000"
    sctqEME2000 = M.TrajQuery(boa , sc, 'Earth' , coordFrame)
    crosstrackdir = M.AngMomentumDir( sctqEME2000 )  #this function requires an update to MONTE ver 166
    return crosstrackdir        

def crossDirsECI(boa, dir1, dir2, time):  # takes two directions at a particular time (epoch), and finds the crossproduct vector: dir1 x dir2 ; returns Fixed direction
    # and the angle between them. Can be used to define an eigenslew from one direction to another;  the crossdir is the eigenvector for the slew
    coordFrame = "EME2000"
    unitdir1 = dir1.unit( time, coordFrame )  #this is a Dbl3Vec type
    unitdir2 = dir2.unit( time, coordFrame )
    crossVec = unitdir1.cross( unitdir2 )
    angle = unitdir1.angle( unitdir2 )
    crossdir = M.FixedDir( boa, coordFrame, crossVec)
    return crossdir, angle

def set_CrosstrackECI(boa, sc, time):  # test function only as an example for prior functions.  
                                        # Does not work since in mst there has been no trajectory generated for the spacecraft at any time
    veldir = set_VelDirECI(boa, sc)
    posdir = set_NadirECI(boa, sc)
    cdir, angle = crossDirsECI(boa, veldir, posdir, time)
    return cdir

def findxyzECI(boa, dir, time):  #finds the direction expressed in ECI coordinate system.
    coordFrame = "EME2000"
    dblvec = dir.unit( time, coordFrame )
    print("The direction in ECI is: ", dblvec)


if __name__ == "__main__":
    # test find dir functions

    #from .frames import make_Dir_Frame
    from frames import make_Dir_Frame

    bfn = '../../../Users/reynerso/Documents/Python/mst/mst/config/default.boa' 
    boa = M.BoaLoad(bfn) #(os.path.dirname(__file__) +'/'+ config.DEFAULT_BOA)

    #need a Monte Epoch, arbitrary...  
    epoch = M.Epoch(('2021-Mar-21 00:00:00 UTC'))
    epoch2 = M.Epoch(('2021-Sep-21 00:00:00 UTC'))

    timeint = M.TimeInterval( epoch, epoch2)  #all needed for make_Dir_Frame; needs th object
    dur = timeint.duration()
    th = TimeHandler( epoch, dur.days() )  


    # test making a DirectionFrame using a fixed direction and the Sun direction:

    sundir = set_SunDirECI(boa)

    findxyzECI( boa, sundir, epoch ) # prints the unit direction for sun in ECI coords
    
    print("Sun Direction: ")
    fixedir = set_FixedDirECI(boa, 0.0, 0.0, 1.0) # this direction will be used as the Z axis for the new Direction Frame

    dframe = make_Dir_Frame( boa, "NewDirFrame", th, fixedir, sundir ) #sets a DirFrame using fixedir (Z) and sundir as ref dir so (ref x z) = y; x = y x z

    # find sundir in the new dir frame:
    dblvec = sundir.unit( epoch, "NewDirFrame" )
    print("The Sun direction in NewDirFrame is: ", dblvec)  
    
    # set axes directions in the NewDirFrame:
    xdir = set_FixedDir( boa, "NewDirFrame", 1.0, 0.0, 0.0 )
    ydir = set_FixedDir( boa, "NewDirFrame", 0.0, 1.0, 0.0 )
    zdir = set_FixedDir( boa, "NewDirFrame", 0.0, 0.0, 1.0 )
    print("Axes in the NewDirFrame:")
    print(xdir)
    print(ydir)
    print(zdir)
    print("Unit vector form:")
    print( "X: ", xdir.unit(epoch, "NewDirFrame") )    
    print( "Y: ", ydir.unit(epoch, "NewDirFrame") )   
    print( "Z: ", zdir.unit(epoch, "NewDirFrame") )   

    # find the new axis directions in ECI coords:
    xdirECI = xdir.unit( epoch, "EME2000" )
    ydirECI = ydir.unit( epoch, "EME2000" )
    zdirECI = zdir.unit( epoch, "EME2000" )

    '''
    xdirECI = M.FixedDir( boa, "EME2000", xdir )
    ydirECI = M.FixedDir( boa, "EME2000", ydir )
    zdirECI = M.FixedDir( boa, "EME2000", zdir )
    '''

    print("Axes in the EME2000 frame (ECI):")
    print("X: ", xdirECI)
    print("Y: ", ydirECI)
    print("Z: ", zdirECI) 











   











    

