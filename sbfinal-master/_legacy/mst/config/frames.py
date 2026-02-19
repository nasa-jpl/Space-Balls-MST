from numpy import deg2rad
import Monte as M
import mpy.units as units
import MonteUI.setup as ui

#from mst import config
from config_default import *


def set_FixedFrame(boa, epoch): # creates ab Earth Intertial at Launch frame, rel to epoch and IAU Earth Fixed
    frameName = "Earth Inertial at Launch"
    frame = M.FixedEpochFrame( boa, epoch, "INERTIAL", frameName, "IAU Earth Fixed" )
    return frame, frameName

def set_NadirFrame(boa, th, sc):  #creates a nadir ref frame in boa for SC to Body nadir vector = X, Y is near velocity direction, Z=XxY 
    frameName = "Nadir Ref"
    BodyName = "Earth"
    relFrameName = "EME2000"
    frame = M.BodyPosDirFrame( boa, frameName, relFrameName, th.interval, sc.name, BodyName)
    # The position direction of the body relative to the center is used for the X-axis. So towards zenith; -X is nadir direction
    # The Z-Axis is defined as the angular momentum vector of the body relative to the center = orbit normal and 
    # the Y-axis is simply the Z-Axis crossed with the X-Axis
    return frame, frameName

def make_Zenith_Frame(boa, th, sc): #create a Zenith frame that is 180 deg from the Nadir frame
    frameName = "Zenith Ref"
    zangle = 180*units.deg
    axis = M.Rotation.Z
    r = M.Rotation( axis, zangle )
    oframe = M.OffsetFrame( boa, frameName, "Nadir Ref", th.interval, r)
    return oframe, frameName

def rotate_Frames(boa, t, frame1, frame2):  #finds the rotation bertween 2 frames at epoch t
    # use names of the frames for frame 1 and frame 2
    # returns a Rotation object that can be used to rotate a vector to another frame
    coord = M.CoordSetBoa.read( boa )
    frame1_to_frame2 = coord.rotation( t, frame2, frame1 )
    #print("Frame rotation from ", frame1, " to ", frame2 )
    #print("The rotation matrix is: ", frame1_to_frame2)
    return frame1_to_frame2

def make_Dir_Frame(boa, frameName, th, direct, direct2):  # defines a direction frame 
    #sets a DirFrame using direct (Z) and direct2 as ref dir so (ref x z) = y; x = y x z
    dframe = M.DirectionFrame( boa, frameName, "EME2000", th.interval, direct, direct2) 
    return dframe

def make_Rotating_Frame(boa, refEpoch, angleFromZ, rotrate):
    # creates a rotating frame relative to a base Frame, given a rotation axis whose angle is set relative to the 
    # z axis of the base Frame; rotating at a particular fixed rate
    frameName = "Rotating SC Frame"
    baseFrameName = "Nadir Ref" #"EME2000"
    rotframe = ui.NewPolynomialFrame(
        boa, 
        Frame = frameName,
        BaseFrame = baseFrameName,
        RefEpoch = refEpoch,
        W = [ angleFromZ * units.deg, rotrate * units.deg/units.sec ] 
        )
    return rotframe, frameName







