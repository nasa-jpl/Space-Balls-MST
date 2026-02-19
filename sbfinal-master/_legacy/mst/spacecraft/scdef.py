#from hvt.config import DEFAULT_INERTIAL_FRAME
import Monte as M
import mpy.units as units
import math
import pandas
import os
#import warnings
#from pdb import set_trace as bp
from ..config import frames as fms

class Spacecraft(object):

    # Initialize spacecraft parameters
    # only name is needed and other params can be set later (or never for loaded trajectories)

    def __init__(self, name, shape='sphere', area = 1.0 *units.m*units.m, rad =0.5 *units.m*units.m, len = 1.0 *units.m*units.m,  mass = 100 * units.kg, cd = 2.2, diffReflect = 0.0, diffDegrade = 1.0, state = [] ): # add diffReflect
        super().__init__()
        self.name = name
        self.shape = shape
        self.area = area
        self.rad = rad
        self.len = len
        self.mass = mass
        self.cd = cd
        self.diffReflect = diffReflect
        self.diffDegrade = diffDegrade
        self.state = state
        
        #self.difref = difref # diffReflect


    def init_shape(self, boa):
        #creates a default spherical shape based on the projected area self.area
        A = self.area / units.m / units.m
        SphElem = M.ScSphere("Sphere", math.sqrt( A/math.pi ) * units.m , self.diffReflect, self.diffDegrade )
        self.shape = M.ScShape( boa, self.name + " Shape" )
        self.shape.insert( SphElem )


    def init_plate_shape(self, boa, direct):
        #creates a default plate shape based on the projected area self.area, and a direction
        #can specify ONE_SIDED or TWO_SIDED as a 'type'
        A = self.area
        PlateElem = M.ScPlate("Plate", "ONE_SIDED", A, direct, 0.0, self.diffReflect, 0.0, self.diffDegrade )
        self.shape = M.ScShape( boa, self.name + " Shape" )
        self.shape.insert( PlateElem )


    def init_cyl_shape(self, boa, direct):
        #creates a default cylinder shape based on a radius, length, and a direction for the axis
        CylElem = M.ScCylinder("Cylinder", self.rad, self.len, direct, 0.0, self.diffReflect, 0.0, self.diffDegrade )
        self.shape = M.ScShape( boa, self.name + " Shape" )
        self.shape.insert( CylElem )


    def init_cube_shape(self, boa, th, direct, direct2):
        #creates a default cube shape based on the area for each side in self.area, a direction, and the dirinputs with 3 fixed directions
        A = self.area
        self.shape = M.ScShape( boa, self.name + " Shape" )
        # create a direction frame from direction 1: direct and direction 2: first vector in dirinputs
        # direct2 = M.Direction()  # dir frame is created using the attmode input type

        # first find the Direction frame from direct  and direct2
        frameName = "cube frame"
        dframe = fms.make_Dir_Frame(boa, frameName, th, direct, direct2) #returns a Direction Frame obj: M.DirectionFrame( boa, "cube frame", "EME2000", timeint, direct, direct2) 
        #find the direction in a frame at a particular epoch: somedirvector = direction.unit( epoch, "EME2000" )
        #define some fixed directions in this new Dir frame:
        # dir format:        dirDbl3Vec = M.Dbl3Vec( x, y, z )
        xdblvec = M.Dbl3Vec( 1.0 , 0.0 , 0.0 )
        ydblvec = M.Dbl3Vec( 0.0 , 1.0 , 0.0 )
        zdblvec = M.Dbl3Vec( 0.0 , 0.0 , 1.0 )
        #FixeDir format:        fixedDir = M.FixedDir(boa, coordFrame, dirDbl3Vec)
        xdirbody = M.FixedDir(boa, frameName, xdblvec)
        ydirbody = M.FixedDir(boa, frameName, ydblvec)
        zdirbody = M.FixedDir(boa, frameName, zdblvec)
        #make the cube plates referenced to the cube frame, put the 6 normal directions into array platedirections 
        platedirections = [ xdirbody, ydirbody, zdirbody, -xdirbody, -ydirbody, -zdirbody]
        pcount = 0
        for platedir in platedirections:   # iterate for all 6 directions; make a plate; put into self.shape, the SC shape
            PlateElem = M.ScPlate("Plate"+str(pcount), "ONE_SIDED", A, platedir, 0.0, self.diffReflect, 0.0, self.diffDegrade )
            self.shape.insert( PlateElem )


    def init_rough_sphere_shape(self, boa, th, direct, direct2, frameName, csvfilename): #, filename):
        #creates a rough sphere shape based on 2 directions, and the csv filename with fixed directions and areas for mult plates
        # first test:  sphere128.csv
        # A = self.area
       
        self.shape = M.ScShape( boa, self.name + " Shape" )
        # create a direction frame from direction 1: direct and direction 2: first vector in dirinputs
        # direct2 = M.Direction()  # dir frame is created using the attmode input type

        # first find the Direction frame from direct and direct2

        #frameName example: "sphere 128 frame"
        dframe = fms.make_Dir_Frame(boa, frameName, th, direct, direct2) #returns a Direction Frame obj: M.DirectionFrame( boa, "cube frame", "EME2000", timeint, direct, direct2) 

        #find the direction in a frame at a particular epoch: somedirvector = direction.unit( epoch, "EME2000" )
        #define some fixed directions in this new Dir frame:
        # dir format:        dirDbl3Vec = M.Dbl3Vec( x, y, z )

        #cycle thru each row in csv:  X Y Z A is the format

        #csvfilename example: 'sphere128.csv'
        print(" ")
        print("Sphere Test Data: ")     
        print('os.path.dirname(__file__): ' , os.path.dirname(__file__)+'/'+  csvfilename ) #

        filename = os.path.dirname(__file__) + '/' + csvfilename

        df = pandas.read_csv(filename) #'hrdata.csv', index_col='Name', parse_dates=['Hire Date'])
        print(df)
        print(" ")
        
        for index, row in df.iterrows():
            print(row['x'], row['y'], row['z'], row['A'], row['difRef'] )
            vec = M.Dbl3Vec( row['x'], row['y'], row['z'] )
            area = row['A'] * units.m * units.m
            dirbody = M.FixedDir(boa, frameName, vec)
            PlateElem = M.ScPlate("Plate"+str(index), "ONE_SIDED", area, dirbody, 0.0, row['difRef'], 0.0, self.diffDegrade )
            #PlateElem = M.ScPlate("Plate"+str(index), "ONE_SIDED", area, dirbody, 0.0, self.diffReflect, 0.0, self.diffDegrade )
            self.shape.insert( PlateElem )

        print(" ")
   

'''
if __name__ == "__main__":  # test of rough sphere shape
    pass
'''



















    





