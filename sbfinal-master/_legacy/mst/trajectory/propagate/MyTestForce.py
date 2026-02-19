#As Ted Drain posted in the MonteForum https://monte.jpl.nasa.gov/forum/index.php?topic=1569.msg5859#msg5859
#Added the last 4 lines

import Monte as M
from mpy.units import *


class MyForceModel (object):
    def __init__( self, boa, body ):
        self.boa = boa
        self._body = body
        self.query = M.TrajQuery( boa, self._body, "Jupiter" )
        #self.mass = M.MassBoa.read( boa, self._body )
       
    def body( self ):
        return self._body

    def name( self ):
       return "J2"

    def compute( self, time, frame ):
        #m = self.mass.mass( time )
        s = self.query.state( time, frame )
        return M.Dbl3Vec( 0, 0, 0 )
 
    def __getstate__( self ):
        return { "boa" : self.boa, "body" : self._body }
    
    def __setstate__( self, state ):
        self.__init__( state["boa"], state["body"] )
 
    def format( self, indentLevel ):
        s =  "Juno: J2\n"
        return s
 
def create(boa, body , modelName ):
    obj = MyForceModel( boa, body )
    py = M.PySimpleForce( obj )
    return py


'''

print(' ')
namelist = M.ForceFactory.names()
print('Before: M.ForceFactory.names(): ', namelist)

M.ForceFactory.add( 'Simple TestForce', create )  # Adds a python force creation function to the factory
print(' ')
print("Simple TestForce force has been added to ForceFactory ")
print(' ')

namelist = M.ForceFactory.names()
print('After: M.ForceFactory.names(): ', namelist)
print(' ')

'''