import Monte as M
import weakref
from mpy.units import *

class SimpleAlbedoForce:
   def __init__( self, boa, body, area=0.7*m ):
      self._body = body
      self.setBoa( boa )
      self.area = area

   def name( self ):
      return "Fake Sun Pressure"

   def body( self ):
      return self._body

   def setBoa( self, boa ):
      self.boa = weakref.ref( boa )
      self.query = M.TrajQuery( boa, self._body, "Sun" )
      self.mass = M.MassBoa.read( boa, self._body )

   def compute( self, time, frame ):
      # Get the position vector as a Unit3Vec
      pos = self.query.state( time, frame ).uPos()
      m = self.mass.mass( time )
      flux = 1e14 *km*kg/sec**2

      # Compute the acceleration as a Unit3Vec
      accel = flux / ( m * pos.mag()**2 ) * pos.unit()

      # Convert to Monte std units in a Dbl3Vec for the force.
      return accel.value()

   def format( self, indentLevel=0 ):
      return f"{self._body}: Fake SRP"

   # Support pickle - this is needed for UI support and to load/save
   # the force to a boa.
   def __getstate__( self ):
      return { "version" : 1, "boa" : self.boa(), "body" : self._body,
               "area" : self.area }

   def __setstate__( self, d ):
      if d.get( "version", 0 ) != 1:
         raise M.ErrorLog( f"Unknown version (expected 1) in setstate "
                           f"data: {d}" )

      self.__init__( d["boa"], d["body"], d["area"] )


def create( boa, body, args ):
   return M.PyForce( SimpleAlbedoForce( boa, body ) )


# Register the force into Monte.  This must be run one time and
# can be activated by importing this module in your script or
# Options.mpy file.


#M.ForceFactory.add( 'Simple SRP', create )

