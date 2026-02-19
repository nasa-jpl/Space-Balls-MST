import Monte as M
import mpy.units as units
import math
#import warnings
#from pdb import set_trace as bp
from mst import config
from mst import input_tools
from mst.input_tools import csvinput
from .suncoverage import find_sun_coverage, thermal_profile

fixedFrameName = 'IAU Earth Fixed' 
framePoleName = 'IAU Earth Pole'

# General objective:  define a thermal object in which Temperature and IR radiation pressure can be found as a function of date/time and Earth location

class Thermal(object):

    # Initialize a thermal object
    # only a body is needed

    def __init__(self, boa, bodyname):
        super().__init__()
        self.boa = boa
        self.bodyname = bodyname
        self.loadpoly(boa)


    def loadpoly(self, boa):
        # earth_shape = M.ShapeBoa.read( boa, "Earth" ) - shape already loaded in config.earth.shape.py
        # create a polyhedral shape for the orbited body and store it in the object as self.polyshape
        shapename = config.earth.shape.check_shape( boa, self.bodyname)
        if shapename == 'Earth Poly Shape':
            #print("This is already the polyhedral shape: Earth Poly Shape.  No need to reload for new Coverage object")
            self.polyshape = config.earth.shape.get_polyshape(boa, 'Earth')
        else:
            #print("Setting the Earth shape to: Earth Poly Shape")
            self.polyshape = config.earth.shape.set_polyshape( boa ) 


    def load_lonlat_faces(self):  # loads the faces and puts into self.lonlatfaces

        lonlatfaces = config.earth.shape.find_face_center_longlats( self.boa, self.bodyname)
        self.lonlatfaces = lonlatfaces


    def find_closest_face(self, lon, lat): # given a facelist with Face objects, finds the closest face center given a long and lat

        mindist = 1000000.0
        for face in self.lonlatfaces:
            facenum = face.num
            facelon = face.long
            facelat = face.lat
            facepos = self.polyshape.faceCenter(facenum)
            rx, ry, rz = config.earth.shape.Earth_Surface_Radius( lon, lat, 0.0 )
            pos = M.Dbl3Vec( rx, ry, rz )
            distvec = facepos - pos
            dist = distvec.mag()
            #print ("face ", facenum, " dist from position: ", rx, ry, rz , " is : ", dist)
            if dist < mindist:
                mindist = dist
                mindistface = facenum
                minlon = facelon
                minlat = facelat
        print("the closest face to longitude ",lon ," and latitude ", lat  )
        print("is: ", mindistface, " with long and lat of: ", minlon, minlat,  " at a distance of: ", mindist , " km from the face center.")
        
        return mindistface
    

    def find_LonLatFaceNums(self, highlat, lowlat, lon=0.0):  #creates a list of face indices that span the latitude inputs at a longitude.
        self.load_lonlat_faces()
        facelist = []

        lats = range(lowlat, highlat)

        for lat in lats:
            print(" ")
            print("Latitude - for finding closest face: ", lat)
            mindistfacenum = self.find_closest_face(lon,lat)
            facelist.append( [mindistfacenum, lat, lon] )

        return facelist


    def SolarCoverageData(self, boa, body, start_time, stop_time, sampletime, faceindex):  # add faceindices, latitudes later for interations

        #fluxes = []
        #powers = []

        # Sun Coverage data over a specified period
        scfaces = find_sun_coverage( boa, body,  start_time , stop_time, sampletime)
        #faceindex = 3
        avgpwr, avgflux, totenergy, sunrise, sunset= thermal_profile( scfaces, faceindex )

        #print(" ")
        #print("The pwr and flux")
        '''
        for faceindex in faceindices:

            avgpwr, avgflux = thermal_profile( scfaces, faceindex )

            fluxes.append(avgflux)
            powers.append(avgpwr)
        '''
        return avgpwr, avgflux, totenergy, sunrise, sunset


    def loop_thrulats( self , boa, body, start_time, stop_time, sampletime ):
        # loops thru specified latitude range at 0 longitude, finds the faces, determnines the ouptput data and returns:
        # output data for each lat point: [faceindex, lat, lon, avgpwr, avgflux, totenergy, sunrise, sunset] 

        looplist = []
        # find the list of faces that span the latitudes of interest
        highlat = 10
        lowlat = 0
        facelist = self.find_LonLatFaceNums( highlat, lowlat, lon=0.0)

        # For that day (time range) find the thermal profile at that face.  

        for face in facelist:

            faceindex = face[0]
            lat = face[1]
            lon = face[2]
            avgpwr, avgflux, totenergy, sunrise, sunset = self.SolarCoverageData( boa, body, start_time, stop_time, sampletime, faceindex)
            looplist.append( [faceindex, lat, lon, avgpwr, avgflux, totenergy, sunrise, sunset] )

        # Store the profile, avg flux, avg energy, and the epoch (time) for the first point and the last point in the profile

        return looplist










