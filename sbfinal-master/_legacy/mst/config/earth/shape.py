import Monte as M
import mpy.units as units

from mst import config
from ..config_default import *

#from ..config_default import *

import math
sqrt=math.sqrt
sin=math.sin
cos=math.cos
pi=math.pi #3.1415926535897931


def set_shape(boa):
    
    #Enter Shape Data for Earth (used for finding Long and Lat )
    e = M.Ellipsoid( EARTH_RADIUS, EARTH_FLATTENING )
    shape = M.EllipsoidShape( boa, 'Earth Shape #1', 'Earth', e )
    M.EllipsoidShapeBoa.write( boa, shape )  #writes the shape to the boa
    print(' ')
    traj = M.TrajSetBoa.read( boa )
    trajobjs = traj.getAll()
    print('Objects in the boa 3: ', trajobjs)

    # change BodyData object for Earth to set the shape name
    data = M.BodyDataBoa.read( boa, 'Earth' )
    data.setShape( 'Earth Shape #1' )


def set_polyshape(boa):  
    # Create a polyhedral shape from the current ellipsoid shape
    earthBodydata = M.BodyDataBoa.read( boa, 'Earth' )
    earthBodyShapeName = earthBodydata.shape()
    print('The Earth Ellipsoid shape name is: ', earthBodyShapeName)
    earthBodyShapeModel = earthBodydata.shapeModel()
    print('The Earth shape model is: ', earthBodyShapeModel)
    print('The frame is: ', earthBodydata.frame() )

    longspacing = EARTH_POLY_GRID_LONG
    latspacing = EARTH_POLY_GRID_LAT
    earthPolyShape = M.PolyhedralShape( boa, "Earth Poly Shape", earthBodyShapeModel, longspacing, latspacing)#earthBodyShapeName, 36, 18)  #optional default is: int numLong = 36, int numLat = 18

    print('The name of the Earth Polyhedral shape is: ', earthPolyShape.name() )
    print("The number of faces for this polyhedral is: ", earthPolyShape.numFaces() )

    # change BodyData object for Earth to use the shape
    #data = M.BodyDataBoa.read( boa, 'Earth' )
    #earthBodydata.setShape( "Earth Poly Shape" )
    #M.PolyhedralShapeBoa.write( boa, earthPolyShape )  #writes the shape to the boa

    newEarthBodydata = M.BodyData( boa, 'Earth', earthBodydata.frame() , earthPolyShape.name() )

    earthBodydata2 = M.BodyDataBoa.read( boa, 'Earth' )
    earthBodyShapeName2 = earthBodydata2.shape()
    print('The Earth Polyhedral shape name is: ', earthBodyShapeName2)
    earthBodyShapeModel2 = earthBodydata2.shapeModel()
    print('The Earth shape model is: ', earthBodyShapeModel2)

    #earthBodyShapeModel2 = earthBodydata.shapeModel()
    #print('The Earth shape model is: ', earthBodyShapeModel2)

    return earthBodyShapeModel2


def check_shape(boa, bodyname):
    #Returns the type of shape the bodyname has.  Ie use "Earth" as the input bodyname
    bodyData = M.BodyDataBoa.read( boa, bodyname)
    BodyShapeName = bodyData.shape()
    #print("Check Shape: The shape of the body is: ", BodyShapeName )
    return BodyShapeName


def get_polyshape(boa, bodyname):
    #returns the shape model object for a bodyname
    bodyData = M.BodyDataBoa.read( boa, bodyname)
    BodyShapeName = bodyData.shape()
    #print("The body shape name is: ", BodyShapeName, " returning the shape model object to calling function")
    bodyShapeModel = bodyData.shapeModel()
    return bodyShapeModel


def load_earth_poly(boa, bodyname):
    #returns a polyshape object for Earth

    # earth_shape = M.ShapeBoa.read( boa, "Earth" ) - shape already loaded in config.earth.shape.py
    # create a polyhedral shape for the orbited body and store it in the object as self.polyshape
    shapename = check_shape( boa, bodyname)
    if shapename == 'Earth Poly Shape':
        #print("This is already the polyhedral shape: Earth Poly Shape.  No need to reload for new Coverage object")
        polyshape = get_polyshape(boa, 'Earth')
    else:
        #print("Setting the Earth shape to: Earth Poly Shape")
        polyshape = set_polyshape( boa ) 
    return polyshape


class FaceLonLat():

    # Initialize a face long - lat object to store long lat data for each face
    #
    def __init__(self, facenum, facelongitude, facelatitude):
        
        self.num = facenum
        self.long = facelongitude
        self.lat = facelatitude
        #self.area = facearea
        #self.tprof = [] # holds thermal profile


def find_face_center_longlats(boa, bodyname):  #given a polyshape with faces, find the lon and lats of the centers ;return list of FaceLonLat objects
    polyshape = load_earth_poly(boa, bodyname)
    numPolyFaces = polyshape.numFaces() 
    #print("The number of faces for this polyhedral is: ", polyshape.numFaces() )

    # loop through each face in the polyhedron

    facenum = 0
    lonlatfaces = []  # contains a list of scface (SolCovFace) objects

    while facenum < numPolyFaces:

        facestate = M.State( polyshape.faceCenter(facenum) ) 
        facelongitude = M.Spherical.longitude(facestate).convert('deg')
        facelatitude = M.Spherical.latitude(facestate).convert('deg')  
        #facearea = polyshape.faceArea(facenum)
        facelonlat = FaceLonLat(facenum, facelongitude, facelatitude)
        #print("facenum, facelongitude, facelatitude : ", facenum, facelongitude, facelatitude)
        lonlatfaces.append(facelonlat)
        facenum = facenum + 1

    return lonlatfaces    


def find_closest_face(boa, bodyname, facelist, lon, lat): # given a facelist with Face objects, finds the closest face center given a long and lat
    mindist = 1000000.0
    #surfarea = 0.0
    polyshape = load_earth_poly(boa, bodyname)
    rx, ry, rz = config.earth.shape.Earth_Surface_Radius( lon, lat, 0.0 )
    pos = M.Dbl3Vec( rx, ry, rz )

    for face in facelist:
        facenum = face.num
        facelon = face.long
        facelat = face.lat
        facepos = polyshape.faceCenter(facenum)
        distvec = facepos - pos
        #print("facepos, pos, magnitudes: ", facepos.mag(), " ", pos.mag() )
        dist = distvec.mag()
        #print ("face ", facenum, " dist from position: ", rx, ry, rz , " is : ", dist)

        if dist < mindist:
            mindist = dist
            mindistface = facenum
            minlon = facelon
            minlat = facelat
            closest_face = face
        #ancillary area calc
        #surfarea = surfarea + face.area

    print("the closest face to longitude ",lon ," and latitude ", lat  )
    print("is: ", mindistface, " with long and lat of: ", minlon, minlat,  " at a distance of: ", mindist , " km from the face center.")
    #print("Surface area in view: ", surfarea)
    return closest_face


def Earth_Surface_Radius( lon, L, H ):  # for Earth, finds the radius in km on the elipsoid given a longitude or latitude; add height factor H
    ae = 6378.137 #equatorial radius, km
    ec = 0.081819301 # earth eccentricity, not flattening = 0.00335281066475 
    rz=abs(ae/sqrt(1-(ec*sin(L*pi/180))**2)+H)*cos(L*pi/180)
    zq=abs(ae*(1-ec**2)/sqrt(1-(ec*sin(L*pi/180))**2)+H)*sin(L*pi/180)
    Rx = rz*cos(lon*pi/180)
    Ry = rz*sin(lon*pi/180)
    Rz = zq
    #print("Position at Earth surface for lon: ", lon, " lat:  ", L, " Rx, Ry, Rz: ", Rx, Ry, Rz)
    return Rx, Ry, Rz


#Note: adapted from:
'''
# Determines a station position vector in GCI given long, lat, station height, gha,
# and central body geometry

# long = station longitude in degrees East
# L = geodetic latitude, N (is positive)
# gha = Greenwich hour angle, degrees
# H = height above reference ellipsoid, km
# ae = equatorial radius, km
# ec = eccentricity of ellipsoid (N-S: bulge at equator)

def stapos(long, L, gha, H, ae, ec):
    gha = mod(gha+long,360)
    rz=abs(ae/sqrt(1-(ec*sin(L*pi/180))**2)+H)*cos(L*pi/180)
    zq=abs(ae*(1-ec)**2/sqrt(1-(ec*sin(L*pi/180))**2)+H)*sin(L*pi/180)
    Rx = rz*cos(gha*pi/180)
    Ry = rz*sin(gha*pi/180)
    Rz = zq

    return Rx, Ry, Rz
'''




