#from hvt.config import DEFAULT_INERTIAL_FRAME
import Monte as M
import mpy.units as units
import math
#import warnings
#from pdb import set_trace as bp
from mst import config
from mst import input_tools
from mst.input_tools import csvinput

# imports for basemap and plots
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from itertools import chain
import numpy as np

def draw_map(m, scale=0.2):  # Used for plotting map
    # draw a shaded-relief image
    m.shadedrelief(scale=scale)
    
    # lats and longs are returned as a dictionary
    lats = m.drawparallels(np.linspace(-90, 90, 13))
    lons = m.drawmeridians(np.linspace(-180, 180, 13))

    # keys contain the plt.Line2D instances
    lat_lines = chain(*(tup[1][0] for tup in lats.items()))
    lon_lines = chain(*(tup[1][0] for tup in lons.items()))
    all_lines = chain(lat_lines, lon_lines)
    
    # cycle through these lines and set the desired style
    for line in all_lines:
        line.set(linestyle='-', alpha=0.3, color='w')


# frame definitions
fixedFrameName = 'IAU Earth Fixed' 
framePoleName = 'IAU Earth Pole'

class Face():

    # Initialize a face object to store data and objects for each face
    # takes the place of a list object - facelist or faceobj in prior versions
    # instantiate in the Coverage object, find_faces method
    # other values added by othe Coverage object methods when executed: 
    # .sunangle, .projsunarea, .solpwr, .albedoval, .albedoratio, .maxalbedofluxratio, iralbedoval, iralbedoratio

    def __init__( self, facenum, projarea, relpos, normvec, facelongitude, facelatitude, facearea, scdist, vza ):
        
        self.num = facenum
        self.projarea = projarea  # has units: km**2
        self.relpos = relpos # this is the direction, a unit vector
        self.normvec = normvec # vector normal for face, a unit vector
        self.scdist = scdist # float value (no units attached), units are in km
        self.long = facelongitude
        self.lat = facelatitude
        self.area = facearea
        self.vertexlonglat = []
        self.vza = vza

class Coverage(object):

    # Initialize a coverage object
    # only a satellite name and a body is needed

    def __init__(self, boa, scname, bodyname):
        super().__init__()
        self.scname = scname
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


    def find_faces(self, boa, epoch_of_interest):

        # finds the coverage at a particular point in time.  Stores the faces of the polyhedral in view.
        # mirrors pattern in analysis/sensor/coverage/find_poly_coverage

        # stores results of faces found in the coverage area in the object as:
        #   self.facelist: list that contains data for each face: [facenum = index, projarea, relpos, facelongitude, facelatitude]
        #   self.faceindices: a list of just indices of faces that can be seen at the epoch of interest
        #   self.totalprojarea:  sums the proj area of each face relative to the sc point of view. Results in an equivalent circular view area for an instrument

        numPolyFaces = self.polyshape.numFaces() 
        #print("The number of faces for this polyhedral is: ", self.polyshape.numFaces() )

        query = M.TrajQuery( boa, self.scname, self.bodyname, framePoleName ) 
        oscStateEarth = query.state( epoch_of_interest, fixedFrameName ) # ie frameFixedName = 'IAU Earth Fixed'

        #print("the magnitude of the state in fixedFrame is : ", oscStateEarth.posMag(), " with position: ", oscStateEarth.pos() )

        dir_to_Earth_Center = M.PositionDir( query )  # find direction from one body to another - SC to Earth dir, query is in the framePole cs
        #print("Direction from Earth to SC - framePole cs: (if value is -1 then the line doesn't intersect the shape) " , dir_to_Earth_Center )
        #print("PositionDir.unit: fixedFrame ref", dir_to_Earth_Center.unit( epoch_of_interest, fixedFrameName ) ) #designate as unitvec in the fixedframe cs
        #print("PositionDir.unit: framePole ref", dir_to_Earth_Center.unit( epoch_of_interest, framePoleName ) ) #designate as unitvec in the framePoleName cs

        unitvec = dir_to_Earth_Center.unit( epoch_of_interest, fixedFrameName ) # finds the direction to the Earth center in the fixed frame cs

        # find the face norms:
        #print("Normal vectors to each face that is in view: ")
        facelist = []   # list of Face objects
        faceindices = [] # list of face indices
        facenum = 0
        totalprojarea = 0.0
        totsurfarea = 0.0

        while facenum < numPolyFaces:
            normvec = self.polyshape.faceNorm(facenum)  # normal vector in the polyshape frame, fixed frame
            normang = normvec.angle(unitvec)
            totsurfarea = totsurfarea + self.polyshape.faceArea(facenum)

            if normang < ( 90.0 )*units.deg:            
                #print("Face index: ", facenum, " Normal Vector: ", normvec.vec(), ' Angle-unitvec: ', normang, ' Norm check:  ', normvec.mag() )
                facestate = M.State( self.polyshape.faceCenter(facenum) ) 
                relpos = oscStateEarth.pos() - facestate.pos() # relative position of the face center wrt the spacecraft
                dist_to_sc = relpos.mag()  # distance to the spacecraft
                dotprod = normvec.dot(relpos)  
                cosvang = dotprod/dist_to_sc # cos of the angle between normvec and relpos dir - will use below for proj area calc
                vang = math.acos( cosvang )  # in radians
                #print("Angle between normal and face-to-sc vector: ", vang , "facestate.frame: ", facestate.frame() )
                
                if vang < math.pi/2: # count only faces within the 90 deg angular constraint
                    faceindices.append(facenum)
                    # the following is used to create facelist and totalprojected area for albedo calcs...
                    facearea = self.polyshape.faceArea(facenum)
                    #projarea = math.fabs( math.cos(normang/units.deg) ) * facearea   # believed to be incorrect - use vang instead of normang
                    projarea = cosvang * facearea  # new equation
                    facelongitude = M.Spherical.longitude(facestate).convert('deg')
                    facelatitude = M.Spherical.latitude(facestate).convert('deg')   
                    #print("COUNTED >>>>>>>>>>>>>>>>>>>>>>>>>>   face area: ", facearea, " projarea: ", projarea, " distance to sc: ", dist_to_sc, " long: ", facelongitude, " lat: ", facelatitude, " facenum: ", facenum )
                    relposunitvec = relpos.unit()
                    vza = vang * 180.0 / math.pi  # viewing zenith angle in degrees
                    face = Face( facenum, projarea, relposunitvec, normvec, facelongitude, facelatitude, facearea, dist_to_sc, vza )
                    facelist.append(face)    #facelist contains: [ faceindex, projected area, relative position vector, long, lat, facearea ]
                    totalprojarea = totalprojarea + projarea

            facenum = facenum + 1

        self.facelist = facelist
        self.faceindices = faceindices
        self.totalprojarea = totalprojarea

        #print("")
        #print("Number of faces in view: ", len(facelist), " Total projected area: ", totalprojarea)
        #print("Total surface area: ", totsurfarea)
        #print("")


    def VZA_SCdist_out(self): # for faces output the VZA and SC distance (aka slant range)
        print("Face num, VZA(deg), Slant Range(km), Longitude, Latitude:")
        for face in self.facelist:
            facenum = face.num
            vza = face.vza
            scdist = face.scdist
            flong = face.long
            flat = face.lat
            print(facenum,",",vza,",",scdist,",",flong,",",flat)


    def find_face_vertices(self): # given a facelist, finds the vertices of each face, puts the 3 vertices in a list; appends to the Face object: face.vertices
                                # finds long and lat for each vertex and stores in face.vertexlonglat
        for face in self.facelist:
            facenum = face.num
            v0 = self.polyshape.faceVertex( facenum, 0 )
            v1 = self.polyshape.faceVertex( facenum, 1 ) 
            v2 = self.polyshape.faceVertex( facenum, 2 )
            face.vertices = [v0, v1, v2]
            v0state = M.State( v0 )
            v1state = M.State( v1 )
            v2state = M.State( v2)
            v0long = M.Spherical.longitude(v0state).convert('deg')
            v0lat = M.Spherical.latitude(v0state).convert('deg') 
            v1long = M.Spherical.longitude(v1state).convert('deg')
            v1lat = M.Spherical.latitude(v1state).convert('deg')             
            v2long = M.Spherical.longitude(v2state).convert('deg')
            v2lat = M.Spherical.latitude(v2state).convert('deg')    
            face.vertexlonglat = [ v0long, v0lat, v1long, v1lat, v2long, v2lat ]   
            #print("Vertex long lats: ", facenum, " ", face.vertexlonglat )


    def find_closest_face(self, lon, lat): # given a facelist with Face objects, finds the closest face center given a long and lat
        mindist = 1000000.0
        rx, ry, rz = config.earth.shape.Earth_Surface_Radius( lon, lat, 0.0 )
        pos = M.Dbl3Vec( rx, ry, rz )

        for face in self.facelist:
            facenum = face.num
            facelon = face.long
            facelat = face.lat
            facepos = self.polyshape.faceCenter(facenum)
            distvec = facepos - pos
            print("facepos, pos, magnitudes: ", facepos.mag(), " ", pos.mag() )
            dist = distvec.mag()
            print ("face ", facenum, " dist from position: ", rx, ry, rz , " is : ", dist)

            if dist < mindist:
                mindist = dist
                mindistface = facenum
                minlon = facelon
                minlat = facelat
        print("the closest face to longitude ",lon ," and latitude ", lat  )
        print("is: ", mindistface, " with long and lat of: ", minlon, minlat,  " at a distance of: ", mindist , " km from the face center to the subsat point.")


    def plot_face_centers(self):  # show basemap plot of the coverage area represented by the center of the faces

        # ref: see aslo /output_tools/plotting_tools/basemap_test0.py

        # set up the main plot:

        fig = plt.figure(figsize=(8, 6), edgecolor='w')

        #cylindrical projection
        m = Basemap(projection='cyl', resolution=None,
                    llcrnrlat=-90, urcrnrlat=90,
                    llcrnrlon=-180, urcrnrlon=180, )

        draw_map(m)

        # Map (long, lat) to (x, y) for plotting  - single test point
        #x, y = m(-122.3, 47.6)
        #plt.plot(x, y, 'ok', markersize=5)
        #plt.text(x, y, ' Seattle', fontsize=12)

        # Do this for all points in the facelist...

        # Go thru each face in the facelist in view
        for face in self.facelist:
            #facenum = face.num
            facelon = face.long
            facelat = face.lat
            x, y = m(facelon, facelat)
            plt.plot(x, y, 'ok', markersize=1)

        plt.title('Satellite Coverage Plot')
        plt.show()


    def plot_face_vertices(self):  # show basemap plot of the coverage area represented by the vertices of the faces

        fig = plt.figure(figsize=(8, 6), edgecolor='w')
        #cylindrical projection
        m = Basemap(projection='cyl', resolution = None,
                    llcrnrlat= -90, urcrnrlat=90,
                    llcrnrlon= -180, urcrnrlon=180, )                
        draw_map(m)

        # Go thru each face in the facelist in view
        
        for face in self.facelist:
            #vertex0
            vlon = face.vertexlonglat[0]
            vlat = face.vertexlonglat[1]
            x, y = m(vlon, vlat)
            plt.plot(x, y, 'ok', markersize=1)
            #vertex1
            vlon = face.vertexlonglat[2]
            vlat = face.vertexlonglat[3]
            x, y = m(vlon, vlat)
            plt.plot(x, y, 'ok', markersize=1)
            #vertex2
            vlon = face.vertexlonglat[4]
            vlat = face.vertexlonglat[5]
            x, y = m(vlon, vlat)
            plt.plot(x, y, 'ok', markersize=1)

        plt.title('Satellite Coverage Plot')
        plt.show()


    def find_area(self):  # for the faces in view, finds the surface or coverage area on the planet's/body's surface that is in view.
        # Surface Area calculation        
        totalarea = 0.0

        for facenum in self.facelist: #coverfaces:
            area = self.polyshape.faceArea(facenum.num) #[0])
            totalarea = totalarea + area

        print("Total coverage area: ", totalarea)


    def find_sun_angle(self,sundir): # for the faces in view, finds the angle to the sun given the sundir in fixedframe coords

        # sun vector finding:
        for faceobj in self.facelist: #coverfaces:
            normvec = self.polyshape.faceNorm(faceobj.num) #[0])
           #faceobj.append( normvec )
            sunangle = normvec.angle( sundir )
            #print("")
            #print("The sun angle for face ", faceobj[0], " is ", sunangle )
            faceobj.sunangle = sunangle #.append( sunangle ) #facelist now contains list with: [ faceindex, projected area, relative position vector, long, lat, facearea. sun angle ]

            # proj area of face rel to sun:
            #print("sunangle.value: ",sunangle.value() * 180 / math.pi )   # sunangle.value() will be in radians
            sunanglevaldeg = sunangle.value() * 180 / math.pi 

            if sunanglevaldeg <= 90.0:  # count as an effective sun area if sun angle < 90 deg, otherwise in shadow.
                area = self.polyshape.faceArea(faceobj.num) #[0])
                cosfactor = math.cos( sunangle.value() )
                projsunarea = area * cosfactor
            else:
                projsunarea = 0.0 * units.km * units.km
                cosfactor = 0.0
            
            #print("projsunarea: ", projsunarea)

            faceobj.projsunarea = projsunarea #.append( projsunarea ) #facelist now contains list with: [ faceindex, projected area, relative position vector, long, lat, facearea, sun angle, proj sun area ]
            faceobj.cosfactor = cosfactor


    def find_raa(self,sundir): # for the faces in view, finds the raangle to the sun given the sundir in fixedframe coords

        print( "sundir: ", sundir) # in the same fixed frame, unit vector
        # sun vector finding:
        for faceobj in self.facelist: #coverfaces:
            normvec = faceobj.normvec #polyshape.faceNorm(faceobj.num) #[0])
            rpuv = faceobj.relpos # relative position Unit vector
            dir1 = normvec.cross( rpuv )  # cross the normal vector with the rpuv to get the first vector in the surface plane for the SC
            dir2 = sundir.cross(normvec)  # cross the sun vector and the normal vector to get the second vector in the surface plane for the sun
            dotprod = dir1.dot( dir2 )  # determine the angle using the dot product of these 2 unit vectors dir1 and dir2
            angledif = math.acos(dotprod) # would normally divide dotprod by the magnitudes, but they are 1 for unit vectors
            angledifdeg = angledif * 180 / math.pi
            raa = 180 - angledifdeg
            faceobj.raa = raa
            print( "facenum: ", faceobj.num, " normvec: ", normvec, " rpuv: ", rpuv, " dir1: ", dir1, " dir2: ", dir2, " angledifdeg: ", angledifdeg, " raa: ", raa )


    def find_input_flux(self, solarflux): # finds the net flux for each face an then totals the flux based on total projected area - store in self; local solar flux stored in each faceobj
        totsolpwr = 0.0
        for faceobj in self.facelist: #
            projsunarea = faceobj.projsunarea #[7] # proj area of face in sun dir in in faceobj position 7
            solpwr = projsunarea.value()* 1000000 * solarflux  # converting from km^2 to m^2,  solarflux in units of W/m^2; solpwr is W
            #print("")
            #print("The input sun power for face ", faceobj.num , " is ", solpwr, " Watts.")  
                      
            faceobj.solpwr = solpwr #.append( solpwr ) #facelist now contains list with: [ faceindex, projected area, relative position vector, long, lat, facearea, sun angle, proj sun area, solar power ]
            totsolpwr = totsolpwr + solpwr

            faceobj.localsolarflux = solarflux * faceobj.cosfactor  # local solar flux used for csv file flux conversion

        self.suninputpwr = totsolpwr


    def find_local_solar_flux(self, solarflux): # finds the net flux for each face an then totals the flux based on total projected area - store in self; local solar flux stored in each faceobj
        #print("")
        #print("Max solar flux: ", solarflux )
        #print("")
        for faceobj in self.facelist: #
            faceobj.localsolarflux = solarflux * faceobj.cosfactor  # local solar flux used for csv file flux conversion
            #print("")
            #print("The local solar flux for face ", faceobj.num , " is ", faceobj.localsolarflux, " Watts/m^2.")  


    def find_temp_flux(self): #, solarflux):  # finds the temperature of each face, resulting flux and totals the flux based on tot. proj area - store in self
        # use solarflux in future for less than daily variances

        totirpwr = 0.0
        for faceobj in self.facelist: #
            projarea = faceobj.projarea  # proj area of face in sc dir in in faceobj 
            lat = faceobj.lat

            # find temperature and IR flux
            e = 1.0 # 0.98  #emissivity
            temp = config.earth.temperature.find_mean_temp(lat)
            Wb = config.earth.temperature.find_irflux(temp,e)
            faceobj.wb = Wb

            irpwr = projarea.value()* 1000000 * Wb    #converting from km^2 to m^2,  Wb = IRflux in units of W/m^2; IRpwr is W
            #print("")
            #print("The input IR power for face ", faceobj.num , " is ", irpwr, " Watts.")        
            
            faceobj.irpwr = irpwr   
            totirpwr = totirpwr + irpwr
        self.irinputpwr = totirpwr


    def assign_albedo_usecsv(self, albedofile):  # given an albedo file, assigns an albedo to each face in view. This value is a function of geographical area and cloud cover and time of year
        # deprecated for assign_albedo_usedf
        for faceobj in self.facelist:
            longitude = faceobj.long #[3]
            latitude = faceobj.lat #[4]
            #filename = albedofile
            filename = '~/Documents/Python/Spaceball/sb_26Apr22/sb/data/CERES_SYN1deg-Day_Terra-Aqua-MODIS_Ed4.1_Subset_20210314_SWreflect.csv' #use for testing
            albedoval = csvinput.find_albedo_from_csv(filename, latitude, longitude)
            faceobj.albedoval = albedoval #.append( albedoval )
            #facelist now contains list with: [ faceindex, projected area, relative position vector, long, lat, facearea, sun angle, proj sun area, solar power, albedoval ]
            #print("")
            #print("The albedo flux value ratio for face ", faceobj[0], " is ", albedoval, " - unitless ratio of albedo to solar flux")    


    def assign_albedo_usedf(self, df):  # given an albedo file, assigns an albedo to each face in view. This value is a function of geographical area and cloud cover and time of year
        
        for faceobj in self.facelist:
            longitude = faceobj.long #[3]
            latitude = faceobj.lat #[4]
            albedoval = csvinput.find_albedo_from_df( df, latitude, longitude)
            #print("")
            #print('Using data frame... for lat/lon = ', latitude, longitude,'albedo is: ', albedoval )
            faceobj.albedoval = albedoval #.append( albedoval )
            #facelist now contains list with: [ faceindex, projected area, relative position vector, long, lat, facearea, sun angle, proj sun area, solar power, albedoval ]
            
            albedoflux = albedoval * faceobj.localsolarflux  # this is not needed for IR format, csv file has actual flux in W/m^2

            # albedoratio = csvinput.convert_albedo( albedoval, solarval=1361)  - this is older version - incorrect
            #print("")
            #print("The albedo flux value ratio for face ", faceobj[0], " is ", albedoratio, " - unitless ratio of albedo to solar flux")    
            faceobj.albedoflux = albedoflux #.append( albedoratio )
            #facelist now contains list with: [ faceindex, projected area, relative position vector, long, lat, facearea, sun angle, proj sun area, solar power, albedoval, albedoratio ]


    def assign_swmean_usedf(self, df):  # given an albedo file, assigns an albedo to each face in view.

        for faceobj in self.facelist:
            longitude = faceobj.long #[3]
            latitude = faceobj.lat #[4]
            swmeanval = csvinput.find_swmean_from_df( df, latitude, longitude)
            faceobj.swmeanflux = swmeanval
        print("************* Assigned SWMEAN values to faces in view ****************************************************************************")

    def assign_cloudfrac_usedf(self, df):  # given an albedo file, assigns an albedo to each face in view.

        for faceobj in self.facelist:
            longitude = faceobj.long #[3]
            latitude = faceobj.lat #[4]
            cloudfracval = csvinput.find_cloudfrac_from_df( df, latitude, longitude)
            faceobj.cloudfrac = cloudfracval
        print("************* Assigned CloudFrac values to faces in view ****************************************************************************")

    def assign_surftype_usedf(self, df):  # given an albedo file, assigns an albedo to each face in view.

        for faceobj in self.facelist:
            longitude = faceobj.long #[3]
            latitude = faceobj.lat #[4]
            surftypeval = csvinput.find_surftype_from_df( df, latitude, longitude)
            faceobj.surftype = surftypeval            
        print("************* Assigned SurfaceType values to faces in view ****************************************************************************")

    def assign_iralbedo_usedf(self, df):  # given an IR albedo file, assigns an IR albedo to each face in view. This value is a function of geographical area and cloud cover and time of year
        
        for faceobj in self.facelist:
            longitude = faceobj.long #[3]
            latitude = faceobj.lat #[4]
            iralbedoval = csvinput.find_albedo_from_df( df, latitude, longitude)
            #print("")
            #print('Using data frame... for lat/lon = ', latitude, longitude,'albedo is: ', albedoval )
            #faceobj.iralbedoval = iralbedoval #.append( albedoval )
            #facelist now contains list with: [ faceindex, projected area, relative position vector, long, lat, facearea, sun angle, proj sun area, solar power, albedoval ]

            #iralbedoflux = iralbedoval * faceobj.localsolarflux

            #iralbedoratio = csvinput.convert_albedo( iralbedoval, solarval=1361)
            #print("")
            #print("The albedo flux value ratio for face ", faceobj[0], " is ", albedoratio, " - unitless ratio of albedo to solar flux")    
            #faceobj.iralbedoratio = iralbedoratio #.append( albedoratio )
            
            faceobj.iralbedoflux =  iralbedoval  #iralbedoflux #.append( albedoratio )


    def find_output_flux(self, solarflux): # based on reflected solar factors for each face, finds the output flux for each face and the max albedo flux ratio to solar flux

        #totsolpwr = self.suninputpwr # for entire area in view
        for faceobj in self.facelist: #
            facearea = faceobj.area / units.km / units.km * 1000000  #[5] / units.km / units.km * 1000000   # surface area of face in m^2       
            solpwr = faceobj.solpwr #[8]  #power input
            reflectance = 0.2941  # should be an input albedo ratio = albedoval/ solar flux (rel to sun)  where solar flux rts = actual solar flux / proj area of face rts
            maxfluxout = reflectance * solpwr / facearea  # max albedo flux out perfectly reflecting all, emissivity = 1
            #print("")
            #print("facearea, solpwr = ",  facearea, solpwr )       
            maxalbedofluxratio = maxfluxout / solarflux.value() 
            #print("")
            #print("The max albedo flux value ratio for face ", faceobj.num, " is ", maxalbedofluxratio, " - unitless ratio of albedo to solar flux")     
            faceobj.maxalbedofluxratio = maxalbedofluxratio #.append( maxalbedofluxratio )
            #facelist now contains list with: [ faceindex, projected area, relative position vector, long, lat, facearea, sun angle, proj sun area, solar power, albedoval, albedoratio, maxalbedofluxratio ]                   

        #self.albedoflux = totsolpwr/self.totalprojarea
        #print("")
        #print("The total albedo flux is", self.albedoflux, " W/m^2." )   
        #return self.albedoflux


    def find_output_irflux(self, solarflux): # based on reflected solar factors for each face and date temp proile, finds the output flux for each face and the max albedo flux ratio to solar flux

        for faceobj in self.facelist: #
            # 
            #irpwr = faceobj.irpwr #power input relative to the sc - use proj area to convert back to W/m^2 and Wb
            #maxfluxout = irpwr/faceobj.projarea  # max albedo flux out perfectly reflecting all, emissivity = 1
            maxfluxout = faceobj.wb
            maxiralbedofluxratio = maxfluxout / solarflux.value() 
            #print("")
            #print("The max IR albedo flux value ratio for face ", faceobj.num, " is ", maxiralbedofluxratio, " - unitless ratio of albedo to solar flux")     
            faceobj.maxiralbedofluxratio = maxiralbedofluxratio         



if __name__ == "__main__":  #Run stmt: ...mst/analysis/coverage/coverage_calcs.py

    # test load albedo

    #from mst import config # .boas as boaset
    #from mst.config.boas import set_boa
    #from ... import set_boa

    #boa = config.boas.set_boa() #boainit)
    #boa = "filename.boa"
    bfn = '../../../Users/reynerso/Documents/Python/mst/mst/config/default.boa' 
    boa = M.BoaLoad(bfn) #(os.path.dirname(__file__) +'/'+ config.DEFAULT_BOA)
    filepath = '../../../Users/reynerso/Documents/Python/Spaceball/sb_23Jul22/sb/mst/trajectory/propagate/'

    # construct simulation for coverage object

    print(" ")
    print("Running Single Coverage object for the start epoch: ")
    cover1 = Coverage( boa, sc.name, 'Earth' )
    cover1.find_faces( boa, th.start_epoch )
    cover1.find_area()

    #test find closest face...
    print("test find closesd face: ")
    cover1.find_closest_face( 1, -1 )
    
    #test plot face centers
    cover1.plot_face_centers()

    #test plot face vertices
    cover1.find_face_vertices()
    cover1.plot_face_vertices()

    #test output of VZA and Slant Range (dist to SC), Long, Lat
    cover1.VZA_SCdist_out() 
    











        






    

    

