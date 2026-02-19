# deprecated file - Coverage object now in coverage_calcs.py under analysis/coverage/

import Monte as M
import mpy.units as units
from analysis import headers
import pandas as pd
import matplotlib.pyplot as plt
import mpy.traj.coverage as cover
import math

from mst import config


fixedFrameName = 'IAU Earth Fixed' 
framePoleName = 'IAU Earth Pole'


# trying mpy.traj.coverage...  seems broken and doesn't vary with altitude
def find_poly_coverage2(boa, polyshape, sensor, startEpoch, stopEpoch, outframeName, sampletime, userfile): # finds facets of polyshape in view

    #polyshape = config.earth.shape.set_polyshape( boa )   # create a polyshape for the Earth type: PolyShapeRegion

    print("Coverage of polyshape facets: ")

    epoch_of_interest = startEpoch + 1*units.sec

    #psregion = polyshape.coverage( epoch_of_interest, sensor )

    dMode = M.PolyhedralShape.DETECT_ANY_TWO
    coverage = cover.computeRange( boa, polyshape, sensor, epoch_of_interest,  epoch_of_interest + 1.0*units.sec, 1.0*units.sec, detectMode = dMode )

    # get the cumulative observed region that includes the number of times
    # each face was seen by the sensor over the duration of the study.
    region = coverage.cumulativeObsRegion()

    # number of faces:
    print( f"number of faces observed: {region.numFaces()}" )

    # how much of the body was seen?
    print( f"total percent observed: {region.percentTotalSurface()}" )

    # how many faces were not seen?
    print( f"total num faces unseen: {coverage.numUnobsFaces()}" )

    # get the region containing all the faces never seen
    unseen = region.inverse()    


#original: 
def find_poly_coverage3(boa, sc, sensor, startEpoch, stopEpoch, outframeName, sampletime, userfile): # finds facets of polyshape in view

    polyshape = config.earth.shape.set_polyshape( boa )   # create a polyshape for the Earth type: PolyShapeRegion

    print("Coverage of polyshape facets: ")

    epoch_of_interest = startEpoch + 1*units.sec

    psregion = polyshape.coverage( epoch_of_interest, sensor )  # test of coverage function, but doesn't seem to vary with altitude!

    facelist = psregion.faces() # list of facets by int index

    for facenum in facelist:
        print("Face index: ", facenum, " Area: ", polyshape.faceArea(facenum), " Center position: ", polyshape.faceCenter(facenum) )

    numfaces = psregion.numFaces()
    print("The number of faces in view is: ", numfaces ) 

    totsurfpct = psregion.percentTotalSurface()
    print("The percent of the total surface seen is: ", totsurfpct )

    regionSurfArea = psregion.surfaceArea()
    print("The amount of surface area seen is: ", regionSurfArea )


    #num of faces in the poly - for iterations:    
    numPolyFaces = polyshape.numFaces() 
    print("The number of faces for this polyhedral is: ", numPolyFaces)


    # find closest distance to Earth

    query = M.TrajQuery( boa, sc.name, 'Earth', fixedFrameName ) #framePoleName ) 
    #oscState = query.state( epoch_of_interest, framePoleName ) # ie framePoleName = 'IAU Earth Pole')
    oscStateEarth = query.state( epoch_of_interest, fixedFrameName ) # ie frameFixedName = 'IAU Earth Fixed'

    print("the magnitude of the state is : ", oscStateEarth.posMag(), " with position: ", oscStateEarth.pos() )

    dir_to_Earth_Center = M.PositionDir( query )  # find direction from one body to another - SC to Earth dir

    print("Direction from Earth to SC: (if value is -1 then the line doesn't intersect the shapw) " , dir_to_Earth_Center )

    unitvec = dir_to_Earth_Center.unit( epoch_of_interest, fixedFrameName )
    distance_to_Earth = polyshape.rayIntersect( oscStateEarth, -unitvec)  #computes the closest distance from the SC to the polyhedral along a vector unitvec

    print("Distance from SC to Earth's surface:  ", distance_to_Earth)
    print("")


    # find distances to all the face centers in view:

    print("Distances from SC to each face: ")
    for facenum in facelist:
        relpos = oscStateEarth.pos() - polyshape.faceCenter(facenum) 
        print("Face index: ", facenum, " Relative position: ", relpos.mag() )
    print("")

    # designate the nadir vector in sc cords:
    nadirvec = M.Dbl3Vec( -1.0, 0.0, 0.0 )

    # find the face norms:
    print("Normal vectors to each face: ")
    facenorms = []
    for facenum in facelist:
        normvec = polyshape.faceNorm(facenum)
        facenorms.append(normvec)
        print("Face index: ", facenum, " Normal Vector: ", normvec.vec(), ' Angle: ', normvec.angle(nadirvec), ' Norm check:  ', normvec.mag() )

    # return the Earth shape to an ellipsoid (needed for long/lat calcs to work)
    config.earth.shape.set_shape( boa ) 



#interate thru all the faces to find those in view: 
def find_poly_coverage(boa, sc, polyshape, startEpoch, stopEpoch, outframeName, sampletime, userfile): # finds facets of polyshape in view

    #polyshape = config.earth.shape.set_polyshape( boa )   # create a polyshape for the Earth type: PolyShapeRegion

    print("Coverage of polyshape facets: ")

    #choose a time step relative to start: 3140 is near equatorial plane for this sc  ; use at least 1.0 to prevent function error
    deltat = 0.0 * units.sec    # 2380.0*units.sec  is near 0 longitude; 3140.0*units.sec is near 0 latitude
    epoch_of_interest = startEpoch + deltat 


    '''
    # attempt to use polyshape.coverage - doesn't vary with altitude

    psregion = polyshape.coverage( epoch_of_interest, sensor )

    facelist = psregion.faces() # list of facets by int index

    for facenum in facelist:
        print("Face index: ", facenum, " Area: ", polyshape.faceArea(facenum), " Center position: ", polyshape.faceCenter(facenum) )

    numfaces = psregion.numFaces()
    print("The number of faces in view is: ", numfaces ) 

    totsurfpct = psregion.percentTotalSurface()
    print("The percent of the total surface seen is: ", totsurfpct )

    regionSurfArea = psregion.surfaceArea()
    print("The amount of surface area seen is: ", regionSurfArea )
    '''


    #num of faces in the poly - for iterations:    
    numPolyFaces = polyshape.numFaces() 
    print("The number of faces for this polyhedral is: ", polyshape.numFaces() )



    # find distance to Earth surface (polyshape)

    query = M.TrajQuery( boa, sc.name, 'Earth', framePoleName ) 
    #oscState = query.state( epoch_of_interest, framePoleName ) # ie framePoleName = 'IAU Earth Pole')
    oscStateEarth = query.state( epoch_of_interest, fixedFrameName ) # ie frameFixedName = 'IAU Earth Fixed'

    print("the magnitude of the state in fixedFrame is : ", oscStateEarth.posMag(), " with position: ", oscStateEarth.pos() )

    dir_to_Earth_Center = M.PositionDir( query )  # find direction from one body to another - SC to Earth dir, query is in the framePole cs
    #print("Direction from Earth to SC - framePole cs: (if value is -1 then the line doesn't intersect the shape) " , dir_to_Earth_Center )
    print("PositionDir.unit: fixedFrame ref", dir_to_Earth_Center.unit( epoch_of_interest, fixedFrameName ) ) #designate as unitvec in the fixedframe cs
    print("PositionDir.unit: framePole ref", dir_to_Earth_Center.unit( epoch_of_interest, framePoleName ) ) #designate as unitvec in the framePoleName cs

    unitvec = dir_to_Earth_Center.unit( epoch_of_interest, fixedFrameName ) # finds the direction to the Earth center in the fixed frame cs
    distance_to_Earth = polyshape.rayIntersect( oscStateEarth, -unitvec) #computes the closest distance from the SC to the polyhedral along a vector unitvec
    print("Distance from SC to Earth's surface:  ", distance_to_Earth)
    print("")
    print("")

    '''
    #attempt to use a nadir frame and rotMatrix to rotate a vector
    # designate the nadir vector in sc cords:
    nadirvec = M.Dbl3Vec( -1.0, 0.0, 0.0 )  # this is in the rotating frame with -X down towards nadir
    # rotate vector from SC coords to Body(Earth) Fixed coords, outframeName is assumed Nadir Frame
    rotMatrix = config.frames.rotate_Frames(boa, epoch_of_interest, fixedFrameName, outframeName) # this rotates from the rotating nadir frame to the fixed frame
    nadirvec_fixedFrame =  rotMatrix * nadirvec  #  this is the nadir vector in the fixed Frame
    print("Nadir vector in the Fixed Frame: ", nadirvec_fixedFrame)
    print("")

    #Compare to dir_to_Earth_Center
    print("dir_to_Earth_Center: ", unitvec)
    print("")
    # compute angle between nadir and unitvec - they should be close to either 0 or 180 deg
    nadir_to_ecvec = nadirvec_fixedFrame.angle(unitvec)
    print("Angle between nadir vector and EC unitvec: ", nadir_to_ecvec )
    print("")

    # find angle threshold based on altitude and central body radius
    print("Angle threshold based on altitude and central body radius: ")
    Re = 6378.14 
    angthreshold = math.asin( Re/( Re + distance_to_Earth/units.km) ) * 180.0 / math.pi
    print("Threshold : ", angthreshold, " degrees")
    #threshold = 59.82 deg  estimated    
    print("")
    '''

    # find the face norms:
    print("Normal vectors to each face that is in view: ")
    #facenorms = []  # normal  vector face list
    facelist = []   # list of face indices and more
    faceindices = [] # list of face indices
    facenum = 0
    #pwrdensity = 100.0 /(units.m)**2 #   # W/m^2 nominal value for reflected solar albedo only
    totalprojarea = 0.0

    while facenum < numPolyFaces:
        normvec = polyshape.faceNorm(facenum)  # normal vector in the polyshape frame, fixed frame
        normang = normvec.angle(unitvec)
        #if normvec.angle(nadirvec_fixedFrame) > ( 90.0)*units.deg:
        if normang < ( 90.0 )*units.deg:            
            #print("Face index: ", facenum, " Normal Vector: ", normvec.vec(), ' Angle: ', normvec.angle(nadirvec_fixedFrame), ' Norm check:  ', normvec.mag() )
            print("Face index: ", facenum, " Normal Vector: ", normvec.vec(), ' Angle-unitvec: ', normang, ' Norm check:  ', normvec.mag() )

            facestate = M.State( polyshape.faceCenter(facenum) ) 
            relpos = oscStateEarth.pos() - facestate.pos() #polyshape.faceCenter(facenum)
            dist_to_sc = relpos.mag()
            dotprod = normvec.dot(relpos)
            vang = math.acos( dotprod/dist_to_sc )
            print("Angle between normal and face-to-sc vector: ", vang , "facestate.frame: ", facestate.frame() )

            if vang < math.pi/2: # count only faces within the 90 deg angular constraint
                #facenorms.append(normvec)
                facearea = polyshape.faceArea(facenum)
                projarea = math.fabs( math.cos(normang/units.deg) ) * facearea 
                facelongitude = M.Spherical.longitude(facestate).convert('deg')
                facelatitude = M.Spherical.latitude(facestate).convert('deg')   
                #netpwr = pwrdensity * projarea / (math.pi * 4.0 * dist_to_sc**2) # how power decreases due to spreading at the distance to the s/c
                print("COUNTED >>>>>>>>>>>>>>>>>>>>>>>>>>   face area: ", facearea, " projarea: ", projarea, " distance to sc: ", dist_to_sc, " long: ", facelongitude, " lat: ", facelatitude )
                facelist.append([facenum, projarea, relpos, facelongitude, facelatitude])  #facelist contains: [ faceindex, projected area, relative position vector, long, lat ]
                faceindices.append(facenum)
                totalprojarea = totalprojarea + projarea

        facenum = facenum + 1

    print("")
    print("Number of faces in view: ", len(facelist), " Total projected area: ", totalprojarea)
    #print("facestate.frame: ", facestate.frame() )
    print("")


    '''
    # find distances to all the face centers in view:

    print("Distances from SC to each face: ")

    facenum = 0

    for facenum in facelist:
        relpos = oscStateEarth.pos() - polyshape.faceCenter(facenum)
        longitude = M.Spherical.longitude(oscStateEarth).convert('deg')
        latitude = M.Spherical.latitude(oscStateEarth).convert('deg') 
        #print("Face index: ", facenum, " Relative position: (mag, x, y, z) ", relpos.mag(), relpos, " angle rel to nadir: ", nadirvec_fixedFrame.angle(-relpos), " long: ", longitude,  " lat: ", latitude )
        print("Face index: ", facenum, " Relative position: (mag, x, y, z) ", relpos.mag(), relpos, " angle rel to unitvec: ", unitvec.angle(relpos), " long: ", longitude,  " lat: ", latitude )

        facenum = facenum + 1

    print("")


    # Limit the distances to the horizon threshold

    print("Distances from SC to each face within the distance threshold: ")

    Re = 6378.14 
    threshold = math.sqrt( ( Re + distance_to_Earth/units.km)**2 - Re**2 )
    print("Threshold : ", threshold)
    #threshold = 3708.9  estimated

    coverfaces = []
    facenum = 0


    for facenum in facelist:
        facestate = M.State( polyshape.faceCenter(facenum) )  
        relpos = oscStateEarth.pos() - facestate.pos()
        if relpos.mag()< threshold:
            # long - lat of each face:
            facelongitude = M.Spherical.longitude(facestate).convert('deg')
            facelatitude = M.Spherical.latitude(facestate).convert('deg')            

            #print("Face index: ", facenum, " Relative position: (mag, x, y, z) ", relpos.mag(), relpos, " angle rel to nadir: ", nadirvec_fixedFrame.angle(-relpos) , " long: ", facelongitude,  " lat: ", facelatitude, " dlong: ",longitude-facelongitude, " dlat: ", latitude-facelatitude )  )
            #print("Face index: ", facenum, " Rel pos (mag): ", relpos.mag(), " ang rel to nadir: ", nadirvec_fixedFrame.angle(-relpos) , " long: ", facelongitude,  " lat: ", facelatitude, " dlong: ",longitude-facelongitude, " dlat: ", latitude-facelatitude )  
            print("Face index: ", facenum, " Rel pos (mag): ", relpos.mag(), " ang rel to unitvec: ", unitvec.angle(relpos) , " long: ", facelongitude,  " lat: ", facelatitude, " dlong: ",longitude-facelongitude, " dlat: ", latitude-facelatitude )  

            coverfaces.append(facenum)

        facenum = facenum + 1
    print("Number of faces in coverage: ", len(coverfaces))
    print("")   
    '''

    #SC position, long and lat:
    longitude = M.Spherical.longitude(oscStateEarth).convert('deg')
    latitude = M.Spherical.latitude(oscStateEarth).convert('deg') 
    print( "SC long and lat: ", longitude, " ", latitude )


    # Surface Area calculation
    
    totalarea = 0.0
    #totalprojarea = 0.0
    for facenum in facelist: #coverfaces:
        area = polyshape.faceArea(facenum[0])
        totalarea = totalarea + area
        #pa = facenum[1]
        #totalprojarea = totalprojarea + pa
    
    print("Total coverage area: ", totalarea)
    #print("Total projected area: ", totalprojarea)

    Re = 6356.785 #at poles; at equator: 6378.14 
    threshold = math.sqrt( ( Re + distance_to_Earth/units.km)**2 - Re**2 )
    print("Threshold : ", threshold)
    #threshold = 3708.9  estimated at equator

    print("")   
    print("Slant range disc area from threshold distance (includes some unseen area): ", threshold*threshold*math.pi ) 
    rd = Re*math.sin(threshold/Re )  
    print("radial distance using threshold: ", rd  )
    print("radial disk circular area: ", math.pi*rd*rd)

    print("frame for the polyshape: ", polyshape.frame() )

    ''' 
    # start albedo vector calc...
    print("")
    print("This is the internal calc in coverage.py...")
    print("")
    c = 2.998e8 #speed of light, m/s

    vectortally = M.Dbl3Vec(0.0, 0.0, 0.0) #start the vector to add cummulative vectors

    for face in facelist:
        index = face[0]
        projarea = face[1]
        relposdir = -face[2].unit()
        longitude = face[3]
        latitude = face[4]

        arearatio = projarea/totalprojarea

        #albedovalue = find_albedo_val( longitude, latitude)  # function that extracts the albedo value given a long and lat
        albedovalue = 0.52527  #placeholder  (note 1.75 factor needed  0.3 * 0.52527; this could be a function of the altitude and view angle from sc)
                
        #solarflux = find_solar_flux( epoch )  # function that computes solar flux at Earth given a day of year or epoch
        solarflux = 1360.0 #placeholder W/m^2

        albedoflux = solarflux * albedovalue
        albedopressmag = arearatio * albedoflux / c

        albedovec = albedopressmag * relposdir

        vectortally = vectortally + albedovec

        print("face: ", index, " albedo press mag: ", albedopressmag, " albedo vec: ", albedovec, " vector tally: ", vectortally)

    print("")
    print("vector tally result: magnitude: ", vectortally.mag(), " components: ", vectortally, " unit vec: ", vectortally.unit() )
    #note: for the albedovalue of 0.52527 expect a magnitude of 1.36e-6
    print("")
    '''


    '''
    # finding the ring of faces that produce the coverage pattern


    print("Faces on the ring of the coverage pattern: ")

    facenum = 0
    ringlist = [] #holds indices of faces on the ring

    for facenum in facelist:

        #create angle from nadir to the face, ATF
        relpos = oscStateEarth.pos() - polyshape.faceCenter(facenum) 
        angle_to_face = nadirvec_fixedFrame.angle(-relpos) 
        #print("Face index: ", facenum, " Relative position: (mag, x, y, z) ", relpos.mag(), relpos, " angle rel to nadir: ", nadirvec_fixedFrame.angle(-relpos) )
        
        #create angle from the nadir vector to the face, or it's earth central angle, ECA
        normvec = polyshape.faceNorm(facenum)  # normal vector in the polyshape frame, fixed frame
        #print("Face index: ", facenum, " Normal Vector: ", normvec.vec(), ' Angle: ', normvec.angle(nadirvec_fixedFrame), ' Norm check:  ', normvec.mag() )
        earth_central_angle = normvec.angle(nadirvec_fixedFrame)
        
        # face is on the ring if the sum of the ECA and ATF  is close to 90 deg, within a tolerance
        tol = 0.5* units.deg
        critangle = angle_to_face + earth_central_angle

        if (critangle < 90.0*units.deg + tol) and ( critangle > 90.0*units.deg - tol ): #add to ringlist if on the ring
            ringlist.append(facenum)
            print("Face found on the ring: ", facenum, " crit angle: ", critangle, " Relative position: (mag, x, y, z) ", relpos.mag(), relpos )

        facenum = facenum + 1

    print("Number of faces on the ring: ", len(ringlist))

    print("")    

   '''

    # return the Earth shape to an ellipsoid (needed for long/lat calcs to work)
    config.earth.shape.set_shape( boa ) 

    #print("types:")
    #print("facelist: ", type(facelist))
    #print("totalarea: ", type( totalarea/(units.km)**2  ) )
    #print("totalprojarea: ", type(totalprojarea))
    #print("longitude: ", type(longitude))
    #print("latitude: ", type(latitude))    

    return facelist, totalarea, totalprojarea, longitude, latitude



