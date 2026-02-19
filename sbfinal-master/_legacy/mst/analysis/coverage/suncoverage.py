#from hvt.config import DEFAULT_INERTIAL_FRAME
import Monte as M
import mpy.units as units
#import math
#import warnings
#from pdb import set_trace as bp
from mst import config #, analysis
#from mst import input_tools
#from mst.input_tools import csvinput
from .coverage_calcs import Coverage
from ..solarflux import find_solar_flux, find_solar_flux_atbody, find_sun_dir

import math

fixedFrameName = 'IAU Earth Fixed' 
framePoleName = 'IAU Earth Pole'


class SolCovFace():

    # Initialize a solar face object to store data and objects for each face
    #

    def __init__(self, facenum, facelongitude, facelatitude, facearea):
        
        self.num = facenum
        self.long = facelongitude
        self.lat = facelatitude
        self.area = facearea
        self.tprof = [] # holds thermal profile


def single_sun_coverage(boa, sunname, body, time):  # test function - used to test single time step to create the net sun input

    #iterates from start time to stop time to find the net albedo pressure vectors at each time step

    # from analysis/coverage/coverage_calcs.py create the Coverage object and run the method to find the faces

    # Coverage object
    cover1 = Coverage( boa, sunname, body )
    cover1.find_faces( boa, time )
    #cover1.find_area()

    # find the XXX for the coverage object
    # 
    #print("The number of faces for this polyhedral is: ", self.polyshape.numFaces() )


    facelist = cover1.facelist  # list of faces seen
    #faceindices = cover1.faceindices # list of indices seen
    totalprojarea = cover1.totalprojarea  
    
    print("Faces computed for time: ", time)

    #print("")
    print("Number of faces in view: ", len(facelist), " Total projected area: ", totalprojarea)
    print("")

    sundir = find_sun_dir( boa, body, time, fixedFrameName) # finds the direction of the sun
    cover1.find_sun_angle( sundir ) # computes the sun angle on each face
    solarFlux = find_solar_flux_atbody( boa, "Earth", time )  # function that computes solar flux at Earth given a day of year from 'time' epoch
    cover1.find_input_flux( solarFlux) # compute the input sun power; puts in cover1.suninputpwr    
    
    return cover1


def find_sun_coverage( boa, body, startEpoch, stopEpoch, sampletime): #, userfile): #  Finds solar input at the body for a series of epochs at the given sample time step

    covobjects = []

    pointtimes = M.Epoch.range( startEpoch, stopEpoch, sampletime * units.sec ) 

    # Find input power to the coverage objects
    for time in pointtimes:

        covobj = single_sun_coverage(boa, "Sun", body, time ) 
        covobj.time = time      
        covobjects.append(covobj)

    print(" ")
    print("len(covobjects)  = ", len(covobjects))    

    polyshape = config.earth.shape.get_polyshape(boa, 'Earth')
    numPolyFaces = polyshape.numFaces() 
    print("The number of faces for this polyhedral is: ", polyshape.numFaces() )

    # loop through each face in the polyhedron

    facenum = 0
    totalprojarea = 0.0
    totsurfarea = 0.0
    scfaces = []  # contains a list of scface (SolCovFace) objects


    while facenum < numPolyFaces:

        facestate = M.State( polyshape.faceCenter(facenum) ) 
        facelongitude = M.Spherical.longitude(facestate).convert('deg')
        facelatitude = M.Spherical.latitude(facestate).convert('deg')  
        facearea = polyshape.faceArea(facenum)
        scface = SolCovFace(facenum, facelongitude, facelatitude, facearea)
        scface.solpwr = 0
        
        for cover in covobjects:  # go thru each coverage object

            for coverface in cover.facelist:   # go thru each facelist in the coverage object

                if coverface.num == facenum:  # finds the face corresponding to the facenum in the facelist
                    scface.solpwr  = scface.solpwr + coverface.solpwr  # tally the solar power for each face
                    scface.tprof.append([ cover.time, coverface.solpwr ])  #adds thermal profile data [ time, power ] to the tprof list
        
        scface.energy = scface.solpwr * sampletime  # Watts * secs = Joules, base value on the last tally total in .solpwr

        scfaces.append(scface)
        #print(" ")
        #print("face ", facenum  ," latitude: ", facelatitude , " scface.solpwr (sum of Watts points) = ", scface.solpwr, " scface.energy (Joules) = ", scface.energy )    

        facenum = facenum + 1


    print(" ")
    print("len(scfaces)  = ", len(scfaces))    

    return scfaces

    # loop through each face and plot latitude vs power for the time period.  Expressing power as input power / face surface area.  Convert the plot to a daily lat vs power relation.


def thermal_profile( scfaces, faceindex ):  # for a face (index = 0 thru max # faces - 1), output the thermal profile avg power and flux for the SolCovFace.tprof array of power
    print(" ")
    print(" ")
    print("Running thermal profile for face number: ", faceindex)
    scface = scfaces[faceindex]
    numpoints = len(scface.tprof)
    print("total number of time points: ", numpoints)
    print("location:  long: ", scface.long, " lat: ", scface.lat)
    print("face area, sq. km: ", scface.area)
    sunrise = scface.tprof[0][0]
    print("Sunrise: ", scface.tprof[0][0])
    sunset = scface.tprof[numpoints-1][0]
    print("Sunset: " ,  scface.tprof[numpoints-1][0])

    if numpoints == 0:
        avgpwr = 0.0

    else:
        pwrtally = 0.0

        for point in scface.tprof:
            #print(" ")
            #print( "time: ", point[0], " power, W: ", point[1] )
            pwrtally = pwrtally + point[1]  # tally to find average pwr - will sum all these then divide by the number of data points to avg.
        
        avgpwr = pwrtally/numpoints
        avgflux = avgpwr/scface.area
        avgenergy = scface.energy/scface.area

    print(" ")
    print("total avg power, W: ",avgpwr, " scface.solpwr: ", scface.solpwr , " avg pwr per unit area (km^2): ", avgflux ) 
    print("Energy: scface.energy/scface.area:  Energy per unit area (Joules/km^2): ", avgenergy )
    #if numpoints >= 2:
    #    timeint = scface.tprof[1][0] - scface.tprof[0][0] 
    #else:
    #    timeint = 0
    #print("numpoints: ", numpoints, " timeint: ",timeint)    
    #print("tot energy based on avg pwr over duration: (avgpwr * duration)", avgpwr * numpoints * timeint )
    totenergy = scface.energy
    print("scface.energy = ", scface.energy)

    return avgpwr, avgflux, totenergy, sunrise, sunset


    # Create a data frame that has the daily accum power for each face, long, lat, solar input start time (sun rise). solar input stop time (sunset)


    # find a relation to daily input power and temperature vs. time





