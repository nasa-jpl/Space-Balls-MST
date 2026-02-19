import Monte as M
import mpy.units as units
from analysis import headers
import pandas as pd
import matplotlib.pyplot as plt

fixedFrameName = 'IAU Earth Fixed' 
framePoleName = 'IAU Earth Pole'

def find_body_position(boa, sc, sensor, startEpoch, stopEpoch, outframeName, sampletime, userfile): # finds body position
    
    #'EME2000' ref replaced with outframeName = name of desired output frame

    vecs = []
    times =[]
    longlat= []
    coord = M.CoordSetBoa.read( boa ) # use for coord rotation to outframeName

    print ('Body Position')  

    # Body Position Vector Relative to a Sensor - Print Section

    pointtimes = M.Epoch.range( startEpoch, stopEpoch, sampletime * units.sec )
    
    header = headers.make_header( userfile )
    print("header: ", header)
    print ('Printing Body Position every ' +str(sampletime)+ ' secs (m):')  #section for screen print - can cut out
    print ('Epoch   %s  ' % (pointtimes[0].format()))
    print ('Seconds    Longitude    Lattitude     X              Y              Z             R  ')

    for time in pointtimes:

        query = M.TrajQuery( boa, sc.name, 'Earth', framePoleName ) 
        #oscState = query.state( epoch, framePoleName ) # ie framePoleName = 'IAU Earth Pole')
        oscStateEarth = query.state( time, fixedFrameName ) # ie frameFixedName = 'IAU Earth Fixed'        

        reltime = (time-startEpoch).convert('sec')
        longitude = M.Spherical.longitude(oscStateEarth).convert('deg')
        latitude = M.Spherical.latitude(oscStateEarth).convert('deg')

        statestring = '%6.2f %3.5f %3.5f %.15f %.15f %.15f %.15f' %( reltime  #section for screen print - can cut out
                    ,longitude
                    ,latitude
                    ,sensor.bodyPos(time, 'Earth', outframeName)[0]*1000# default position is in km
                    ,sensor.bodyPos(time, 'Earth', outframeName)[1]*1000
                    ,sensor.bodyPos(time, 'Earth', outframeName)[2]*1000
                    ,sensor.bodyPos(time, 'Earth', outframeName).mag()*1000
                    ) 
        print( statestring )

        #eme2000ToNadir = coord.rotation(time, outframeName, 'EME2000') #rotation for going from EME2000 to outframeName

        nadirvec = sensor.bodyPos(time, 'Earth', outframeName)  #

        vecs.append(nadirvec * 1000) #store vectors
        times.append(reltime)
        longlat.append([ longitude, latitude])
        #vectimes.append([earthAlbedoPress.accel(time,'EME2000')*1000, reltime])

    print ('')

    outset = [times, longlat, vecs]

    return outset #, vectimes