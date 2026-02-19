# analysis for ground track and altitude

import Monte as M
import mpy.units as units
#from analysis import headers
from .. import headers
import pandas as pd
import matplotlib.pyplot as plt

fixedFrameName = 'IAU Earth Fixed' 
framePoleName = 'IAU Earth Pole'


def groundtrack_latlong( boa, sc, epoch ): # single state point

    query = M.TrajQuery( boa, sc.name, 'Earth', framePoleName ) 
    oscState = query.state( epoch, framePoleName ) # ie framePoleName = 'IAU Earth Pole')
    oscStateEarth = query.state( epoch, fixedFrameName ) # ie frameFixedName = 'IAU Earth Fixed'
    func = M.OptMltanFunc( boa, "Mean Local Solar time", sc.name , "Earth", epoch )
    mltanlist = func.values()
    # sma = M.Conic.semiMajorAxis(oscState).convert('km')

    print ('Printing ground track elements:')
    print ('In True coordinates:')
    print ('      Epoch                      SMA       Radius     Height       Lat       Lon      MLTAN(conic) MLTAN(optfunc)')
    print ('%s  %.3f  %9.4f  %9.4f  %9.4f  %9.4f  %9.4f  %s' %(epoch.format()
            ,M.Conic.semiMajorAxis(oscState).convert('km')
            ,M.Conic.radius(oscState).convert('km')
            ,M.Geodetic.height(oscStateEarth).convert('km')
            #,M.Conic.eccentricity(oscState)
            #,M.Conic.argumentOfPeriapsis(oscState).convert('deg')
            #,M.Conic.inclination(oscState).convert('deg')
            #,M.Conic.argumentOfLatitude(oscState).convert('deg')
            ,M.Spherical.latitude(oscStateEarth).convert('deg')
            ,M.Spherical.longitude(oscStateEarth).convert('deg')
            #,M.Spherical.velocity(oscState).convert('m/s')
            ,M.Conic.meanLocalTimeAtNode(oscStateEarth).convert('hours')
            ,mltanlist[0].convert('hours')
            ) )
    print ('')


def find_geodeticht_longlat(boa, sc, startEpoch, stopEpoch, sampletime, userfile): # finds position and velocity vs time (no outframe needed)
    
    #'EME2000' ref replaced with outframeName = name of desired output frame

    heights = []
    times =[]
    longlat= []
    mltans = []
    #coord = M.CoordSetBoa.read( boa ) # use for coord rotation to outframeName

    print ('Geodetic Height and Long/Lat')  

    # Print Section

    pointtimes = M.Epoch.range( startEpoch, stopEpoch, sampletime * units.sec )
    
    header = headers.make_header( userfile )
    print("header: ", header)
    print ('Printing geodetic height and Long/Lat every ' +str(sampletime)+ ' secs (m/s**2):')  #section for screen print - can cut out
    print ('Epoch   %s  ' % (pointtimes[0].format()))
    print ('Seconds    Height   Long    Lat    MLTAN')

    for time in pointtimes:

        query = M.TrajQuery( boa, sc.name, 'Earth', framePoleName ) 
        #oscState = query.state( time, framePoleName ) # ie framePoleName = 'IAU Earth Pole')
        oscStateEarth = query.state( time, fixedFrameName ) # ie frameFixedName = 'IAU Earth Fixed'        

        reltime = (time-startEpoch).convert('sec')
        longitude = M.Spherical.longitude(oscStateEarth).convert('deg')
        latitude = M.Spherical.latitude(oscStateEarth).convert('deg')
        height = M.Geodetic.height(oscStateEarth).convert('km')
        mltan = M.Conic.meanLocalTimeAtNode(oscStateEarth).convert('hours')

        statestring = '%6.2f %.3f %3.5f %3.5f %2.3f' %( reltime  #section for screen print - can cut out
                    ,height        
                    ,longitude
                    ,latitude
                    ,mltan
                    )

        print( statestring )
        print ('')

        heights.append(height)
        times.append(reltime)
        longlat.append([ longitude, latitude])
        mltans.append( mltan )

    print ('')

    outset = [times, heights, longlat, mltans]
    return outset 


def geodeticht_longlat_csv(outset, outpath, startEpoch, sampletime, userfile):  #csv output

    fileName = outpath + '/' + 'geodeticht_longlat.csv'

    outfile = open( fileName, 'w' )

    header = headers.make_header( userfile )
    print("header: ", header)
    outfile.write( header + '\n')

    headerline1='Printing Geodetic Height and Long/Lat every ' +str(sampletime)+ ' secs (m/s**2):\n\
    Epoch   %s \n' % (startEpoch.format()) + \
    'Secs_from_Epoch, Height, Long, Lat, MLTAN\n' 
    print(headerline1)
    outfile.write( headerline1 + '\n')

    times = outset[0]
    heights = outset[1]
    longlats = outset[2]
    mltans = outset[3]
    longs = []
    lats = []
    hts = []
    mls = []
    
    count = 0
    for time in times:
        longitude = longlats[count][0]
        latitude = longlats[count][1]
        height = heights[count] 
        mltan = mltans[count]  

        longs.append(longitude)
        lats.append(latitude)
        hts.append(height)
        mls.append(mltan)
       
        outstring = '%6.3f, %.3f, %3.5f, %3.5f, %2.3f' %( time
            ,height
            ,longitude
            ,latitude
            ,mltan
            )   

        #print( outstring )
        outfile.write( outstring + '\n')
        count = count + 1
    
    outfile.close()


'''

    #datalen = [len(times),len(longlats),len(vecs),len(longs),len(lats),len(xvals),len(yvals),len(zvals), len(vmag) ]# check vector lengths
    #print('datalen: ',datalen)
    
    if userfile.PLOT[ 'ploton' ] &  userfile.PLOT[ 'position_plot' ]: 

        data = {
            #'longitude': longs,
            #'latitude': lats,
            'position_x': xvals,
            'position_y': yvals,
            'position_z': zvals,
            'position_mag': posmag,       
        }

        row_labels = times 
        df = pd.DataFrame(data = data, index = row_labels)

        df.plot()
        plt.savefig( outpath + '/' + 'position.png' )
        plt.show()
        plt.close()

        #longlatdf = df.loc[:,['longitude','latitude']]
        #longlatdf.plot()
        #plt.show()
    
        print('Position df: \n', df)
        print(' ')

    if userfile.PLOT[ 'ploton' ] &  userfile.PLOT[ 'velocity_plot' ]: 

        data = {
            #'longitude': longs,
            #'latitude': lats,
            'velocity_x': vxvals,
            'velocity_y': vyvals,
            'velocity_z': vzvals,
            'velocity_mag': velmag,       
        }

        row_labels = times 
        df = pd.DataFrame(data = data, index = row_labels)

        df.plot()
        plt.savefig( outpath + '/' + 'velocity.png' )
        plt.show()
        plt.close()

        #longlatdf = df.loc[:,['longitude','latitude']]
        #longlatdf.plot()
        #plt.show()
    
        print('Velocity df: \n', df)        
        print(' ')


'''