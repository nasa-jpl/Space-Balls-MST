import Monte as M
import mpy.units as units
#from analysis import headers
from .. import headers
from mst.config.time_handling import convert_Epoch_to_DateStr
import pandas as pd
import matplotlib.pyplot as plt
import os

fixedFrameName = 'IAU Earth Fixed' 
framePoleName = 'IAU Earth Pole'

def find_albedo_accel(boa, sc, earthAlbedoPress, startEpoch, stopEpoch, outframeName, sampletime, userfile ): # finds albedo acceleration given an Albedo Pressure object 
    
    #'EME2000' ref replaced with outframeName = name of desired output frame

    vecs = []
    times =[]
    longlat= []
    coord = M.CoordSetBoa.read( boa ) # use for coord rotation to outframeName

    print ('Albedo Acceleration')  

    # Albedo Accel Print Section

    pointtimes = M.Epoch.range( startEpoch, stopEpoch, sampletime * units.sec )
    
    header = headers.make_header( userfile )
    print("header: ", header)
    print ('Printing albedo+thermal acceleration every ' +str(sampletime)+ ' secs (m/s**2):')  #section for screen print - can cut out
    print ('Epoch   %s  ' % (pointtimes[0].format()))
    print ('Seconds    Longitude    Lattitude     AX              AY              AZ             AMAG ')

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
                    ,earthAlbedoPress.accel(time, 'EME2000')[0]*1000# default accel is in km/s**2
                    ,earthAlbedoPress.accel(time, 'EME2000')[1]*1000
                    ,earthAlbedoPress.accel(time, 'EME2000')[2]*1000
                    ,earthAlbedoPress.accel(time, 'EME2000').mag()*1000
                    ) 
        #print( statestring )

        eme2000ToNadir = coord.rotation(time, outframeName, 'EME2000') #rotation for going from EME2000 to outframeName

        nadirvec = eme2000ToNadir * earthAlbedoPress.accel(time, 'EME2000') #transform operation to find vector in nadir frame

        vecs.append(nadirvec * 1000) #store vectors
        times.append(reltime)
        longlat.append([ longitude, latitude])
        #vectimes.append([earthAlbedoPress.accel(time,'EME2000')*1000, reltime])

    print ('')

    outset = [times, longlat, vecs]

    return outset #, vectimes


def albedo_accel_txt(outset, startEpoch):  # text file output, no commas - deprecated

    fileName = 'albedo_accel.txt'

    outfile = open( fileName, 'w' )

    headerline1='Printing albedo+thermal acceleration every 60 secs (m/s**2):'
    print(headerline1)
    outfile.write( headerline1 + '\n')
    headerline2='Epoch   %s  ' % (startEpoch.format())
    print (headerline2)
    outfile.write( headerline2 + '\n')
    headerline3='Seconds     AX              AY              AZ             AMAG '
    print (headerline3)
    outfile.write( headerline3 + '\n')

    times = outset[0]
    vecs = outset[1]
    timelen=len(times)
    count = 0

    for time in times:
        x = vecs[count][0]#.value()
        y = vecs[count][1]#.value()
        z = vecs[count][2]#.value()  
        mag = vecs[count].mag()  

        outstring = '%6.2f %.15f %.15f %.15f %.15f' %( time
            ,x 
            ,y 
            ,z 
            ,mag
            )   

        print( outstring )
        outfile.write( outstring + '\n')
        count = count + 1
    
    outfile.close()


def albedo_accel_csv(outset, outpath, startEpoch, sampletime, userfile, initstartEpoch):  #csv output

    datestring = convert_Epoch_to_DateStr(startEpoch)
    
    fileName = outpath + '/csvoutput/' + 'albedo_accel_'+datestring+'.csv'
    fileName2 = outpath + '/csvoutput/' + 'albedo_accel_totrun.csv'
    
    #check if filename2 exists already.  If so, delete it since the file is appended to below
    if os.path.exists(fileName2):
        if startEpoch == initstartEpoch:  # it must match only when they are equal or it will keep deleting each day
            os.remove(fileName2)

    outfile = open( fileName, 'w' )
    outfile2 = open( fileName2, 'a' )

    header = headers.make_header( userfile )
    print("header: ", header)
    outfile.write( header + '\n')

    headerline1='Printing albedo+thermal acceleration every ' +str(sampletime)+ ' secs (m/s**2):\n\
    Epoch   %s \n' % (startEpoch.format()) + \
    'Secs_from_Epoch, Long, Lat, Ar, Ai, Ac, AMAG\n' # ric = radial, intrack, crosstrack
    print(headerline1)
    outfile.write( headerline1 + '\n')

    times = outset[0]
    longlats = outset[1]
    longs = []
    lats = []
    vecs = outset[2]
    xvals = []
    yvals = []
    zvals = []
    vmag = []

    count = 0
    for time in times:
        longitude = longlats[count][0]
        latitude = longlats[count][1]
        x = vecs[count][0]#.value()
        y = vecs[count][1]#.value()
        z = vecs[count][2]#.value()  
        mag = vecs[count].mag()  
        longs.append(longitude)
        lats.append(latitude)
        xvals.append(x)
        yvals.append(y)
        zvals.append(z)
        vmag.append(mag) #derived value for the df only - using loop to create

        outstring = '%6.2f, %3.5f, %3.5f, %.15f, %.15f, %.15f, %.15f' %( time
            ,longitude
            ,latitude
            ,x 
            ,y 
            ,z 
            ,mag
            )   

        #print( outstring )
        outfile.write( outstring + '\n')
        if time < 86400:
            outfile2.write( outstring + '\n')       
        count = count + 1
    
    outfile.close()
    outfile2.close()    
    
    #datalen = [len(times),len(longlats),len(vecs),len(longs),len(lats),len(xvals),len(yvals),len(zvals), len(vmag) ]# check vector lengths
    #print('datalen: ',datalen)
    
    if userfile.PLOT[ 'ploton' ] &  userfile.PLOT[ 'albedo_plot' ]: 

        data = {
            #'time': times,
            #'longitude': longs,
            #'latitude': lats,
            'albedo_accel_r': xvals,
            'albedo_accel_i': yvals,
            'albedo_accel_c': zvals,
            'albedo_accel': vmag,       
        }
        row_labels = times 
        df = pd.DataFrame(data = data, index = row_labels)

        df.plot()
        plt.savefig( outpath + '/pngoutput/' + 'albedo_accel'+datestring+'.png' )        
        plt.show()
        plt.close()

        #longlatdf = df.loc[:,['longitude','latitude']]

        #longlatdf.plot()
        #plt.show()
    
        print('Albedo Accel df: \n', df)