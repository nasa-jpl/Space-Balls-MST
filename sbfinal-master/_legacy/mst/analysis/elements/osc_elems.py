# osculating elemtents scripts, non AN related

import Monte as M
import mpy.units as units
#from analysis import headers
from .. import headers
from mst.config.time_handling import convert_Epoch_to_DateStr
import pandas as pd
import matplotlib.pyplot as plt

fixedFrameName = 'IAU Earth Fixed' 
framePoleName = 'IAU Earth Pole'


def osc_elements_latlongV( boa, sc, epoch ): # single state point

    query = M.TrajQuery( boa, sc.name, 'Earth', framePoleName ) 
    oscState = query.state( epoch, framePoleName ) # ie framePoleName = 'IAU Earth Pole')
    oscStateEarth = query.state( epoch, fixedFrameName ) # ie frameFixedName = 'IAU Earth Fixed'

    # sma = M.Conic.semiMajorAxis(oscState).convert('km')

    print ('Printing osculating elements:')
    print ('In True coordinates:')
    print ('      Epoch                      SMA        Ecc      ArgPeri      Incl         ArgLat     Lat       Lon      Velocity')
    print ('%s  %.3f  %.6f  %8.2f     %8.5f  %9.4f   %9.4f  %9.4f  %11.4f' %(epoch.format()
            ,M.Conic.semiMajorAxis(oscState).convert('km')
            ,M.Conic.eccentricity(oscState)
            ,M.Conic.argumentOfPeriapsis(oscState).convert('deg')
            ,M.Conic.inclination(oscState).convert('deg')
            ,M.Conic.argumentOfLatitude(oscState).convert('deg')
            ,M.Spherical.latitude(oscStateEarth).convert('deg')
            ,M.Spherical.longitude(oscStateEarth).convert('deg')
            ,M.Spherical.velocity(oscState).convert('m/s')
            ) )
    print ('')


def find_pos_vel(boa, sc, startEpoch, stopEpoch, outframeName, sampletime, userfile): # finds position and velocity vs time 
    
    #'EME2000' ref replaced with outframeName = name of desired output frame

    posvecs = []
    velvecs = []
    times =[]
    longlat= []
    coord = M.CoordSetBoa.read( boa ) # use for coord rotation to outframeName

    print ('Position and Velocity')  

    # Print Section

    pointtimes = M.Epoch.range( startEpoch, stopEpoch, sampletime * units.sec )
    
    header = headers.make_header( userfile )
    print("header: ", header)
    print ('Printing position and velocity every ' +str(sampletime)+ ' secs (m/s**2):')  #section for screen print - can cut out
    print ('Epoch   %s  ' % (pointtimes[0].format()))
    print ('Seconds   Long    Lat        X           Y           Z           POSmag           VX          VY          VZ           VELmag')

    for time in pointtimes:

        query = M.TrajQuery( boa, sc.name, 'Earth', framePoleName ) 
        oscState = query.state( time, framePoleName ) # ie framePoleName = 'IAU Earth Pole')
        oscStateEarth = query.state( time, fixedFrameName ) # ie frameFixedName = 'IAU Earth Fixed'        

        reltime = (time-startEpoch).convert('sec')
        longitude = M.Spherical.longitude(oscStateEarth).convert('deg')
        latitude = M.Spherical.latitude(oscStateEarth).convert('deg')
        x = M.Cartesian.x(oscState).convert('km')
        y = M.Cartesian.y(oscState).convert('km')
        z = M.Cartesian.z(oscState).convert('km')
        vx = M.Cartesian.dx(oscState).convert('km/s')
        vy = M.Cartesian.dy(oscState).convert('km/s')
        vz = M.Cartesian.dz(oscState).convert('km/s')
        posvec = M.Dbl3Vec(x,y,z)
        velvec = M.Dbl3Vec(vx,vy,vz)

        statestring = '%6.2f %3.5f %3.5f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f' %( reltime  #section for screen print - can cut out
                    ,longitude
                    ,latitude
                    ,M.Cartesian.x(oscState).convert('km')
                    ,M.Cartesian.y(oscState).convert('km')
                    ,M.Cartesian.z(oscState).convert('km')
                    ,posvec.mag()
                    ,M.Cartesian.dx(oscState).convert('km/s')
                    ,M.Cartesian.dy(oscState).convert('km/s')
                    ,M.Cartesian.dz(oscState).convert('km/s')
                    ,velvec.mag()
                    )

        print( statestring )
        print ('')

        # use for coord transformations ie to nadir LVLH
        
        eme2000ToNadir = coord.rotation(time, outframeName, framePoleName) #'EME2000') #rotation for going from EME2000 to outframeName
        nadirposvec = eme2000ToNadir * posvec  #transform operation to find vector in nadir frame
        nadirvelvec = eme2000ToNadir * velvec

        posvecs.append(nadirposvec) #store vectors
        velvecs.append(nadirvelvec)
        times.append(reltime)
        longlat.append([ longitude, latitude])

    print ('')

    outset = [times, longlat, posvecs, velvecs]
    return outset 


def pos_vel_csv(outset, outpath, startEpoch, sampletime, userfile):  #csv output

    datestring = convert_Epoch_to_DateStr(startEpoch)
    fileName = outpath + '/csvoutput/' + 'pos_vel_'+datestring+'.csv'

    outfile = open( fileName, 'w' )

    header = headers.make_header( userfile )
    print("header: ", header)
    outfile.write( header + '\n')

    headerline1='Printing Position and Velocity every ' +str(sampletime)+ ' secs (m/s**2):\n\
    Epoch   %s \n' % (startEpoch.format()) + \
    'Secs_from_Epoch, Long, Lat, X, Y, Z, Pmag, Vx, Vy, Vz, Vmag\n' # ric = radial, intrack, crosstrack
    print(headerline1)
    outfile.write( headerline1 + '\n')

    times = outset[0]
    longlats = outset[1]
    longs = []
    lats = []
    pvecs = outset[2]
    xvals = []
    yvals = []
    zvals = []
    posmag = []
    vvecs = outset[3]
    vxvals = []
    vyvals = []
    vzvals = []
    velmag = []

    count = 0
    for time in times:
        longitude = longlats[count][0]
        latitude = longlats[count][1]
        x = pvecs[count][0]#.value()
        y = pvecs[count][1]#.value()
        z = pvecs[count][2]#.value()  
        pmag = pvecs[count].mag()  
        vx = vvecs[count][0]#.value()
        vy = vvecs[count][1]#.value()
        vz = vvecs[count][2]#.value()  
        vmag = vvecs[count].mag()     
        longs.append(longitude)
        lats.append(latitude)
        xvals.append(x)
        yvals.append(y)
        zvals.append(z)
        posmag.append(pmag) #derived value for the df only - using loop to create
        vxvals.append(vx)
        vyvals.append(vy)
        vzvals.append(vz)
        velmag.append(vmag) #derived value for the df only - using loop to create        

        outstring = '%6.3f, %3.5f, %3.5f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f' %( time
            ,longitude
            ,latitude
            ,x 
            ,y 
            ,z 
            ,pmag
            ,vx
            ,vy
            ,vz
            ,vmag
            )   

        #print( outstring )
        outfile.write( outstring + '\n')
        count = count + 1
    
    outfile.close()
    
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
        plt.savefig( outpath + '/pngoutput/' + 'position_'+datestring+'.png' )
        
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
        plt.savefig( outpath + '/pngoutput/' + 'velocity_'+datestring+'.png' )
        plt.show()
        plt.close()

        #longlatdf = df.loc[:,['longitude','latitude']]
        #longlatdf.plot()
        #plt.show()
    
        print('Velocity df: \n', df)        
        print(' ')




