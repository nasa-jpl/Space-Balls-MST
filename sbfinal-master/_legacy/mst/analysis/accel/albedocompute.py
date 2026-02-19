import Monte as M
import mpy.units as units
#from analysis import headers
from .. import headers
import pandas as pd
import matplotlib.pyplot as plt
import math

from ..coverage import coverage_calcs
from ..solarflux import find_solar_flux, find_sun_dir
from mst.input_tools import csvinput
from mst.config.files import file_sort, last_Xchars
from mst.config.time_handling import find_UT_hour


fixedFrameName = 'IAU Earth Fixed' 
framePoleName = 'IAU Earth Pole'

#placeholder for future file gen and plot functions
#def find_albedo_accel(boa, sc, earthAlbedoPress, startEpoch, stopEpoch, outframeName, sampletime, userfile): # finds albedo acceleration given an Albedo Pressure object 

# these functions assume the following has been run:

# 
    # polyshape = config.earth.shape.set_polyshape( boa )   # create a polyshape for the Earth type: PolyShapeRegion
    # facelist, totalarea, totalprojarea, longitude, latitude = analysis.sensor.coverage.find_poly_coverage( boa, sc, polyshape, th.start_epoch, th.stop_epoch, outFrameName, sampletime, userfile)
 
#

def net_albedo_vector(facelist, totalprojarea, solarFlux, userfile):   #(boa, polyshape, facelist, totalarea, totalprojarea, longitude, latitude):
    # produces a net albedo pressure vector given all the polyhedral faces from polyshape given in the facelist
    # The weight of each face is given by the ratio:  facearea/totalprojarea
    # facelist format from  analysis.sensor.coverage.py:  [ faceindex, projected area, relative position vector, long, lat ]
    c = 2.998e8 #speed of light, m/s

    vectortally = M.Dbl3Vec(0.0, 0.0, 0.0) #start the vector to add cummulative vectors

    for face in facelist:
        #index = face.num #[0]
        projarea = face.projarea #[1]
        relposdir = face.relpos #[2].unit()
        dist_to_sc = face.scdist 
        #print('dist_to_sc', dist_to_sc, ' type: ', type(dist_to_sc) )
        
        #longitude = face.long #[3]
        #latitude = face.lat #[4]

        #choose method in user file: source from csv file or solar flux input/output method

        if ( userfile.INPUT[ 'use_albedofile'] | userfile.INPUT[ 'use_albedofolder'] ) :
            #albedoratio = face.albedoratio # is the albedo from the csv
            #albedoflux = face.albedoflux # is the albedo flux computed from csv

            datafluxval = face.albedoflux # is the albedo flux computed from csv
            albedoflux = datafluxval * projarea / ( math.pi * dist_to_sc**2 )/units.km/units.km # make unitless
            arearatio = 1.0  # so equation still works below

            #print('datafluxval: ', datafluxval, ' projarea: ',projarea, ' dist_to_sc: ',dist_to_sc, ' albedoflux: ', albedoflux)

        else:    
            albedoratio = face.maxalbedofluxratio  # calculated from coverage_calcs find_output_flux function
            
            #albedovalue = find_albedo_val( longitude, latitude)  # function that extracts the albedo value given a long and lat
            #albedovalue = 0.52527  #placeholder  (note 1.75 factor needed  0.3 * 0.52527; this could be a function of the altitude and view angle from sc)
            #solarflux = find_solar_flux( epoch )  # function that computes solar flux at Earth given a day of year or epoch
            #solarflux = 1360.0 #placeholder W/m^2  # now computed in find_solar_flux function in analysis.solarflux.py
            #albedoflux = solarFlux * albedovalue

            albedoflux = solarFlux * albedoratio
            arearatio = projarea/totalprojarea

        albedopressmag = arearatio * albedoflux / c  # units of pressure are in Newtons/m^2
        albedoforcemag = albedopressmag * userfile.INPUT[ 'area' ]  # Ex for parameters from userfile: userfile.PLOT[ 'ploton' ]
        albedoaccelmag = albedoforcemag / userfile.INPUT[ 'mass' ] 

        albedovec = albedoaccelmag * relposdir
        #print('albedoaccelmag: ', albedoaccelmag, ' relposdir: ', relposdir, ' type: relposdir: ', type(relposdir) , )
        #print('albedovec: ', albedovec, ' type: ', type(albedovec)  )
        #dblalbedovec = albedovec.value()
        #print('albedovec: ', dblalbedovec, ' type: ', type(dblalbedovec)  )
        vectortally = vectortally + albedovec

        #print("face: ", index, " albedo press mag: ", albedopressmag, " albedo vec: ", albedovec, " vector tally: ", vectortally)
    '''
    #print("")
    if vectortally.mag() > 0:
        print("vector tally result: magnitude: ", vectortally.mag(), " components: ", vectortally, " unit vec: ", vectortally.unit() )
    else: 
        print("vector tally result: magnitude: ", vectortally.mag(), " components: ", vectortally, )
    #note: for the albedovalue of 0.52527 expect a magnitude of 1.36e-6
    '''
    return vectortally


def net_iralbedo_vector(facelist, totalprojarea, solarFlux, userfile):   #(boa, polyshape, facelist, totalarea, totalprojarea, longitude, latitude):
    # produces a net albedo pressure vector given all the polyhedral faces from polyshape given in the facelist
    # The weight of each face is given by the ratio:  facearea/totalprojarea
    # facelist format from  analysis.sensor.coverage.py:  [ faceindex, projected area, relative position vector, long, lat ]
    c = 2.998e8 #speed of light, m/s

    vectortally = M.Dbl3Vec(0.0, 0.0, 0.0) #start the vector to add cummulative vectors

    for face in facelist:

        projarea = face.projarea #[1]
        relposdir = face.relpos #[2].unit()
        dist_to_sc = face.scdist 

        #choose method in user file: source from csv file or solar flux input/output method

        if ( userfile.INPUT[ 'use_iralbedofile'] | userfile.INPUT[ 'use_iralbedofolder'] ):    # userfile.INPUT[ 'use_iralbedofile'] :
            #albedoratio = face.iralbedoratio # is the albedo from the csv;  
            #albedoflux = face.iralbedoflux
            datafluxval = face.iralbedoflux # is the IR albedo flux found from cover.assign_iralbedo_usedf, W/m^2
            albedoflux = datafluxval * projarea / ( math.pi * dist_to_sc**2 )/units.km/units.km # make unitless
            arearatio = 1.0  # so equation still works below            
        else:    
            albedoratio = face.maxiralbedofluxratio  # calculated from coverage_calcs find_outputir_flux function
            albedoflux = solarFlux * albedoratio

            arearatio = projarea/totalprojarea

        albedopressmag = arearatio * albedoflux / c  # units of pressure are in Newtons/m^2
        albedoforcemag = albedopressmag * userfile.INPUT[ 'area' ]  # Ex for parameters from userfile: userfile.PLOT[ 'ploton' ]
        albedoaccelmag = albedoforcemag / userfile.INPUT[ 'mass' ] 

        albedovec = albedoaccelmag * relposdir

        vectortally = vectortally + albedovec

        #print("face: ", index, " albedo press mag: ", albedopressmag, " albedo vec: ", albedovec, " vector tally: ", vectortally)
    '''
    #print("")
    if vectortally.mag() > 0:
        print("vector tally result: magnitude: ", vectortally.mag(), " components: ", vectortally, " unit vec: ", vectortally.unit() )
    else: 
        print("vector tally result: magnitude: ", vectortally.mag(), " components: ", vectortally, )
    #note: for the albedovalue of 0.52527 expect a magnitude of 1.36e-6
    '''
    return vectortally


def create_albedo_force( albedo_vector ):

    #given an albedo vector, create the force object to pass
    # ref: https://monte.jpl.nasa.gov/monte/doc/160.1/source/Monte/PyForce.html
    # https://monte.jpl.nasa.gov/monte/doc/160.1/source/Monte/PySimpleForce.html#monte-pysimpleforce
    
    pass

def find_albedo_forces(boa, sc, body, start, stop):

    #iterates from start time to stop time to find the net albedo force vectors at each time step

    pass


def find_sceneid(surftype, cloudfrac): #surftype is 1 to 4, cloudfrac is between 0 and 1

    if cloudfrac > 0.95:
        sid = 12 
    if surftype == 4:
        if cloudfrac < 0.05:
            sid = 1
        elif cloudfrac <= 0.5:
            sid = 6
        elif cloudfrac <= 0.95:
            sid = 9
    else:
        if cloudfrac < 0.05:
            if surftype == 1:
                sid = 2
            if surftype == 2:
                sid = 4
            if surftype == 3:
                sid = 3
        elif cloudfrac < 0.5:
            sid = 7
        elif cloudfrac <= 0.95:
            sid = 10

    return sid


def find_vzabin(vza): # divide into 7 bins based on value of vza in degrees
    if vza < 15:
        bin = 1
    elif vza < 27:
        bin = 2
    elif vza < 39:
        bin = 3
    elif vza < 51:
        bin = 4
    elif vza < 63:
        bin = 5
    elif vza < 75:
        bin = 6
    elif vza < 91:
        bin = 7
    return bin

def find_raabin(raa): # divide into 8 bins based on value of raa in degrees
    if raa < 9:
        bin = 1
    elif raa < 30:
        bin = 2
    elif raa < 60:
        bin = 3
    elif raa < 90:
        bin = 4
    elif raa < 120:
        bin = 5
    elif raa < 150:
        bin = 6
    elif raa < 171:
        bin = 7
    elif raa <= 180:
        bin = 8
    return bin

def find_szabin(sza): # divide into 10 bins based on value of sza in degrees
    if sza < 25.84:
        bin = 1
    elif sza < 36.87:
        bin = 2
    elif sza < 45.57:
        bin = 3
    elif sza < 53.13:
        bin = 4
    elif sza < 60:
        bin = 5
    elif sza < 66.42:
        bin = 6
    elif sza < 72.54:
        bin = 7
    elif sza < 78.46:
        bin = 8
    elif sza < 84.26:
        bin = 9
    elif sza <= 90:
        bin = 10
    return bin


def find_albedo_pressures(boa, sc, body, start_epoch, stop_epoch):  # test function - used to test single time step to create the net output vector to screen

    #iterates from start time to stop time to find the net albedo pressure vectors at each time step

    # from analysis/coverage/coverage_calcs.py create the Coverage object and run the method to find the faces

    # Coverage object
    cover1 = coverage_calcs.Coverage( boa, sc.name, body )
    cover1.find_faces( boa, start_epoch )
    #cover1.find_area()

    # find the net albedo vector for the coverage object
    albedo_vector = net_albedo_vector(cover1.facelist, cover1.totalprojarea)

    print("Albedo pressure computed for time: ", start_epoch)



def find_albedo_press(boa, sc, startEpoch, stopEpoch, outframeName, sampletime, userfile): # finds albedo acceleration using Coverage objects
    
    #'EME2000' ref replaced with outframeName = name of desired output frame
    body = 'Earth'  #body to find the current sun direction
    vecs = []
    times =[]
    longlat= []
    #solarFlux = find_solar_flux( boa, sc, startEpoch )  # use here if the time duration is short ( < 1 day )
    coord = M.CoordSetBoa.read( boa ) # use for coord rotation to outframeName

    if userfile.INPUT['use_albedofolder']:

        folder = userfile.INPUT[ 'albedofolder' ]
        file_list = file_sort(folder)  # sorting files in folder into a list sorted by last x chars
        print( 'Ordered file list in folder ', folder )
        for filename in file_list:
            print( "filename: ", filename )

    else:    
        albedofilename = userfile.INPUT[ 'albedofile' ]
        df = csvinput.read_albedo_from_csv(albedofilename)

    if userfile.INPUT['use_anisofiles']: # assign data frames for each of the 3 file types needed for anisotropy calcs

        df1 = csvinput.read_albedo_from_csv(userfile.INPUT[ 'swmeanfile' ])
        df2 = csvinput.read_albedo_from_csv(userfile.INPUT[ 'cloudfracfile' ])
        df3 = csvinput.read_albedo_from_csv(userfile.INPUT[ 'surftypefile' ])

    print(' ')
    print ('Albedo Pressure Acceleration - from Coverage Objects')  


    # Albedo Accel Print Section

    #samples = 105  # for testing - specify number of samples to generate
    #pointtimes = M.Epoch.range( startEpoch, startEpoch + samples * sampletime * units.sec  , sampletime * units.sec ) # set time points - temp
    #after testing is complete:
    
    pointtimes = M.Epoch.range( startEpoch, stopEpoch, sampletime * units.sec ) 
    

    header = headers.make_header( userfile )
    print("header: ", header)
    print(" ")
    print(" ")
    print ('Printing albedo pressure acceleration every ' +str(sampletime)+ ' secs (m/s**2):')  #section for screen print - can cut out
    print ('Epoch   %s  ' % (pointtimes[0].format()))
    print(" ")
    print ('Seconds    Longitude    Lattitude     AX              AY              AZ             AMAG ')
    print(" ")


    for time in pointtimes:

        if userfile.INPUT['use_albedofolder']:

            # find the hour in UT from 0 to 23:
            filenum = 0
            filenum = find_UT_hour( userfile.start, time )

            # find the filename for the right hour
            albedofilename = folder + file_list[ filenum + 1 ]
            print("chosen filenum / filename at time ", time, " is: ", filenum, " ", albedofilename)

            df = csvinput.read_albedo_from_csv(albedofilename)

        query = M.TrajQuery( boa, sc.name, 'Earth', framePoleName ) 
        #oscState = query.state( epoch, framePoleName ) # ie framePoleName = 'IAU Earth Pole')
        oscStateEarth = query.state( time, fixedFrameName ) # ie frameFixedName = 'IAU Earth Fixed'        

        reltime = (time-startEpoch).convert('sec')
        longitude = M.Spherical.longitude(oscStateEarth).convert('deg')
        latitude = M.Spherical.latitude(oscStateEarth).convert('deg')

        solarFlux = find_solar_flux( boa, sc, time )  # function that computes solar flux at Earth given a day of year from 'time' epoch
        
        #Coverage object compute section:
        cover1 = coverage_calcs.Coverage( boa, sc.name, "Earth" )
        cover1.find_faces( boa, time )  # find new faces at each time step
        sundir = find_sun_dir( boa, body, time, fixedFrameName) # finds the direction of the sun
        cover1.find_sun_angle( sundir ) # computes the sun angle on each face

        if ( userfile.INPUT[ 'use_albedofile'] | userfile.INPUT[ 'use_albedofolder'] ): 
            #cover1.assign_albedo( albedofilename ) #deprecated - too slow
            cover1.find_local_solar_flux( solarFlux) #  finds the localsolarflux for each face 
            cover1.assign_albedo_usedf( df ) # loads the csv file into a pandas data frame, given an albedo file, assigns an albedo to each face in view.
            if ( userfile.INPUT['use_anisofiles'] ):  # run through anisotropy routines
                cover1.assign_swmean_usedf( df1 )
                cover1.assign_cloudfrac_usedf( df2 )
                cover1.assign_surftype_usedf( df3 )

        else:    
            cover1.find_input_flux( solarFlux) # compute the input sun power; puts in cover1.suninputpwr ; finds the localsolarflux for each face while doing that and stores in faceobj
            cover1.find_output_flux( solarFlux )  # based on reflected solar factors for each face, finds the output flux for each face and the max albedo flux ratio to solar flux

        # find the net albedo vector for the coverage object - already in nadir frame
        albedo_vector = net_albedo_vector(cover1.facelist, cover1.totalprojarea, solarFlux, userfile)        

        statestring = '%6.2f %3.5f %3.5f %.15f %.15f %.15f %.15f' %( reltime  #section for screen print - can cut out
                    ,longitude
                    ,latitude
                    ,albedo_vector[0] #*1000# default accel is in km/s**2
                    ,albedo_vector[1] #*1000
                    ,albedo_vector[2] #*1000
                    ,albedo_vector.mag() #*1000
                    ) 
        print(" ")            
        print("StateString: ")
        print( statestring )
        print(" ")
        
        eme2000ToNadir = coord.rotation(time, outframeName, fixedFrameName )  #rotation for going from fixedFrame to outframeName
        nadirvec = eme2000ToNadir * albedo_vector   #transform operation to find vector in nadir frame

        vecs.append(nadirvec)  #(albedo_vector) # * 1000) #store vectors
        times.append(reltime)
        longlat.append([ longitude, latitude])
        #vectimes.append([earthAlbedoPress.accel(time,'EME2000')*1000, reltime])

    print ('')

    outset = [times, longlat, vecs]

    return outset #, vectimes


def find_iralbedo_press(boa, sc, startEpoch, stopEpoch, outframeName, sampletime, userfile): # finds IR albedo acceleration using Coverage objects
    
    #'EME2000' ref replaced with outframeName = name of desired output frame
    body = 'Earth'  #body to find the current sun direction
    vecs = []
    times =[]
    longlat= []
    #solarFlux = find_solar_flux( boa, sc, startEpoch )  # use here if the time duration is short ( < 1 day )
    coord = M.CoordSetBoa.read( boa ) # use for coord rotation to outframeName

    if userfile.INPUT['use_iralbedofolder']:

        folder = userfile.INPUT[ 'iralbedofolder' ]
        file_list = file_sort(folder)  # sorting files in folder into a list sorted by last x chars
        print( 'Ordered file list in folder ', folder )
        for filename in file_list:
            print( "filename: ", filename )

    else:    
        iralbedofilename = userfile.INPUT[ 'iralbedofile' ]
        df = csvinput.read_albedo_from_csv(iralbedofilename)

    print(' ')
    print ('IR Albedo Pressure Acceleration - from Coverage Objects')  


    # IR Albedo Accel Print Section
    
    pointtimes = M.Epoch.range( startEpoch, stopEpoch, sampletime * units.sec ) 
    
    header = headers.make_header( userfile )
    print("header: ", header)
    print(" ")
    print(" ")
    print ('Printing IR albedo pressure acceleration every ' +str(sampletime)+ ' secs (m/s**2):')  #section for screen print - can cut out
    print ('Epoch   %s  ' % (pointtimes[0].format()))
    print(" ")
    print ('Seconds    Longitude    Lattitude     AX              AY              AZ             AMAG ')
    print(" ")


    for time in pointtimes:

        if userfile.INPUT['use_iralbedofolder']:

            # find the hour in UT from 0 to 23:
            filenum = 0
            filenum = find_UT_hour( userfile.start, time )

            # find the filename for the right hour
            albedofilename = folder + file_list[ filenum + 1]
            print("chosen filenum / filename at time ", time, " is: ", filenum, " ", albedofilename)

            df = csvinput.read_albedo_from_csv(albedofilename)

        query = M.TrajQuery( boa, sc.name, 'Earth', framePoleName ) 
        #oscState = query.state( epoch, framePoleName ) # ie framePoleName = 'IAU Earth Pole')
        oscStateEarth = query.state( time, fixedFrameName ) # ie frameFixedName = 'IAU Earth Fixed'        

        reltime = (time-startEpoch).convert('sec')
        longitude = M.Spherical.longitude(oscStateEarth).convert('deg')
        latitude = M.Spherical.latitude(oscStateEarth).convert('deg')

        #Coverage object compute section:
        cover1 = coverage_calcs.Coverage( boa, sc.name, "Earth" )
        cover1.find_faces( boa, time )  # find new faces at each time step
        sundir = find_sun_dir( boa, body, time, fixedFrameName) # finds the direction of the sun
        cover1.find_sun_angle( sundir ) # computes the sun angle on each face

        solarFlux = find_solar_flux( boa, sc, time )  # function that computes solar flux at Earth given a day of year from 'time' epoch

        if ( userfile.INPUT[ 'use_iralbedofile'] | userfile.INPUT[ 'use_iralbedofolder'] ):
            #cover1.assign_albedo( albedofilename ) #deprecated - too slow
            #cover1.assign_albedo_usedf( df ) # loads the csv file into a pandas data frame, given an albedo file, assigns an albedo to each face in view.
            cover1.find_local_solar_flux( solarFlux) #  finds the localsolarflux for each face 
            cover1.assign_iralbedo_usedf( df ) # loads the csv file into a pandas data frame, given an IR albedo file, assigns an IR albedo to each face in view.

        else:    
            cover1.find_temp_flux() # solarFlux)
            #cover1.find_input_flux( solarFlux) # compute the input sun power; puts in cover1.suninputpwr
            #these are removed for this version - don't need sun dir/angle info:
            #sundir = find_sun_dir( boa, body, time, fixedFrameName) # finds the direction of the sun
            #cover1.find_sun_angle( sundir ) # computes the sun angle on each face
            #cover1.find_output_flux( solarFlux )  # based on reflected solar factors for each face, finds the output flux for each face and the max albedo flux ratio to solar flux
            cover1.find_output_irflux( solarFlux ) # based on temperature for each face, finds the output IR flux for each face and the max albedo flux ratio to solar flux

        # find the net albedo vector for the coverage object - already in nadir frame
        albedo_vector = net_iralbedo_vector(cover1.facelist, cover1.totalprojarea, solarFlux, userfile)        

        statestring = '%6.2f %3.5f %3.5f %.15f %.15f %.15f %.15f' %( reltime  #section for screen print - can cut out
                    ,longitude
                    ,latitude
                    ,albedo_vector[0] #*1000# default accel is in km/s**2
                    ,albedo_vector[1] #*1000
                    ,albedo_vector[2] #*1000
                    ,albedo_vector.mag() #*1000
                    ) 
        print(" ")            
        print("StateString: ")
        print( statestring )
        print(" ")
        
        eme2000ToNadir = coord.rotation(time, outframeName, fixedFrameName )  #rotation for going from fixedFrame to outframeName
        nadirvec = eme2000ToNadir * albedo_vector   #transform operation to find vector in nadir frame

        vecs.append(nadirvec)  #(albedo_vector) # * 1000) #store vectors
        times.append(reltime)
        longlat.append([ longitude, latitude])
        #vectimes.append([earthAlbedoPress.accel(time,'EME2000')*1000, reltime])

    print ('')

    outset = [times, longlat, vecs]

    return outset #, vectimes


def find_total_albedo_press(boa, sc, startEpoch, stopEpoch, outframeName, sampletime, userfile, outpath):
    print(' ')
    print(' ')
    print('Total Albedo press calc: ')
    print(' ')

    # compute SW albedo vector
    outset = find_albedo_press(boa, sc, startEpoch, stopEpoch, outframeName, sampletime, userfile)

    if userfile.PLOT[ 'albedo_cov_plot' ]:
        print(' ')
        print('Running albedo_press_csv')
        albedo_press_csv(outset, outpath, startEpoch, sampletime, userfile)

    # compute LW albedo vector        
    outset2 = find_iralbedo_press(boa, sc, startEpoch, stopEpoch, outframeName, sampletime, userfile)

    if userfile.PLOT[ 'iralbedo_cov_plot' ]:
        print(' ')
        print('Running iralbedo_press_csv')
        iralbedo_press_csv(outset2, outpath, startEpoch, sampletime, userfile)

    # combine them to find the total vector
    totvec = []
    count = 0
    for time in outset[0]:
        albedo_vec = outset[2][count]    
        iralbedo_vec = outset2[2][count]     
        totalbedo_vec = albedo_vec + iralbedo_vec
        totvec.append( totalbedo_vec )
        count = count + 1
        print('Time:', time, ' albedovec: ', albedo_vec, ' iralbedovec: ', iralbedo_vec, ' totalbedovec: ', totalbedo_vec )

    outset3 = [ outset[0], outset[1], totvec ]

    return outset3



def albedo_press_csv(outset, outpath, startEpoch, sampletime, userfile):  #csv output

    fileName = outpath + '/' + 'albedo_press.csv'

    outfile = open( fileName, 'w' )

    header = headers.make_header( userfile )
    print("header: ", header)
    outfile.write( header + '\n')

    headerline1='Printing albedo pressure vector every ' +str(sampletime)+ ' secs (units?):\n\
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

        print( outstring )
        outfile.write( outstring + '\n')
        count = count + 1
    
    outfile.close()
    
    #datalen = [len(times),len(longlats),len(vecs),len(longs),len(lats),len(xvals),len(yvals),len(zvals), len(vmag) ]# check vector lengths
    #print('datalen: ',datalen)

    
    if userfile.PLOT[ 'ploton' ] &  userfile.PLOT[ 'albedo_cov_plot' ]: 

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
        plt.savefig( outpath + '/' + 'albedo_cov_accel.png' )
        plt.show()
        plt.close()

        #longlatdf = df.loc[:,['longitude','latitude']]

        #longlatdf.plot()
        #plt.show()
    
        print('Albedo Coverage Accel df: \n', df)


def iralbedo_press_csv(outset, outpath, startEpoch, sampletime, userfile):  #csv output

    fileName = outpath + '/' + 'iralbedo_press.csv'

    outfile = open( fileName, 'w' )

    header = headers.make_header( userfile )
    print("header: ", header)
    outfile.write( header + '\n')

    headerline1='Printing IR albedo pressure vector every ' +str(sampletime)+ ' secs (units?):\n\
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

        print( outstring )
        outfile.write( outstring + '\n')
        count = count + 1
    
    outfile.close()
    
    #datalen = [len(times),len(longlats),len(vecs),len(longs),len(lats),len(xvals),len(yvals),len(zvals), len(vmag) ]# check vector lengths
    #print('datalen: ',datalen)

    
    if userfile.PLOT[ 'ploton' ] &  userfile.PLOT[ 'iralbedo_cov_plot' ]: 

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
        plt.savefig( outpath + '/' + 'iralbedo_cov_accel.png' )
        plt.show()
        plt.close()

        #longlatdf = df.loc[:,['longitude','latitude']]

        #longlatdf.plot()
        #plt.show()
    
        print('IR Albedo Coverage Accel df: \n', df)


def totalbedo_press_csv(outset, outpath, startEpoch, sampletime, userfile):  #csv output

    fileName = outpath + '/' + 'totalbedo_press.csv'

    outfile = open( fileName, 'w' )

    header = headers.make_header( userfile )
    print("header: ", header)
    outfile.write( header + '\n')

    headerline1='Printing Total Albedo pressure vector every ' +str(sampletime)+ ' secs (units?):\n\
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

        print( outstring )
        outfile.write( outstring + '\n')
        count = count + 1
    
    outfile.close()
    
    #datalen = [len(times),len(longlats),len(vecs),len(longs),len(lats),len(xvals),len(yvals),len(zvals), len(vmag) ]# check vector lengths
    #print('datalen: ',datalen)

    
    if userfile.PLOT[ 'ploton' ] &  userfile.PLOT[ 'totalbedo_cov_plot' ]: 

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
        plt.savefig( outpath + '/' + 'totalbedo_cov_accel.png' )
        plt.show()
        plt.close()

        #longlatdf = df.loc[:,['longitude','latitude']]

        #longlatdf.plot()
        #plt.show()
    
        print('Total Albedo Coverage Accel df: \n', df)

