#!/usr/bin/env python

import os, sys

#from sb.mst import user_inputs
#from mst.config.time_handling import TimeHandler
#from mst.config.earth.shape import set_shape
#from mst.config.earth.gravity.gravity import set_grav
#from mst.config.earth.atmosphere import set_atmos

if __name__ == "__main__":
    # add mst directory to path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
    print("sys.path: ", sys.path)

from mst import config, analysis, input_tools, ground, output_tools, spacecraft, trajectory, sensors #, util
#import config, analysis, input_tools, ground, output_tools, spacecraft, trajectory, sensors #, util
#from analysis import accel, elements, headers, coverage, groundref
from .analysis import accel, elements, headers, coverage, groundref

import time
import importlib
import argparse
#import multiprocessing
import Monte as M
from pdb import set_trace as bp

#from config.earth.gravity.gravity import set_grav


def main(path_to_input_file=None): #filename, parallel=False):

    if path_to_input_file is not None:
        print("entering if path_to_input_file is not None:")
        userfile = importlib.import_module(path_to_input_file, 'mst')
        importlib.reload(userfile)
    else:
        userfile = userfile_global

    # Use this section for running cases and parallel execution:

    """Runs all mst cases from an input file
    
    Args:
        filename (str): path to input file
        parallel (bool): flag to run in parallel. Defaults to False
    """
    '''
    # load cases
    cases = input_tools.HvtCaseSet(filename)

    if parallel:
        # determine number of processes
        cpu_processes = multiprocessing.cpu_count()
        num_cases = len(cases)
        num_processes = min(cpu_processes, num_cases)

        # set up parallel poolpwd
        pool = multiprocessing.Pool(processes=num_processes)

        # run
        pool.map(run, cases)
    else:
        # run serially
        for case in cases:
            print(case)
            run(case)
    '''

    # Monte config and other setup scripts:
    #execfile('/home/reynerso/Python/setupMonteMD.py')
    #execfile('/home/reynerso/Python/setupOsmean.py')
    #import osmean


    # Set environment: forces, boas, bodies, spacecraft, ground stations, ...

    # Setup boa
    
    boa = config.boas.set_boa() #boainit)
    '''
    # sent to: /config/boas.py
    '''

    # Orbiting bodies:

    # Earth - call set_earth
    
    body = "Earth"
        
    # Earth - gravity
    GravModel = config.earth.gravity.grav.set_grav()
    #import config.earth.gravity.ggm02c_monte_grv as GGM02C

    # Earth - atmosphere and solar flux
    AtmModel, SolarFluxMag = config.earth.atmosphere.set_atmos()
    ''' 
    #sent to: /config/earth/atmosphere.py
    '''

    # Earth - shape as an Ellipsoid
    config.earth.shape.set_shape(boa)
    '''
    #sent to: /config/earth/shape.py
    '''

    # Create an Earth Polyhedral shape:
    #config.earth.shape.set_polyshape(boa)

    # Spacecraft params
    # there are 2 names demonstrated here:  one for a loaded trajectory; another to use for propagating a trajectory

    # Ex for loading a boa:
    #sc40059 = spacecraft.sc_40059.set_sc('40059')
    #scname = sc40059.name  # use this name for the loaded sc
    
    #sent to sc_40059.py:
    #scname = '40059' # OCO-2 pre-loaded trajectory

    #Use this line to set the spacecraft for propagation.  Ex:                   
    #sc = spacecraft.sc_OCO2.set_sc('OCO2', boa)  #use this object for the propagated sc


    # Timing
    # Use settings from config/config.default.py file: 
    # th = config.time_handling.TimeHandler(config.config_default.SIM_START, config.config_default.SIM_DURATION)

    #use times from user_inputs.py file:
    print('setting the start epoch: ', userfile.SETTINGS['startepoch'])
    th = config.time_handling.TimeHandler(userfile.SETTINGS['startepoch'])
    th.dur_fm_stop(userfile.SETTINGS['endepoch'])


    #Enter Ground Station Data (do before defining sc to allow for direction definition for the sc)

    gs = ground.stations.Altadena.set_gs(boa)
    
    '''
    # sent to: ground/stations/Altadena.py and ground/stations/gsdef.py
    '''


    # sc = spacecraft.sc_SB.set_sc('SB', boa)  #use this object for the propagated sc
    sc = spacecraft.sc_user1.set_sc('SB', boa, userfile.sat1, gs, th, userfile.SETTINGS)  # prop sc using user_inputs.py

    '''
    #sent to: sc_OCO2.py ;  used as example for an imported trajectory
    '''


    # Define Nadir frame for sensor
    NadirFrame, NadirFrameName = config.frames.set_NadirFrame(boa, th, sc)

    # Define Zenith frame given the Nadir frame is set
    ZenithFrame, ZenithFrameName = config.frames.make_Zenith_Frame(boa, th, sc)


    # Sensor on Spacecraft:

    # Create a sensor on SB
    sen = sensors.sensor_SB.set_sensor( "SB_Nadir_Sensor", boa, NadirFrameName, "Spaceball") # ZenithFrameName, "Spaceball") # 



    # Simulation related


    # Inputs/outputs/plot related
    #sent to: __init__.py in ..mst/output_tools/

    '''
    import cristo    
    # sent to: config.config_default.py as DEFAULT_BOA_OUT
    outputName = 'OCO2_out.boa'
    # sent to: config.config_default.py as DEFAULT_STK_OUT
    stkoutfileName = 'stkout_17Jun24.e'
    '''

    # Load and propagate trajectories

    # Note: use time frame as defined above

    # Setup TrajQueries for Analysis (prior to propagations)

    query = M.TrajQuery(boa, sc.name,'Earth', 'IAU Earth Pole')  # used in integrator as a stop condition too
    #queryEME2000 = TrajQuery(boa, 'Earth', scname, 'EME2000')  # scname is used for the boa loaded sc
    suntq = M.TrajQuery(boa ,'Sun' ,'Earth' ,'IAU Earth Pole')
    #suntqEME2000 = TrajQuery(boa ,'Sun' ,'Earth' ,'EME2000')
    #moontqEME2000 = TrajQuery(boa ,'Moon' ,'Earth' ,'EME2000')


    # Frames
    '''
    # sent to: config.frames.py
    '''
    
    EIALframe, EIALName  = config.frames.set_FixedFrame(boa, th.start_epoch)


    # Set initial state prior to propagation

    '''
    # sent to: trajectory.trajdef.py

    '''

    #initState = trajectory.trajdef.set_State( boa, sc.name, th.start_epoch, EIALName ) # hardcoded state in trajdef.py
    initState = trajectory.trajdef.set_State_UserConfig( boa, userfile.sat1, th.start_epoch, "EME2000") # EIALName ) # user_inputs.py defines state


    # Propagation Section

    # Set Forces
    forces = []

    # load forces of interest here using forceChooser
    #earthAlbedoPress = trajectory.propagate.forceChooser.set_intforces( boa, sc, forces)
    pressForces = trajectory.propagate.forceChooser.set_intforces( boa, sc, body, forces, userfile, th)

    print('\n\nForces Loaded: ', pressForces , '\n\n')


    '''
    # sent to: trajectory.propagate.gravity.py and integrate.py
    '''

    # integrate trajectory

    endTime = trajectory.integrate.run_integrator(boa, sc, th, initState, forces,  "EME2000", userfile) # EIALName, userfile)

    '''
    sent to: trajectory.integrate.py
    '''



    # Analysis section

    header = headers.make_header( userfile )
    print("header: ", header)


    # Output file sampling rate:
    sampletime = userfile.SETTINGS['output_sample_time']

    # Long Lat Alt MLTAN:
    analysis.groundref.groundtrack_latlong( boa, sc, th.start_epoch )


    # Define analysis period of interest

    ascNodeTimes, period = analysis.elements.find_antimes_period( boa, sc, th.start_epoch, th.stop_epoch)

    '''
    # sent to : analysis.elements.osc_an.py
    '''


    # Find SC state and parameters of interest  

    # print the elements at the start: osc_elements_latlongV( boa, sc, epoch )

    analysis.elements.osc_elements_latlongV( boa, sc, th.start_epoch)
    # fields:     Epoch        SMA        Ecc      ArgPeri      Incl       ArgLat     Lat       Lon      Velocity
    
    '''
    # sent to : analysis.elements.osc_an.py
    '''
    
    # print the elements for each AN from start to end - New addition 11Aug22
    analysis.elements.osc_elements_ANs_PV( boa, sc, ascNodeTimes, th.start_epoch )


    # find_someDailyANtimes( boa, sc, startEpoch, stopEpoch, dayInterval ): filters some AN times every <dayInterval> number of days
    ascNodeSomeTimes = analysis.elements.find_someDailyANtimes( boa, sc, th.start_epoch, th.stop_epoch, 5)

    '''
    # sent to : analysis.elements.osc_an.py
    '''

    #ascNodeSomeTimes = analysis.elements.osc_an.filter_someANtimes( ascNodeTimes, period)
    # See Defaults for dayInterval; daySeconds: elements.osc_an.filter_someANtimes( ascNodeTimes, period, dayInterval = 5, daySeconds = 86400 )

    # someAN_osc_elements_latlongV( boa, sc, th, ascNodeSomeTimes, top=4)  : note : default is to show the first (top) 4
    analysis.elements.someAN_osc_elements_latlongV( boa, sc, th, ascNodeSomeTimes )

    '''
    # sent to : analysis.elements.osc_an.py
    '''

    # someAN_PV(boa, sc, th, ascNodeSomeTimes, top=4)

    analysis.elements.someAN_PV( boa, sc, th, ascNodeSomeTimes)

    '''
    # sent to : analysis.elements.osc_an.py
    '''

    num = 16
    analysis.elements.someAN_PV( boa, sc, th, ascNodeTimes, num)

    '''
    # sent to : analysis.elements.osc_an.py
    '''

    # Acceleration Data

    outpath = os.path.dirname(__file__)
    print("os.outpath.dirname(__file__):", outpath)

    outFrame, outFrameName = config.frames.set_NadirFrame(boa, th, sc)  #setting the output frame to Nadir Frame
    print("outFrame: ", outFrame)


    # Uncomment to use a rotating frame instead of the Nadir frame above
    '''
    angleFromZ = 0.0  # deg
    rotrate = 3.0 # deg/s
    outFrame, outFrameName = config.frames.make_Rotating_Frame(boa, th.start_epoch, angleFromZ, rotrate)
    '''
    
    analysis.accel.gendata.gen_accel_data(boa, sc, pressForces, th, outFrameName, outpath, sampletime, userfile, th.start_epoch ) #creates data files for each force in pressForces


    # Position and Velocity Data
    
    if (userfile.PLOT['position_plot'] | userfile.PLOT['velocity_plot']):

        # first plot using RIC (LVLH) coord sys

        outset = analysis.elements.find_pos_vel(boa, sc, th.start_epoch, th.stop_epoch, outFrameName, sampletime, userfile)
        analysis.elements.pos_vel_csv(outset, outpath, th.start_epoch, sampletime, userfile)

        # next plot using EME2000 (note: the plots and csv files will overwrite the above unless you save them as a different name)

        outset = analysis.elements.find_pos_vel(boa, sc, th.start_epoch, th.stop_epoch, "EME2000", sampletime, userfile)
        analysis.elements.pos_vel_csv(outset, outpath, th.start_epoch, sampletime, userfile)
    

    # Geodetic Height and Long/Lat Data

    outset = analysis.groundtrack.find_geodeticht_longlat(boa, sc, th.start_epoch, th.stop_epoch, sampletime, userfile) 
    analysis.groundtrack.geodeticht_longlat_csv(outset, outpath, th.start_epoch, sampletime, userfile)  


    '''
    # output albedo accel to csv:

    #outset = analysis.accel.find_albedo_accel(boa, earthAlbedoPress, th.start_epoch, th.stop_epoch, outFrameName )
    #analysis.accel.albedo_accel_csv(outset, outpath, th.start_epoch)

    # outFrameName = ZenithFrameName # make output rel to Zenith frame

    # Body Position data relative to Sensor
    #outset = analysis.sensor.bodyPosition.find_body_position(boa, sc, sen, th.start_epoch, th.stop_epoch, outFrameName, sampletime, userfile)
   
    

    # Coverage object testing

    print(" ")
    print("Running Single Coverage object for the start epoch: ")
    cover1 = analysis.coverage.coverage_calcs.Coverage( boa, sc.name, 'Earth' )
    cover1.find_faces( boa, th.start_epoch )
    cover1.find_area()

    sundir = analysis.solarflux.find_sun_dir( boa, 'Earth' , th.start_epoch, 'IAU Earth Fixed' )
    cover1.find_raa(sundir)

    #test output of VZA and Slant Range (dist to SC), Long, Lat
    cover1.VZA_SCdist_out() 
    

    
    
    #test find closest face...
    print("test find closesd face: ")
    cover1.find_closest_face( 1, -1 )
    
    #test plot face centers
    cover1.plot_face_centers()

    #test plot face vertices
    cover1.find_face_vertices()
    cover1.plot_face_vertices()
    

    # coverage calcs for albedo vector determination

    polyshape = config.earth.shape.set_polyshape( boa )   # create a polyshape for the Earth type: PolyShapeRegion
    facelist, totalarea, totalprojarea, longitude, latitude = analysis.sensor.coverage.find_poly_coverage( boa, sc, polyshape, th.start_epoch, th.stop_epoch, outFrameName, sampletime, userfile)
    #key: facelist contains info on all faces in coverage, totalarea = coverage area, totalprojarea = proj area of cap for ratios, longitude = sat long, lattiude = sc lat
    
    #Calc the net albedo vector:
    #analysis.accel.albedocompute.net_albedo_vector(facelist, totalprojarea)
    albedo_vector = analysis.accel.albedocompute.net_albedo_vector(cover1.facelist, cover1.totalprojarea)
    


    # functions now put in analysis.accel.gendata.gen_accel_data:
    #analysis.accel.albedocompute.find_albedo_pressures(boa, sc, 'Earth', th.start_epoch, th.stop_epoch) # test for onetime point
    #outset = analysis.accel.albedocompute.find_albedo_press(boa, sc, th.start_epoch, th.stop_epoch, outFrameName, sampletime, userfile)

    #print("Running analysis.sensor.coverage.find_poly_coverage2 ...")  # note: this doesn't seem to work right
    #analysis.sensor.coverage.find_poly_coverage2( boa, polyshape, sen, th.start_epoch, th.stop_epoch, outFrameName, sampletime, userfile)



    #test of find_sun_dir:

    #Epoch format: M.Epoch('2021-Apr-1 00:00:00 UTC')
    epochstring = '2021-Mar-21 00:00:00 UTC'  #yr+'-'+mo+'-'+dy+' '+hr+':'+mn+':'+sc+' UTC'   
    ttime = M.Epoch(epochstring)
    print('epoch defined:',epochstring)
    print(ttime)
    tframe =  'IAU Earth Fixed' 
    sundir_at_time = analysis.solarflux.find_sun_dir( boa, 'Earth', ttime, tframe)

    
    # Sun Coverage data
    scfaces = analysis.coverage.suncoverage.find_sun_coverage( boa, body,  th.start_epoch ,  th.stop_epoch , sampletime)
    #analysis.coverage.suncoverage.single_sun_coverage(boa, "Sun", "Earth", th.start_epoch )
    faceindex = 3
    analysis.coverage.suncoverage.thermal_profile( scfaces, faceindex )
    

    #test Earth_Surface_Radius calc
    tlon = 0.0
    tlat = 90.0
    theight = 0.0
    Rx, Ry, Rz = coverage.coverage_calcs.Earth_Surface_Radius( tlon, tlat, theight )
    print("Rx, Ry, Rz (in km)= ", Rx, Ry, Rz)

    
    # Thermal Profile test
    to = analysis.coverage.thermal_profile.Thermal(boa, 'Earth')
    to.loadpoly(boa) 
    # to.SolarCoverageData(boa, body, th.start_epoch, th.stop_epoch, sampletime)
    looplist = to.loop_thrulats(boa, body, th.start_epoch, th.stop_epoch, sampletime)
    print(" created looplist: ")
    print(" faceindex, lat, lon, avgpwr, avgflux, totenergy, sunrise, sunset ")
    for loop in looplist:
        print( loop )


    #Testing Long - Lat of all polyhedral faces
    lonlatlist = config.earth.shape.find_face_center_longlats(boa, 'Earth')
    print("ran find_face_center_longlats")
    lon = -5
    lat = -83.3
    closest_face = config.earth.shape.find_closest_face(boa, 'Earth', lonlatlist, lon, lat)

    '''

    # Setup for next analyses...    

    # endEpoch = Epoch('2027-June-17 00:00:00 UTC') # 2024-June-17 23:59:00 UTC for a day; 2027-June-17 00:00:00 UTC for 3 yr
    # pointtimes = M.Epoch.range( initTime, endEpoch, 60 * sec )



    # add plugins for stk out,  bsp out (cristo), other desired analyses and outputs


    print("Ran mst.main")


if __name__ == "__main__":
    # read command-line inputs
    parser = argparse.ArgumentParser(description="Mst Tool")
    parser.add_argument("input_file", type=str, help="path to input file")
    parser.add_argument("--parallel", action="store_true",
                        help="procecc in parallel")
    args = parser.parse_args()
    print("input file: ", args.input_file)
    userfile_global = importlib.import_module(args.input_file)

    # run
    runStart = time.time()
    main() #args.input_file, args.parallel)

    runStop = time.time()
    print("Total Run Time: %f minutes" % ((runStop - runStart)/60.))



