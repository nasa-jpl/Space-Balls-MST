#!/usr/bin/env python

import os, sys
# daily spherical harmonics for the radiation

# turn forces on & off
# xyz and norm of different acc in a single fig
# sensitivities wrt OE & A/m
# bug when turnign spin ON - select spin freq & direction

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

    '''
    # Sensor on Spacecraft:

    # Create a sensor on SB
    sen = sensors.sensor_SB.set_sensor( "SB_Nadir_Sensor", boa, NadirFrameName, "Spaceball") # ZenithFrameName, "Spaceball") # 
    '''


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
    #suntq = TrajQuery(boa ,'Sun' ,'Earth' ,'IAU Earth Pole')
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

    print("the start_epoch: ", th.start_epoch)

    #daystartEpoch = th.start_epoch.dayStart("UTC")  # defines the Epoch at the start of the day - may only work with newer versions of Monte
    cdateStartEpoch = M.CalDate(th.start_epoch) #.date("UTC")
    yr = str(cdateStartEpoch.year())
    mo = str(cdateStartEpoch.month())
    dy = str(cdateStartEpoch.day())
    hr = str(cdateStartEpoch.hour())
    mn = str(cdateStartEpoch.minute())
    sec = str(cdateStartEpoch.second())
    #Epoch format: M.Epoch('2021-Apr-1 00:00:00 UTC')
    epochstring = yr+'-'+mo+'-'+dy+' '+hr+':'+mn+':'+sec+' UTC'   
    daystartEpoch = M.Epoch(epochstring)
    initstartEpoch = daystartEpoch

    oneday = M.Duration(1,0,0,0.0)  #duration of 1 day
    twohours = M.Duration(0,2,0,0.0) 
    dayendEpoch = daystartEpoch + oneday + twohours


    # Daily Loop for changing Albedo SH file: 

    # preserve the th.startepoch and th.stopepoch in another th object:  mainth
    mainth = config.time_handling.TimeHandler(userfile.SETTINGS['startepoch'])
    mainth.dur_fm_stop(userfile.SETTINGS['endepoch'])


    # Count how many days are being looped thru

    loopduration = mainth.duration.days() #num of days, returns float
    loopdays = int(loopduration)
    print("days to loop: ", loopdays)
    days = loopdays

    # reset the th to go from the first day to the next
    th = config.time_handling.TimeHandler(daystartEpoch)  # default is 1 day to set the stop epoch in th (so don't execute a dur_fm_stop method)
    th.dur_fm_stop(dayendEpoch)


    while days >= 0:

        print('Day: ', days)

        query = M.TrajQuery(boa, sc.name,'Earth', 'IAU Earth Pole') 

        # main loop functions here:

        # execute prior loop functions
        # key change is to update the albedo SH file in the forceChooser.set_intforces function each iteration


        # Propagation Section

        # Set Forces
        forces = []

        # load forces of interest here using forceChooser
        #earthAlbedoPress = trajectory.propagate.forceChooser.set_intforces( boa, sc, forces)
        pressForces = trajectory.propagate.forceChooser.set_intforces( boa, sc, body, forces, userfile, th)

        print('\n\nForces Loaded: ', pressForces , '\n\n')

        # integrate trajectory

        endTime = trajectory.integrate.run_integrator(boa, sc, th, initState, forces,  "EME2000", userfile) # EIALName, userfile)
        #where endTime is prop.info().endTime(), or the time at the end of hte propagation


        # Analysis section

        header = headers.make_header( userfile )
        print("header: ", header)
        # Output file sampling rate:
        sampletime = userfile.SETTINGS['output_sample_time']


        # Acceleration Data

        outpath = os.path.dirname(__file__)
        print("os.outpath.dirname(__file__):", outpath)

        outFrame, outFrameName = config.frames.set_NadirFrame(boa, th, sc)  #setting the output frame to Nadir Frame
        print("outFrame: ", outFrame)

        # Uncomment to use a rotating frame instead of the Nadir frame above
        if userfile.INPUT['framename'] == 'Rotating':

            angleFromZ = 0.0  # deg
            rotrate = userfile.INPUT['spinrate'], #3.0 # deg/s
            #rr = int(rotrate)
            print('rotrate: ', rotrate)
            print('rotrate type: ', type(rotrate))#, ' rr: ', rr)
            rotratefloat = float('.'.join(str(elem) for elem in rotrate))
            print('rotratefloat: ', rotratefloat)

            outFrame, outFrameName = config.frames.make_Rotating_Frame(boa, th.start_epoch, angleFromZ, rotratefloat)
        

        analysis.accel.gendata.gen_accel_data(boa, sc, pressForces, th, outFrameName, outpath, sampletime, userfile, initstartEpoch ) #creates data files for each force in pressForces

        

        # Clean up/ Prep for next iteration Section:

        # Set the new state at the next start time/epoch:  initState is now the state at the end of the day
        initState = query.state( dayendEpoch - twohours, 'EME2000')  #, 'IAU Earth Pole') # since dayendEpoch > endTime, this fails as it is not in the traj interval
        # note: two hours was added to ensure a full days propagation (which didn't happen when 1 day was specified)

        #timedif = dayendEpoch - endTime #duration object in Monte;  dayendEpoch is the daystartEpoch + oneday

        # Set the new start time/epoch for the next iteration:  

        # change the th object:  new start time needs to be changed and the new end time does also
        daystartEpoch = daystartEpoch + oneday # set to the next day
        #daysleft = daystartEpoch - mainth.stop_epoch
        #if daysleft.days() >= 1:
        #    dayendEpoch = daystartEpoch + oneday
        #else:
        #    dayendEpoch = mainth.stop_epoch

        #dayendEpoch = daystartEpoch + timedif + oneday  # this should make it at the beginning of the next day
        dayendEpoch = daystartEpoch + oneday + twohours

        print("CalDate start Epoch at: ", cdateStartEpoch)
        print("Init start Epoch: ", initstartEpoch)

        print("Setting the next day start Epoch at: ", daystartEpoch)
        print("Setting the next day end Epoch at: ", dayendEpoch)

        #redefining the th object to the new bounds of the next day
        th.start_epoch = daystartEpoch
        th.stop_epoch = dayendEpoch
        th.dur_fm_stop(dayendEpoch)
        print("")
        print("th.display in main loop:")
        th.display()
        print("")

        days = days - 1

        print("Advancing to next day...")
        print("")
        #input("Hit Enter key...")

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



