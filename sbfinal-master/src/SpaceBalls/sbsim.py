
from __future__ import annotations

import os, sys
import re
import time
from importlib import import_module

import Monte as M
import mpy.units as units

import numpy as np
import importlib
from scipy.io import savemat
from scipy.special import lpmn
from astropy.time import Time as Time_astropy

from SpaceBalls.paths import CONFIG_DIR, INPUT_DIR, OUTPUT_DIR
sys.path.insert(0, str(CONFIG_DIR.parent))  # parent of 'config'
sys.path.insert(0, str(INPUT_DIR.parent))  # parent of 'config'
import config.constants as constants
from SpaceBalls.utils import load_input_file, progress_bar, get_Earth_SH_gravity_model#, get_mesh_from_fibonacci_sphere

import SpaceBalls.radiation_settings as rad_settings
import SpaceBalls.sph_meshing as sph_meshing
import SpaceBalls.astropy_functions as astropy_functions
import SpaceBalls.input_database_manager as input_manager


class SpaceBallsSim():

    def __init__(self, input_name='sb_inputs', input_mode='file', verbose=True):    # input_mode: 'file' or 'hash'

        if input_mode=="hash":
            print("Loading config dict")
            self.config_info = input_manager.get_dicts(input_name)
            print(self.config_info)
        elif input_mode=="file":
            self.userfile = load_input_file(input_name)
            #TODO: self.config_info = old_usefile_to_config_info(self.userfile) # convert the old userfiles to the new format

        self.input_name = input_name
        self.verbose = verbose

        #if self.userfile.INTEGRATION_SETTINGS['n_rings_albedo']=='default':
        #    self.n_rings_albedo = None
        #else:
        #    self.n_rings_albedo = self.userfile.INTEGRATION_SETTINGS['n_rings_albedo']
        
        self.n_rings_albedo = self.config_info['integration_settings']['n_rings']
        
        #self.wgs84_albedo = self.userfile.FORCE_SETTINGS['WGS84_alb edo_shape']
        #self.out_dir_csv = './output_files/' + self.input_name + '/data_output/'
        #self.out_dir_png = './output_files/' + self.input_name + '/img_output/'
        
        self.out_dir_csv = os.path.join(OUTPUT_DIR, self.input_name) # + '/data_output/'
        #self.out_dir_png = '/media/monte_share/output_files/' + self.input_name + '/img_output/'
        
        if os.path.exists(self.out_dir_csv): # TODO: not robust if the directories exist but they are empty!
            #sc_boa = self.input_name + '_' + self.spacecraft.name + '_out.boa'
            #previous_sc_boa = M.BoaLoad(sc_boa)
            # retrieve number of existing propagated days
            self.existing_days = len(next(os.walk(self.out_dir_csv))[1]) - 2
            #assert(self.existing_days>0)
            #if self.existing_days<0:
            
            self.existing_days = 0
        else:
            self.existing_days = 0

        print(self.config_info)
        EEI_truth_name = "EEI_truth_" + str(self.config_info['force_settings']['EEI_truth'])
        self.EEI_truth_settings = rad_settings.radiation_settings_from_EEI_truth_name(EEI_truth_name)
        
        self._set_boa()

        Shape._set_earth_shape(self.boa, self.EEI_truth_settings["earth_shape"]) # we're here
        if self.verbose: print("Earth shape set")

        self._set_time_handler()

        # Ground station stuff to be tackled later

        self._create_spacecraft_set()

        # block to be conducted for each sc (TBC)
        self.spacecraft = self.sc_array[0]  # only 1 spacecraft - see note inside create_spacecraft_set() method
        self.NadirFrame = Frames.set_NadirFrame(self.boa, self.th, self.spacecraft.name)
        self.ZenithFrame = Frames.make_Zenith_Frame(self.boa, self.th)
        self.EIALframe = Frames.set_FixedFrame(self.boa, self.th.start_epoch)
        
        self.SolarFrame = Frames.set_SolarPointingFrame(self.boa, self.th)
        
        self.sampletime = self.config_info['delta_t_out'] # self.userfile.OUTPUT_SETTINGS['output_sample_time']
        self.output_pointtimes = M.Epoch.range( self.th.start_epoch, self.th.stop_epoch, self.sampletime * units.sec ) # TODO: remove duplication in output manager

        # previously in OutputManager:
        self.coord = M.CoordSetBoa.read( self.boa )

        self.GravModel = get_Earth_SH_gravity_model("GOCO06s")

    def run_propagation_arc(self, t0: Time, duration_hours):
        
        th = TimeHandler(t0.epoch, duration_hours/24) # hard-coded conversion to days
        self.spacecraft._set_initial_state_Monte(self.boa, 0, th.start_epoch) 

        self.force_manager = ForceManager(self, th)
        self.force_manager.set_forces()
        self.run_integrator(th)
        #self.write_output('', th)
        
    
    def run_daily_loop(self, day_to_stop=None):
        
        #if self.verbose: print("days to loop: ", self.loop_ndays)

        #self.daily_th = TimeHandler(self.start_Time, 
        #                            duration=np.min([1, self.th.duration.days()]))
        
        days = self.existing_days
        self.spacecraft._set_initial_state_Monte(self.boa, days, self.daily_th_array[days].start_epoch)

        max_day = day_to_stop if day_to_stop is not None else self.loop_ndays
        
        #while days <= (self.loop_ndays-1):
        while days <= (max_day):
            
            self.daily_th = self.daily_th_array[days]  # other methods need the daily th

            if self.verbose: print(f"Running day {days+1}/{self.loop_ndays}")
            self.force_manager = ForceManager(self, self.daily_th)
            self.force_manager.set_forces()

            daily_out_subdir = 'day_' + str(days) + '/'
            self.run_integrator(self.daily_th)
            self.write_output(daily_out_subdir, self.daily_th)
            
            self.spacecraft.initState = self.query.state( self.daily_th.stop_epoch, 'EME2000') # update it for the next day
            # TODO: extend to multiple spacecraft?

            days = days + 1

        
    def _set_boa(self): #boa):

        # DEFAULT_BOA = r'default.boa'
        # sys.path.insert(0, os.path.dirname(__file__))
        # default_boa_dir = os.path.dirname(__file__)+'/config/'+ DEFAULT_BOA
        # if self.verbose: print('config.DEFAULT_BOA: ', default_boa_dir)

        boa_name = self.EEI_truth_settings["ephemerides"]
        boa_dir = os.path.join(CONFIG_DIR, 'boa_files', boa_name+'.boa')
        if self.verbose: print(f"Loading boa file {boa_name}...")

        self.boa = M.BoaLoad(boa_dir)

        self.trajset = M.TrajSetBoa.read(self.boa)
        trajobjs = self.trajset.getAll()
        
        if self.verbose: print('Objects in the boa: ', trajobjs)

        earthBody = M.BodyDataBoa.read(self.boa, "Earth")

        if self.verbose: print('Earth data from BodyDataBoa: ', earthBody)

        boaControl = M.ErrorControlBoa.read(self.boa)
        boaControl.setAction('Missing EarthAngle data' , M.ErrorAction.IGNORE )

        if self.verbose: print("Boa set")

    def _set_time_handler(self):
        
        jd_window = self.EEI_truth_settings["jd_interval"]
        jd_0, jd_f = jd_window[0] + self.config_info['orbit']['delta_jd_0'], jd_window[1]
        
        self.start_Time = Time(jd_0) #Time(self.userfile.TIMESTART)
        self.finish_Time = Time(jd_f) # Time(self.userfile.TIMEFINISH)

        self.th = TimeHandler(self.start_Time.epoch)
        self.th.dur_fm_stop(self.finish_Time.epoch)
        
        self.loop_ndays = int(np.ceil(self.th.duration.days()))
        
        self.daily_th_array = [None] * self.loop_ndays
        for i in range(self.loop_ndays):
            start_epoch = self.start_Time.epoch + i*86400*units.sec
            end_epoch = self.start_Time.epoch + (i+1)*86400*units.sec
            self.daily_th_array[i] = TimeHandler(start_epoch)
            self.daily_th_array[i].dur_fm_stop(end_epoch)
        
        if self.verbose: print("Time handler set")

    
    def _create_spacecraft_set(self):
        
        #n = self.userfile.SC_INPUT['N_sc']
        n = 1 # several sc at once does not seem to be more efficient (file sbsim2) - use parallel proc. instead
        self.sc_array = [None]*n

        for i in range(n):
            name = self.input_name  #self.userfile.SC_INPUT['name'] + '_' + str(i)
            sc_i = Spacecraft(self.boa, name, self.th.start_epoch, self.config_info['sc_params'], self.config_info['orbit'], self.existing_days)
            self.sc_array[i] = sc_i

        if self.verbose: print("SC set created")

    def run_integrator(self, th: TimeHandler):

        self._set_propagation(th)
        self._run_propagation(th)


    def _set_propagation(self, th: TimeHandler):

        # myforce & low thrust force ???

        self.query = M.TrajQuery(self.boa, self.spacecraft.name, 'Earth', 'IAU Earth Pole')
        integState = M.IntegrationState(self.boa)
        integState.addState(th.start_epoch, self.spacecraft.name, "Earth", "EME2000", self.spacecraft.initState) # ie frameName = "Earth Inertial at Launch"
        integState.addMass(th.start_epoch, self.spacecraft.name, self.spacecraft.mass)
        self.prop = M.DivaPropagator(self.boa, "DIVA", integState)
        self.prop.setMinStep(1e-9*units.sec)
        self.prop.setMaxStep(1e9*units.sec)

        for f in self.force_manager.forces:
            self.prop.addForce(f(self.boa, self.spacecraft.name))

        M.DivaTraj(self.boa, self.spacecraft.name, self.prop)
        M.DivaMass(self.boa, self.spacecraft.name, self.prop)

        #from config.config_default import EARTH_RADIUS
        StopEvents = [ M.CoordinateEvent(self.query, M.Conic.semiMajorAxis(),  constants.earth_radius(monte_units=True) + 100*units.km , M.CoordinateEvent.DECREASING ) ]
        self.prop.setStopEvents( StopEvents )


    def _run_propagation(self, th: TimeHandler):

        integTimeStart = M.Epoch.now()
        if self.verbose: 
            print(f"Integration start time: {integTimeStart}")
            print("Running integration...")

        output_name = self.input_name + '_' + self.spacecraft.name + '_out.boa'
        self.boa2 = M.BoaFile(output_name, M.BoaFile.TRUNCATE)  #creates a new Boa file, if named file exists then erase the old file

        self.prop.create( self.boa2, th.start_epoch, th.stop_epoch )  # main propagation method. 
        self.boa.mergeTree(self.boa2)

        endTime = self.prop.info().endTime()
        integTimeEnd = M.Epoch.now()
        self.integration_time = integTimeEnd - integTimeStart
        trajLifetime = (endTime - th.start_epoch).convert('days')

        if self.verbose:
            if self.integration_time.seconds().value()<60:
                print(f"Integration time: {self.integration_time.seconds()} seconds")
            elif self.integration_time.hours()<1:
                print(f"Integration time: {self.integration_time.minutes()} minutes")
            else:
                print(f"Integration time: {self.integration_time.hours()} hours")

            print ('Trajectory Lifetime is ',trajLifetime,' days.')


    def write_output(self, daily_out_subdir, th: TimeHandler):
        
        data_out_dir = os.path.join(self.out_dir_csv, daily_out_subdir) 
        ##img_out_dir = self.out_dir_png + daily_out_subdir
        
        os.makedirs(data_out_dir, exist_ok=True) 
        #os.makedirs(img_out_dir, exist_ok=True)
        
        #self.matlab_output = self.userfile.OUTPUT_SETTINGS['matlab_output']
        
        # self.out_frame = Frames.set_NadirFrame(self.boa, self.daily_th, self.spacecraft.name) #TODO: frame flexibility
        # out frame seems useless

        # rotating frame stuff TBD here

        if self.verbose: print("Generating Accel data files ... ")
        output_manager = OutputManager(self, th)

        self.all_acc_hist = [None] * len(self.force_manager.force_names)
        self.all_acc_norm_hist = [None] * len(self.force_manager.force_names)

        # OUTPUT_VARS = ['jd_vec', 'aero', 'erp', 'srp', 'total_grav', 'xyz_ecef', 'xyz_sun_frame']

        np.save(os.path.join(data_out_dir, 'jd_vec'+'.npy'), output_manager.jd_array)
        #np.savetxt(data_out_dir + 'jd_vec.csv', output_manager.jd_array)

        for i, pf in enumerate(self.force_manager.pressforces):

            acc_hist = output_manager.find_accel_hist(pf)            
            file_name = self.force_manager.force_names[i]
            
            # output_manager.write_acc_csv(acc_hist, data_out_dir, file_name)
            output_manager.write_acc(acc_hist, data_out_dir, file_name)
            
            self.all_acc_hist[i] = acc_hist
            self.all_acc_norm_hist[i] = np.linalg.norm(acc_hist, ord=2, axis=1)

        self.all_acc_hist = np.stack(self.all_acc_hist, axis=2)  # size: [nsteps, 3, nforces]

        # longs, lats, heights = output_manager.find_lonlat_hist()

        #np.savetxt(self.out_dir_csv + 'visibility_caps.csv', output_manager.all_visibility_cap_lonlats, 
        #           header='Lat (deg), Lon (deg), height (km)') # TODO: organize more consistently

        # file_name = 'lat_lon_h.csv'
        # output_manager.write_latlonheight_csv(lats, longs, heights, data_out_dir, file_name)
        
        output_manager.find_r_hist()

        file_name = 'xyz_ecef'
        output_manager.save_cartesian_hist(output_manager.r_ecef_hist, data_out_dir, file_name)
        
        file_name = 'vxvyvz_ecef'
        output_manager.save_cartesian_hist(output_manager.v_ecef_hist, data_out_dir, file_name)
        
        file_name = 'xyz_sun_frame'
        output_manager.save_cartesian_hist(output_manager.r_sunframe_hist, data_out_dir, file_name)
        
        all_rot_mat = output_manager.get_all_ECEF2RIC_rotations()
        np.save(os.path.join(data_out_dir, 'ecef2ric_hist'), np.array(all_rot_mat))


        # file_name = 'r_sun_ecef.csv'
        # output_manager.save_r_sun_hist('IAU Earth Fixed', data_out_dir, file_name)
        
        #self.groundtrack_hist = [longs, lats, heights]

        """
        if 'visibility_cap_hist' in self.matlab_output:
            savemat(data_out_dir + '/visibility_caps.mat', 
                    {'my_array': np.array(output_manager.all_visibility_cap_lonlats)})

        if 'r_sun_hist_astropy' in self.matlab_output:
            all_r_Sun = [None] * len(output_manager.pointtimes)
            for i, epoch in enumerate(output_manager.pointtimes):
                _, _, r_Sun = astropy_functions.find_sun_ecef(epoch)
                all_r_Sun[i] = r_Sun
            savemat(data_out_dir + '/r_Sun_hist.mat', {'my_array': np.array(all_r_Sun)})

        if 'R_ijk2ric' in self.matlab_output:
            #np.savetxt(out_dir_data + '/rot_mattrices_ijk2ric.dat', np.array(all_rot_mat))
            all_rot_mat = output_manager.get_all_ECEF2RIC_rotations()
            savemat(data_out_dir + '/rot_matrices_ijk2ric.mat', {'my_array': np.array(all_rot_mat)})
        
        
        if self.userfile.OUTPUT_SETTINGS['make_acc_plots']:
            output_manager.make_acc_hist_plot(self.all_acc_norm_hist, self.force_manager.force_names, 
                                            'Acceleration norm (m/s$^2$)', self.out_dir_png, 'acc_norms', scale="log", title=None,
                                            make_animation=False)

            filenames = ['acc_radial', 'acc_intrack', 'acc_crosstrack']
            ylabels = ['Radial acceleration', 'In-track acceleration', 'Cross-track acceleration']

            for i in range(3):
                output_manager.make_acc_hist_plot(np.transpose(self.all_acc_hist[:,i,:]), self.force_manager.force_names, 
                                            ylabels[i] + '(m/s$^2$)', self.out_dir_png, filenames[i], scale="symlog", 
                                            title=None, add_sum_curve=True, make_animation=False)
        

        # output_manager.animate_accel_plot() # deprecated

        # output_manager.plot_groundtrack_with_erp_maps(self.out_dir_png, 'groundtrack_erp')

        if self.userfile.OUTPUT_SETTINGS['save_ERP_mesh_history']:
            output_manager.save_monte_mesh_hist(data_out_dir)
            output_manager.evaluate_SH_at_monte_mesh_hist(data_out_dir)
        """

        #output_manager.animate_groundtrack_with_erp_maps(self.out_dir_png, 'groundtrack_erp_animation')



class TimeHandler():

    # Initialize epoch and duration in Days (optional, default is 1 day)
    
    def __init__(self, start_epoch, duration=1.0 ):  #duration is in days
    #def __init__(self, start_Time, duration=1.0 ):  #duration is in days

        #self.start_Time = start_Time
        #self.start_epoch = self.start_Time.epoch
        self.start_epoch = start_epoch
        self.duration = duration
        self.stop_epoch = self.start_epoch + duration * 86400*units.sec # TODO: we are then saving this using UTC scale, but then the factor 86400 is not fully true
        self.interval = M.TimeInterval(self.start_epoch, self.stop_epoch) # create Monte Interval object for start and stop epochs

    def dur_fm_stop(self, stop_epoch): # make a duration object from self.start and an input stop epoch ; set the new Interval
    #def dur_fm_stop(self, stop_Time): # make a duration object from self.start and an input stop epoch ; set the new Interval
        #self.stop_Time = stop_Time
        #self.stop_epoch = self.stop_Time.epoch
        self.stop_epoch = stop_epoch
        self.duration = self.stop_epoch - self.start_epoch 
        self.interval = M.TimeInterval(self.start_epoch, self.stop_epoch) # create Monte Interval object for start and stop epochs

    def display(self):
        print("Timehandler object display start and stop epochs:")
        print("start_epoch: ",self.start_epoch)
        print("stop_epoch: ",self.stop_epoch)




class Frames():

    @staticmethod
    def set_FixedFrame(boa, epoch): # creates ab Earth Intertial at Launch frame, rel to epoch and IAU Earth Fixed
        frameName = "Earth Inertial at Launch"
        frame = M.FixedEpochFrame(boa, epoch, "INERTIAL", frameName, "IAU Earth Fixed" )
        return frame, frameName

    @staticmethod
    def set_NadirFrame(boa, th: TimeHandler, sc_name):  #creates a nadir ref frame in boa for SC to Body nadir vector = X, Y is near velocity direction, Z=XxY 
        frameName = "Nadir Ref"
        BodyName = "Earth"
        relFrameName = "EME2000"
        frame = M.BodyPosDirFrame( boa, frameName, relFrameName, th.interval, sc_name, BodyName)
        
        # The position direction of the body relative to the center is used for the X-axis. So towards zenith; -X is nadir direction
        # The Z-Axis is defined as the angular momentum vector of the body relative to the center = orbit normal and 
        # the Y-axis is simply the Z-Axis crossed with the X-Axis
        return frame, frameName
    
    @staticmethod
    def set_SolarPointingFrame(boa, th: TimeHandler):
        frameName = 'Solar Ref'
        relFrameName = 'EME2000'
        frame = M.BodyPosDirFrame(boa, frameName, relFrameName, th.interval, 'Earth', 'Sun')
        
        return frame

    @staticmethod
    def make_Zenith_Frame(boa, th): #create a Zenith frame that is 180 deg from the Nadir frame
        frameName = "Zenith Ref"
        zangle = 180*units.deg
        axis = M.Rotation.Z
        r = M.Rotation( axis, zangle )
        oframe = M.OffsetFrame( boa, frameName, "Nadir Ref", th.interval, r)
        return oframe, frameName
    
    @staticmethod
    def set_NadirECI(boa, sc_name):  # sets the direction from the SC to Earth nadir direction
        coordFrame = "EME2000" # TODO: frames (??)
        sctqEME2000 = M.TrajQuery(boa , sc_name, 'Earth' , coordFrame)
        nadirdir = M.NadirDir( sctqEME2000 )
        return nadirdir     
    
    @staticmethod
    def set_SunDirECI(boa): # finds the sun direction and creates a direction object relative to ECI (EME2000) - moves with time
        coordFrame = "EME2000" # TODO: frames (??)
        suntqEME2000 = M.TrajQuery(boa ,'Sun' ,'Earth' , coordFrame)
        sundir = M.PositionDir( suntqEME2000 )
        return sundir   



class Shape():

    def __init__(self):
        pass

    @staticmethod
    def _set_earth_shape(boa, shape_tag):
    
        #Enter Shape Data for Earth (used for finding Long and Lat )

        R_Earth = constants.earth_radius(monte_units=True)   # TODO: WGS84 is hard-coded
        #f_Earth = constants.earth_flattening()
        if shape_tag == "spherical":
            f_Earth = 0
        else:
            f_Earth = constants.earth_flattening(model=shape_tag)

        e = M.Ellipsoid( R_Earth, f_Earth )
        shape = M.EllipsoidShape( boa, 'Earth Shape #1', 'Earth', e )
        M.EllipsoidShapeBoa.write( boa, shape )  # writes the shape to the boa

        # change BodyData object for Earth to set the shape name
        data = M.BodyDataBoa.read( boa, 'Earth' )
        data.setShape( 'Earth Shape #1' )



class OutputManager():

    def __init__(self, sim_instance, th: TimeHandler):

        self.boa = sim_instance.boa
        #self.settings_dict = sim_instance.userfile.FORCE_SETTINGS
        self.verbose = sim_instance.verbose
        self.sc = sim_instance.spacecraft
        self.th = th  # TODO: not call it "daily" - why not?
        self.force_manager = sim_instance.force_manager
        self.sampletime = sim_instance.sampletime
        self.n_rings = sim_instance.n_rings_albedo
        self.trajset = sim_instance.trajset

        self.pointtimes = M.Epoch.range( self.th.start_epoch, self.th.stop_epoch, self.sampletime * units.sec )
        self.coord = sim_instance.coord  # M.CoordSetBoa.read( self.boa )

        self.jd_array = [pointtime.julianDate('UTC') for pointtime in self.pointtimes]

        os.makedirs(sim_instance.out_dir_csv, exist_ok=True)   # TODO: daily subdirs data/png should probably be removed
        #os.makedirs(sim_instance.out_dir_png, exist_ok=True)
    

    def find_r_hist(self):
        
        r_ecef_hist = [None] * len(self.pointtimes)
        v_ecef_hist = [None] * len(self.pointtimes)
        r_sunframe_hist = [None] * len(self.pointtimes)
        
        for i, time in enumerate(self.pointtimes):
            query = M.TrajQuery( self.boa, self.sc.name, 'Earth', 'IAU Earth Pole' )  # TODO: self.query?
            oscStateEarth = query.state( time, 'IAU Earth Fixed'  ) 

            r_ecef_hist[i] = [M.Cartesian.x(oscStateEarth).convert('km'), 
                              M.Cartesian.y(oscStateEarth).convert('km'),
                              M.Cartesian.z(oscStateEarth).convert('km')]
            
            v_ecef_hist[i] = [M.Cartesian.dx(oscStateEarth).convert('km/s'), 
                              M.Cartesian.dy(oscStateEarth).convert('km/s'),
                              M.Cartesian.dz(oscStateEarth).convert('km/s')]
            
            oscStateSunFrame = query.state( time, 'Solar Ref'  )
            r_sunframe_hist[i] = [M.Cartesian.x(oscStateSunFrame).convert('km'), 
                                M.Cartesian.y(oscStateSunFrame).convert('km'),
                                M.Cartesian.z(oscStateSunFrame).convert('km')]
            
        self.r_ecef_hist = np.array(r_ecef_hist)
        self.v_ecef_hist = np.array(v_ecef_hist)
        self.r_sunframe_hist = np.array(r_sunframe_hist)



    def find_lonlat_hist(self): # TODO: change name or separate

        longs = [None] * len(self.pointtimes)
        lats = [None] * len(self.pointtimes)
        geo_heights = [None] * len(self.pointtimes)

        for i, time in enumerate(self.pointtimes):
            query = M.TrajQuery( self.boa, self.sc.name, 'Earth', 'IAU Earth Pole' )  # TODO: self.query?
            oscStateEarth = query.state( time, 'IAU Earth Fixed'  ) 
            
            longs[i] = M.Spherical.longitude(oscStateEarth).convert('deg')
            lats[i] = M.Spherical.latitude(oscStateEarth).convert('deg')
            geo_heights[i] = M.Geodetic.height(oscStateEarth).convert('km')
        
        self.groundtrack_hist = [longs, lats, geo_heights]

        self._save_visibility_cap_hist()

        return longs, lats, geo_heights

    
    # def _save_visibility_cap_hist(self):
# 
    #     self.all_visibility_cap_lonlats = [None] * len(self.r_ecef_hist)
# 
    #     for i, r_ecef in enumerate(self.r_ecef_hist):
    #         visibility_lonlats = sph_meshing.find_tangent_cone_x_Earth_loci(r_ecef)
    #         # vis_lons, vis_lats = get_clean_lonlat_vecs_for_plot(visibility_lonlats[:,0], visibility_lonlats[:,1])
# 
    #         self.all_visibility_cap_lonlats[i] = visibility_lonlats #np.column_stack((vis_lons, vis_lats))
# 
    #     #out_mat = np.column_stack((t_hist, lat_vec, lon_vec, h_vec))
    #     #np.savetxt(out_dir + fileName, out_mat, header='t-t0 (sec), Lat (deg), Lon (deg), height (km)')


    def save_r_sun_hist(self, frame_name, out_dir, filename):
        
        all_r_sun = np.zeros((len(self.pointtimes), 3))
        for i, time in enumerate(self.pointtimes):
            all_r_sun[i,:] = self.trajset.state(time, 'Sun', 'Earth', frame_name, 1).pos().toArray() # "EME2000": ECI (non-rot), "IAU Earth Pole": ??
        
        self.save_cartesian_hist(all_r_sun, out_dir, filename)
    

    def get_all_ECEF2RIC_rotations(self):
        
        all_rot_mat = [None] * len(self.pointtimes)
        
        for i, time in enumerate(self.pointtimes):
            iauToNadir = self.coord.rotation(time, "Nadir Ref", 'IAU Earth Fixed')
            all_rot_mat[i] = iauToNadir.m().toArray()
            
        return all_rot_mat

            
    def find_accel_hist(self, pf):

        nadir_acc_vecs = [None] * len(self.pointtimes)
        # TODO: are accelerations in the Nadir frame including coriolis / frame rotation effects?
        if pf.__class__ == M.Gravity:
            for i, time in enumerate(self.pointtimes):
                nadir_acc_vecs[i] = pf.accel(time, 'Earth', 'Nadir Ref')
        elif pf.__class__ == M.AlbedoPressure:
            for i, time in enumerate(self.pointtimes):
                eme2000ToNadir = self.coord.rotation(time, "Nadir Ref", 'EME2000')   # TODO: introduce frame flexibility
                acc_EME = pf.accel(time, 'EME2000')
                nadir_acc_vecs[i] = eme2000ToNadir * acc_EME
        else:
            for i, time in enumerate(self.pointtimes):
                nadir_acc_vecs[i] = pf.accel(time, 'Nadir Ref')

        return nadir_acc_vecs
    
    
    def get_all_ECEF2RIC_rotations(self):
        
        all_rot_mat = [None] * len(self.pointtimes)
        
        for i, time in enumerate(self.pointtimes):
            iauToNadir = self.coord.rotation(time, "Nadir Ref", 'IAU Earth Fixed')
            all_rot_mat[i] = iauToNadir.m().toArray()
            
        return all_rot_mat
    

    def write_acc_csv(self, acc_vecs, out_dir, fileName):
        
        np.savetxt(out_dir + fileName, np.array(acc_vecs), header='Ar (m/s^2), Ai (m/s^2), Ac (m/s^2)')

    def write_acc(self, acc_vecs, out_dir, fileName):
        np.save(os.path.join(out_dir, fileName+'.npy'), np.array(acc_vecs))


    def write_latlonheight_csv(self, lat_vec, lon_vec, h_vec, out_dir, fileName):
                
        out_mat = np.column_stack((lat_vec, lon_vec, h_vec))
        np.savetxt(out_dir + fileName, out_mat, header='Lat (deg), Lon (deg), height (km)')


    def save_cartesian_hist(self, out_mat, out_dir, file_name):

        #out_mat = r_ecef_hist #np.column_stack((r_ecef_hist))
        #np.savetxt(out_dir + file_name, out_mat, header='X (km), Y (km), Z (km)')
        np.save(os.path.join(out_dir, file_name+'.npy'), out_mat)



    # def save_monte_mesh_hist(self, out_dir):
    #     
    #     n_steps = len(self.pointtimes)
    #     #############
    #     self.all_ring_centers_hist = [None] * n_steps
    #     areas_hist = [None] * n_steps
# 
    #     for i in range(n_steps):
    #         all_ring_nodes, all_ring_connects, all_ring_centers, all_element_areas = sph_meshing.get_monte_mesh(self.r_ecef_hist[i], self.n_rings)
    #         self.all_ring_centers_hist[i] = np.vstack(all_ring_centers)
    #         areas_hist[i] = all_element_areas
# 
    #     savemat(out_dir + '/all_element_centers_hist.mat', {'my_array': np.array(self.all_ring_centers_hist)})
    #     savemat(out_dir + '/all_el_area_hist_by_ring.mat', {'my_array': np.array(areas_hist)})
    #                                                             # TODO: organize i/o stream more consistently

        

    def evaluate_SH_at_monte_mesh_hist(self, out_dir):

        a_sh_map = self.force_manager.albedo_sh_map
        e_sh_map = self.force_manager.emissivity_sh_map
        Nmax = self.force_manager.Nmax
        n_steps = len(self.pointtimes)

        a_hist = [None] * n_steps
        e_hist = [None] * n_steps
        
        for i in range(n_steps):
            element_centers = self.all_ring_centers_hist[i]
            a_array = [None] * len(element_centers)
            e_array = [None] * len(element_centers)

            for j, center in enumerate(element_centers):
                lon_rad, lat_rad = np.deg2rad(center)
                u = np.cos(np.pi/2 - lat_rad)
                # Pnm = legendre(Nmax, u, normalization='unnorm')  # TODO: pysh tools' legendre function can't be used since pyshtools is not in monte168!
                Pnm = lpmn(Nmax, Nmax, u) # TODO: check normalization!!! Not verified against pyshtools!!!

                m_idx = np.arange(Nmax+1)  # shape (nmax+1,)
                m_array = np.tile(m_idx, (Nmax+1,1))  # broadcast to shape (nmax+1, nmax+1)

                a_array[j] = np.sum( Pnm * (a_sh_map[0,:,:]*np.cos(m_array*lon_rad) + a_sh_map[1,:,:]*np.sin(m_array*lon_rad)) )
                e_array[j] = np.sum( Pnm * (e_sh_map[0,:,:]*np.cos(m_array*lon_rad) + e_sh_map[1,:,:]*np.sin(m_array*lon_rad)) )

            a_hist[i] = a_array
            e_hist[i] = e_array

        
        savemat(out_dir + '/all_element_centers_a_hist.mat', {'my_array': np.array(a_hist)})
        savemat(out_dir + '/all_element_centers_e_hist.mat', {'my_array': np.array(e_hist)})
                




class ForceManager():

    def __init__(self, sim_instance, th):
        self.boa = sim_instance.boa
        self.settings_dict = sim_instance.config_info['force_settings']
        self.rad_settings_dict = sim_instance.EEI_truth_settings # this ensures that all radiation force settings are consistent with the set EEI
        self.verbose = sim_instance.verbose
        self.sc = sim_instance.spacecraft
        self.th = th # sim_instance.daily_th   # TODO
        self.n_rings_albedo = sim_instance.n_rings_albedo
        self.GravModel = sim_instance.GravModel
        #self.wgs84_albedo = sim_instance.wgs84_albedo 

        self.forces = []
        self.pressforces = []

    def set_forces(self):

        if self.verbose: print('\n\nsettings: ', self.settings_dict, '\n\n')
        self.force_names = []

        self._load_gravity(self.sc.name)
        self._load_drag()   # WE'RE HERE 
        self._load_srp()
        self._load_albedo()
        
        # if self.settings_dict[ 'aero_force' ]:
        #     self._load_drag()
# 
        # if self.settings_dict[ 'srp_force' ]:
        #     self._load_srp()
# 
        # #if self.settings_dict[ 'albedoforce' ]:
        # self._load_albedo()

        if self.verbose: print("Set forces: ", self.forces)


    def _load_gravity(self, scname):

        self.forces.append(M.GravityForce)
        grav = M.Gravity(self.boa, scname)

        Nmax = 12 #self.settings_dict['Nmax_grav']
        
        # EarthHarmonics = M.SphHarmonics(self.boa, "Earth", M.CoordName.IauEarthFixed, self.GravModel.meanr,  # TODO: is this output useless?? (the line is not)
        #                                 self.GravModel.jCof[:(Nmax+1)], Nmax, self.GravModel.cCof, self.GravModel.sCof, self.GravModel.normalized)
        Earth_SH = M.SphHarmonics(self.boa, "Earth", "Earth", M.CoordName.IauEarthFixed, self.GravModel.meanr,
                                  self.GravModel.normalized, self.GravModel.jCof, self.GravModel.cCof, self.GravModel.sCof, []) # new constructor TODO: Nmax
        Earth_SH.setDegreeCS(Nmax)
        Earth_SH.setDegreeJ(Nmax)
        
        Earth_SOI = constants.earth_SOI(monte_units=True)
        grav.insert(M.GravityNode("", "", "Earth", Earth_SOI, M.GravityNode.SPHERICAL))
        
        for body in (self.settings_dict['third_bodies']).split('|'):
            grav.insert( M.GravityNode("", "", body, 0*units.km))    #, GravityNode.NEWTONIAN) )
            #self.force_names.append(body) # TODO: third bodies (?)

        self.pressforces.append(grav)
        self.force_names.append("total_grav") # TODO: third bodies (?)

        if self.verbose: print("gravity force loaded")
        

    def _load_drag(self):

        def get_atmospheric_model():
            F107_ap_model_file = os.path.join(CONFIG_DIR, 'earth', 'atmosphere', 'CurF10_extended.txt') 
            SolarFluxMag = 0.50  # TODO: clarify if this can be modified and why it is in "atmospheric model"
            return F107_ap_model_file, SolarFluxMag

        self.forces.append(M.AtmDragForce)

        AtmModel, SolarFluxMag = get_atmospheric_model()
        fluxdata = open(AtmModel).readlines()
        fluxUnit = M.UnitDbl(1e-22,'kg/sec/sec')
        fluxTable = M.MonthlyTableFlux( self.boa, 'Flux Model')
        yearInitialized = False

        for i, line in enumerate(fluxdata):
            if i >= 7:
                data = line.strip().split()
                
                if data[1] == 'JAN':
                    fluxYear = int(float(data[0]))
                    fluxVals = [ M.HermiteTable( [0.05, 0.5, 0.95], [float(data[4]), float(data[3]), float(data[2])]).interp( SolarFluxMag )[0]*fluxUnit ]
                    magVals = [ M.HermiteTable( [0.05, 0.5, 0.95], [float(data[7]), float(data[6]), float(data[5])]).interp( SolarFluxMag )[0] ]
                    yearInitialized = True
                elif yearInitialized:
                    fluxVals.append( M.HermiteTable( [0.05, 0.5, 0.95], [float(data[4]), float(data[3]), float(data[2])]).interp( SolarFluxMag )[0]*fluxUnit )
                    magVals.append( M.HermiteTable( [0.05, 0.5, 0.95], [float(data[7]), float(data[6]), float(data[5])]).interp( SolarFluxMag )[0] )
            
                if data[1] == 'DEC' and yearInitialized:
                    fluxTable.insertFlux( fluxYear, fluxVals )
                    fluxTable.insertGeo( fluxYear, magVals )

        densityModel = M.Dtm(self.boa,'DTM','Flux Model' )

        atm = M.AtmDrag(self.boa, self.sc.name, self.sc.cd, self.sc.shape_label, 'DTM') # 'OCO2 Shape', 'DTM')
        self.pressforces.append(atm)
        self.force_names.append('aero')

        if self.verbose:
            print ('The mass is %.1f kg.' %( self.sc.mass.value() ) )
            print ('The area is %.1f m^2.' %( self.sc.area.value()*1.e6 ))
            print ('The solar flux is at %.0f%%.' %( SolarFluxMag*100. ))
            print ('drag loaded')

            print('aeroPress.__class__: ', atm.__class__ )    

        
    def _load_srp(self):
        
        self.forces.append(M.SolarPressureForce)

        time = self.th.start_epoch
        TSI_source = self.rad_settings_dict["TSI_source"]
        jD = np.mean([self.th.start_epoch.julianDate('UTC'), self.th.stop_epoch.julianDate('UTC')])

        # from solarflux import find_solar_force_atbody
        # force = find_solar_force_atbody( self.boa, "Earth", time)  # Why not just use the historical value? To be revised
        # solarenergy = find_solar_flux_atbody( boa, body, time )
        solar_energy_1AU = rad_settings.get_TSI_1AU(jD, TSI_source)

        # TODO: make more robust with monte_units=True
        c = constants.light_speed(units='m/s', monte_units=False)  #units m/s^2
        au = constants.astronomical_unit(units='m', monte_units=False) # AU converted to units of meters
        solar_flux = solar_energy_1AU / c * au**2 * units.N   # what Monte calls "Solar flux"

        #create the object:
        solarRadPress = M.SolarPressure(self.boa, self.sc.name, self.sc.shape_label, solar_flux)

        #enable shadowing:
        solarRadPress.addShadowBody(M.BodyName.Earth)

        # TODO: if smooth shadow disabled
        if self.settings_dict["smooth_shadow"] == 0:
            solarRadPress.setShadowThreshold(1 * units.radian)
            print("The Sun is a point-source now")

        self.pressforces.append(solarRadPress)
        self.force_names.append('srp')

        if self.verbose:
            print('srpPress.__class__: ', solarRadPress.__class__ )


    def _load_albedo(self):

        self.forces.append(M.AlbedoForce)
        null_mat = M.Matrix(1, 1)
        self.earthAlbedo = M.Albedo(self.boa, "Earth", constants.earth_radius(monte_units=True), null_mat, null_mat, null_mat, null_mat)
        # self.Nmax = self.rad_settings_dict['Nmax']
        
        datestring = str(self.th.start_epoch.date('UTC')).split()[0]
        a_CS_mats, e_CS_mats = rad_settings.get_a_and_e_sh_maps(datestring, self.rad_settings_dict)
        #self.albedo_sh_map, self.emissivity_sh_map = get_ae_sh_maps_numpy(a_CS_mats, e_CS_mats, self.Nmax). # deprecated: was used to generate plots inside the simulator
                
        if a_CS_mats is not None:
            self.earthAlbedo.setAlbedoCosineCof(a_CS_mats[0])
            self.earthAlbedo.setAlbedoSineCof(a_CS_mats[1])
        
        if e_CS_mats is not None:
            self.earthAlbedo.setThermalCosineCof(e_CS_mats[0])
            self.earthAlbedo.setThermalSineCof(e_CS_mats[1])

        maxVehDist = 1000000.0 * units.km
        self.earthAlbedo.setModelSphere( maxVehDist )

        threshAngle = 180.0 * units.deg  # if you want to limit the arc angle set smaller
        earthAlbedoPress = M.AlbedoPressure( self.boa, self.sc.name, 
                                                 self.sc.name+' Shape',  self.sc.name+' Shape', 
                                                 ["Earth"], threshAngle )
        
        #if self.wgs84_albedo:
        #    earthAlbedoPress.setShape('Earth Shape #1')
        
        if self.n_rings_albedo is not None:
            earthAlbedoPress.setNumRings( int(self.n_rings_albedo) )

        self.pressforces.append(earthAlbedoPress)
        self.force_names.append('erp')
        if self.verbose: print('earthAlbedoPress.__class__: ', earthAlbedoPress.__class__ )



class Time:
    def __init__(self,time):
        if isinstance(time, dict):
            self.year = time['year']
            self.month = time['month']
            self.day = time['day']
            self.hour = time['hour']
            self.minute = time['minute']
            self.second = time['second']
            #self.julian = jultime(self.hour,self.minute,self.second,self.day,self.month,self.year)
            #self.gstdeg = gstdeg(self.julian)

        else:   # assume float as jd
            time_ap = Time_astropy(time, format='jd', scale='utc')
            self.year = time_ap.datetime.year
            self.month = time_ap.datetime.month
            self.day = time_ap.datetime.day
            self.hour = time_ap.datetime.hour
            self.minute = time_ap.datetime.minute
            self.second = time_ap.datetime.second
        
        self.covert_Time_to_Epoch()
        self.convert_to_date_str()

    def displaytime(self, time):
        yr = str(time.year)
        mo = str(time.month)
        dy = str(time.day)
        hr = str(time.hour)
        mn = str(time.minute)
        sc = str(time.second)

        if time.minute < 10:
            minzero = "0"
        else:
            minzero = "" # use to add extra leading zeros if necessary
        if time.second < 10:
            seczero = "0"
        else:
            seczero = "" # use to add extra leading zeros if necessary

        self.stringout = hr+":"+minzero+mn+":"+seczero+sc+"  "+mo+"/"+dy+"/"+yr

    def covert_Time_to_Epoch(self):  #Converts Time object in user_inputs.py to a Monte Epoch in UTC
        yr = str(self.year)
        mo = str(self.month)
        dy = str(self.day)
        hr = str(self.hour)
        mn = str(self.minute)
        sc = str(self.second)

        #Epoch format: M.Epoch('2021-Apr-1 00:00:00 UTC')
        self.epochstring = yr+'-'+mo+'-'+dy+' '+hr+':'+mn+':'+sc+' UTC'   # TODO: review datestr vs epochstr
        self.epochstring_astropy = yr+'-'+mo+'-'+dy+'T'+hr+':'+mn+':'+ f"{int(np.round(self.second)):02d}"   # TODO: review datestr vs epochstr
        self.epoch = M.Epoch(self.epochstring)

        return self.epoch
    
    def convert_to_date_str(self):
        cdateStartEpoch = M.CalDate(self.epoch) 
        year = str(cdateStartEpoch.year())
        month = str(cdateStartEpoch.month())
        day = str(cdateStartEpoch.day())

        if cdateStartEpoch.month() < 10:
            month = "0"+month

        if cdateStartEpoch.day() < 10:
            day = "0"+day

        self.datestring = year+"-"+month+"-"+day





class Spacecraft():
    # Initialize spacecraft parameters
    # only name is needed and other params can be set later (or never for loaded trajectories)

    #def __init__(self, name, sim_instance: SpaceBallsSim): # , state_0=[] 
    def __init__(self, boa, name, start_epoch, sc_config_dict: dict, sc_kep0_dict: dict, existing_days: int): # , state_0=[] 
    #def __init__(self, name, shape='sphere', area=1.0, rad=0.5, length = 1.0,
    #            mass=100, cd=2.2, diffReflect=0.0, diffDegrade=1.0, boa=None, kep_0=None): # , state_0=[] 

        self.name = name
        self.shape_name = sc_config_dict['shape']                # [m^2]
        self.area = sc_config_dict['area']  * units.m * units.m                                      # [m^2]
        self.equivalent_radius = np.sqrt( self.area /np.pi )         # [m]
        self.mass = sc_config_dict['mass'] * units.kg             # [kg]
        self.cd = sc_config_dict['CD']                            # []
        self.diffReflect = sc_config_dict['diffReflect']          # [] 
        self.diffDegrade = sc_config_dict['diffDegrade']          # []
        self.specReflect = sc_config_dict['specReflect']          # [] 
        self.specDegrade = sc_config_dict['specDegrade']          # []

        self.start_epoch = start_epoch
        #self.png_out_folder = sim_instance.out_dir_png
        
        if existing_days==0:
            self.kep_0 = sc_kep0_dict # sim_instance.userfile.SC_KEP_0
        else: # TODO: use previous boa file properly
            last_day_dir = sim_instance.out_dir_csv + 'day_' + str(sim_instance.existing_days) 
            try:
                cart_r_hist = np.loadtxt(last_day_dir + '/xyz_ecef.csv')  # [km]
            except:
                cart_r_hist = np.load(last_day_dir + '/xyz_ecef.npy')  # [km]
            self.cart_r_0 = cart_r_hist[0,:]
            
            cart_v_hist_files = [os.path.join(last_day_dir,'vxvyvz_ecef.'+ext) for ext in ['csv', 'npy']] #last_day_dir + '/vxvyvz_ecef.csv'

            if os.path.isfile(cart_v_hist_files[0]):
                cart_v_hist = np.loadtxt(cart_v_hist_files[0], skiprows=1)
                self.cart_v_0 = cart_v_hist[0,:]  # [km/s]
            elif os.path.isfile(cart_v_hist_files[1]):
                cart_v_hist = np.load(cart_v_hist_files[1])
                self.cart_v_0 = cart_v_hist[0,:]  # [km/s]

            else:
                print("Warning: velocity history file not found! Estimating initial velocity from position history")
                jd_vec = np.loadtxt(last_day_dir + '/jd_vec.csv')
                
                # first order derivaitve seems to increase eccentricity too much (factor 10); try central difference
                # self.cart_v_0 = (cart_r_hist[1,:] - cart_r_hist[0,:]) / ((jd_vec[1]-jd_vec[0]) * 86400.0)  # [km/s]
                
                last_day_dir_prev = sim_instance.out_dir_csv + 'day_' + str(sim_instance.existing_days - 1)
                jd_vec_prev = np.loadtxt(last_day_dir_prev + '/jd_vec.csv') 
                cart_r_hist_prev = np.loadtxt(last_day_dir_prev + '/xyz_ecef.csv')  # [km]
                # cart_r0_prev = cart_r_hist_prev[-1,:]
                self.cart_v_0 = (cart_r_hist[1,:] - cart_r_hist_prev[-1,:]) / ((jd_vec[1]-jd_vec_prev[-1]) * 86400.0)  # [km/s]          
            
        #self._set_initial_state_Monte()

        if self.shape_name != 'sphere':
            # for non-sphere shapes
            self.direct =  Frames.set_NadirECI(boa, self.name)
            self.direct2 = Frames.set_SunDirECI(boa) # TBC with other config options

            # for cylindrical shape:
            #if self.shape_name=="cylinder":
            #    self.rad = rad * units.m
            #    self.len = length * units.m

        self._init_shape(boa)
    
    def _init_shape(self, boa, make_sc_plot=False):
        #creates a default spherical shape based on the projected area self.area
        self.shape_label = self.name + " Shape"
        self.shape = M.ScShape( boa, self.shape_label)

        if self.shape_name=="sphere":
            self._define_sphere_shape()
        else:
            parts = re.match(r'([a-zA-Z]+)(\d+)', self.shape_name)
            if parts is not None and parts.group(1)=='sphere':
                # sided sphere
                n_faces = int(parts.group(2))
                self._define_plated_shape(boa, n_faces, load_csv=False)
                
        #else:
        #    self._define_plated_shape()

        if make_sc_plot:
            self.plot_sc_shape()


    def _define_sphere_shape(self):

        # area_val = self.area / units.m / units.m
        # radius = math.sqrt( area_val/math.pi ) * units.m
        SphElem = M.ScSphere("Sphere", self.equivalent_radius, self.diffReflect, self.diffDegrade )
        self.shape.insert( SphElem )

    def _define_plated_shape(self, boa, n_faces, load_csv=False):
        
        #frame_name = self.shape_name + '_frame'
        frame_name = "Nadir Ref"  # define the orientation of the plates in the Nadir ref (?)

        if load_csv: # legacy
            if self.shape_name == "sphere128":
                csvfilename = './config/shape_files/sphere128-1m.csv'  # TODO: warning that this seems to be a sphere of radius 0.5 m only
                data = np.loadtxt(csvfilename, delimiter=',', skiprows=1)
                normal_vecs = data[:,:3]
                areas = data[:,3]
        
        else:
            print("Fibonacci sphere here - not active until import dependencies are sorted out")
            #normal_vecs, areas = sph_meshing.get_mesh_from_fibonacci_sphere(
            #    self.equivalent_radius.convert('m') , n_faces)
            radius_meters = self.equivalent_radius.convert('m')
            print(f"self.equivalent_radius.convert('m'): {radius_meters}")
            mesh = sph_meshing.get_faceted_sphere_mesh(radius_meters, n_faces)
            vertices, centroids, normal_vecs, areas, edge_conns, hull = mesh



        for i, (vec, area) in enumerate(zip(normal_vecs, areas)):
            vec = M.Dbl3Vec(vec[0], vec[1], vec[2])
            area = area * units.m * units.m
            
            # frame name for each plate? apparently not
            # frame_name = self.shape_name + '_' + str(i) + '_frame'
            
            dirbody = M.FixedDir(boa, frame_name, vec)  # "(one) typical use is to define a direction that lies along an axis of a coordinate frame"
            
            PlateElem = M.ScPlate("Plate_" + str(i), # name
                                  "ONE_SIDED",       # type
                                  area,              # area [m^2]
                                  dirbody,           # normal direction [,,]
                                  self.specReflect,  # specular reflectivity coefficient
                                  self.diffReflect,  # diffuse reflectivity coefficient
                                  self.specDegrade,  # specular reflectivity degradation factor
                                  self.diffDegrade   # diffuse reflectivity degradation factor # TODO: different coefficients per plate?
                                )
            self.shape.insert( PlateElem )


    def _set_initial_state_Monte(self, boa, existing_days, start_epoch=None):
        #set_State_UserConfig( boa, sc, epoch, frameName ):
    
        # Set initial state prior to propagation
        if start_epoch is None:
            start_epoch = self.start_epoch
        
        if existing_days==0:
            self.initState = M.State (
                                M.Conic.semiMajorAxis( self.kep_0['a_0'] *units.km ),    # 7378.14 mean alt 1000km, 7037.34 is osc, mean is 7028.14; mean alt. 650km  
                                M.Conic.eccentricity( self.kep_0['e_0'] ),               # 0.0012550 is for frozen orbit
                                M.Conic.inclination( self.kep_0['i_0'] * units.deg ),    # 99.48 deg for ss orbit at altitude 1000km, 97.99 is inclination for meab alt 650 km    
                                M.Conic.longitudeOfNode( self.kep_0['RAAN_0'] *units.deg),  # MLTAN 6AM for 90 deg; 12PM for 180 deg   
                                M.Conic.argumentOfPeriapsis( self.kep_0['w_0'] *units.deg), 
                                M.Conic.trueAnomaly( self.kep_0['theta_0'] *units.deg),  # just choosing a true anomaly: use (359.9-initArgPeri)        
                                # # EME is inertial equatorial, so this should be correct
                                M.StateInfo (boa, start_epoch, self.name, "Earth", "EME2000" ) # frameName = "Earth Inertial at Launch" 
                                
                                # # That would be wrt the rotating frame:
                                # M.StateInfo (self.boa, start_epoch, self.name, "Earth", "IAU Earth Fixed" ) # frameName = "Earth Inertial at Launch" 
                            )    
        else:
            self.initState = M.State( M.Unit3Vec(self.cart_r_0[0] *units.km, 
                                                 self.cart_r_0[1] *units.km,
                                                 self.cart_r_0[2] *units.km),
                                      M.Unit3Vec(self.cart_v_0[0] *units.km/units.sec,
                                                 self.cart_v_0[1] *units.km/units.sec,
                                                 self.cart_v_0[2] *units.km/units.sec),
                                      M.StateInfo (boa, start_epoch, self.name, "Earth", "IAU Earth Fixed" ) # frameName = "Earth Inertial at Launch"
                                           
            ).rotate("EME2000")
            # from the documentation: "This (.rotate()) does not call the high precision rotation version of CoordSet.rotation().
            # It uses the regular rotation() call without the body location being passed in."
            # TODO: it does not matter because ECEF and EME2000 have a common origin, right?




# TODO: separate set of functions?


