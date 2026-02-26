TIMESTART = {
        'year':2018,
        'month':1,
        'day':1,
        'hour':0,
        'minute':0,
        'second':0.0,
        }

TIMEFINISH = {
        'year':2023,
        'month':1,
        'day':1,
        'hour':0,
        'minute':0,
        'second':0,
}

SC_INPUT = {
        'N_sc': 1,
        'name': "Spaceball",
        'mass': 90.0,   # kg
        'shape': "sphere", # choose name of shape type: "icosahedron", "buckyball", "cylinder", "plate", "sphere", "cube", or "sphere128"  
        'area': 0.7854, # m^2  radius = 0.5 m, a = 0.785 m^2
        #'radius': 0.5, # m radius for cylinder Shape def
        #'length': 1.0, # m length for cylinder Shape def
        'cd': 2.0,
        'diffReflect': 0.0,  # diffuse reflectivity coeff
        'diffDegrade': 1.0,  # degradation of diffuse reflectivity coeff
        
        'specReflect': 0.0,  # specular reflectivity coeff
        'specDegrade': 1.0,  # degradation of specular reflectivity coeff

        # Direction choosing - use for sc shapes requiring a direction
        'dirmode': 'nadir',  #chose mode type:  fixed, velocity, zenith, nadir, ground, sun, moon, (FUTURE:crosstrack for monte 165 and higher) 
        'attmode': 'sun',  #chose mode type:  fixed, velocity, zenith, nadir, ground, sun, moon, (FUTURE:crosstrack for monte 165 and higher) 
            #attmode uses the dirmode as the first direction and attmode to find the second direction that helps define the direction frame
}

SC_KEP_0 = { 'a': 6378.14 + 1500, #+ 894,      # 6878.14, (500km) #7378.14, #1000 km mean alt  #semi-maj axis in km
             'e': 0.0,          # 0.001, # eccentricity
             'i': 102,         # 97.40,(500 km for SS orbit) #100.66,(1250 km) #99.48,(1000 km) #58.5064, #inclination
             'o': 120.0,         # for 21Mar to zero longitude: 358.88, #not RAAN here but long. of node: 180.0 deg = 12 pm MLTAN, 90.0 = 6am
             'u': 0.0,          # Arg of perigee
             'theta': 120.0,     # true anomaly
           } 


FORCE_SETTINGS = { 
        'Earth_grav': "GGM02C",
        'third_body_grav': ['Sun', 'Moon'],
        'srp_force' : True,
        #'erp_forces': ['Albedo', 'Thermal'],   # 'Albedo', 'Thermal'
        #'erp_sh_files': "historic_SYN1deg", #"historic",
        #'erp_sh_Nmax': 45,
        'aero_force' : True,
        'EEI_truth': "EEI_truth_1",     # this defines the settings for all radiation-related forces

        #?? 'covalbedoforce' : False,
        #?? 'comboforce' : True,
        #?? 'myforce' : False,
        #?? 'lowthrustforce' : False,
        #?? 'sat1_propagate': True,
}

INTEGRATION_SETTINGS = {
    'n_rings_albedo': 25
}

OUTPUT_SETTINGS = {
    'output_sample_time': 60,
    'save_ERP_mesh_history': False,
    'make_acc_plots': False,
    'matlab_output' : ['']      # available options: ['visibility_cap_hist', 'r_sun_hist_astropy', 'R_ijk2ric']
}