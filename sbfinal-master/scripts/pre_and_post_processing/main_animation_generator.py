from SpaceBalls.postproc_EEI_estimation import generate_constellation_day_animations, generate_constellation_stacking_animations

generate_constellation_stacking_animations('case_5_years', sc_names=['sc_C1', 'sc_C2', 'sc_C3'], 
                                           constellation_tag='constellation_C',
                                           n_days=365, n_lon=360, n_lat=180, frame='SFF')

generate_constellation_day_animations('case_5_years', sc_names=['sc_C1', 'sc_C2', 'sc_C3'],
                                        constellation_tag='constellation_C',
                                        day_idxs=[0, 182], max_hours=12)

