from SpaceBalls.sbsim import SpaceBallsSim, Time
import SpaceBalls.input_database_manager as input_db_manager

t0 = Time({'year': 2018,
        'month': 1,
        'day': 1,
        'hour': 0,
        'minute': 0,
        'second': 0})

for i in range(20):
    simulator = SpaceBallsSim(input_name='SB_R800_0', #'LAGEOS_'+str(100+i), 
                            input_mode='hash')
    
    simulator.run_propagation_arc(t0, 24)