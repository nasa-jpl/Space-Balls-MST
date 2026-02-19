import numpy as np
import importlib
import sys
import os

def convert_to_shtools_array(Nmax, date_str, rad_type):

    folder = './mst/trajectory/propagate/'
    sys.path.append(os.path.abspath(folder))
    sys.path.append(os.path.abspath(folder + 'shfiles/'))

    module_name = rad_type + 'Cosine_' + date_str + '_Nmax' + str(Nmax)
    module = importlib.import_module(module_name)
    C_mat = getattr(module, rad_type.lower() + 'CosineCof')
    C_mat = np.array(C_mat.toArray())

    module_name = rad_type + 'Sine_' + date_str + '_Nmax' + str(Nmax)
    module = importlib.import_module(module_name)
    S_mat = getattr(module, rad_type.lower() + 'SineCof')
    S_mat = np.array(S_mat.toArray())

    shtools_array = np.zeros((2, Nmax+1, Nmax+1))
    shtools_array[0] = C_mat
    shtools_array[1] = S_mat

    return shtools_array


all_dates = (
    ['2018-01-0' + str(i) for i in range(1,7)] + ['2018-03-21'] + 
    ['2022-12-' + str(i) for i in range(25,31)]
)
n = len(all_dates)

thermal_array = [None] * n
albedo_array = [None] * n

for i, date_str in enumerate(all_dates):
    thermal_array[i] = convert_to_shtools_array(50, date_str, 'Thermal')
    albedo_array[i] = convert_to_shtools_array(50, date_str, 'Albedo')



#thermal_50 = convert_to_shtools_array(50, "2018-01-01", 'Thermal')
#albedo_50 = convert_to_shtools_array(50, "2018-01-01", 'Albedo')

#thermal_30 = convert_to_shtools_array(30, "20210321_12", 'Thermal')
#albedo_30 = convert_to_shtools_array(30, "20210321_12", 'Albedo')
