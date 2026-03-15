import os, sys
import numpy as np
import importlib
from SpaceBalls.paths import CONFIG_DIR
sys.path.insert(0, str(CONFIG_DIR.parent))  # parent of 'config'


def get_TSI_1AU(jD, TSI_source):
    
    if TSI_source=="LASP":
        TSI_mat = np.loadtxt(os.path.join(CONFIG_DIR, 'TSI', 'TSI_LASP.txt'), skiprows=133)
        # JD_vec_nom = np.array(TSI_mat[:,1])
        JD_vec_avg = np.array(TSI_mat[:,2])
        TSI_1AU_vec = np.array(TSI_mat[:,4])

        TSI_Earth_vec = np.array(TSI_mat[:,9])

        idxs_to_rm = TSI_1AU_vec==0
        JD_vec_avg_clean = JD_vec_avg[~idxs_to_rm]
        TSI_1AU_vec_clean = TSI_1AU_vec[~idxs_to_rm]

        TSI_Earth_vec_clean = TSI_Earth_vec[~idxs_to_rm]

        solar_energy_1AU = np.interp(jD, JD_vec_avg_clean, TSI_1AU_vec_clean) # circa 1360 W/m^2 (classical solar flux)
        solar_energy_Earth = np.interp(jD, JD_vec_avg_clean, TSI_Earth_vec_clean)
        
        return solar_energy_1AU
        
    elif TSI_source=="CERES":
        TSI_mat = np.loadtxt(os.path.join(CONFIG_DIR, 'TSI', 'TSI_CERES.txt'), delimiter=',') # col 1: jd; col 2: TSI_1AU
        jd_vec = np.array(TSI_mat[:, 0])
        idx = np.where(jd_vec==jD)
        solar_energy_1AU = TSI_mat[idx, 1][0][0]
        
        return solar_energy_1AU
    

def get_reduced_matrix(M_in, Nmax):
    M_aux = M_in.slice(0, Nmax).transpose()
    M_out = M_aux.slice(0, Nmax).transpose()

    return M_out


def get_CS_mats(type_str, date_str, Nmax_file, Nmax_crop, mode="historic_SYN1deg", subfolder=None, name_tail=None):
    
    if (subfolder is None) and (name_tail is None):
        subfolder = ''

        if mode=="historic_SYN1deg":
            name_tail = date_str + '_Nmax' + str(Nmax_file)
            subfolder = 'SYN1deg_2018_2022.'
        elif mode=="default":
            name_tail = "default"
        elif mode=="cristopher":
            assert(date_str=='2021-12-31')
            subfolder = 'cristopher.'
            name_tail = date_str + 'T232945_Nmax50'
        elif mode=="new":
            name_tail = date_str + '_Nmax' + str(Nmax_file)
            subfolder = 'all_2018_2022_new.'
        elif mode=="test":
            name_tail = 'C00'
            subfolder = 'test.'
            Nmax_crop = 0
        
     # why does it work without the "shfiles." thing outside of tests??
    module_name = 'config.earth.albedo_and_thermal.' + subfolder + type_str + 'Cosine_' + name_tail
    module = importlib.import_module(module_name)
    C_mat = getattr(module, type_str.lower() + 'CosineCof')
    C_mat = get_reduced_matrix(C_mat, Nmax_crop+1)

    module_name = 'config.earth.albedo_and_thermal.' + subfolder + type_str + 'Sine_' + name_tail
    module = importlib.import_module(module_name)
    S_mat = getattr(module, type_str.lower() + 'SineCof')
    S_mat = get_reduced_matrix(S_mat, Nmax_crop+1)

    return C_mat, S_mat



def get_a_and_e_sh_maps(datestring, rad_config_dict):
    
        albedo_types = rad_config_dict["earth_components"]
        Nmax_file = 2 if rad_config_dict["sh_mode"]=="default" else 50  # TODO: hard-coded, TBD
        Nmax = rad_config_dict["Nmax"]
        #sys.path.append('./config/shfiles')

        albedo_CS_mats = None
        emissivity_CS_mats = None

        for i, type_str in enumerate(albedo_types):
            C_mat, S_mat = get_CS_mats(type_str, datestring, Nmax_file, Nmax,
                                       mode=rad_config_dict["sh_mode"])
            if type_str=="Albedo":
                albedo_CS_mats = [C_mat, S_mat]

            elif type_str=="Thermal":
                emissivity_CS_mats = [C_mat, S_mat]
                
        return albedo_CS_mats, emissivity_CS_mats



def get_ae_sh_maps_numpy(a_CS_mats, e_CS_mats): # deprecated?
    
    if a_CS_mats is not None:
        a_C_mat = a_CS_mats[0].toArray()
        a_S_mat = a_CS_mats[1].toArray()
        a_sh_map = np.stack([a_C_mat, a_S_mat], 0)
        
    if e_CS_mats is not None:
        e_C_mat = e_CS_mats[0].toArray()
        e_S_mat = e_CS_mats[1].toArray()
        e_sh_map = np.stack([e_C_mat, e_S_mat], 0)
        
    return a_sh_map, e_sh_map

def convert_sh_maps_monte_to_numpy(C_mat, S_mat):
    return np.stack([C_mat.toArray(), S_mat.toArray()], 0)


def get_ae_sh_maps_numpy_new(mode, date):

    if mode=="historic_SYN1deg":
        source = 'SYN1deg_2018_2022'

    dir = os.path.join(CONFIG_DIR, 'earth', 'albedo_and_thermal', source, 'numpy_format')
    albedo_array = np.load(os.path.join(dir, 'Albedo_' + date + '.npy'))
    emissivity_array = np.load(os.path.join(dir, 'Thermal_' + date + '.npy'))

    return albedo_array, emissivity_array


def radiation_settings_from_EEI_truth_name(EEI_truth_name):

    rad_config_dict = {}

    if EEI_truth_name=="EEI_truth_1":
        rad_config_dict["TSI_source"] = "CERES"
        rad_config_dict["earth_components"] = ["Albedo", "Thermal"] # ["Albedo", "Thermal"], case sensitive
        rad_config_dict["sh_mode"] = "historic_SYN1deg"
        rad_config_dict["Nmax"] = 45
        rad_config_dict["jd_interval"] = [2458119.5, 2459945.5]
        rad_config_dict["ephemerides"] = "boa_0"
        rad_config_dict["earth_shape"] = "spherical"    
        # 2018-01-01 00:00:00.000 to 2023-01-01 00:00:00.000
        
    else:
        print(f"{EEI_truth_name} not recognized!")

    return rad_config_dict
