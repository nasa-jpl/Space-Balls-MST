import sys, os
import xarray as xr
import numpy as np
from SpaceBalls.plotter import Plotter
from SpaceBalls.sph_meshing import RegularLatLonGrid
from SpaceBalls.paths import MEDIA_DIR, CONFIG_DIR
from SpaceBalls.sph_meshing import fit_sh_field, expand_sh

ceres_grid = RegularLatLonGrid(alt_km=0, n_lat=180, n_lon=360)
base_dir = os.path.join(MEDIA_DIR, 'true_EEI', 'ceres_dataproducts')
out_dir = os.path.join(CONFIG_DIR, 'earth', 'albedo_and_thermal', 'SYN1deg_l179', 'numpy_format')
os.makedirs(out_dir, exist_ok=True)

#ebaf_fname = 'CERES_EBAF-TOA_Ed4.2.1_Subset_201801-202512.nc'
syn1deg_fname_1 = 'CERES_SYN1deg-Day_Terra-Aqua-NOAA20_Ed4.2_Subset_20180101-20220105.nc'

#ebaf_dataset = xr.open_dataset(os.path.join(base_dir, ebaf_fname))

syn1deg_dataset_1 = xr.open_dataset(os.path.join(base_dir, syn1deg_fname_1))
time_array = syn1deg_dataset_1['time'].to_numpy()

sw_daily_array = syn1deg_dataset_1['toa_sw_all_daily'].to_numpy()
a_daily_array = syn1deg_dataset_1['toa_alb_all_daily'].to_numpy()
a_daily_array[np.isnan(a_daily_array)] = 0

lw_daily_array = syn1deg_dataset_1['toa_lw_all_daily'].to_numpy()
solar_daily_array = syn1deg_dataset_1['toa_solar_all_daily'].to_numpy()
avg_solar_daily_array = ceres_grid.compute_surf_integral(np.moveaxis(solar_daily_array, 0, -1), average=True)
e_daily_array = lw_daily_array / avg_solar_daily_array[:,None,None]

n_days = len(time_array)

for day_idx in range(len(time_array)):
    print(f"Computing day {day_idx+1}/{len(time_array)}")
    day_label = np.datetime_as_string(time_array[day_idx], unit='D')

    a_day = np.flipud(a_daily_array[day_idx,:,:])
    sh_a = fit_sh_field(ceres_grid.vectorize_if_needed(a_day), 
                        ceres_grid.stacked_grid_latlon[:,1], 
                        ceres_grid.stacked_grid_latlon[:,0], lmax=179)
    print(f"Saving {os.path.join(out_dir, 'Albedo_'+day_label)}")
    np.save(os.path.join(out_dir, 'Albedo_'+day_label), sh_a.coeffs)
    
    e_day = np.flipud(e_daily_array[day_idx,:,:])
    sh_e = fit_sh_field(ceres_grid.vectorize_if_needed(e_day), 
                        ceres_grid.stacked_grid_latlon[:,1], 
                        ceres_grid.stacked_grid_latlon[:,0], lmax=179)
    np.save(os.path.join(out_dir, 'Thermal_'+day_label), sh_e.coeffs)
    print(f"Saving {os.path.join(out_dir, 'Thermal_'+day_label)}")