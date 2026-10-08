
from SpaceBalls.radiation_settings import radiation_settings_from_EEI_truth_name
from SpaceBalls.radiation_fluxes_preprocessing import compute_radiation_maps, get_toa_grid_from_rad_config
from SpaceBalls.sph_meshing import RegularLatLonGrid, QuadratureGrid

if __name__ == '__main__':

    all_EEIs = ["EEI_truth_1"] # , "EEI_truth_4", "EEI_truth_42"] # "EEI_truth_35", 
    compute_altitude = True
    compute_toa = False

    for EEI_truth_name in all_EEIs:
    #EEI_truth_name = "EEI_truth_41"
        print(f"Now computing day 0 for {EEI_truth_name}")

        if compute_toa:
            # TOA case:
            rad_config = radiation_settings_from_EEI_truth_name(EEI_truth_name)
            toa_grid = get_toa_grid_from_rad_config(rad_config, 
                                                    quadrature_order=None, # None for default
                                                    )
            compute_radiation_maps(EEI_truth_name, grid=toa_grid, 
                                selected_days_idxs=None, 
                                compute_SFF=False)

        if compute_altitude:
            # altitude case:
            altitude = 800
            grid_order = 325
            grid = QuadratureGrid(alt_km=altitude, 
                                order=grid_order,    
                                flattening=0  # altitude grid is fully spherical
                                )

            compute_radiation_maps(EEI_truth_name, grid=grid, 
                                   selected_days_idxs=[0], 
                                   compute_SFF=True)

"""
# 2 cores seems to work for a 1deg x Leb131 grid - let's run this for now
    compute_radiation_maps("EEI_truth_1", 
                           altitude_array = [800],
                           degrees_bins=1, 
                           selected_days_idxs = [0, 182],
                           n_cores=1)

    

    # get_file_names_and_existence
    # 
    # 
    # compute_daily_hist_net_at_altitude
    # get_daily_hist_toa

"""