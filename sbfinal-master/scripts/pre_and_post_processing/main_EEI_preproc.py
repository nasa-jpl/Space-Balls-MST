from SpaceBalls.radiation_fluxes_preprocessing import compute_radiation_maps    
from SpaceBalls.sph_meshing import RegularLatLonGrid, QuadratureGrid

if __name__ == '__main__':

    EEI_truth_name = "EEI_truth_35"
    #grid = RegularLatLonGrid(alt_km=0, n_lat=18, n_lon=36)
    grid = QuadratureGrid(alt_km=0, order=131)
    compute_radiation_maps(EEI_truth_name, grid=grid, 
                           selected_days_idxs=None, n_cores=1)

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