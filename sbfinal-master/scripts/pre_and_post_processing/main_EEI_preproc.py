from SpaceBalls.radiation_fluxes_preprocessing import compute_radiation_maps    

if __name__ == '__main__':
    compute_radiation_maps("EEI_truth_1", altitude_array = [800], selected_days_idxs = [0, 182],
                           n_cores=12)