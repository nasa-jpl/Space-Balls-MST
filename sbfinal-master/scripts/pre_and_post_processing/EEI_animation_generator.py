from SpaceBalls.radiation_fluxes_preprocessing import compute_daily_time_series_toa, get_day_jd_array
from SpaceBalls.radiation_settings import radiation_settings_from_EEI_truth_name, get_TSI_1AU
from SpaceBalls.sph_meshing import QuadratureGrid
from SpaceBalls.plotter import Plotter
import numpy as np

EEI_name = "EEI_truth_3"
rad_config = radiation_settings_from_EEI_truth_name(EEI_name)

day_idx = 5
mid_day_jd = rad_config['jd_interval'][0] + 0.5 + day_idx
grid = QuadratureGrid(0, order=201, flattening=0)

TSI_1AU_day = get_TSI_1AU(mid_day_jd, rad_config["TSI_source"])

rad_hist, net_hist = compute_daily_time_series_toa(mid_day_jd, rad_config, TSI_1AU_day, grid)

day_jd_hist = get_day_jd_array(mid_day_jd)
Plotter.make_map_animation_arbitrary_frame(net_hist[:,:10], day_jd_hist[:10], grid, 
                                           'Net', '.', 'test_animation',
                                           R_mat_hist=np.stack([np.eye(3) for _ in range(len(day_jd_hist))]),
                                           add_satellite_positions=False,
                                           add_groundtracks=False)