import os
import numpy as np
import time
from astropy.time import Time, TimeDelta
import astropy.units as u
import multiprocessing

import warnings
from cryptography.utils import CryptographyDeprecationWarning
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)

from SpaceBalls.sph_meshing import get_sphere_grid, get_spherical_grid_cell_areas
from multiprocessing import Pool

if __name__ == '__main__':
    
    n_days = 1
    h_orbit_km = 800
    degrees_CV_bins = 1
    mode = "daily_true_fluxes_through_time"   # "daily_hist_toa"  # "daily_avg_toa"
    n_cores = multiprocessing.cpu_count() - 1
    
    n_lon = 360 // degrees_CV_bins
    n_lat = 180 // degrees_CV_bins
    lon_vec, lat_vec, lon_edges_vec, lat_edges_vec = get_sphere_grid(n_lon, n_lat)
    
    # Orbit control volume:
    R_cv = EARTH_RADIUS + h_orbit_km
    grid_r_el_CV = lonlat_to_r(lon_vec, lat_vec, h_orbit_km, "spherical")
    grid_u_el_CV = grid_r_el_CV / (R_cv* np.ones((n_lat, n_lon,3)))
    stacked_grid_r_el_CV = np.reshape(grid_r_el_CV, (n_lon*n_lat, 3))
    stacked_grid_u_el_CV = stacked_grid_r_el_CV / (R_cv * np.ones((n_lon*n_lat,1)))
    grid_cell_areas_CV = get_spherical_grid_cell_areas(lat_edges_vec, lon_edges_vec, R_cv)
    stacked_grid_cell_areas_CV = np.reshape(grid_cell_areas_CV, (n_lon*n_lat,))
    
    # TOA: 
    n_lon_toa = 360
    n_lat_toa = 180
    lon_vec_toa, lat_vec_toa, lon_edges_vec_toa, lat_edges_vec_toa = get_sphere_grid(n_lon_toa, n_lat_toa)
    grid_r_el_toa = lonlat_to_r(lon_vec_toa, lat_vec_toa, 0, "spherical")
    grid_u_el_toa = grid_r_el_toa / (EARTH_RADIUS * np.ones((n_lat_toa, n_lon_toa,3)))
    #stacked_grid_r_el = np.reshape(grid_r_el_toa, (n_lon*n_lat, 3))
    #stacked_grid_u_el = stacked_grid_r_el / (EARTH_RADIUS * np.ones((n_lon*n_lat,1)))  # same as reshape grid u el
    grid_cell_areas_toa = get_spherical_grid_cell_areas(lat_edges_vec_toa, lon_edges_vec_toa)
    earth_surface_area = 4 * np.pi * EARTH_RADIUS**2
    #n_elements = np.shape(stacked_grid_r_el)[0]
    
    for i in range(n_days):
        
        # Avg TOA map:
        if mode=="daily_avg_toa":
            net_toa_map = np.loadtxt('output_files/true_EEI/grid_180x360/daily_avg_net_toa/day_'
                                    + str(i) + '.txt')
            
            all_F_outgoing_radial_CV_map = get_flux_map_CV_2(stacked_grid_r_el_CV, stacked_grid_u_el_CV,
                        grid_r_el_toa, grid_u_el_toa, grid_cell_areas_toa, 
                        net_toa_map, n_lat, n_lon, n_workers=7)
            
            np.savetxt('output_files/true_EEI/grid_180x360/daily_avg_net_' + str(h_orbit_km) + 
                    'km_' + str(degrees_CV_bins) + 'deg/day_' + str(i) + '.txt', all_F_outgoing_radial_CV_map)
            # Plotter.plot_geo_data(all_F_outgoing_radial_CV_map, lon_edges_vec, lat_edges_vec, file_name='avg_net_CV_5deg')
            #get_true_EEI_toa(daily_jd_vec_array, n_lon=360, n_lat=180)
        
        elif mode=="daily_hist_toa":
            out_dir = 'output_files/true_EEI/grid_'+str(n_lat)+'x'+str(n_lon)+'/daily_net_' + str(h_orbit_km) + 'km_' + str(degrees_CV_bins) + 'deg' + '_hist/'
            os.makedirs(exist_ok=True)
            
            # historic TOA map:
            n_steps = 1440
            EEI_day_hist = np.loadtxt('output_files/true_EEI/grid_'+str(n_lat)+'x'+str(n_lon)+
                                    '/daily_net_toa_hist/day_' + str(i) + '.txt'
                                    ).reshape((n_lat, n_lon, n_steps))
            EEI_day_hist_CV = np.zeros_like(EEI_day_hist)
            
            for j in range(n_steps):
                print(f"computing time step {j}/{n_steps}...")
                t1 = time.time()
                EEI_day_hist_CV[:,:,j] = get_flux_map_CV_2(stacked_grid_r_el_CV, stacked_grid_u_el_CV,
                        grid_r_el_toa, grid_u_el_toa, grid_cell_areas_toa, 
                        EEI_day_hist[:,:,j], n_lat, n_lon, n_workers=n_cores)
                t2 = time.time()
                print(f"Time with multiprocessing ({n_cores} cores: {t2-t1})")
                
            np.savetxt(out_dir  + 'day_' + str(i) + '.txt',
                        np.reshape(EEI_day_hist_CV, (n_lon*n_lat, n_steps)))
        
        elif mode=="daily_true_fluxes_through_time":
            
            # duplicated from get_EEI_truth: get daily jd arrays
            epoch_0 = Time('2018-01-01 00:00:00.000', format='iso', scale='utc')
            epoch_f = Time('2023-01-01 00:00:00.000', format='iso', scale='utc')
            n_days = (epoch_f - epoch_0).to(u.day).value.astype(int)
            
            # we go day by day to prevent memory overload
            epoch = epoch_0
            dt = TimeDelta(60 * u.second)
            n = 86400 / dt.sec # = 1440
            
            daily_jd_vec_array = [None] * n_days
            
            for i in range(n_days):
                
                # next day:
                epoch_0_day = epoch
                epoch_f_day = epoch_0_day + TimeDelta(1.0 * u.day)
                day_epoch_vec = epoch_0_day + np.arange(n) * dt
                daily_jd_vec_array[i] = day_epoch_vec.jd
                epoch = epoch_f_day
            
            
            out_dir = 'output_files/true_EEI/grid_'+str(n_lat)+'x'+str(n_lon)+'/daily_net_' + str(h_orbit_km) + 'km_' + str(degrees_CV_bins) + 'deg' + '_true_hist/'
            os.makedirs(out_dir, exist_ok=True)
            n_steps = 1440
            
            get_true_EEI_orbit([daily_jd_vec_array[0]], h_orbit_km, out_dir, 
                               n_workers=n_cores)
        
        print("done")



def get_flux_map_CV_2(stacked_grid_r_el_CV, stacked_grid_u_el_CV,
                    grid_r_el_toa, grid_u_el_toa, grid_cell_areas_toa, 
                    net_toa_map, n_lat_CV, n_lon_CV, n_workers=None):
    
    n_elements_CV = np.shape(stacked_grid_r_el_CV)[0] # should be equal to n_lat_CV * n_lon_CV
    print("Entering get flux map function...")
    if n_workers is None:
        t1 = time.time()
        pass
    else:
        print("Using multiprocessing...")
        args_list = [
            (stacked_grid_r_el_CV[j,:], stacked_grid_u_el_CV[j,:]) for j in range(n_elements_CV)
        ]
        t1 = time.time()
        with Pool(processes=n_workers, initializer=init_worker_2, 
                    initargs=(grid_r_el_toa, grid_u_el_toa, grid_cell_areas_toa, net_toa_map)) as pool:
            results = pool.starmap(get_net_fluxes_at_CV_element_j_simple, args_list)

        all_F_outgoing_radial_CV = results # np.array([res[0] for res in results])
        
        t2 = time.time()
        print(f"Time with multiprocessing ({n_workers} workers): {t2-t1}")
                        
    
    print('')
    all_F_outgoing_radial_CV_map = np.reshape(all_F_outgoing_radial_CV, (n_lat_CV, n_lon_CV))
    
    return all_F_outgoing_radial_CV_map


def get_net_fluxes_at_CV_element_j_simple(r_el_j_CV, u_el_j_CV): # , area_el_j_CV):
    
    g = _global
    grid_r_el = g['grid_r_el_toa']
    grid_u_el = g['grid_u_el_toa']
    grid_cell_areas_toa = g['grid_cell_areas_toa']
    net_toa_map = g['net_toa_map']
    
    all_r_rel = r_el_j_CV[None,None,:] - grid_r_el
    all_r_rel_norm = np.linalg.norm(all_r_rel,axis=2)
    all_u_rel_expanded = all_r_rel/all_r_rel_norm[:,:,None]
    
    all_cos_alpha = np.sum((all_u_rel_expanded)*grid_u_el, axis=2)
    all_cos_alpha[all_cos_alpha<0] = 0

    all_F_outgoing =  net_toa_map[:,:,None] * (1/np.pi) * grid_cell_areas_toa[:,:,None] * all_cos_alpha[:,:,None] * (1/(all_r_rel_norm**2))[:,:,None] * all_r_rel/(all_r_rel_norm[:,:,None])
    F_outgoing = np.sum(all_F_outgoing, axis=(0,1))
    F_radial_j = np.dot(F_outgoing, u_el_j_CV)
    
    
    return F_radial_j


_global = {}

def init_worker_2(grid_r_el_toa, grid_u_el_toa, grid_cell_areas_toa, net_toa_map):
    _global['grid_r_el_toa'] = grid_r_el_toa
    _global['grid_u_el_toa'] = grid_u_el_toa
    _global['grid_cell_areas_toa'] = grid_cell_areas_toa
    _global['net_toa_map'] = net_toa_map
    
    
    
def lonlat_to_r(lon_vec, lat_vec, h_toa, mode):
    
    if mode=="spherical":
        #n = len(lon_vec)*len(lat_vec)
        #(lon_mesh, lat_mesh) = np.meshgrid(lon_vec, lat_vec)
        lon_mesh_vectd, lat_mesh_vectd, mesh_shape = get_reshaped_grid(lon_vec,lat_vec)
        
        x, y, z = spherical_to_cartesian(EARTH_RADIUS + h_toa, 
                                         np.radians(lat_mesh_vectd), 
                                         np.radians(lon_mesh_vectd))
        x = np.reshape(x.value, mesh_shape)
        y = np.reshape(y.value, mesh_shape)
        z = np.reshape(z.value, mesh_shape)
        
        return np.stack((x, y, z), 2)
        
    else:
        print("Mode {mode} not implemented!")
        
        
        
        
        
# sun-fixed frame stuff:

def ecef_field_hist_to_sunframe(jd_vec, field_hist):
    
    n_steps = np.shape(field_hist)[2]
    n_lat, n_lon = np.shape(field_hist)[:2]
    lon_vec, lat_vec, lon_edges_vec, lat_edges_vec = get_sphere_grid(n_lon, n_lat)
    
    grid_r_el = lonlat_to_r(lon_vec, lat_vec, 0, "spherical")
    stacked_grid_r_el = np.reshape(grid_r_el, (n_lon*n_lat, 3))
    stacked_grid_u_el = stacked_grid_r_el / (EARTH_RADIUS * np.ones((n_lon*n_lat,1)))
    
    R_ECEF_to_SunFrame_hist = np.loadtxt('./output_files/R_ECEF_to_SunFrame_hist.txt')
    R_ECEF_to_SunFrame_hist = R_ECEF_to_SunFrame_hist.reshape(R_ECEF_to_SunFrame_hist.shape[0], 
                                    R_ECEF_to_SunFrame_hist.shape[1]//3,
                                    3)
    jd_R_vec = np.loadtxt('./output_files/jd_R_ECEF_to_SunFrame_hist.txt')
    
    idx_mask = get_idx_mask(jd_vec, jd_R_vec)
    R_hist = R_ECEF_to_SunFrame_hist[idx_mask,:,:]
    assert(np.shape(R_hist)[0]==n_steps)
    
    field_hist_sunframe = np.zeros((n_lat, n_lon, n_steps))
    
    for i in range(n_steps):
        progress_bar(i, n_steps)
        field_i = field_hist[:,:,i]
        R_i = R_hist[i,:,:]
        
        grid_u_el_rot = (R_i @ stacked_grid_u_el.T).T
        _, lat_rot, lon_rot = cartesian_to_spherical(grid_u_el_rot[:,0], grid_u_el_rot[:,1], grid_u_el_rot[:,2])  # astropy
        
        P_i = np.array((lat_rot.degree, lon_rot.degree)).T
        Z_i = field_i.reshape(n_lon*n_lat)
        
        LAT_vec, LON_vec, _ = get_reshaped_grid(lat_vec, lon_vec)
        Pq = np.array((LAT_vec, LON_vec)).T
        Z_interp = griddata(P_i, Z_i, Pq, method="linear")  # scipy interpolate
        field_hist_sunframe[:,:,i] = Z_interp.reshape((n_lat, n_lon), order='F')  # seems to be right ...
        
        # Plotter.plot_geo_data(Z_interp.reshape((n_lat, n_lon)), lon_edges_vec, lat_edges_vec, 
        #                       file_name='test_sunframe_1')
        # Plotter.plot_geo_data(Z_interp.reshape((n_lat, n_lon), order='C'), lon_edges_vec, lat_edges_vec, 
        #                       file_name='test_sunframe_C')
        # Plotter.plot_geo_data(Z_interp.reshape((n_lat, n_lon), order='F'), lon_edges_vec, lat_edges_vec, 
        #                       file_name='test_sunframe_F')
        # Plotter.plot_geo_data(Z_interp.reshape((n_lat, n_lon), order='A'), lon_edges_vec, lat_edges_vec, 
        #                       file_name='test_sunframe_A')
    return field_hist_sunframe, R_hist

def get_reshaped_grid(x,y):
    
    n = len(x)*len(y)
    X_grid, Y_grid = np.meshgrid(x,y)
    #X_vectorized = np.reshape(X_grid, (n,1)) 
    #Y_vectorized = np.reshape(Y_grid, (n,1))
    X_vectorized = np.reshape(X_grid, n) 
    Y_vectorized = np.reshape(Y_grid, n)
    
    return X_vectorized, Y_vectorized, np.shape(X_grid)


def get_idx_mask(target_jd_vec, full_jd_vec):
    
    tol = 1e-7  # choose based on your JD resolution
    idx = np.searchsorted(target_jd_vec, full_jd_vec)
    idx = np.clip(idx, 1, len(target_jd_vec)-1)

    mask = np.minimum(
        np.abs(full_jd_vec - target_jd_vec[idx]),
        np.abs(full_jd_vec - target_jd_vec[idx-1])
    ) < tol
    
    return mask


# plotter class:
    @classmethod
    def make_map_animation_arbitrary_frame(cls, map_grid_3D_rotated, R_mat_hist, jd_vec, lon_edges_vec, 
                                           lat_edges_vec, data_label, out_dir, filename, f_width=1, f_height=1, fade_coastlines=False):
        
        coastlines = cfeature.NaturalEarthFeature(
                    category='physical',
                    name='coastline',
                    scale='110m'
                )

        geoms = list(coastlines.geometries())
        all_lon_lat_points = np.empty((0,2))
        
        for geom in geoms:
            lonlat_i = np.array(geom.xy).T
            all_lon_lat_points = np.vstack((all_lon_lat_points, lonlat_i, np.zeros((1,2))*np.nan))
        n_coastline_points = np.shape(all_lon_lat_points)[0]
                
        # plt.figure()
        # plt.plot(all_lon_lat_points[:,0], all_lon_lat_points[:,1])
        # plt.savefig('test_coastlines.png')  # it works! we're here
        
        all_r_coastlines = spherical_to_cartesian(np.ones(n_coastline_points)*EARTH_RADIUS, 
                                                  np.radians(all_lon_lat_points[:,1]), np.radians(all_lon_lat_points[:,0]))
        all_r_coastlines_T = np.array(all_r_coastlines)  # coastlines in ECEF

        # t0:
        field_0 = map_grid_3D_rotated[:,:,0]
        R_0 = R_mat_hist[0,:,:]
        all_r_coastlines_0_T = R_0 @ all_r_coastlines_T
        
        _, lat, lon = cartesian_to_spherical(all_r_coastlines_0_T[0,:], all_r_coastlines_0_T[1,:], all_r_coastlines_0_T[2,:])
        lon_coast_rotated, lat_coast_rotated = cls.get_clean_lonlat_vecs_for_plot(lon.degree, lat.degree)
        
        fig, ax = plt.subplots(figsize=(2 * cls.fig_width * f_width, cls.fig_height * f_height),
                                    dpi=200)
        
        max_abs = np.nanmax(np.abs(map_grid_3D_rotated))
        norm = colors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
        
        map = ax.pcolormesh(
                lon_edges_vec,
                lat_edges_vec,
                field_0,
                shading="auto",
                cmap="seismic",
                norm=norm
            )
        cl, = ax.plot(lon_coast_rotated, lat_coast_rotated, color='k', linestyle=':', linewidth=0.5)
        # --- Colorbar ---
        cbar = fig.colorbar(map, ax=ax, shrink=0.88, pad=0.03)
        cbar.ax.set_ylabel(data_label)
        
        def update(frame):
            
            datestr = Time_astropy(jd_vec[frame], format='jd').to_datetime().strftime("%Y-%m-%d %H:%M:%S")
            ax.set_title(datestr)
            map.set_array(map_grid_3D_rotated[:,:,frame])
            
            R_0 = R_mat_hist[frame,:,:]
            all_r_coastlines_0_T = R_0 @ all_r_coastlines_T
            
            _, lat, lon = cartesian_to_spherical(all_r_coastlines_0_T[0,:], all_r_coastlines_0_T[1,:], all_r_coastlines_0_T[2,:])
            lon_coast_rotated, lat_coast_rotated = cls.get_clean_lonlat_vecs_for_plot(lon.degree, lat.degree)
            cl.set_data(lon_coast_rotated, lat_coast_rotated)
            
            return map,
        
        animation_1 = animation.FuncAnimation(fig, update, frames=np.shape(map_grid_3D_rotated)[2], 
                                               blit=True)
        animation_1.save(out_dir + filename, fps=24, writer='ffmpeg')
        