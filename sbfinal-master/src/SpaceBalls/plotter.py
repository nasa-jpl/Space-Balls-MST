



import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as colors
import numpy as np
import scipy
import os, sys
import matplotlib.animation as animation


from astropy.time import Time
from astropy.coordinates import spherical_to_cartesian, cartesian_to_spherical

from SpaceBalls.sph_meshing import get_sphere_grid
from SpaceBalls.utils import get_clean_lonlat_vecs_for_plot, get_two_perp_unit_vectors
from SpaceBalls.paths import CONFIG_DIR, MEDIA_DIR
sys.path.insert(0, str(CONFIG_DIR.parent)) 
import config.constants as constants

class Plotter:
    
    # plt.rcParams.update({
    #     "text.usetex": True,
    #     "font.family": "serif",
    #     "font.serif": ["Computer Modern Roman"],
    #     "axes.unicode_minus": False
    # })
    
    fig_width = 4  # for half the page
    fig_height = 4
    
    line_w = 1.1
    line_colors = ['black', 'blue', 'red', 'green', 'magenta', 'cyan']
    linestyles = ['-', '-.', '--', ':']
    
    font_size = 9
    font_size_red = 8

    @classmethod
    def plot_data(cls, curves_dict, scatter_dict, xlabel, ylabel, f_height=1, f_width=1, scatter_alpha=0.5):
        fig, ax = plt.subplots(figsize=(cls.fig_width*f_width, cls.fig_height*f_height), dpi=200)

        col_idx = 0
        for i, (label, (x,y)) in enumerate(curves_dict.items()):
            ax.plot(x, y, label=label, # label=cls.raw(label), 
                    color=cls.line_colors[col_idx],
                     lw=cls.line_w, linestyle=cls.linestyles[i])
            col_idx = col_idx+1
        
        for i, (label, (x,y)) in enumerate(scatter_dict.items()):
            ax.scatter(x, y, label=label, # label=cls.raw(label), 
                        color='cyan', #cls.line_colors[col_idx], 
                           marker='.',
                           s=8, alpha=scatter_alpha)
            col_idx = col_idx+1

        ax.set_ylabel(ylabel, fontsize=cls.font_size)
        ax.set_xlabel(xlabel, fontsize=cls.font_size)

        # Add a publication-quality grid (subtle and behind the data)
        ax.grid(
            visible=True,
            which='both',  # Apply to both major and minor ticks if needed
            axis='both',
            color='gray',
            linestyle='--',
            linewidth=0.5,
            alpha=0.6
        )

        # Optional: Place grid behind plot elements (e.g., lines)
        ax.set_axisbelow(True)

        fig.show()
        #ax.set_xlim(datetime_array[0], datetime_array[-1])


    @classmethod
    def plot_time_series(cls, jd_vec, y_dict, 
                         y_dict_right=None, y_scatter_dict=None, scatter_alpha=1,
                         f_height=1, f_width=1, 
                         ylabel=None, xlabel=None, 
                         yscale="linear", linthresh=1e-10, 
                         out_dir=None, file_name=None, out_fmt='.png',
                         broken_x_axis_block_length_days = None,
                         jd_vlines=None, title=None,
                         add_averages=False):
        
        max_points = 10_000
        datetime_array = cls.jd_to_datetime(jd_vec)
        
        
        if broken_x_axis_block_length_days is None:
            fig, ax = plt.subplots(figsize=(cls.fig_width*f_width, cls.fig_height*f_height), dpi=200)
        
        else:
            left_block_lims = (jd_vec[0], jd_vec[0]+broken_x_axis_block_length_days)
            right_block_lims = (jd_vec[-1]-broken_x_axis_block_length_days, jd_vec[-1])
            fig = plt.figure(figsize=(cls.fig_width*f_width, cls.fig_height*f_height), dpi=200)
            #ax = brokenaxes(xlims=(left_block_lims, right_block_lims))
        
        for i, (label, y) in enumerate(y_dict.items()):
            n = len(y)
            # if n>max_points:
            #     stride = n//max_points
            #     datetime_array_plot = datetime_array[::stride]
            #     y = y[::stride]
            # else:
            #     datetime_array_plot = datetime_array
            
            col_idx = np.remainder(i, len(cls.line_colors))
            ax.plot(datetime_array, y, label=label, # label=cls.raw(label), 
                    color=cls.line_colors[col_idx],
                     lw=cls.line_w, linestyle=cls.linestyles[i])
            if add_averages:
                    ax.axhline(np.mean(y), color=cls.line_colors[col_idx])
            
        
        j = -1
        if y_scatter_dict is not None:
            for j, (label, (jd, y)) in enumerate(y_scatter_dict.items()):
                    
                col_idx = np.remainder(i+j+1, len(cls.line_colors))
                ax.scatter(cls.jd_to_datetime(jd), y, label=label, #label=cls.raw(label), 
                           color='cyan', #cls.line_colors[col_idx], 
                           marker='.',
                           s=8, alpha=scatter_alpha, edgecolors='none', rasterized=True)
                if add_averages:
                    ax.axhline(np.mean(y), color=cls.line_colors[col_idx])
            
        locator = mdates.AutoDateLocator() # minticks=3, maxticks=7)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        
        ax.set_xlim(datetime_array[0], datetime_array[-1])
        
        if i+j >= 0:
            ax.legend(frameon=True, fontsize=cls.font_size_red, loc='best')
        else:
            assert ylabel is not None

        if jd_vlines is not None:
            for jd in jd_vlines:
                ax.axvline(cls.jd_to_datetime(jd))
            #ax.vlines(cls.jd_to_datetime(jd_vlines), -1,1) # TODO: actual plot limits
        
        #ax.set_ylabel(cls.raw(ylabel), fontsize=cls.font_size)
        ax.set_ylabel(ylabel, fontsize=cls.font_size)
        if xlabel is not None:
            #ax.xlabel(cls.raw(xlabel), fontsize=cls.font_size)
            ax.xlabel(xlabel, fontsize=cls.font_size)
        
        if yscale=="symlog":
            ax.set_yscale('symlog', linthresh=linthresh)

        elif yscale=="log":
            ax.yscale('log')
            
        ax.grid(False)
        ax.tick_params(direction='in', length=3, width=0.8, labelsize=8)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        if title is not None:
            ax.set_title(title)

        fig.tight_layout()
        fig.show()
        
        if file_name is not None:
            out_dir = out_dir if out_dir is not None else './' 
            os.makedirs(out_dir, exist_ok=True)       
            fig.savefig(os.path.join(out_dir,file_name+out_fmt), dpi=600, bbox_inches='tight')
    
        


    @classmethod
    def plot_geo_data(cls, data, lon_edges_vec, lat_edges_vec, data_label='',
                      f_height=1, f_width=1, 
                      out_dir=None, file_name=None, title=None, out_fmt='.png', draw_grid=False):
        
        fig, ax = plt.subplots(figsize=(2 * cls.fig_width*f_width, cls.fig_height*f_height), dpi=200)
        max_abs = np.max(np.abs(data))
        norm = colors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
        
        #cmap = plt.get_cmap("plasma")
        #norm = colors.BoundaryNorm(np.unique(data), ncolors=cmap.N, clip=True)

        # Use pcolormesh (preferred for gridded Earth maps)
        mesh = ax.pcolormesh(
            lon_edges_vec,
            lat_edges_vec,
            data,
            shading="auto",
            cmap="seismic", 
            norm=norm
        )
        
        cbar = fig.colorbar(mesh, ax=ax, shrink=0.8, pad=0.03)
        cbar.ax.set_ylabel(data_label)

        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Latitude [deg]")
        if title is not None:
            ax.set_title(title)

        fig.tight_layout()
        fig.show()

        if draw_grid:
            cls.draw_grid(fig, )
        
        if file_name is not None:
            out_dir = out_dir if out_dir is not None else './' 
            os.makedirs(out_dir, exist_ok=True)      
            fig.savefig(out_dir+file_name+out_fmt, dpi=600, bbox_inches='tight')



    @classmethod
    def make_map_animation(cls, map_grid_3D, jd_vec, n_lon, n_lat, 
                           data_label, out_dir, filename, fade_coastlines=False,
                           sat_h_lat_lon_hist_array=None, sat_hist_jd_array=None,
                           orbital_periods_minutes=None):
        
        sat_hist_steps_minutes = [np.mean(np.diff(sat_jd_vec)) * 24 * 60 for sat_jd_vec in sat_hist_jd_array]

        
        lon_vec, lat_vec, lon_edges_vec, lat_edges_vec = get_sphere_grid(n_lon, n_lat)
        ax, fig = cls.get_coastline_axes(fade_coastlines=fade_coastlines)
        
        max_abs = np.nanmax(np.abs(map_grid_3D))
        norm = colors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
        
        map = ax.pcolormesh(
                lon_edges_vec,
                lat_edges_vec,
                map_grid_3D[:,:,0],
                shading="auto",
                cmap="seismic",
                norm=norm
            )
        
        # --- Colorbar ---
        cbar = fig.colorbar(map, ax=ax, shrink=0.9, pad=0.03)
        cbar.ax.set_ylabel(data_label) 
        fig.tight_layout()


        #initialize satellite positions
        satellite_positions = cls.initialize_satellite_position_scatter(ax)

        # prepare satellite location history:
        all_lon_hist = [h_lat_lon_hist[:,2] for h_lat_lon_hist in sat_h_lat_lon_hist_array]
        all_lat_hist = [h_lat_lon_hist[:,1] for h_lat_lon_hist in sat_h_lat_lon_hist_array]

        all_lat_hist_map_steps, all_lon_hist_map_steps = cls.prepare_sat_groundtrack_hist(jd_vec, sat_hist_jd_array, sat_h_lat_lon_hist_array)

        n_satellites = len(sat_hist_jd_array)
        groundtrack_array = cls.initialize_groundtrack_plots(ax, n_satellites)


        def update(frame):
            
            datestr = Time(jd_vec[frame], format='jd').to_datetime().strftime("%Y-%m-%d %H:%M:%S")
            ax.set_title(datestr)
            map.set_array(map_grid_3D[:,:,frame])

            satellite_positions.set_offsets(np.column_stack((all_lon_hist_map_steps[:,frame], all_lat_hist_map_steps[:,frame])))
            #cls.update_groundtrack_plots(frame, all_lon_hist, all_lat_hist, groundtrack_array, orbital_periods_minutes, 
            #                             sat_hist_steps_minutes, lons_0to360=False)
            idxs_sat_hist_at_frame = [np.argmin(np.abs(sat_jd_vec - jd_vec[frame])) for sat_jd_vec in sat_hist_jd_array]
            cls.update_groundtrack_plots(idxs_sat_hist_at_frame, all_lon_hist, all_lat_hist, groundtrack_array, orbital_periods_minutes, 
                                         sat_hist_steps_minutes)


            return map,

        animation_1 = animation.FuncAnimation(fig, update, frames=np.shape(map_grid_3D)[2], 
                                               blit=True)
        
        animation_1.save(os.path.join(out_dir, filename), fps=16, writer='ffmpeg')
    
    @classmethod
    def prepare_sat_groundtrack_hist(cls, jd_vec, sat_hist_jd_array, sat_h_lat_lon_hist_array):
        
        assert(len(sat_h_lat_lon_hist_array)==len(sat_hist_jd_array))
        n_satellites = len(sat_hist_jd_array)
        all_lat_hist = [None]*n_satellites
        all_lon_hist = [None]*n_satellites
        
        for i in range(n_satellites):    
            all_lat_hist[i] = np.interp(jd_vec, sat_hist_jd_array[i], sat_h_lat_lon_hist_array[i][:,1])
            all_lon_hist[i] = np.interp(jd_vec, sat_hist_jd_array[i], sat_h_lat_lon_hist_array[i][:,2])
        all_lat_hist, all_lon_hist = np.array(all_lat_hist), np.array(all_lon_hist)

        return all_lat_hist, all_lon_hist

    @classmethod
    def initialize_satellite_position_scatter(cls, ax):
        scatter = ax.scatter([], [], color=(0, 1, 0), marker='o')
        
        return scatter
    
    @classmethod
    def initialize_groundtrack_plots(cls, ax, n_satellites, linestyle=':'):
        groundtrack_array = [None] * n_satellites
        for i in range(n_satellites):
            groundtrack_array[i], = ax.plot([], [], linestyle, color=(0,1,0))

        return groundtrack_array
    
    @classmethod
    def update_groundtrack_plots(cls, idx_at_frame, all_lon_hist, all_lat_hist, groundtrack_array, orbital_periods_minutes, 
                                 sat_hist_steps_minutes, lons_0to360=True):
        n_satellites = len(orbital_periods_minutes)

        for i in range(n_satellites):
            n_steps_full_hist = len(all_lon_hist[i])
            n_steps_1_groundtrack = int(orbital_periods_minutes[i]/sat_hist_steps_minutes[i])  # steps / mins/step = steps
            track_idxs = cls.get_track_idxs(idx_at_frame[i], n_steps_1_groundtrack, n_steps_full_hist)
            lon_track = all_lon_hist[i][track_idxs]
            lat_track = all_lat_hist[i][track_idxs]
            if not(lons_0to360):
                lon_track[lon_track>180] = lon_track[lon_track>180]-360
            lons, lats = get_clean_lonlat_vecs_for_plot(lon_track, lat_track)
            groundtrack_array[i].set_data(lons, lats)

    @classmethod
    def get_track_idxs(cls, j, N, T): # TODO: use get_window_idxs in utils?
        half = int(N/2 + 0.025*N)

        # tentative centered window
        start = j - half
        end   = start + N

        # slide if out of bounds
        if start < 0:
            start = 0
            end = N
        elif end > T:
            end = T
            start = T - N

        return np.arange(start, end)


    @classmethod
    def make_map_animation_arbitrary_frame(cls, map_grid_3D_rotated,
                                            jd_vec, n_lon, n_lat, 
                                            data_label, out_dir, filename, 
                                            R_mat_hist=None,
                                            add_coastlines=True, fade_coastlines=False,
                                            groundtrack_linstyle=':',
                                            add_satellite_positions=True,
                                            sat_h_lat_lon_hist_array=None, sat_hist_jd_array=None,
                                            orbital_periods_minutes=None,
                                            f_width=1, f_height=1):
        lon_vec, lat_vec, lon_edges_vec, lat_edges_vec = get_sphere_grid(n_lon, n_lat)
        if add_coastlines: assert(R_mat_hist is not None)
        if add_satellite_positions: assert((sat_h_lat_lon_hist_array is not None) and (sat_hist_jd_array is not None))

        map_time_step_minutes = np.mean(np.diff(jd_vec)) * 24 * 60
        sat_hist_steps_minutes = [np.mean(np.diff(sat_jd_vec)) * 24 * 60 for sat_jd_vec in sat_hist_jd_array]

        if add_coastlines:
            import cartopy.feature as cfeature
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
                    
            
            all_r_coastlines = spherical_to_cartesian(np.ones(n_coastline_points)*constants.earth_radius(units='km'), 
                                                        np.radians(all_lon_lat_points[:,1]), np.radians(all_lon_lat_points[:,0]))
            all_r_coastlines_T = np.array(all_r_coastlines)  # coastlines in ECEF

        n_points_equator = 360
        theta_vec = np.linspace(0, 2*np.pi, n_points_equator, endpoint=False)
        v, w = get_two_perp_unit_vectors(np.array([0,0,1]))
        all_r_equator = constants.earth_radius(units='km') * (np.outer(v, np.cos(theta_vec)) + np.outer(w, np.sin(theta_vec)))
        

        # t0:
        field_0 = map_grid_3D_rotated[:,:,0]
        R_0 = R_mat_hist[0,:,:]

        all_r_equator_rotated = R_0 @ all_r_equator
        _, lat, lon = cartesian_to_spherical(all_r_equator_rotated[0,:], all_r_equator_rotated[1,:], all_r_equator_rotated[2,:])
        lon_eq_rotated, lat_eq_rotated = get_clean_lonlat_vecs_for_plot(lon.degree, lat.degree)

        r_poles_ECEF = np.transpose(np.array([[0,0,1],[0,0,-1]])*constants.earth_radius(units='km'))
        r_poles_rotated = R_0 @ r_poles_ECEF
        _, lat, lon = cartesian_to_spherical(r_poles_rotated[0,:], r_poles_rotated[1,:], r_poles_rotated[2,:])
        lon_poles_rotated, lat_poles_rotated = get_clean_lonlat_vecs_for_plot(lon.degree, lat.degree)

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
        
        if add_coastlines:
            
            all_r_coastlines_0_T = R_0 @ all_r_coastlines_T
            _, lat, lon = cartesian_to_spherical(all_r_coastlines_0_T[0,:], all_r_coastlines_0_T[1,:], all_r_coastlines_0_T[2,:])
            lon_coast_rotated, lat_coast_rotated = get_clean_lonlat_vecs_for_plot(lon.degree, lat.degree)

            if fade_coastlines:
                cl, = ax.plot(lon_coast_rotated, lat_coast_rotated, color='k', linestyle=':', linewidth=0.5)
            else:
                cl, = ax.plot(lon_coast_rotated, lat_coast_rotated, color='k', linestyle='-', linewidth=1)

        eq, = ax.plot(lon_eq_rotated, lat_eq_rotated, color='k', linestyle='--', linewidth=0.5)
        poles = ax.scatter([], [], color='k', marker='*')

        # --- Colorbar ---
        cbar = fig.colorbar(map, ax=ax, shrink=0.88, pad=0.03)
        cbar.ax.set_ylabel(data_label)

        #initialize satellite positions
        if add_satellite_positions:
            satellite_positions = cls.initialize_satellite_position_scatter(ax)

        # prepare satellite location history:
        all_lon_hist = [h_lat_lon_hist[:,2] for h_lat_lon_hist in sat_h_lat_lon_hist_array]
        all_lat_hist = [h_lat_lon_hist[:,1] for h_lat_lon_hist in sat_h_lat_lon_hist_array]

        all_lat_hist_map_steps, all_lon_hist_map_steps = cls.prepare_sat_groundtrack_hist(jd_vec, sat_hist_jd_array, sat_h_lat_lon_hist_array)

        n_satellites = len(sat_hist_jd_array)
        groundtrack_array = cls.initialize_groundtrack_plots(ax, n_satellites, linestyle=groundtrack_linstyle)
    
        def update(frame):
            
            datestr = Time(jd_vec[frame], format='jd').to_datetime().strftime("%Y-%m-%d %H:%M:%S")
            ax.set_title(datestr)
            map.set_array(map_grid_3D_rotated[:,:,frame])
            
            R_0 = R_mat_hist[frame,:,:]

            if add_coastlines:
                all_r_coastlines_0_T = R_0 @ all_r_coastlines_T            
                _, lat, lon = cartesian_to_spherical(all_r_coastlines_0_T[0,:], all_r_coastlines_0_T[1,:], all_r_coastlines_0_T[2,:])
                lon_coast_rotated, lat_coast_rotated = get_clean_lonlat_vecs_for_plot(lon.degree, lat.degree)
                cl.set_data(lon_coast_rotated, lat_coast_rotated)

            all_r_equator_rotated = R_0 @ all_r_equator
            _, lat, lon = cartesian_to_spherical(all_r_equator_rotated[0,:], all_r_equator_rotated[1,:], all_r_equator_rotated[2,:])
            lon_eq_rotated, lat_eq_rotated = get_clean_lonlat_vecs_for_plot(lon.degree, lat.degree)
            eq.set_data(lon_eq_rotated, lat_eq_rotated)

            r_poles_rotated = R_0 @ r_poles_ECEF
            _, lat, lon = cartesian_to_spherical(r_poles_rotated[0,:], r_poles_rotated[1,:], r_poles_rotated[2,:])
            lon_poles_rotated, lat_poles_rotated = get_clean_lonlat_vecs_for_plot(lon.degree, lat.degree)
            poles.set_offsets(np.column_stack((lon_poles_rotated, lat_poles_rotated)))

            if add_satellite_positions:
                satellite_positions.set_offsets(np.column_stack((all_lon_hist_map_steps[:,frame], all_lat_hist_map_steps[:,frame])))
            
            idxs_sat_hist_at_frame = [np.argmin(np.abs(sat_jd_vec - jd_vec[frame])) for sat_jd_vec in sat_hist_jd_array]
            cls.update_groundtrack_plots(idxs_sat_hist_at_frame, all_lon_hist, all_lat_hist, groundtrack_array, orbital_periods_minutes, 
                                         sat_hist_steps_minutes)
            
            return map,
    
        animation_1 = animation.FuncAnimation(fig, update, frames=np.shape(map_grid_3D_rotated)[2], 
                                                blit=True)
        animation_1.save(os.path.join(out_dir,filename), fps=16, writer='ffmpeg')
        


    @classmethod
    def plot_histogram(cls, data_vec, xlabel, 
                       normalized=True, add_kernel=False,
                       f_height=1, f_width=1, 
                       out_dir=None, file_name=None, out_fmt='.png'):
        
        n_bins = cls.get_n_bins(len(data_vec))
        
        plt.figure(figsize=(cls.fig_width*f_width, cls.fig_height*f_height))
        hist_out = plt.hist(data_vec, bins=n_bins, 
                 density=normalized, alpha=0.25, color='C0', 
                 edgecolor='black', linewidth=0.4, 
                 label=r'Histogram')
        
        ylabel=r'Probability density' if normalized else r'Count'
        plt.ylabel(ylabel, fontsize=cls.font_size)
        plt.xlabel(xlabel, fontsize=cls.font_size)
        #plt.ylim(bottom=0)
        
        plt.legend(frameon=False, fontsize=cls.font_size)
        plt.grid(False)
        plt.tick_params(direction='in', length=3, width=0.8, labelsize=cls.font_size_red)
        
        if add_kernel:
            #hist_out = plt.hist(data_vec, bins=n_bins, density=True)
            edges = hist_out[1]
            pdf = hist_out[0]
            bin_centers = (edges[1:] + edges[:-1]) / 2

            x_vec_plot = np.linspace(edges[0], edges[-1], 400)
            y_interp = scipy.interpolate.Akima1DInterpolator(bin_centers, pdf) #, extrapolate=True)
            y_vec = y_interp(x_vec_plot)
            
            plt.plot(x_vec_plot, y_vec, color='black', lw=1.1)
            
            if file_name is None:
                return y_interp
        
        for spine in plt.gca().spines.values():
            spine.set_linewidth(0.8)
        plt.tight_layout()
        
        if file_name is not None:
            out_dir = out_dir if out_dir is not None else './'        
            plt.savefig(out_dir+file_name+out_fmt, dpi=600, bbox_inches='tight')
    

# full ChatGPT:

    @classmethod
    def plot_linear_regression(cls, x, y, xlabel, ylabel,
                            add_confidence=True,
                            f_height=1, f_width=1,
                            out_dir=None, file_name=None, out_fmt='.png'):
        
        # Fit linear regression
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
        
        # Regression line
        x_fit = np.linspace(np.min(x), np.max(x), 200)
        y_fit = intercept + slope * x_fit
        
        plt.figure(figsize=(cls.fig_width*f_width, cls.fig_height*f_height))
        
        # Scatter
        plt.scatter(x, y, s=15, alpha=0.25, color='C0')
        
        # Fit line
        plt.plot(x_fit, y_fit, color='k', linewidth=1.5,
                label=rf'Fit: $y = {slope:.3g}x + {intercept:.3g}$')
        
        # Optional confidence band (95%)
        if add_confidence:
            n = len(x)
            t_val = scipy.stats.t.ppf(0.975, n-2)
            y_pred = intercept + slope*x
            s_err = np.sqrt(np.sum((y - y_pred)**2) / (n-2))
            
            x_mean = np.mean(x)
            conf = t_val * s_err * np.sqrt(
                1/n + (x_fit - x_mean)**2 / np.sum((x - x_mean)**2)
            )
            
            plt.fill_between(x_fit, y_fit-conf, y_fit+conf,
                            color='C1', alpha=0.2,
                            label='95% CI')
            
        # R^2 and p-value textbox
        textstr = (
            rf"$R^2 = {r_value**2:.3f}$" "\n"
            rf"$\hat\lambda = {slope:.2e}$/day"
        )

        plt.text(0.05, 0.85, textstr,
                transform=plt.gca().transAxes,
                fontsize=cls.font_size_red,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, linewidth=0.4))

        
        plt.xlabel(xlabel, fontsize=cls.font_size)
        plt.ylabel(ylabel, fontsize=cls.font_size)
        
        plt.legend(frameon=False, fontsize=cls.font_size)
        plt.grid(False)
        plt.tick_params(direction='in', length=3, width=0.8,
                        labelsize=cls.font_size_red)
        plt.show()
        
        # Save if requested
        if out_dir is not None and file_name is not None:
            out_path = os.path.join(out_dir, file_name + out_fmt)
            plt.savefig(out_path, bbox_inches='tight', dpi=600)
        
        return slope, intercept, r_value, p_value, std_err


    @classmethod
    def get_coastline_axes(cls, f_width=1, f_height=1, fade_coastlines=False):
        import cartopy.crs as ccrs

        _, ax_coast = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()},  # PlateCarree
                                   figsize=(2 * cls.fig_width * f_width, cls.fig_height * f_height),
                                    dpi=200)
        #
        if fade_coastlines:
            import cartopy.feature as cfeature
            coast = cfeature.NaturalEarthFeature(
                'physical', 'coastline', '110m',    # or 50m / 10m
                edgecolor='k', facecolor='none'
            )

            # Add with custom linestyle
            ax_coast.add_feature(coast, linestyle=':', linewidth=0.5)
            
        else:
            ax_coast.coastlines()

        gl = ax_coast.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        fig = ax_coast.get_figure()

        return ax_coast, fig


    @staticmethod
    def jd_to_datetime(jd_array):
        t = Time(jd_array, format='jd')
        return t.to_datetime()

    @staticmethod
    def get_n_bins(n_points):
        
        # Sturge's rule:
        k = np.ceil(np.log2(n_points)+1)
        return int(np.min([k, 20]))
    