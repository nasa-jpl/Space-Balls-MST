


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as colors
import numpy as np
import scipy
import os, sys
import matplotlib.animation as animation
from matplotlib import ticker


from astropy.time import Time
from datetime import datetime, timedelta
from astropy.coordinates import spherical_to_cartesian, cartesian_to_spherical
from scipy.stats import norm

from SpaceBalls.sph_meshing import get_sphere_grid, sphere_field_interp, Grid, RegularLatLonGrid
from SpaceBalls.utils import get_clean_lonlat_vecs_for_plot, get_two_perp_unit_vectors
from SpaceBalls.paths import CONFIG_DIR, MEDIA_DIR
sys.path.insert(0, str(CONFIG_DIR.parent)) 
import config.constants as constants

matplotlib.rcParams.update({

    # Font
    "font.size": 10,
    "font.family": "STIXGeneral",  # A clean, professional font
    "mathtext.fontset": "stix",

    # Axes
    "axes.linewidth": 0.8,
    "axes.labelsize": 10,
    "axes.titlesize": 10,

    # Lines
    "lines.linewidth": 1.5,
    "lines.markersize": 4,

    # Ticks
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.direction": "in",
    "ytick.direction": "in",

    # Legend
    "legend.fontsize": 9,
    #"legend.frameon": False,
    "legend.fontsize": 9,
    "legend.frameon": True,
    "legend.framealpha": 0.7,    # semi-transparent white box
    "legend.facecolor": "white",
    "legend.edgecolor": "none",   # no border
    "legend.fancybox": False, 

    # Grid (usually off in publications)
    "axes.grid": False,

    # Figure
    "figure.dpi": 150
})

plt.rcParams['axes.grid'] = False          # Off by default
plt.rcParams['axes.axisbelow'] = True      # Grid sits behind data plots
plt.rcParams['grid.color'] = '#e0e0e0'     # Major grid color
plt.rcParams['grid.linewidth'] = 0.4       # Major grid weight
plt.rcParams['grid.linestyle'] = '-'       # Major grid style

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
    line_colors = ['black', 'blue', 'red', 'green', 'magenta', 'cyan', 'yellow']
    linestyles = ['-', '--'] * 20 #['-', '-.', '--', ':']
    
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
                         scatter_marker='.', scatter_size=8, scatter_color='cyan',
                         f_height=1, f_width=1, 
                         ylabel=None, xlabel=None, 
                         yscale="linear", linthresh=1e-10, 
                         out_dir=None, file_name=None, out_fmt='.png',
                         broken_x_axis_block_length_days = None,
                         jd_vlines=None, title=None,
                         add_averages=False,
                         ylim=None, zero_hline=False, alpha_array=None, lw_array=None,
                         col_array=None, note_top_left=None):
        
        max_points = 10_000
        datetime_array = cls.jd_to_datetime(jd_vec)
        if lw_array is None:
            lw_array = np.ones(len(y_dict.keys())) * cls.line_w

        if alpha_array is None:
            alpha_array = np.ones(len(y_dict.keys()))
        
        if col_array is None:
            col_array = cls.line_colors
        
        
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
                    color=col_array[col_idx],
                     lw=lw_array[i], linestyle=cls.linestyles[i],
                     alpha=alpha_array[i])
            if add_averages:
                    ax.axhline(np.mean(y), color=cls.line_colors[col_idx])
            
        
        j = -1
        if y_scatter_dict is not None:
            for j, (label, (jd, y)) in enumerate(y_scatter_dict.items()):
                    
                col_idx = np.remainder(i+j+1, len(cls.line_colors))
                ax.scatter(cls.jd_to_datetime(jd), y, label=label, #label=cls.raw(label), 
                           color=scatter_color, #cls.line_colors[col_idx], 
                           marker=scatter_marker,
                           s=scatter_size, alpha=scatter_alpha, edgecolors='none', rasterized=True)
                if add_averages:
                    ax.axhline(np.mean(y), color=cls.line_colors[col_idx])
            
        locator = mdates.AutoDateLocator() # minticks=3, maxticks=7)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        
        ax.set_xlim(datetime_array[0], datetime_array[-1])
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])
        
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

        if zero_hline:
            ax.axhline(0, color='red')

        if note_top_left is not None:
            cls.add_text_note_top_left(ax, note_top_left)

        fig.tight_layout()
        fig.show()
        
        if file_name is not None:
            out_dir = out_dir if out_dir is not None else './' 
            os.makedirs(out_dir, exist_ok=True)       
            fig.savefig(os.path.join(out_dir,file_name+out_fmt), dpi=600, bbox_inches='tight')

        return fig, ax


    @classmethod
    def plot_animated_time_series(cls, jd_vec, y_dict, 
                                   animated_keys=None,
                                   y_scatter_dict=None, scatter_alpha=1,
                                   scatter_marker='.', scatter_size=8, scatter_color='cyan',
                                   f_height=1, f_width=1, 
                                   ylabel=None, xlabel=None, 
                                   out_dir=None, file_name=None, out_fmt='.mp4',
                                   title=None, ylim=None, zero_hline=False, 
                                   alpha_array=None, lw_array=None, col_array=None,
                                   fps=16, note_top_left=None,
                                   add_moving_vline=True,
                                   moving_vline_jd_array=None,
                                   add_avg_window=False,
                                   window_days=None):

        """
        Create an animated time series plot with curves drawn progressively.
        
        Args:
            jd_vec: X-axis data (Julian Date array)
            y_dict: Dictionary with {label: y_data_array}
            animated_keys: List of keys to animate. If None, animates all keys
            y_scatter_dict: Optional dict with {label: (jd_scatter, y_scatter)}
            scatter_alpha, scatter_marker, scatter_size, scatter_color: Scatter styling
            f_height, f_width: Figure dimensions multipliers
            ylabel, xlabel, title: Axis labels
            out_dir, file_name, out_fmt: Output file path (default .mp4)
            ylim: Y-axis limits [min, max]
            zero_hline: Add horizontal line at y=0
            alpha_array, lw_array, col_array: Per-curve styling arrays
            fps: Animation frames per second
            note_top_left: Text annotation for top-left corner
        """
        from matplotlib.animation import FuncAnimation
        
        if animated_keys is None:
            animated_keys = list(y_dict.keys())
        
        datetime_array = cls.jd_to_datetime(jd_vec)
        if add_moving_vline:
            if (moving_vline_jd_array is None):
                moving_vline_datetimes = datetime_array
            else:
                moving_vline_datetimes = cls.jd_to_datetime(moving_vline_jd_array)
        n_frames = len(jd_vec)
        
        if lw_array is None:
            lw_array = np.ones(len(y_dict.keys())) * cls.line_w
        
        if alpha_array is None:
            alpha_array = np.ones(len(y_dict.keys()))
        
        if col_array is None:
            col_array = cls.line_colors
        
        # Calculate y-limits if not provided
        if ylim is None:
            all_data = [y_dict[key] for key in animated_keys]
            ylim = [np.nanmin(np.concatenate(all_data)) - 0.1,
                    np.nanmax(np.concatenate(all_data)) + 0.1]
        
        fig, ax = plt.subplots(figsize=(cls.fig_width*f_width, cls.fig_height*f_height), dpi=200)
        
        # Initialize line objects for animated keys
        lines = {}
        for i, (label, y) in enumerate(y_dict.items()):
            col_idx = np.remainder(i, len(cls.line_colors))
            
            if label in animated_keys:
                line, = ax.plot([], [], label=label, 
                               color=col_array[col_idx],
                               lw=lw_array[i], 
                               linestyle=cls.linestyles[i],
                               alpha=alpha_array[i])
                lines[label] = line
            else:
                # Plot non-animated lines as static
                ax.plot(datetime_array, y, label=label,
                       color=col_array[col_idx],
                       lw=lw_array[i],
                       linestyle=cls.linestyles[i],
                       alpha=alpha_array[i])
        
        # Add scatter plot if provided
        scatter_obj = None
        if y_scatter_dict is not None:
            for j, (label, (jd, y)) in enumerate(y_scatter_dict.items()):
                col_idx = np.remainder(len(y_dict) + j + 1, len(cls.line_colors))
                scatter_obj = ax.scatter(cls.jd_to_datetime(jd), y, label=label,
                                        color=scatter_color,
                                        marker=scatter_marker,
                                        s=scatter_size, 
                                        alpha=scatter_alpha, 
                                        edgecolors='none', 
                                        rasterized=True)
        
        # Add zero line if requested
        if zero_hline:
            ax.axhline(0, color='red', linestyle='--', alpha=0.5)
        
        # Set axis limits and labels
        datetime_array_no_nan = datetime_array[~np.isnan(list(y_dict.values())[0])]
        datetime_span = datetime_array_no_nan[-1] - datetime_array_no_nan[0]
        ax.set_xlim(datetime_array_no_nan[0], datetime_array_no_nan[-1])
        ax.set_ylim(ylim[0], ylim[1])

        if add_moving_vline:
            vline = ax.axvline(moving_vline_datetimes[0], color='red', linestyle='-', lw=0.5)
        
        if add_avg_window:
            shade_right = ax.axvspan(moving_vline_datetimes[0], moving_vline_datetimes[-1], color='gray', alpha=0.3)
            shade_left = ax.axvspan(moving_vline_datetimes[0], moving_vline_datetimes[0], color='gray', alpha=0.3)
        

        
        if xlabel is None:
            xlabel = 'Time'
        ax.set_xlabel(xlabel, fontsize=cls.font_size)
        
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=cls.font_size)
        
        if title is not None:
            ax.set_title(title)
        
        ax.legend(frameon=True, fontsize=cls.font_size_red, loc='best')
        ax.grid(False)
        ax.tick_params(direction='in', length=3, width=0.8, labelsize=8)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        
        if note_top_left is not None:
            cls.add_text_note_top_left(ax, note_top_left)
        
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        
        def update(frame):
            print(f"Currently printing frame {frame}")
            current_datetime = datetime_array[:frame+1]
            for label in animated_keys:
                y_data = y_dict[label][:frame+1]
                lines[label].set_data(current_datetime, y_data)
            
            vline.set_xdata([moving_vline_datetimes[frame], moving_vline_datetimes[frame]])
            
            if add_avg_window:
                xmin, xmax = moving_vline_datetimes[frame], moving_vline_datetimes[-1]
                shade_right.set_x(mdates.date2num(xmin))
                shade_left_right_bound = xmin - timedelta(days=window_days)
                shade_left_width = mdates.date2num(shade_left_right_bound) - mdates.date2num(moving_vline_datetimes[0])
                shade_left.set_width(shade_left_width)
                #shade_left.set_bounds(mdates.date2num(moving_vline_datetimes)[0], 0, #left, bot
                #                      mdates.date2num(xmin-timedelta(days=window_days)), 1) # width, height
            #shade_right.set_xy((xmin_frac, 0))
        
            return list(lines.values()) + [vline] + [shade_right] + [shade_left]
        
        anim = FuncAnimation(fig, update, frames=n_frames, 
                            blit=True)
        
        fig.tight_layout()
        
        if file_name is not None:
            out_dir = out_dir if out_dir is not None else './'
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, file_name + out_fmt)
            anim.save(out_path, writer='ffmpeg', fps=fps)
            print(f"Animation saved to {out_path}")
        else:
            fig.show()
        
        return fig, anim



    @classmethod
    def plot_convergence(cls, x_array, y_array, xlabel, ylabel, f_height=1, f_width=1, out_dir=None, file_name=None, 
                         out_fmt='.png', cols=None, title=None, title_loc='top'):
        
        fig, ax = plt.subplots(figsize=(cls.fig_width*f_width, cls.fig_height*f_height), dpi=200)
        
        if isinstance(y_array, dict):
            for i, (label, y) in enumerate(y_array.items()):
                #col_idx = np.remainder(i, len(cls.line_colors))
                if cols is not None: col = cols[i]
                ax.plot(x_array[i], y, label=label, # label=cls.raw(label), 
                        color=col,#color=cls.line_colors[col_idx],
                         lw=cls.line_w, marker='x')
            ax.legend(frameon=True, fontsize=cls.font_size_red, loc='best')
        else:
            ax.plot(x_array, y_array, marker='x', lw=1.5) # 
        ax.set_xlabel(xlabel, fontsize=cls.font_size)
        ax.set_ylabel(ylabel, fontsize=cls.font_size)
        #ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

        if title is not None:
            if title_loc=="top":
                ax.set_title(title)
            elif title_loc=="inside":
                ax.text(.37,.9,title,
                    horizontalalignment='center',
                    transform=ax.transAxes)
        
        #fig.tight_layout()
        #fig.show()
        
        if file_name is not None:
            out_dir = out_dir if out_dir is not None else './' 
            os.makedirs(out_dir, exist_ok=True)       
            fig.savefig(os.path.join(out_dir,file_name+out_fmt), dpi=600, bbox_inches='tight')

        return fig, ax


    @classmethod
    def plot_geo_data(cls, data, lon_edges_vec, lat_edges_vec, data_label='',
                      f_height=1, f_width=1, 
                      out_dir=None, file_name=None, title=None, out_fmt='.png', draw_grid=False, max_mag=None,
                      make_symmetric_cmap=True):
        
        fig, ax = plt.subplots(figsize=(2 * cls.fig_width*f_width, cls.fig_height*f_height), dpi=200)
        max_abs = np.nanmax(np.abs(data)) if max_mag is None else max_mag
        if make_symmetric_cmap:
            norm = colors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
            cmap = "seismic"
        else:
            norm = None
            cmap = "plasma"

        #cmap = plt.get_cmap("plasma")
        #norm = colors.BoundaryNorm(np.unique(data), ncolors=cmap.N, clip=True)

        # Use pcolormesh (preferred for gridded Earth maps)
        mesh = ax.pcolormesh(
            lon_edges_vec,
            lat_edges_vec,
            data,
            shading="auto",
            cmap=cmap,
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
    def plot_geo_data_new(cls, data, grid: Grid, data_label='',
                      f_height=1, f_width=1, 
                      out_dir=None, file_name=None, title=None, out_fmt='.png', draw_grid=False, max_mag=None,
                      make_symmetric_cmap=True, add_coastlines=True, fade_coastlines=False):
        
        fig, ax = plt.subplots(figsize=(2 * cls.fig_width*f_width, cls.fig_height*f_height), dpi=200)
        max_abs = np.nanmax(np.abs(data)) if max_mag is None else max_mag
        if make_symmetric_cmap:
            norm = colors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
            cmap = "seismic"
        else:
            norm = None
            cmap = "plasma"

        #cmap = plt.get_cmap("plasma")
        #norm = colors.BoundaryNorm(np.unique(data), ncolors=cmap.N, clip=True)

        if "regular" in grid.grid_type_name:  #grid.grid_type_name == "regular_latlon":
            lon_edges_vec = grid.lon_edges_vec
            lat_edges_vec = grid.lat_edges_vec
            if len(np.shape(data))==2:
                data_mat = data
            elif len(np.shape(data))==1:
                data_mat = np.reshape(data, (grid.n_lat, grid.n_lon))
        
        elif grid.grid_type_name == "quadrature":
            regular_grid = RegularLatLonGrid(alt_km=grid.alt_km,
                                             n_lat=180*3, n_lon=360*3)
            data[np.isnan(data)] = 0
            interp_method="griddata_nearest"#"griddata_linear"#"griddata_linear" if grid.n_points>6000 else "griddata_cubic"
            data_vec = sphere_field_interp(grid.stacked_grid_latlon, data, 
                                           regular_grid.stacked_grid_latlon,
                                           method=interp_method)
            data_mat = regular_grid.reshape_if_needed(data_vec)
            lon_edges_vec = regular_grid.lon_edges_vec
            lat_edges_vec = regular_grid.lat_edges_vec
            
            # we're here: interpolate 

        # Use pcolormesh (preferred for gridded Earth maps)
        mesh = ax.pcolormesh(
            lon_edges_vec,
            lat_edges_vec,
            data_mat,
            shading="auto",
            cmap=cmap,
            norm=norm
        )
        
        cbar = fig.colorbar(mesh, ax=ax, shrink=0.8, pad=0.03)
        cbar.ax.set_ylabel(data_label)
        
        if add_coastlines:
            coastlines_lon_lat = cls.get_coastline_lonlat_points()
            lw = 0.5 if fade_coastlines else 1
            lons = coastlines_lon_lat[:,0]
            lons[lons<0] = lons[lons<0] + 360 # TODO: make consistent with input grid
            lats = coastlines_lon_lat[:,1]
            lons, lats = get_clean_lonlat_vecs_for_plot(lons, lats)
            ax.plot(lons, lats, color='k', linestyle=':', linewidth=lw)

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
    def make_map_animation_arbitrary_frame(cls, map_hist,
                                            jd_vec, grid: Grid, 
                                            data_label, out_dir, filename, 
                                            R_mat_hist=None,
                                            add_coastlines=True, fade_coastlines=False,
                                            groundtrack_linstyle=':',
                                            add_satellite_positions=True,
                                            add_groundtracks=True,
                                            sat_h_lat_lon_hist_array=None, sat_hist_jd_array=None,
                                            orbital_periods_minutes=None,
                                            f_width=1, f_height=1, max_abs_colorscale=None,
                                            title_header='',
                                            adaptative_colorbar=False):
        
        if add_coastlines: assert(R_mat_hist is not None)
        if add_satellite_positions: assert((sat_h_lat_lon_hist_array is not None) and (sat_hist_jd_array is not None))

        map_time_step_minutes = np.mean(np.diff(jd_vec)) * 24 * 60
        sat_hist_steps_minutes = [np.mean(np.diff(sat_jd_vec)) * 24 * 60 for sat_jd_vec in sat_hist_jd_array]

        if add_coastlines:
            """
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
            """
            all_lon_lat_points = cls.get_coastline_lonlat_points()
            n_coastline_points = np.shape(all_lon_lat_points)[0]
                    
            
            all_r_coastlines = spherical_to_cartesian(np.ones(n_coastline_points)*constants.earth_radius(units='km'), 
                                                        np.radians(all_lon_lat_points[:,1]), np.radians(all_lon_lat_points[:,0]))
            all_r_coastlines_T = np.array(all_r_coastlines)  # coastlines in ECEF

        n_points_equator = 360
        theta_vec = np.linspace(0, 2*np.pi, n_points_equator, endpoint=False)
        v, w = get_two_perp_unit_vectors(np.array([0,0,1]))
        all_r_equator = constants.earth_radius(units='km') * (np.outer(v, np.cos(theta_vec)) + np.outer(w, np.sin(theta_vec)))
        

        # t0:
        field_0 = map_hist[:,0]
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
        
        max_abs = np.nanmax(np.abs(map_hist)) if max_abs_colorscale is None else max_abs_colorscale
        norm = colors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)

        plotting_grid = RegularLatLonGrid(0, n_lat=180*3, n_lon=360*3)
        field_0 = grid.map_field_to_different_grid(field_0, plotting_grid)
        field_0 = plotting_grid.reshape_if_needed(field_0)
        
        map = ax.pcolormesh(
                plotting_grid.lon_edges_vec,
                plotting_grid.lat_edges_vec,
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

        if add_satellite_positions:
            all_lat_hist_map_steps, all_lon_hist_map_steps = cls.prepare_sat_groundtrack_hist(jd_vec, sat_hist_jd_array, sat_h_lat_lon_hist_array)

        if add_groundtracks:
            n_satellites = len(sat_hist_jd_array)
            groundtrack_array = cls.initialize_groundtrack_plots(ax, n_satellites, linestyle=groundtrack_linstyle)
    
        def update(frame):
            
            datestr = Time(jd_vec[frame], format='jd').to_datetime().strftime("%Y-%m-%d %H:%M:%S")
            ax.set_title(title_header + '  ' + datestr)
            field = grid.map_field_to_different_grid(map_hist[:,frame], plotting_grid)
            field = grid.reshape_if_needed(field)
            map.set_array(field)
            if adaptative_colorbar:
                mag_max = np.max(np.abs(field))
                new_norm = colors.TwoSlopeNorm(vmin=-mag_max, vcenter=0, vmax=mag_max)
                map.set_norm(new_norm)
            
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
            
            if add_groundtracks:
                idxs_sat_hist_at_frame = [np.argmin(np.abs(sat_jd_vec - jd_vec[frame])) for sat_jd_vec in sat_hist_jd_array]
                cls.update_groundtrack_plots(idxs_sat_hist_at_frame, all_lon_hist, all_lat_hist, groundtrack_array, orbital_periods_minutes, 
                                            sat_hist_steps_minutes)
            
            return map,
    
        animation_1 = animation.FuncAnimation(fig, update, frames=np.shape(map_hist)[1], 
                                                blit=True)
        animation_1.save(os.path.join(out_dir,filename), fps=16, writer='ffmpeg')
        



    @classmethod
    def make_map_animation_arbitrary_frame_old(cls, map_grid_3D_rotated,
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
            """
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
            """
            all_lon_lat_points = cls.get_coastline_lonlat_points()()
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

        if add_satellite_positions:
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
                       out_dir=None, file_name=None, out_fmt='.png', fit_normal=False,
                       units=''):
        
        n_bins = cls.get_n_bins(len(data_vec))
        
        plt.figure(figsize=(cls.fig_width*f_width, cls.fig_height*f_height))
        hist_out = plt.hist(data_vec, bins=n_bins, 
                 density=normalized, alpha=0.25, color='C0', 
                 edgecolor='black', linewidth=0.4)
        
        ylabel=r'Probability density' if normalized else r'Count'
        plt.ylabel(ylabel, fontsize=cls.font_size)
        plt.xlabel(xlabel + ' (' + units + ')', fontsize=cls.font_size)
        #plt.ylim(bottom=0)
        
        plt.legend(frameon=False, fontsize=cls.font_size)
        plt.grid(False)
        plt.tick_params(direction='in', length=3, width=0.8, labelsize=cls.font_size_red)
        
        if fit_normal:
            assert(normalized)
            f, p = scipy.stats.normaltest(data_vec)
            mu, std = norm.fit(data_vec)
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            plt.plot(x, norm.pdf(x, mu, std), 'r', linewidth=2)
            plt.text(0.05, 0.95, "$\hat{\mu}="+f"{mu:.2f}\,$"+units+'\n'+
                                  "$\hat{\sigma}="+f"{std:.2f}\,$"+units+'\n'+
                                  "$p="+f"{p:.4e}$", 
                    transform=plt.gca().transAxes, 
                    verticalalignment='top',
                    fontsize=cls.font_size_red
                    )
            plt.axvline(mu, color='r', lw=0.8)
            plt.axvline(mu+std, color='r', lw=0.5, linestyle='--')
            plt.axvline(mu-std, color='r', lw=0.5, linestyle='--')
            

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
                            add_confidence=False,
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
            #rf"$\hat\lambda = {slope:.2e}$/day"
        )

        plt.text(0.05, 0.9, textstr,
                transform=plt.gca().transAxes,
                fontsize=cls.font_size_red,
                verticalalignment='top',
                horizontalalignment='left',
                #bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, linewidth=0.4)
                )

        
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
    def format_time_labels(ax):
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    @classmethod
    def add_text_note_top_left(cls, ax, text):
        ax.text(
            0.02, 0.95,
            text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.6
            )
        )

    @staticmethod
    def get_coastline_lonlat_points():
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

        return all_lon_lat_points


    @staticmethod
    def jd_to_datetime(jd_array):
        t = Time(jd_array, format='jd')
        return t.to_datetime()

    @staticmethod
    def get_n_bins(n_points):
        
        # Sturge's rule:
        k = np.ceil(np.log2(n_points)+1)
        return int(np.min([k, 20]))
    
    @staticmethod
    def adjust_label_to_avoid_ticklabels(ax, padding_pixels=5):
        """
        Horizontally nudges the axis label left or right if it overlaps
        with any of its visible tick labels.
        """
        fig = ax.figure
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        
        axis_label = ax.xaxis.get_label()
        if not axis_label.get_visible() or not axis_label.get_text():
            return
            
        label_bbox = axis_label.get_window_extent(renderer)
        # Calculate the horizontal center of the label
        label_center_x = (label_bbox.xmin + label_bbox.xmax) / 2
        
        max_overlap = 0
        shift_direction = 0  # +1 for right, -1 for left
        
        # Check against all visible tick labels to find the worst conflict
        for ticklabel in ax.xaxis.get_ticklabels():
            if ticklabel.get_visible() and ticklabel.get_text():
                tick_bbox = ticklabel.get_window_extent(renderer)
                
                # Check for standard bounding box intersection
                if (label_bbox.xmin - padding_pixels < tick_bbox.xmax and
                    label_bbox.xmax + padding_pixels > tick_bbox.xmin and
                    label_bbox.ymin - padding_pixels < tick_bbox.ymax and
                    label_bbox.ymax + padding_pixels > tick_bbox.ymin):
                    
                    # Calculate the horizontal center of the tick label
                    tick_center_x = (tick_bbox.xmin + tick_bbox.xmax) / 2
                    
                    # If the tick label is centered to the left of the text, push text right
                    if tick_center_x < label_center_x:
                        overlap = (tick_bbox.xmax + padding_pixels) - label_bbox.xmin
                        if overlap > max_overlap:
                            max_overlap = overlap
                            shift_direction = 1
                    # If the tick label is centered to the right of the text, push text left
                    else:
                        overlap = (label_bbox.xmax + padding_pixels) - tick_bbox.xmin
                        if overlap > max_overlap:
                            max_overlap = overlap
                            shift_direction = -1
                            
        # If a collision is found, calculate the exact shift in axes coordinates
        if max_overlap > 0:
            # Convert pixel shift distance to axes fraction (0.0 to 1.0)
            inv_trans = ax.transAxes.inverted()
            p0 = inv_trans.transform((0, 0))
            p1 = inv_trans.transform((max_overlap, 0))
            delta_x_axes = p1[0] - p0[0]
            
            # Safely grab current X from the position tuple
            current_x, _ = axis_label.get_position()
            new_x = current_x + (shift_direction * delta_x_axes)
            axis_label.set_x(new_x)
            
            # Force a redraw to apply changes safely
            fig.canvas.draw()
                                    
    @staticmethod
    def add_grid(ax, show_minor=False):
        """
        Activates a publication-quality grid on a specific axes object.
        Major grid pulls styling from rcParams; minor grid uses explicit parameters.
        """
        # 1. Activate major grid (inherits valid rcParams)
        ax.grid(True, which='major', zorder=1)
        
        # 2. Handle minor grid
        if show_minor:
            # CRITICAL FIX: Instantiate separate locator objects for X and Y axes
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            
            # Force the axes engine to display minor ticks
            ax.minorticks_on()
            
            # Explicitly style and activate minor elements
            ax.grid(True, which='minor', color="#f5f5f5a5", linestyle=':', linewidth=0.4, zorder=1)
        else:
            ax.grid(False, which='minor')