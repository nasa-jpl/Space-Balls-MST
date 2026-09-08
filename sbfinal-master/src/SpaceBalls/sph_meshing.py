
from __future__ import annotations

#import pyshtools as pysh
import sys, os, importlib
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.interpolate import griddata, Rbf, RBFInterpolator, RectSphereBivariateSpline, SmoothSphereBivariateSpline, LSQSphereBivariateSpline

from astropy.coordinates import get_body, ITRS, SkyCoord, CartesianRepresentation
from astropy.coordinates import spherical_to_cartesian, cartesian_to_spherical
from multiprocessing import Pool
from abc import ABC, abstractmethod
#import stripy.spherical

from SpaceBalls.paths import CONFIG_DIR, MEDIA_DIR
from SpaceBalls.utils import get_two_perp_unit_vectors, progress_bar, get_pairwise_midpoints, get_norm_across_last_dim
#from SpaceBalls.plotter import Plotter
sys.path.insert(0, str(CONFIG_DIR.parent)) 
import config.constants as constants
RE = constants.earth_radius()

class Grid(ABC):
    """
    Abstract base class for grid implementations.
    Provides a common interface for accessing grid attributes like integration_weights.
    """
    def __init__(self, alt_km, flattening=0):
        self.alt_km = alt_km
        self.flattening = flattening
        self.total_area = self.compute_total_area()
        # Common attributes initialized by subclasses
        self.grid_name = None
        self.grid_type_name = None
        self.integration_weights = None
        self.stacked_grid_r = None
        self.stacked_grid_u = None
        self.stacked_grid_latlon = None
        self.n_points = None
        self.r_projected_to_ellipsoid = False
        # Additional common attributes can be added here if needed

    def compute_total_area(self):
        if self.flattening==0:
            return 4 * np.pi * (RE + self.alt_km)**2

        else:
            a = (RE + self.alt_km)
            f = self.flattening
            e = np.sqrt(f * (2 - f))

            return 2 * np.pi * a**2 * (1 + (1/e - e) * np.arctanh(e))

            # TODO: verify (formula given by Gemini) - wrong! a should have been squared
            #c = a * (1 - self.flattening)
            #e = np.sqrt(1 - (c**2 / a**2))
            #return 2 * np.pi * a**2 * (1 + ((c**2 / (a * e)) * np.arctanh(e)))

    def project_r_vectors_to_elliposoid(self):
        f = self.flattening
        lats = np.deg2rad(self.stacked_grid_latlon[:,0])
        R_corrections = get_r_correction_factors_biaxial_ellipsoid(f, lats)
        self.stacked_grid_r = self.stacked_grid_r * R_corrections[:,None]
        self.r_projected_to_ellipsoid = True

        print(f"r vectors projected to ellipsoid")

    def adjust_u_vectors_to_ellipsoid(self):
        assert(self.r_projected_to_ellipsoid)
        f = self.flattening
        denominator_adjustment = np.array([1, 1, (1-f)**2])
        new_u = self.stacked_grid_r / denominator_adjustment[None,:]
        self.stacked_grid_u = new_u / get_norm_across_last_dim(new_u)[...,None]
        

    def adjust_integration_weights_to_ellipsoid(self):
        f = self.flattening
        lats = np.deg2rad(self.stacked_grid_latlon[:,0])
        R_corrections = get_r_correction_factors_biaxial_ellipsoid(f, lats)

        e2 = f * (2 - f) # e (eccentricity) squared
        sqrt_corrections = np.sqrt(1 + (e2 * np.sin(lats) * np.cos(lats) / (e2 * (np.cos(lats))**2 - 1))**2)

        dS_corrections = R_corrections**2 * sqrt_corrections
        self.integration_weights = self.integration_weights * dS_corrections
                


    def compute_surf_integral(self, field_array, average=False):
        field_array = self.vectorize_if_needed(field_array)
        if len(np.shape(field_array))==2:   # axis 1 is time
            #integrated_result = np.sum( self.integration_weights[:, None] * field_array, axis=0)
            integrated_result = np.einsum('i,ij->j', self.integration_weights, field_array) # slightly more efficient
        elif len(np.shape(field_array))==1:
            integrated_result = np.sum( self.integration_weights * field_array)
        # TODO: try chatGPT's version: integrated_result = np.tensordot(self.integration_weights, field_array, axes=(0, 0))
        
        if average:
            integrated_result = integrated_result / self.total_area
        
        return integrated_result
    
    def compute_time_avg_map(self, field_array):
        # last dim is always the time dim
        #return np.mean(field_array, axis=len(np.shape(field_array))-1)
        return np.nanmean(field_array, axis=len(np.shape(field_array))-1)
    
    def map_field_to_different_grid(self, field, new_grid: Grid, method='griddata_nearest'):
        field_new_grid = sphere_field_interp(self.stacked_grid_latlon, field, new_grid.stacked_grid_latlon,
                                             method=method)
        return field_new_grid
        

    @abstractmethod
    def initialize_grid(self):
        """Abstract method to initialize grid-specific attributes."""
        pass
    
    @abstractmethod
    def reshape_if_needed(self, field_array):
        pass
    
    @abstractmethod
    def vectorize_if_needed(self, field_array):
        pass

    @abstractmethod
    def compute_lat_avg_map(self, field_aray):
        pass

    @abstractmethod
    def recompute_grid(self, r_sat):
        pass
        

class RegularLatLonGrid(Grid):
    """
    Subclass for regular latitude-longitude grids.
    """
    def __init__(self, alt_km, n_lat:int=None, n_lon:int=None, lmax:int=None, quad_type:str=None, flattening=0):
        super().__init__(alt_km, flattening)
        if (lmax is None) and (quad_type is None) and (n_lat is not None) and (n_lon is not None): 
            # regular equispaced grid
            self.grid_name = 'grid_'+str(n_lat)+'x'+str(n_lon)
            self.grid_type_name = 'regular_latlon'
            self.lon_vec, self.lat_vec, self.lon_edges_vec, self.lat_edges_vec = get_sphere_grid(n_lon, n_lat)
            self.initialize_grid()

            if self.flattening==0:
                areas = get_spherical_grid_cell_areas(self.lat_edges_vec, self.lon_edges_vec, R=RE+self.alt_km)
            else:
                areas = get_ellipsoid_grid_cell_areas(self.lat_edges_vec, self.lon_edges_vec, f=self.flattening, Req=RE+self.alt_km)

            self.integration_weights = np.reshape(areas, (self.n_lat * self.n_lon))

        elif (lmax is not None) and (quad_type is not None) and (n_lat is None) and (n_lon is None): 
            
            assert(quad_type=="GLQ") # it seems that DH is basically equispaced
            self.grid_name = 'grid_' + quad_type + '_n' + str(lmax)
            self.grid_type_name = 'regular_' + quad_type
            pysh_grid = get_pysh_grid(quad_type, lmax)
            
            #self.lon_vec, self.lat_vec = pysh_grid.lons(), pysh_grid.lats()
            self.lon_edges_vec, self.lat_vec = pysh_grid.lons(), pysh_grid.lats()

            lon_midpoints = get_pairwise_midpoints(self.lon_edges_vec)
            self.lon_vec = lon_midpoints

            lat_midpoints = get_pairwise_midpoints(self.lat_vec)
            self.lat_edges_vec = np.concatenate(([90], lat_midpoints, [-90]))

            self.initialize_grid()

            pysh_lat_weigths = pysh_grid.weights
            areas = get_spherical_grid_cell_areas(self.lat_edges_vec, self.lon_edges_vec, R=RE+self.alt_km)
            sph_area = 4 * np.pi * (RE+self.alt_km)**2
            #lat_band_areas = np.sum(areas, axis=1)
            weights_array = np.zeros_like(areas)
            for i, w_i in enumerate(pysh_lat_weigths):
                #weights_array[i] = np.ones(self.n_lon) * w_i * self.total_area / (2 * self.n_lon)
                weights_array[i] = np.ones(self.n_lon) * w_i * sph_area / (2 * self.n_lon)
            
            self.integration_weights = np.reshape(weights_array, (self.n_lat * self.n_lon))
            if self.flattening != 0:
                self.adjust_integration_weights_to_ellipsoid()

        else:
            print("Wrong initialization of Regular LatLonGrid!")

        #self.adjust_integration_weights_to_ellipsoid()
            
    def initialize_grid(self):

        self.n_lat = len(self.lat_vec)
        self.n_lon = len(self.lon_vec)
        
        #if self.flattening == 0:  # for now
        self.stacked_grid_r, self.stacked_grid_u = get_stacked_spherical_grid_els(
            self.lon_vec, self.lat_vec, 
            self.alt_km, total_R=RE+self.alt_km)
        
        lon_mesh_vectd, lat_mesh_vectd, _ = get_reshaped_grid(self.lon_vec, self.lat_vec)
        self.stacked_grid_latlon = np.column_stack((lat_mesh_vectd, lon_mesh_vectd))

        self.n_points = len(self.stacked_grid_r)
        if self.flattening != 0:
            self.project_r_vectors_to_elliposoid()
            self.adjust_u_vectors_to_ellipsoid()
    
    def reshape_if_needed(self, field_array):
        shape = np.shape(field_array)
        if len(shape)==1:
            return field_array.reshape((self.n_lat, self.n_lon))
        elif len(shape)==2: # time hist in the second axis
            n_steps = shape[1]
            return field_array.reshape((self.n_lat, self.n_lon, n_steps))
        elif len(shape)==3: # stacked time series of vectors on grid
            n_steps = shape[2]
            assert(shape[1]==3)
            return field_array.reshape((self.n_lat, self.n_lon, 3, n_steps))


    def vectorize_if_needed(self, field_array):
        shape = np.shape(field_array)
        if len(shape)==2:
            if np.prod(shape)==self.n_lat*self.n_lon: # single snaptshot of field in a 2D latlon grid
                return field_array.reshape((self.n_lat*self.n_lon))
            elif shape[0]==self.n_lat*self.n_lon: # stretched time hist of field with time in axis 1
                return field_array
        
        elif len(shape)==3:
            n_steps = shape[2] # assume last dim is the time dim - what about F maps?
            return field_array.reshape((self.n_lat*self.n_lon, n_steps)) # checked to give the same as for loop
            #out_array = np.zeros((self.n_lat*self.n_lon, n_steps))
            #for i in range(n_steps):
            #    out_array[:,i] = field_array[:,:,i].reshape((self.n_lat*self.n_lon))
            #return out_array
    
    def compute_lat_avg_map(self, field_aray):
        # mean works because element areas are constant at every latitude
        return np.mean(field_aray, axis=1)  

    def recompute_grid(self, r_sat):
            pass
    
    
class RegularQuadratureGrid(Grid):

    def __init__(self, alt_km, l_max, flattening=0):
        pass


class QuadratureGrid(Grid):
    """
    Subclass for quadrature grids using Lebedev rules.
    """
    def __init__(self, alt_km, order:int, flattening=0, default_womersley=False):
        super().__init__(alt_km, flattening)
        self.grid_name = 'grid_quad_n'+str(order)
        self.grid_type_name = 'quadrature'
        self.order = order
        self.default_womersley = default_womersley
        self.initialize_grid()

    def initialize_grid(self):

        #if self.flattening == 0:
        if (self.order <= 131) and not(self.default_womersley): 
            from scipy.integrate import lebedev_rule  # not available in the scipy version of monte168
            u_el, weights = lebedev_rule(self.order) # 131 is maximum available in scipy
            self.n_points = len(weights)
        else:
            all_fnames = os.listdir(os.path.join(CONFIG_DIR, 'maths','spherical_designs_womersley'))
            fname = [f for f in all_fnames if f"ss{self.order:03d}" in f]
            assert len(fname)==1
            u_el = np.loadtxt(os.path.join(CONFIG_DIR, 'maths','spherical_designs_womersley',fname[0])).T
            self.n_points = np.shape(u_el)[1]
            weights = np.ones(self.n_points) * (4 * np.pi) / self.n_points
            #raise NotImplementedError("Orders > 131 not implemented")
        
        self.stacked_grid_u = np.transpose(u_el)
        self.stacked_grid_r = self.stacked_grid_u * (RE + self.alt_km)
        
        (_, all_lat, all_lon) = cartesian_to_spherical(u_el[0, :], u_el[1, :], u_el[2, :])
        self.stacked_grid_latlon = np.column_stack((all_lat.deg, all_lon.deg))
        self.integration_weights = weights * (RE + self.alt_km)**2
        #self.n_points = len(self.integration_weights)

        if self.flattening != 0:
            self.project_r_vectors_to_elliposoid()
            self.adjust_u_vectors_to_ellipsoid()
            self.adjust_integration_weights_to_ellipsoid()

    def recompute_grid(self, r_sat):
        pass
    
    def reshape_if_needed(self, field_array):
        return field_array # quadrature grids cannot be reshaped to a regular matrix
    
    def vectorize_if_needed(self, field_array):
        return field_array
    
    def compute_lat_avg_map(self, field_aray):
        raise NotImplementedError("Lat avg. not implemented (yet?) for non-regular grids")


class KnockeGridMONTE(Grid):

    def __init__(self, n_rings:int, alt_km=0, flattening=0, r_sat_0=np.array([0,0,0])):
        super().__init__(alt_km, flattening)
        self.grid_name = 'grid_knocke_monte_nR'+str(n_rings)
        self.grid_type_name = 'knocke_monte'
        self.n_rings = n_rings

        self.initialize_grid(r_sat_0)

    def initialize_grid(self, r_sat_0):
        #if self.flattening == 0:
        _, _, all_ring_centers, all_element_areas_at_ring = get_monte_mesh(r_sat_0, self.n_rings, alt_km=self.alt_km)
        self.stacked_grid_r = np.hstack(all_ring_centers).T
        self.stacked_grid_u = self.stacked_grid_r / (RE + self.alt_km)

        n_els_per_ring = [len(ring_mat.T) for ring_mat in all_ring_centers]
        areas = np.repeat(all_element_areas_at_ring, n_els_per_ring) # should all be equal
        self.integration_weights = areas
        self.n_points = len(areas)

        (_, all_lat, all_lon) = cartesian_to_spherical(self.stacked_grid_u[:, 0], 
                                                        self.stacked_grid_u[:, 1], 
                                                        self.stacked_grid_u[:, 2])
        self.stacked_grid_latlon = np.column_stack((all_lat.deg, all_lon.deg))
        if self.flattening != 0:
            self.project_r_vectors_to_elliposoid()
            self.adjust_u_vectors_to_ellipsoid()
            self.adjust_integration_weights_to_ellipsoid()

        #else:
        #    raise NotImplementedError("Knocke MONTE grid is only defined for perfectly spherical TOA")

    def recompute_grid(self, r_sat):
        #print(f"Recomputing Knocke grid at {r_sat}")
        self.initialize_grid(r_sat)
    
    def reshape_if_needed(self, field_array):
        return field_array # quadrature grids cannot be reshaped to a regular matrix
    
    def vectorize_if_needed(self, field_array):
        return field_array
    
    def compute_lat_avg_map(self, field_aray):
        raise NotImplementedError("Lat avg. not implemented (yet?) for non-regular grids")

class KnockeGrid(Grid):
    pass

"""
class SphereGrid():

    def __init__(self, grid_type, alt_km, flattening=0, n_lat=None, n_lon=None, order=None):
        self.grid_type = grid_type
        
        if grid_type=="regular_latlon":
            assert n_lat is not None and n_lon is not None
            
            if flattening==0: # for now
                lon_vec, lat_vec, lon_edges_vec, lat_edges_vec = get_sphere_grid(n_lon, n_lat)
                self.lon_edges, self.lat_edges = lon_edges_vec, lat_edges_vec
                self.stacked_grid_r, self.stacked_grid_u = get_stacked_spherical_grid_els(
                    lon_vec, lat_vec, 
                    alt_km, total_R=RE+alt_km)
                lon_mesh_vectd, lat_mesh_vectd, _ = get_reshaped_grid(lon_vec,lat_vec)
                self.stacked_grid_latlon = np.column_stack((lat_mesh_vectd, lon_mesh_vectd))
                areas = get_spherical_grid_cell_areas(lat_edges_vec, lon_edges_vec, R=RE+alt_km)
                self.integration_weights = np.reshape(areas, (n_lat*n_lon))

        
        elif grid_type=="quadrature":
            assert order is not None

            if flattening==0:
                if order<=131:
                    u_el, weights = lebedev_rule(order)
                else:
                    pass # not implemented yet
                
                self.stacked_grid_u = np.transpose(u_el)
                self.stacked_grid_r = self.stacked_grid_u * (RE + alt_km)
                
                (_, all_lat, all_lon) = cartesian_to_spherical(u_el[0,:], u_el[1,:], u_el[2,:])
                self.stacked_grid_latlon = np.column_stack((all_lat.deg, all_lon.deg))
                self.integration_weights = weights * (RE + alt_km)**2
"""
def get_pysh_grid(quad_type, lmax): # separate function to avoid the import conflict in debug mode?

    import pyshtools as pysh
    pysh_grid = pysh.SHGrid.from_zeros(lmax=lmax, grid=quad_type)
    return pysh_grid

def fit_sh_field(y_vec, lon_vec, lat_vec, lmax):
    import pyshtools as pysh 
    
    sh_set = pysh.SHCoeffs.from_least_squares(data=y_vec, latitude=lat_vec, longitude=lon_vec, 
                                              lmax=lmax)
    
    return sh_set

def get_r_correction_factors_biaxial_ellipsoid(f, lats):
    return (1-f) / np.sqrt((1-f)**2 * (np.cos(lats))**2 + (np.sin(lats))**2)


# deprecated function
def get_cell_samples_in_regular_grid(lat_hist_vec, lon_hist_vec, x_hist_vec, n_lon=360, n_lat=180):  # TODO: optimize and make it a mehtod of the above classes


    n = len(lat_hist_vec)
    assert(len(lon_hist_vec) == n == len(x_hist_vec))
    
    lon_hist_vec = ensure_0_to_360(lon_hist_vec)
    lon_vec, lat_vec, lon_edges_vec, lat_edges_vec = get_sphere_grid(n_lon, n_lat)
    #count_matrix = np.zeros((n_lat, n_lon), dtype=int)
    x_series_matrix = [ [ [] for _ in range(n_lon) ] for _ in range(n_lat) ]  # ref as acc_series_matrix[lat_idx][lon_idx]

    # split lat_hist and lon_hist in chunks of 1e5 rows:
    lat_hist_vec_split = np.array_split(lat_hist_vec, max(1, n//100000))
    lon_hist_vec_split = np.array_split(lon_hist_vec, max(1, n//100000))
    x_hist_vec_split = np.array_split(x_hist_vec, max(1, n//100000))
    n_chunks = len(lat_hist_vec_split)
    
    
    for i in range(n_chunks): # TODO
        
        lon_hist = lon_hist_vec_split[i]
        lat_hist = lat_hist_vec_split[i]
        
        lon_diffs = np.abs(lon_vec - lon_hist[:, None])
        lat_diffs = np.abs(lat_vec - lat_hist[:, None])
        
        lon_idxs = np.argmin(lon_diffs, axis=1)
        lat_idxs = np.argmin(lat_diffs, axis=1)
        radial_acc_chunk = x_hist_vec_split[i]
        
        for (lon, lat, acc) in zip(lon_idxs, lat_idxs, radial_acc_chunk):
            x_series_matrix[lat][lon].append(acc)
        
        #stretched jd matrix: [x for level1 in jd_series_matrix for level2 in level1 for x in level2]
        #np.append.at(count_matrix, (lat_idxs, lon_idxs), 1)  # order 100 times faster than for loop

        progress_bar(i+1, n_chunks)

    
    return x_series_matrix


def get_sphere_grid(n_lon, n_lat):

    lon_edges_vec = np.linspace(0,360,n_lon+1)
    lat_edges_vec = np.linspace(90,-90,n_lat+1)
    
    lon_vec = 0.5 * (lon_edges_vec[:-1] + lon_edges_vec[1:])
    lat_vec = 0.5 * (lat_edges_vec[:-1] + lat_edges_vec[1:])
    
    return lon_vec, lat_vec, lon_edges_vec, lat_edges_vec


def get_spherical_grid_cell_areas(lat_edges, lon_edges, R=constants.earth_radius()):
    
    lat_edges = np.radians(lat_edges)
    lon_edges = np.radians(lon_edges)
    
    sin_lat_vec = np.sin(lat_edges)
    diff_sin_lat_vec = np.diff(sin_lat_vec)
    diff_lon_lat_vec = np.diff(lon_edges)
    areas_grid = R**2 * np.abs(np.outer(diff_sin_lat_vec, diff_lon_lat_vec)) 
    # same result as with the double for loop (within 1e-14 rel. diff.) and ~500 times faster
    
    return areas_grid

def get_ellipsoid_grid_cell_areas(lat_edges, lon_edges, f, Req=constants.earth_radius()):

    lat_edges = np.radians(lat_edges)
    lon_edges = np.radians(lon_edges)
    F_lat_vec = F_lambda_function(f, lat_edges)
    diff_F_lat_vec = np.diff(F_lat_vec)
    diff_lon_lat_vec = np.diff(lon_edges)
    areas_grid = Req**2 / 2 * np.abs(np.outer(diff_F_lat_vec, diff_lon_lat_vec))

    return areas_grid



def F_lambda_function(f, lats):
    assert(f>0)

    e2 = f * (2 - f)
    q = (1 - f)**2
    A = e2 * (2 - e2)
    e = np.sqrt(e2)
    x = np.sin(lats)
    B = np.sqrt(q**2 + A*x**2)

    return x * B / (q + e2 * x**2) + (q/e) * np.arctanh(e * x / B)


def get_stacked_spherical_grid_els(lon_vec, lat_vec, altitude_km, total_R):
    n_lon = len(lon_vec)
    n_lat = len(lat_vec)

    grid_r_el = lonlat_to_r(lon_vec, lat_vec, altitude_km, "spherical")
    stacked_grid_r_el = np.reshape(grid_r_el, (n_lon*n_lat, 3))
    stacked_grid_u_el = stacked_grid_r_el / ( total_R * np.ones((n_lon*n_lat,1)))

    return stacked_grid_r_el, stacked_grid_u_el


def load_lebedev_grid_aux(lebedev_order, radius_meters):
    
    vertices_file = os.path.join(CONFIG_DIR, 'maths', 'lebedev_grids', 'stacked_r_grid_'+str(lebedev_order)+'.npy')
    if os.path.exists(vertices_file):
        vertices = np.load(vertices_file) * radius_meters
    else: 
        grid = QuadratureGrid(alt_km=-RE+1, order=lebedev_order, default_womersley=False)
        np.save(vertices_file, grid.stacked_grid_r)
        vertices = grid.stacked_grid_r * radius_meters
    
    return vertices


def get_faceted_sphere_mesh(radius_meters, n_faces, default_golden=False):
    
    if n_faces==6:
        # TODO: cube
        a = np.sqrt(np.pi) * radius_meters
        vertices = np.array([
            [-a/2, -a/2, -a/2],
            [ a/2, -a/2, -a/2],
            [ a/2,  a/2, -a/2],
            [-a/2,  a/2, -a/2],
            [-a/2, -a/2,  a/2],
            [ a/2, -a/2,  a/2],
            [ a/2,  a/2,  a/2],
            [-a/2,  a/2,  a/2],
        ])

    if not(default_golden):
        if n_faces==8:
            # Regular (Platonic) octahedron, availabla as a quadrature grid
            grid = QuadratureGrid(alt_km=-RE+radius_meters, order=3)
            vertices = grid.stacked_grid_r

        elif n_faces==20:
            # Regular (Platonic) icosahedron, availabla as a quadrature grid
            grid = QuadratureGrid(alt_km=-RE+radius_meters, order=5, default_womersley=True)
            vertices = grid.stacked_grid_r

        elif n_faces==24:
            # Cube with pyramids (tetrakis hexahedron), available as a quadrature grid
            lebedev_order = 5
            vertices = load_lebedev_grid_aux(lebedev_order, radius_meters)
        
        elif n_faces==32:
            # TODO: load sphere32 file?
            pass

        elif n_faces==48:
            lebedev_order = 7
            vertices = load_lebedev_grid_aux(lebedev_order, radius_meters)
            #grid = QuadratureGrid(alt_km=-RE+radius_meters, order=7, default_womersley=False)
            #vertices = grid.stacked_grid_r

        elif n_faces==60:
            grid = QuadratureGrid(alt_km=-RE+radius_meters, order=7, default_womersley=True)
            vertices = grid.stacked_grid_r
        
        elif n_faces==72:
            lebedev_order = 9
            vertices = load_lebedev_grid_aux(lebedev_order, radius_meters)
            #grid = QuadratureGrid(alt_km=-RE+radius_meters, order=9, default_womersley=False)
            #vertices = grid.stacked_grid_r
        
        elif n_faces==92:
            grid = QuadratureGrid(alt_km=-RE+radius_meters, order=9, default_womersley=True)
            vertices = grid.stacked_grid_r
        
        elif n_faces==96:
            lebedev_order = 11
            vertices = load_lebedev_grid_aux(lebedev_order, radius_meters)
            #grid = QuadratureGrid(alt_km=-RE+radius_meters, order=11, default_womersley=False)
            #vertices = grid.stacked_grid_r

    grid_made = 'vertices' in locals()
    if not(grid_made): #elif n_faces > 96:
        n_vertices = int(2 + n_faces/2)
        print(f"n_vertices: {n_vertices}")
        vertices = fibonacci_sphere(radius_meters, n=n_vertices)

    hull = ConvexHull(vertices)
    if n_faces!=6:
        assert(n_faces == len(hull.simplices))

    triangle_vertices = hull.points[hull.simplices]
    triangle_centroids = np.mean(triangle_vertices, axis=1)

    normal_vecs, areas = get_areas_and_normal_vecs_from_pointmesh(hull)
    edge_conns = get_edge_connectivities(hull)
    
    return vertices, triangle_centroids, normal_vecs, areas, edge_conns, hull

        


def fibonacci_sphere(R, n=1000):  
    
    # from https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere

    phi = np.pi * (np.sqrt(5.) - 1.)  # golden angle

    i = np.arange(n)
    theta = phi * i

    y = R - (i / (n - 1)) * 2 * R  # y from R to -R
    radius = np.sqrt(R**2 - y**2)

    x = np.cos(theta) * radius
    z = np.sin(theta) * radius

    points = np.stack((x, y, z), axis=1)
    
    return points


def get_mesh_from_fibonacci_sphere(radius_meters, n_faces):

    n_vertices = int(2 + n_faces/2)
    vertices = fibonacci_sphere(radius_meters, n=n_vertices)
    
    hull = ConvexHull(vertices)
    assert(n_faces == len(hull.simplices))

    normal_vecs, areas = get_areas_and_normal_vecs_from_pointmesh(hull)

    return normal_vecs, areas, hull

def get_edge_connectivities(hull: ConvexHull):
    triangles = hull.simplices  # shape (n_faces, 3)
    print(f"Number of faces: {len(triangles)}")

    edge_conns = np.vstack([
        triangles[:, [0, 1]],
        triangles[:, [1, 2]],
        triangles[:, [2, 0]],
    ])

    edge_conns = np.sort(edge_conns, axis=1)    # normalize direction
    edge_conns = np.unique(edge_conns, axis=0)  # remove duplicates

    return edge_conns

def get_areas_and_normal_vecs_from_pointmesh(hull: ConvexHull):

    #assert(n_faces == len(hull.simplices))

    n = len(hull.simplices)
    all_normals = np.zeros((n,3))
    all_areas = np.zeros(n)

    for i, idx_triad in enumerate(hull.simplices):
        triangle = hull.points[idx_triad]
        closed_triangle = np.vstack((triangle, triangle[0]))
        triangle_vectors = np.diff(closed_triangle, axis=0)

        face_normal_1 = np.cross(triangle_vectors[0], triangle_vectors[1])
        area = np.linalg.norm(face_normal_1) / 2
        face_normal_2 = -face_normal_1
        face_normal_candidates = [face_normal_1, face_normal_2]

        vec_to_centroid = np.mean(triangle, axis=0)
        vec_to_centroid_norm = np.linalg.norm(vec_to_centroid)
        cos_angles = [np.dot(normal, vec_to_centroid)/(np.linalg.norm(normal)*vec_to_centroid_norm) for normal in face_normal_candidates]

        # the true "outwards" normal vector is the one that forms a smallest angle with the origin-to-facecentroid vector
        j = cos_angles.index(np.max(cos_angles))
        normal = face_normal_candidates[j]

        all_normals[i, :] = normal
        all_areas[i] = area

    all_normals = all_normals / np.linalg.norm(all_normals, axis=1)[:,None]

    return all_normals, all_areas



def improved_normals_and_areas(points):  # mainly authored by ChatGPT with the above function as input
    """
    Compute outward-facing normals and areas for triangular faces defined by 'simplices' on 'points'.

    Parameters:
    - points: (N, 3) array of vertex coordinates.
    - simplices: (M, 3) array of indices into 'points', defining M triangles.

    Returns:
    - normals: (M, 3) array of outward-facing unit normals.
    - areas: (M,) array of triangle areas.
    """
    simplices = ConvexHull(points).simplices

    # Gather triangle vertices: shape (M, 3, 3)
    tri_pts = points[simplices]  # For each triangle, 3 vertices with 3D coords

    # Compute two edges per triangle: v1 = p1 - p0, v2 = p2 - p0
    v1 = tri_pts[:, 1] - tri_pts[:, 0]
    v2 = tri_pts[:, 2] - tri_pts[:, 0]

    # Compute face normals (not yet unit length): cross product v1 × v2, shape (M, 3)
    raw_normals = np.cross(v1, v2)

    # Areas are half the norm of the cross product: ||raw_normal|| / 2
    norm_lengths = np.linalg.norm(raw_normals, axis=1)
    areas = 0.5 * norm_lengths

    # Unit normals (before enforcing outward direction)
    # Avoid division by zero by handling degenerate faces (area=0) if needed
    nonzero = norm_lengths > 0
    unit_normals = np.zeros_like(raw_normals)
    unit_normals[nonzero] = raw_normals[nonzero] / norm_lengths[nonzero, None]

    # Compute centroid of each triangle: mean of 3 vertices, shape (M, 3)
    centroids = tri_pts.mean(axis=1)

    # Determine sign for outward direction: dot(unit_normal, centroid) < 0 means normal points inward
    sign = np.sign(np.einsum('ij,ij->i', unit_normals, centroids))
    # Flip normals where sign < 0 (i.e., if dot product is negative)
    outward_normals = unit_normals.copy()
    outward_normals[sign < 0] *= -1

    return outward_normals, areas




def find_tangent_cone_x_Earth_loci(r_vec, n=250, endpoint_bool=True, shift_deg=0, Re=constants.earth_radius()):

    # NOTE that this assumes a SPHERICAL EARTH!

    shift_rad = np.deg2rad(shift_deg)

    # assume spherical Earth
    #Re = constants.earth_radius()     # [km]

    r_norm = np.linalg.norm(r_vec)
    u = r_vec / r_norm
    v, w = get_two_perp_unit_vectors(u)

    c = Re**2 / (r_norm**2) * r_vec                 # center of circle (proof on paper)
    R_red = np.sqrt(np.abs(Re**2 - np.dot(c,c)))    # radius of circle (Pythagoras)

    # parametric description of circle:
    theta_vec = np.linspace(0 + shift_rad, 2*np.pi + shift_rad, n, endpoint=endpoint_bool)
    sin_vec = np.sin(theta_vec)
    cos_vec = np.cos(theta_vec)
    all_x = c.reshape((3,1)) + R_red*(np.outer(v, cos_vec) + np.outer(w, sin_vec))

    # # transform set of points to Earth coordinates:
    # cart = CartesianRepresentation(x=all_x[0, :], y=all_x[1, :], z=all_x[2, :])
    # coord = ITRS(cart)
    # lon = coord.spherical.lon.to_value()
    # lon = np.where(lon > 180, lon - 360, lon)
    # lat = coord.spherical.lat.to_value()

    # lonlats = np.transpose(np.vstack((lon, lat)))
    # #lonlats = lonlats[np.argsort(lonlats[:, 0])]

    # return lonlats

    return all_x



def get_monte_mesh(r_vec, n_rings, alt_km=0, compute_nodes_and_connects=False):

    Re = constants.earth_radius() + alt_km   # [km]
    cos_phi_vec = get_cos_phi_rings(Re, r_vec, n_rings)

    r_vec_norm = np.linalg.norm(r_vec)
    all_ring_nodes = [None] * n_rings
    all_ring_connects = [None] * n_rings
    all_ring_centers = [None] * (n_rings + 1)           # includes central cap
    all_element_areas_at_ring = [None] * (n_rings + 1)  # includes central cap

    cap_area = ring_area = 2*np.pi*(1-cos_phi_vec[0]) * Re**2
    all_element_areas_at_ring[0] = cap_area
    #print(f"Central cap area: {cap_area}")

    r_vec_aux = Re * r_vec/r_vec_norm
    cap_center = find_tangent_cone_x_Earth_loci(r_vec_aux, 1, endpoint_bool=False, Re=Re)
    all_ring_centers[0] = cap_center
    

    for i in range(n_rings):
        n_sections = 6*(i + 1)
        cos_phi_up = cos_phi_vec[i]
        cos_phi_down = cos_phi_vec[i+1]
        ring_area = 2*np.pi*(cos_phi_up-cos_phi_down) * (Re)**2
        element_area = ring_area / n_sections
        #print(f"Element areas at ring {i}: {element_area}")
        all_element_areas_at_ring[i+1] = element_area  # [km^2]
        
        r_vec_aux = Re/(np.mean([cos_phi_vec[i], cos_phi_vec[i+1]])) * r_vec/r_vec_norm
        ring_i_element_centers = find_tangent_cone_x_Earth_loci(r_vec_aux, n_sections, endpoint_bool=False, shift_deg=360/(2*n_sections), Re=Re)
        all_ring_centers[i+1] = ring_i_element_centers

        if compute_nodes_and_connects:
            r_vec_aux = Re/cos_phi_vec[i] * r_vec/r_vec_norm
            ring_i_upper_bound = find_tangent_cone_x_Earth_loci(r_vec_aux, n_sections, endpoint_bool=False, Re=Re)

            r_vec_aux = Re/cos_phi_vec[i+1] * r_vec/r_vec_norm
            ring_i_lower_bound = find_tangent_cone_x_Earth_loci(r_vec_aux, n_sections, endpoint_bool=False, Re=Re)

            ring_i_node_matrix = np.vstack((ring_i_upper_bound, ring_i_lower_bound))
            #ring_i_connectivity_matrix = np.array([[j, (j+1)%(n_sections), (n_sections+j+1)%(2*n_sections), (n_sections+j)] for j in range(n_sections)])
            ring_i_connectivity_matrix = np.array([[j, (j+1)%(n_sections), n_sections+(j+1)%(n_sections), n_sections+j] for j in range(n_sections)])

            all_ring_nodes[i] = ring_i_node_matrix
            all_ring_connects[i] = ring_i_connectivity_matrix

    return all_ring_nodes, all_ring_connects, all_ring_centers, all_element_areas_at_ring


def get_cos_phi_rings(Re, r_vec, n_rings):

    A_mat = np.zeros((n_rings, n_rings))
    for i in range(n_rings):
        if i>0:
            A_mat[i,i] = -1/(6*i) - 1/(6*(i+1))
            A_mat[i,i-1] = 1/(6*i)
            if i<n_rings-1:
                A_mat[i,i+1] = 1/(6*(i+1))
        else:
            A_mat[0,0] = -1 - 1/6
            A_mat[0,1] = 1/6
    
    cos_beta = Re / np.linalg.norm(r_vec)
    b = np.zeros(n_rings)
    b[0] = -1
    b[-1] = -cos_beta * (1/(6*n_rings))

    cos_phi_vec = np.linalg.solve(A_mat, b)
    cos_phi_vec = np.append(cos_phi_vec, cos_beta)

    return cos_phi_vec



def lonlat_to_r(lon_vec, lat_vec, h_toa_km, mode):
    
    if mode=="spherical":
        #n = len(lon_vec)*len(lat_vec)
        #(lon_mesh, lat_mesh) = np.meshgrid(lon_vec, lat_vec)
        lon_mesh_vectd, lat_mesh_vectd, mesh_shape = get_reshaped_grid(lon_vec,lat_vec)
        
        x, y, z = spherical_to_cartesian(constants.earth_radius(units='km') + h_toa_km, 
                                         np.radians(lat_mesh_vectd), 
                                         np.radians(lon_mesh_vectd))
        x = np.reshape(x.value, mesh_shape)
        y = np.reshape(y.value, mesh_shape)
        z = np.reshape(z.value, mesh_shape)
        
        return np.stack((x, y, z), 2)
        
    else:
        print("Mode {mode} not implemented!")


def get_reshaped_grid(x,y):
    
    n = len(x)*len(y)
    X_grid, Y_grid = np.meshgrid(x,y)
    X_vectorized = np.reshape(X_grid, n) 
    Y_vectorized = np.reshape(Y_grid, n)
    
    return X_vectorized, Y_vectorized, np.shape(X_grid)



# spherical harmonic stuff?

def expand_sh(sh_map, evaluation_lon_vec, evaluation_lat_vec, normalization):
    import pyshtools as pysh
    """
    if 'pyshtools' in sys.modules:
        pysh = sys.modules['pyshtools']
    else:
        pysh = importlib.import_module('pyshtools')
    """
    
    eval_coeffs = pysh.SHCoeffs.from_array(sh_map, normalization=normalization).expand(
                                        lon=evaluation_lon_vec, lat=evaluation_lat_vec, 
                                        backend='ducc', nthreads=10)
    return eval_coeffs


"""
def expand_sh_regular_latlon_grid(sh_map, lon_vec, lat_vec):
    
    import pyshtools as pysh
    
    #n = len(lon_vec)*len(lat_vec)
    #(lon_mesh, lat_mesh) = np.meshgrid(lon_vec, lat_vec)
    lon_mesh_vectd, lat_mesh_vectd, mesh_shape = get_reshaped_grid(lon_vec,lat_vec)
    grid = pysh.SHCoeffs.from_array(sh_map, normalization='unnorm').expand(
                                        lon=lon_mesh_vectd, lat=lat_mesh_vectd)
    grid = np.reshape(grid, mesh_shape)
    
    return grid


def expand_sh_arbitrary_grid(sh_map, stacked_r_array):
    
    import pyshtools as pysh
    
    (_, lat_vec, lon_vec) = cartesian_to_spherical(stacked_r_array[:,0], stacked_r_array[:,1], stacked_r_array[:,2])
    #a_sh_array, e_sh_array = rad_settings.get_ae_sh_maps_numpy_new(rad_config["sh_mode"], day_datestr)
    
    coeffs = pysh.SHCoeffs.from_array(sh_map, normalization='unnorm')
    grid = coeffs.expand(lat=lat_vec.degree, lon=lon_vec.degree)

    return grid
"""

def field_hist_rotation(field_hist, R_hist, grid: Grid, transpose_R=False):
    
    field_hist = grid.vectorize_if_needed(field_hist)
    n_points, n_steps = np.shape(field_hist)

    field_hist_newframe = np.zeros_like(field_hist) #np.zeros((n_lat, n_lon, n_steps))
    print("Rotating field history to Sun-Fixed frame...")

    for i in range(n_steps):
        progress_bar(i, n_steps)
        #field_i = field_hist[:,:,i]
        field_i = field_hist[:,i]
        R_i = np.transpose(R_hist[i,:,:]) if transpose_R else R_hist[i,:,:]
        
        grid_u_el_rot = (R_i @ grid.stacked_grid_u.T).T
        interp2 = sphere_field_interp2(grid_u_el_rot, field_i, grid)
        Plotter.plot_geo_data(interp2.reshape((grid.n_lat, grid.n_lon)),
                              grid.lon_edges_vec, grid.lat_edges_vec, file_name='test_interp2')

        _, lat_rot, lon_rot = cartesian_to_spherical(grid_u_el_rot[:,0], grid_u_el_rot[:,1], grid_u_el_rot[:,2])  # astropy
        interp = sphere_field_interp(lat_rot.deg, lon_rot.deg, field_i, grid)

        Plotter.plot_geo_data(interp.reshape((grid.n_lat, grid.n_lon)),
                              grid.lon_edges_vec, grid.lat_edges_vec, file_name='test_interp')
        stripy.spherical.interpolate
        field_hist_newframe[:,i] = interp

        #P_i = np.array((lat_rot.degree, lon_rot.degree)).T
        #Z_i = field_i #.reshape(n_lon*n_lat)
        
        ##LAT_vec, LON_vec, _ = get_reshaped_grid(lat_vec, lon_vec)
        #_, LAT_vec, LON_vec = grid.stacked_grid_latlon[:,0], grid.stacked_grid_latlon[:,1]  # cartesian_to_spherical(stacked_grid_u_el[:,0], stacked_grid_u_el[:,1], stacked_grid_u_el[:,2])  # astropy
        #Pq = np.array((LAT_vec, LON_vec)).T
        # Pq = grid.stacked_grid_latlon
        # Z_interp = sphere_field_interp(P_i, Z_i, Pq, grid)
# 
        # Z_interp = griddata(P_i, Z_i, Pq, method="linear")  # scipy interpolate
        # field_hist_newframe[:,i] = Z_interp#.reshape((n_lat, n_lon), order='F')  # seems to be right ...
        
    return field_hist_newframe


def sphere_field_interp2(stacked_u_i, V_i, grid: Grid):

    for i in range(5):
        t1 = time.time()
        interpolator = RBFInterpolator(stacked_u_i, V_i, neighbors=i+2, smoothing=0, kernel='linear', degree=-1)
        out = interpolator(grid.stacked_grid_u)
        t2 = time.time()
        print(f"Time to interpolate with neighbors={i}: {t2-t1}")
        print(f"Exactness check: {np.max(np.abs(interpolator(stacked_u_i)-V_i))}")

    return out


def sphere_field_interp(stacked_lat_lon_i, stacked_Zi, stacked_lat_lon_q, method='griddata_nearest'):
    # quest points will be the grid points
    # RectSphereBivariateSpline cannot be used unless the input coordinates are distributed on a regular grid
    # (since we input coordinates rotated to the SFF and/or on quadrature grids, this rarely happens in our usage)
    # SmoothSphereBivariateSpline has been tried but (at least with a regular 1degx1deg grid) it's riddiculously slow
    # RBFInterpolator is fast-ish with low number of neighbors (kernel='linear', degree=-1), but huge artifacts appear near the poles
    
    if method=="griddata_nearest":      # TODO: halo to fix edges?
        return griddata(stacked_lat_lon_i, stacked_Zi, stacked_lat_lon_q, method="nearest")  # scipy interpolate
    
    if method=="griddata_linear":      # TODO: halo to fix edges?
        return griddata(stacked_lat_lon_i, stacked_Zi, stacked_lat_lon_q, method="linear")  # scipy interpolate
    
    if method=="griddata_cubic":      # TODO: halo to fix edges?
        return griddata(stacked_lat_lon_i, stacked_Zi, stacked_lat_lon_q, method="cubic")  # scipy interpolate
    
    elif method=="spline":
        colats_i = np.deg2rad(lat_to_colat_deg(stacked_lat_lon_i[:,0]))
        lons_i = np.deg2rad(ensure_0_to_360(stacked_lat_lon_i[:,1]))
        interp = SmoothSphereBivariateSpline(colats_i, lons_i, stacked_Zi, s=1)

        print("interpolator created")



def field_hist_rotation_old(field_hist, R_hist, transpose_R=False):
    
    #n_steps = np.shape(field_hist)[2]
    n_steps = np.shape(field_hist)[1]
    n_lat, n_lon = np.shape(field_hist)[:2]
    lon_vec, lat_vec, lon_edges_vec, lat_edges_vec = get_sphere_grid(n_lon, n_lat)
    
    grid_r_el = lonlat_to_r(lon_vec, lat_vec, 0, "spherical")
    stacked_grid_r_el = np.reshape(grid_r_el, (n_lon*n_lat, 3))
    stacked_grid_u_el = stacked_grid_r_el / (constants.earth_radius(units='km') * np.ones((n_lon*n_lat,1)))
    

    # assert(np.shape(R_hist)[0]==n_steps)
    field_hist_newframe = np.zeros((n_lat, n_lon, n_steps))
    print("Rotating field history to Sun-Fixed frame...")

    for i in range(n_steps):
        progress_bar(i, n_steps)
        #field_i = field_hist[:,:,i]
        field_i = field_hist[:,i]
        R_i = np.transpose(R_hist[i,:,:]) if transpose_R else R_hist[i,:,:]
        
        grid_u_el_rot = (R_i @ stacked_grid_u_el.T).T
        _, lat_rot, lon_rot = cartesian_to_spherical(grid_u_el_rot[:,0], grid_u_el_rot[:,1], grid_u_el_rot[:,2])  # astropy
        
        P_i = np.array((lat_rot.degree, lon_rot.degree)).T
        Z_i = field_i.reshape(n_lon*n_lat)
        
        LAT_vec, LON_vec, _ = get_reshaped_grid(lat_vec, lon_vec)
        Pq = np.array((LAT_vec, LON_vec)).T
        Z_interp = griddata(P_i, Z_i, Pq, method="linear")  # scipy interpolate
        field_hist_newframe[:,:,i] = Z_interp.reshape((n_lat, n_lon), order='F')  # seems to be right ...
        
    return field_hist_newframe


def field_hist_rotation_multiproc(field_hist, R_hist, grid: Grid, transpose_R=False, n_cores=4):
    
    field_hist = grid.vectorize_if_needed(field_hist) # check done: (field_hist_vec[:,0].reshape((180,360))==field_hist[:,:,0]).all()
    n_points, n_steps = np.shape(field_hist)

    assert(np.shape(R_hist)[0]==n_steps)
    print(f"Rotating field history to Sun-Fixed frame with {n_cores} cores...")

    if transpose_R:
        R_hist = np.transpose(R_hist, (0,2,1))

    args_list = [(field_hist[:,i], R_hist[i,:,:]) for i in range(n_steps)]
    with Pool(processes=n_cores, initializer=init_worker, initargs=(grid,)) as pool:
        results = pool.starmap(rotate_field, args_list)

    return grid.reshape_if_needed(np.stack(results, axis=1))


def field_hist_rotation_multiproc_old(field_hist, R_hist, transpose_R=False, n_cores=4):

    n_steps = np.shape(field_hist)[2]
    n_lat, n_lon = np.shape(field_hist)[:2]
    lon_vec, lat_vec, lon_edges_vec, lat_edges_vec = get_sphere_grid(n_lon, n_lat)
    
    grid_r_el = lonlat_to_r(lon_vec, lat_vec, 0, "spherical")
    stacked_grid_r_el = np.reshape(grid_r_el, (n_lon*n_lat, 3))
    stacked_grid_u_el = stacked_grid_r_el / (constants.earth_radius(units='km') * np.ones((n_lon*n_lat,1)))
    
    assert(np.shape(R_hist)[0]==n_steps)
    print(f"Rotating field history to Sun-Fixed frame with {n_cores} cores...")

    if transpose_R:
        R_hist = np.transpose(R_hist, (0,2,1))
        R_hist_2 = np.array([np.transpose(R_hist[i,:,:]) for i in range(n_steps)])
        assert((R_hist_2 == R_hist).all())

    args_list = [(field_hist[:,:,i], R_hist[i,:,:]) for i in range(n_steps)]
    with Pool(processes=n_cores, initializer=init_worker, initargs=(stacked_grid_u_el, lon_vec, lat_vec)) as pool:
        results = pool.starmap(rotate_field, args_list)
    
    #return np.stack(results, axis=2)




_global = {}

def init_worker_old(stacked_grid_u_el, lon_vec, lat_vec):

    _global['stacked_grid_u_el'] = stacked_grid_u_el
    _global['lon_vec'] = lon_vec
    _global['lat_vec'] = lat_vec

def init_worker(grid: Grid):
    _global['stacked_grid_u_el'] = grid.stacked_grid_u
    _global['stacked_grid_latlon'] = grid.stacked_grid_latlon


def rotate_field(field, R_matrix):

    stacked_grid_u_el = _global['stacked_grid_u_el']
    stacked_grid_latlon = _global['stacked_grid_latlon']

    grid_u_el_rot = (R_matrix @ stacked_grid_u_el.T).T
    _, lat_rot, lon_rot = cartesian_to_spherical(grid_u_el_rot[:,0], grid_u_el_rot[:,1], grid_u_el_rot[:,2])  # astropy
    
    P_i = np.array((lat_rot.degree, lon_rot.degree)).T

    return sphere_field_interp(P_i, field, stacked_grid_latlon)

    #LAT_vec, LON_vec, _ = get_reshaped_grid(lat_vec, lon_vec)
    #Pq = np.array((LAT_vec, LON_vec)).T
    #Z_interp = griddata(P_i, Z_i, Pq, method="linear")  # scipy interpolate
#
    #return Z_interp.reshape((n_lat, n_lon), order='F') 


def rotate_field_old(field, R_matrix):

    stacked_grid_u_el = _global['stacked_grid_u_el']
    lon_vec = _global['lon_vec']
    lat_vec = _global['lat_vec']
    n_lon, n_lat = len(lon_vec), len(lat_vec)

    grid_u_el_rot = (R_matrix @ stacked_grid_u_el.T).T

    _, lat_rot, lon_rot = cartesian_to_spherical(grid_u_el_rot[:,0], grid_u_el_rot[:,1], grid_u_el_rot[:,2])  # astropy
    
    P_i = np.array((lat_rot.degree, lon_rot.degree)).T
    Z_i = field.reshape(n_lon*n_lat)

    LAT_vec, LON_vec, _ = get_reshaped_grid(lat_vec, lon_vec)
    Pq = np.array((LAT_vec, LON_vec)).T
    Z_interp = griddata(P_i, Z_i, Pq, method="linear")  # scipy interpolate

    return Z_interp.reshape((n_lat, n_lon), order='F') 




def interp_zeroes_in_grid_data(data_array, grid: Grid, method='nearest'):
    # data array comes in a single array with as many points as the grid

    zero_idxs = np.where(np.abs(data_array)<1e-9)[0]
    nonzero_idxs = np.where(np.abs(data_array)>=1e-9)[0]

    if len(zero_idxs)==0:
        return data_array

    if len(zero_idxs)>0:
        data_array[zero_idxs] = sphere_field_interp(grid.stacked_grid_latlon[nonzero_idxs, :],
                                                    data_array[nonzero_idxs],
                                                    grid.stacked_grid_latlon[zero_idxs, :],
                                                    method='griddata_'+method) #griddata(P_i, Z_i, Pq, method=method)
        return data_array

# separate file?

def ensure_0_to_360(lon_vec):
    return lon_vec % 360

def ensure_180_to_180(lon_vec):
    lon_vec[lon_vec>180] = lon_vec[lon_vec>180]-360
    return lon_vec

def lat_to_colat_deg(lat_vec):
    return (90-lat_vec)



if __name__=="__main__":

    grid = RegularLatLonGrid(alt_km=0, n_lat=180, n_lon=360)
    grid2 = QuadratureGrid(alt_km=0, order=131)

