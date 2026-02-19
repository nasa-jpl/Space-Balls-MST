import sys, os
import numpy as np
from scipy.spatial import ConvexHull
from scipy.interpolate import griddata
from astropy.coordinates import get_body, ITRS, SkyCoord, CartesianRepresentation
from astropy.coordinates import spherical_to_cartesian, cartesian_to_spherical

from SpaceBalls.paths import CONFIG_DIR, MEDIA_DIR
from SpaceBalls.utils import get_two_perp_unit_vectors, progress_bar
sys.path.insert(0, str(CONFIG_DIR.parent)) 
import config.constants as constants



class SphereSurfaceMesh:

    def __init__(self, type):
        pass


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


def get_stacked_spherical_grid_els(lon_vec, lat_vec, altitude_km, total_R):
            n_lon = len(lon_vec)
            n_lat = len(lat_vec)

            grid_r_el = lonlat_to_r(lon_vec, lat_vec, altitude_km, "spherical")
            stacked_grid_r_el = np.reshape(grid_r_el, (n_lon*n_lat, 3))
            stacked_grid_u_el = stacked_grid_r_el / ( total_R * np.ones((n_lon*n_lat,1)))

            return stacked_grid_r_el, stacked_grid_u_el





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

    return normal_vecs, areas



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
        angles = [np.arccos(np.dot(normal, vec_to_centroid)/(np.linalg.norm(normal)*vec_to_centroid_norm)) for normal in face_normal_candidates]

        # the true "outwards" normal vector is the one that forms a smallest angle with the origin-to-facecentroid vector
        j = angles.index(np.min(angles))
        normal = face_normal_candidates[j]

        all_normals[i, :] = normal
        all_areas[i] = area

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




def find_tangent_cone_x_Earth_loci(r_vec, n=250, endpoint_bool=True, shift_deg=0):

    shift_rad = np.deg2rad(shift_deg)

    # assume spherical Earth
    Re = constants.earth_radius()     # [km]

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

    # transform set of points to Earth coordinates:
    cart = CartesianRepresentation(x=all_x[0, :], y=all_x[1, :], z=all_x[2, :])
    coord = ITRS(cart)
    lon = coord.spherical.lon.to_value()
    lon = np.where(lon > 180, lon - 360, lon)
    lat = coord.spherical.lat.to_value()

    lonlats = np.transpose(np.vstack((lon, lat)))
    #lonlats = lonlats[np.argsort(lonlats[:, 0])]

    return lonlats



def get_monte_mesh(r_vec, n_rings):

    cos_phi_vec = get_cos_phi_rings(r_vec, n_rings)
    Re = constants.earth_radius()     # [km]
    r_vec_norm = np.linalg.norm(r_vec)
    all_ring_nodes = [None] * n_rings
    all_ring_connects = [None] * n_rings
    all_ring_centers = [None] * (n_rings + 1)           # includes central cap
    all_element_areas_at_ring = [None] * (n_rings + 1)  # includes central cap

    cap_area = ring_area = 2*np.pi*(1-cos_phi_vec[0]) * Re**2
    all_element_areas_at_ring[0] = cap_area
    #print(f"Central cap area: {cap_area}")

    r_vec_aux = Re * r_vec/r_vec_norm
    cap_center = find_tangent_cone_x_Earth_loci(r_vec_aux, 1, endpoint_bool=False)
    all_ring_centers[0] = cap_center
    

    for i in range(n_rings):
        n_sections = 6*(i + 1)
        cos_phi_up = cos_phi_vec[i]
        cos_phi_down = cos_phi_vec[i+1]
        ring_area = 2*np.pi*(cos_phi_up-cos_phi_down) * (Re)**2
        element_area = ring_area / n_sections
        #print(f"Element areas at ring {i}: {element_area}")
        all_element_areas_at_ring[i+1] = element_area  # [km^2]
        
        r_vec_aux = Re/cos_phi_vec[i] * r_vec/r_vec_norm
        ring_i_upper_bound = find_tangent_cone_x_Earth_loci(r_vec_aux, n_sections, endpoint_bool=False)

        r_vec_aux = Re/(np.mean([cos_phi_vec[i], cos_phi_vec[i+1]])) * r_vec/r_vec_norm
        ring_i_element_centers = find_tangent_cone_x_Earth_loci(r_vec_aux, n_sections, endpoint_bool=False, shift_deg=360/(2*n_sections))

        r_vec_aux = Re/cos_phi_vec[i+1] * r_vec/r_vec_norm
        ring_i_lower_bound = find_tangent_cone_x_Earth_loci(r_vec_aux, n_sections, endpoint_bool=False)

        ring_i_node_matrix = np.vstack((ring_i_upper_bound, ring_i_lower_bound))
        #ring_i_connectivity_matrix = np.array([[j, (j+1)%(n_sections), (n_sections+j+1)%(2*n_sections), (n_sections+j)] for j in range(n_sections)])
        ring_i_connectivity_matrix = np.array([[j, (j+1)%(n_sections), n_sections+(j+1)%(n_sections), n_sections+j] for j in range(n_sections)])

        all_ring_nodes[i] = ring_i_node_matrix
        all_ring_connects[i] = ring_i_connectivity_matrix
        all_ring_centers[i+1] = ring_i_element_centers

    return all_ring_nodes, all_ring_connects, all_ring_centers, all_element_areas_at_ring


def get_cos_phi_rings(r_vec, n_rings):

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
    
    cos_beta = constants.earth_radius()/np.linalg.norm(r_vec)
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

def expand_sh_grid(sh_map, lon_vec, lat_vec):
    
    import pyshtools as pysh
    
    #n = len(lon_vec)*len(lat_vec)
    #(lon_mesh, lat_mesh) = np.meshgrid(lon_vec, lat_vec)
    lon_mesh_vectd, lat_mesh_vectd, mesh_shape = get_reshaped_grid(lon_vec,lat_vec)
        
    grid = pysh.SHCoeffs.from_array(sh_map, normalization='unnorm').expand(
                                        lon=lon_mesh_vectd, lat=lat_mesh_vectd)
    
    grid = np.reshape(grid, mesh_shape)
    
    return grid




def field_hist_rotation(field_hist, R_hist):
    
    n_steps = np.shape(field_hist)[2]
    n_lat, n_lon = np.shape(field_hist)[:2]
    lon_vec, lat_vec, lon_edges_vec, lat_edges_vec = get_sphere_grid(n_lon, n_lat)
    
    grid_r_el = lonlat_to_r(lon_vec, lat_vec, 0, "spherical")
    stacked_grid_r_el = np.reshape(grid_r_el, (n_lon*n_lat, 3))
    stacked_grid_u_el = stacked_grid_r_el / (constants.earth_radius(units='km') * np.ones((n_lon*n_lat,1)))
    

    assert(np.shape(R_hist)[0]==n_steps)
    field_hist_newframe = np.zeros((n_lat, n_lon, n_steps))
    
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
        field_hist_newframe[:,:,i] = Z_interp.reshape((n_lat, n_lon), order='F')  # seems to be right ...
        
    return field_hist_newframe




# separate file?

def ensure_0_to_360(lon_vec):
    return lon_vec % 360



