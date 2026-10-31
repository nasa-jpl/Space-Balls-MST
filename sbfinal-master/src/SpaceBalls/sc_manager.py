import numpy as np
import os
import pickle
from SpaceBalls.sph_meshing import get_faceted_sphere_mesh
from SpaceBalls.utils import make_dict_hash_key_codex
from SpaceBalls.paths import CONFIG_DIR
import SpaceBalls.input_database_manager as db_manager


def select_solar_panel_facets(mesh, fraction, distribution_mode):
    """Return a Boolean solar-panel mask aligned with mesh[5].simplices.

    mesh is the tuple returned by get_faceted_sphere_mesh. fraction must be
    in [0, 1] and specifies a fraction of facets, not surface area; the count
    is rounded to the nearest integer (half up). distribution_mode is
    "random" (uniform sampling without replacement) or "equatorial" (a belt
    around the z axis, selecting centroids closest to the z=0 plane).
    Equatorial ties follow facet order, so the final ring may be partial.
    The mesh is not modified. np.random.seed can make random selection repeatable.
    """
    if not np.isscalar(fraction) or not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("fraction must be a finite number between 0 and 1")
    if distribution_mode not in ("random", "equatorial"):
        raise ValueError("distribution_mode must be 'random' or 'equatorial'")

    _, centroids, _, _, _, hull = mesh
    n_facets = len(hull.simplices)
    n_panels = int(np.floor(fraction * n_facets + 0.5))
    solar_panel_mask = np.zeros(n_facets, dtype=bool)
    if distribution_mode == "random":
        panel_indices = np.random.choice(n_facets, size=n_panels, replace=False)
    else:
        panel_indices = np.argsort(np.abs(centroids[:, 2]), kind="stable")[:n_panels]
    solar_panel_mask[panel_indices] = True
    return solar_panel_mask


def get_sc_dict(mass, mesh, materials, idx_filters, coeff_wl_split):

    #mesh = get_faceted_sphere_mesh(sat_radius, n_faces, default_golden=False)
    vertices, centroids, normal_vecs, areas, edge_conns, hull = mesh
    n_faces = len(areas)

    sat_dict = {'mass': mass,
                'coeff_wl_split': coeff_wl_split,
                'plate_normals': normal_vecs,
                'plate_areas': areas,
                'avg_area': np.sum(areas) / 4,
                'materials': '|'.join(materials)}
    #TODO: if mesh is None, cannonball sat_dict

    sat_dict['material_plate_idx_filter'] = idx_filters

    sat_dict['plate_ca_SW'] = np.zeros(n_faces)
    sat_dict['plate_cd_SW'] = np.zeros(n_faces)
    sat_dict['plate_cs_SW'] = np.zeros(n_faces)

    if coeff_wl_split:
        sat_dict['plate_ca_LW'] = np.zeros(n_faces)
        sat_dict['plate_cd_LW'] = np.zeros(n_faces)
        sat_dict['plate_cs_LW'] = np.zeros(n_faces)

    for material, idx_filter in zip(materials, idx_filters):
        coeff_dict = db_manager.get_sc_coeffs(material, 'GRACE_handbook')

        sat_dict['plate_ca_SW'][idx_filter] = coeff_dict['ca_SW']
        sat_dict['plate_cd_SW'][idx_filter] = coeff_dict['cd_SW']
        sat_dict['plate_cs_SW'][idx_filter] = coeff_dict['cs_SW']

        if coeff_wl_split:
            sat_dict['plate_ca_LW'][idx_filter] = coeff_dict['ca_LW']
            sat_dict['plate_cd_LW'][idx_filter] = coeff_dict['cd_LW']
            sat_dict['plate_cs_LW'][idx_filter] = coeff_dict['cs_LW']

    return sat_dict