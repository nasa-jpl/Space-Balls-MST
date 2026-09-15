from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations
from pathlib import Path
import re

import numpy as np

import sys, os
from scipy.interpolate import make_interp_spline
from SpaceBalls.sph_meshing import Grid, RegularLatLonGrid
from SpaceBalls.paths import CONFIG_DIR
from SpaceBalls.utils import jd_to_mmddyyyy, get_jd_to_build_interp

REG_1DEG_GRID = RegularLatLonGrid(0, n_lat=180, n_lon=360)
IGBP_CLASSES = np.load(os.path.join(CONFIG_DIR, 'earth', 'ADMs',
                                    'IGBP_land_class', 'IGBPa_1198.npy'))



# ERBE SW ADM bin upper bounds, in degrees.
SZA_UPPER = np.array(
    [25.84, 36.87, 45.57, 53.13, 60.00,
     66.42, 72.54, 78.46, 84.26, 90.00]
)
VZA_UPPER = np.array([15., 27., 39., 51., 63., 75., 90.])
RAA_UPPER = np.array([9., 30., 60., 90., 120., 150., 171., 180.])

_LUT_SHAPE = (12, 10, 7, 8)  # scene, SZA, VZA, RAA
_N_FACTORS = int(np.prod(_LUT_SHAPE))
_FLOAT = re.compile(
    r"(?<![A-Za-z0-9_.])[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][+-]?\d+)?"
)


def _centres(upper):
    return (np.r_[0.0, upper[:-1]] + upper) / 2.0


SZA_CENTRES = _centres(SZA_UPPER)
VZA_CENTRES = _centres(VZA_UPPER)
RAA_CENTRES = _centres(RAA_UPPER)


def _numbers(line: str) -> list[float]:
    return [
        float(token.replace("D", "E").replace("d", "e"))
        for token in _FLOAT.findall(line)
    ]


def _valid_index_column(values, size, base):
    rounded = np.rint(values)
    return (
        np.all(np.abs(values - rounded) < 1e-10)
        and np.all((rounded >= base) & (rounded < base + size))
    )


def _parse_indexed_records(rows):
    """Parse five-column records in any index-column order."""
    if rows.shape != (_N_FACTORS, 5) or not np.all(np.isfinite(rows)):
        return None

    for factor_col in range(5):
        index_cols = [col for col in range(5) if col != factor_col]

        for order in permutations(index_cols):
            raw = rows[:, order]

            for base in (0, 1):
                if not all(
                    _valid_index_column(raw[:, axis], _LUT_SHAPE[axis], base)
                    for axis in range(4)
                ):
                    continue

                indices = np.rint(raw).astype(np.intp) - base
                flat = np.ravel_multi_index(indices.T, _LUT_SHAPE)

                if np.unique(flat).size != _N_FACTORS:
                    continue

                values = np.empty(_N_FACTORS, dtype=np.float64)
                values[flat] = rows[:, factor_col]
                return values.reshape(_LUT_SHAPE)

    return None


def _parse_labelled_table(lines, table_path):
    """Read the bundled scene/SZA blocks without treating labels as factors."""
    values = np.empty(_LUT_SHAPE, dtype=np.float64)
    scene = sza = -1
    vza = 7
    row_pattern = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s+(.+)$")
    azimuth_bounds = np.column_stack((np.r_[0, RAA_UPPER[:-1]], RAA_UPPER)).ravel()

    for lineno, line in enumerate(lines, 1):
        stripped = line.strip()
        try:
            if stripped.startswith("Scene "):
                if scene >= 0 and (sza != 9 or vza != 7):
                    raise ValueError("incomplete scene")
                scene += 1
                if scene >= 12 or int(stripped.split()[1]) != scene + 1:
                    raise ValueError("unexpected scene ID or order")
                sza = -1
            elif stripped.startswith("solar zenith angle range"):
                sza += 1
                bounds = _numbers(line.split("=", 1)[1])
                expected = [0.0 if sza == 0 else SZA_UPPER[sza - 1],
                            SZA_UPPER[sza]] if sza < 10 else None
                if scene < 0 or vza != 7 or bounds != expected:
                    raise ValueError("unexpected SZA bounds or incomplete VZA block")
                vza = 0
            elif (match := row_pattern.match(line)) is not None:
                lower, upper = map(int, match.group(1, 2))
                if (lower, upper) == (0, 9):
                    if not np.array_equal(_numbers(line), azimuth_bounds):
                        raise ValueError("unexpected RAA column bounds")
                    continue
                expected_lower = 0.0 if vza == 0 else VZA_UPPER[vza - 1]
                if (scene < 0 or sza < 0 or vza >= 7
                        or lower != expected_lower or upper != VZA_UPPER[vza]):
                    raise ValueError("unexpected VZA bounds or row order")
                factors = [float(token.replace("D", "E").replace("d", "e"))
                           for token in match.group(3).split()]
                if len(factors) != 8:
                    raise ValueError("expected eight RAA factors")
                values[scene, sza, vza] = factors
                vza += 1
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Invalid ERBE ADM table {table_path}, line {lineno}: {exc}") from exc

    if (scene, sza, vza) != (11, 9, 7):
        raise ValueError(f"Incomplete ERBE ADM table in {table_path}.")
    return values


def _parse_table(table_path: Path) -> np.ndarray:
    """
    Read common ERBE table layouts:

    - labelled scene/SZA blocks with VZA bounds and eight RAA factors per row
    - five-column records: four integer indices plus factor
    - 840 compact rows, each containing the eight RAA factors
    - a plain stream of exactly 6720 decimal factors
    """
    text = table_path.read_text(encoding="ascii", errors="replace")
    lines = text.splitlines()
    if "solar zenith angle range" in text:
        return _parse_labelled_table(lines, table_path)
    rows = [_numbers(line) for line in lines]

    five_col = [row for row in rows if len(row) == 5]
    if five_col:
        lut = _parse_indexed_records(np.asarray(five_col, dtype=float))
        if lut is not None:
            return lut

    eight_value_rows = []
    for line, row in zip(lines, rows, strict=True):
        tokens = _FLOAT.findall(line)
        if (
            len(row) == 8
            and len(tokens) == 8
            and all("." in token or "e" in token.lower() or "d" in token.lower()
                    for token in tokens)
        ):
            eight_value_rows.append(row)

    if len(eight_value_rows) == _N_FACTORS // 8:
        return np.asarray(eight_value_rows, dtype=float).reshape(_LUT_SHAPE)

    decimal_values = [
        float(token.replace("D", "E").replace("d", "e"))
        for line in lines
        for token in _FLOAT.findall(line)
        if "." in token or "e" in token.lower() or "d" in token.lower()
    ]
    if len(decimal_values) == _N_FACTORS:
        return np.asarray(decimal_values, dtype=np.float64).reshape(_LUT_SHAPE)

    raise ValueError(
        f"Unrecognized ERBE ADM layout in {table_path}. "
        f"Expected {_N_FACTORS} factors."
    )


def _brackets(values, centres):
    """Return lower/upper indices and interpolation weights."""
    values = np.clip(values, centres[0], centres[-1])
    hi = np.searchsorted(centres, values, side="right")
    hi = np.clip(hi, 1, len(centres) - 1).astype(np.intp)
    lo = hi - 1
    weight = (values - centres[lo]) / (centres[hi] - centres[lo])
    return lo, hi, weight


@dataclass(frozen=True)
class ERBEShortwaveADM:
    values: np.ndarray

    def __post_init__(self):
        values = np.array(self.values, dtype=np.float64, copy=True, order="C")
        if values.shape != _LUT_SHAPE:
            raise ValueError(f"Expected table shape {_LUT_SHAPE}; got {values.shape}.")
        # The bundled table has strictly positive factors (minimum 0.410).
        # Zero angle bounds in its headings are not anisotropic factors.
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("ERBE ADM factors must be finite and strictly positive.")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)

    def factors(
        self,
        solar_zenith_angles,
        viewing_zenith_angles,
        relative_azimuth_angles,
        scene_types,
        *,
        method="nearest",
        scene_index_base=1,
        out_of_range="clip",
    ):
        """
        Return anisotropic factors for dense grids or selected cells.

        Dense inputs: SZA/scenes have shape (n_toa, nt), and VZA/RAA have
        shape (np, n_toa, nt). Output has shape (np, n_toa, nt).
        Selected inputs: all four arrays have shape (n_cells,). Output has
        shape (n_cells,), including when n_cells is zero.

        method:
          - "nearest": selects the containing ERBE angular bin.
          - "linear": trilinear interpolation over the three angle axes.

        ERBE scene IDs are 1–12 by default. Set scene_index_base=0 for 0–11.
        Angles are in degrees. Exact bin upper bounds belong to the lower bin.
        Linear interpolation clamps to the outer bin centres. Clipping zenith
        angles does not mask night-side or nonvisible cells; mask those upstream.
        """
        if method not in ("nearest", "linear"):
            raise ValueError("method must be 'nearest' or 'linear'.")
        if scene_index_base not in (0, 1):
            raise ValueError("scene_index_base must be 0 or 1.")
        sza = np.asarray(solar_zenith_angles, dtype=float)
        vza = np.asarray(viewing_zenith_angles, dtype=float)
        raa = np.asarray(relative_azimuth_angles, dtype=float)
        scenes = np.asarray(scene_types)

        if sza.ndim == 1:
            if not (sza.shape == vza.shape == raa.shape == scenes.shape):
                raise ValueError("Selected-cell inputs must all have shape (n_cells,).")
        else:
            if sza.ndim != 2 or scenes.shape != sza.shape:
                raise ValueError("SZA and scene_types must both have shape (n_toa, nt).")
            if vza.ndim != 3 or vza.shape != (vza.shape[0],) + sza.shape:
                raise ValueError("VZA must have shape (np, n_toa, nt).")
            if raa.shape != vza.shape:
                raise ValueError("RAA must have the same shape as VZA.")

        if (not np.all(np.isfinite(sza)) or not np.all(np.isfinite(vza))
                or not np.all(np.isfinite(raa))):
            raise ValueError("Angles must be finite.")

        if out_of_range == "clip":
            sza = np.clip(sza, 0.0, 90.0)
            vza = np.clip(vza, 0.0, 90.0)
        elif out_of_range == "raise":
            if np.any((sza < 0) | (sza > 90)) or np.any((vza < 0) | (vza > 90)):
                raise ValueError("SZA and VZA must lie in [0, 90].")
        else:
            raise ValueError("out_of_range must be 'clip' or 'raise'.")

        # Supports signed RAA or conventional [0, 360] relative azimuth.
        raa = np.remainder(raa, 360.0)
        np.subtract(raa, 360.0, out=raa, where=raa > 180.0)
        np.abs(raa, out=raa)

        if (not np.all(np.isfinite(scenes))
                or not np.all(np.equal(scenes, np.rint(scenes)))):
            raise ValueError("scene_types must be integer-valued.")

        if np.any((scenes < scene_index_base) | (scenes >= scene_index_base + 12)):
            raise ValueError("scene_types are outside the supported ERBE range.")
        scene_i = scenes.astype(np.intp) - int(scene_index_base)

        # Dense SZA/scenes broadcast across observers; selected-cell arrays
        # index the table elementwise using the same lookup/interpolation code.
        if method == "nearest":
            sza_i = np.searchsorted(SZA_UPPER, sza, side="left")
            vza_i = np.searchsorted(VZA_UPPER, vza, side="left")
            raa_i = np.searchsorted(RAA_UPPER, raa, side="left")
            return self.values[scene_i, sza_i, vza_i, raa_i]

        s0, s1, ws = _brackets(sza, SZA_CENTRES)
        v0, v1, wv = _brackets(vza, VZA_CENTRES)
        a0, a1, wa = _brackets(raa, RAA_CENTRES)

        def blend_azimuth(s, v):
            lower = self.values[scene_i, s, v, a0]
            upper = self.values[scene_i, s, v, a1]
            upper -= lower
            upper *= wa
            upper += lower
            return upper

        def blend_view(s):
            lower = blend_azimuth(s, v0)
            upper = blend_azimuth(s, v1)
            upper -= lower
            upper *= wv
            upper += lower
            return upper

        # Release corner arrays as we go; only a few full-size factor arrays
        # are live at once, rather than all eight corners plus intermediates.
        lower = blend_view(s0)
        upper = blend_view(s1)
        upper -= lower
        upper *= ws
        upper += lower
        return upper


@lru_cache(maxsize=None)
def _load_cached(resolved_path: str) -> ERBEShortwaveADM:
    return ERBEShortwaveADM(_parse_table(Path(resolved_path)))


def load_erbe_sw_adm(table_path: str | Path) -> ERBEShortwaveADM:
    """Load and parse a table only once per canonical path."""
    return _load_cached(str(Path(table_path).expanduser().resolve()))


def anisotropic_factors(
    solar_zenith_angles,
    viewing_zenith_angles,
    relative_azimuth_angles,
    scene_types,
    *,
    table_path,
    method="nearest",
    scene_index_base=1,
    out_of_range="clip",
):
    """Cached convenience wrapper."""
    return load_erbe_sw_adm(table_path).factors(
        solar_zenith_angles,
        viewing_zenith_angles,
        relative_azimuth_angles,
        scene_types,
        method=method,
        scene_index_base=scene_index_base,
        out_of_range=out_of_range,
    )



def map_igbp_to_erbe(igbp_classes):
    """Map CERES IGBP classes 1–18 to clear-sky ERBE scene IDs.

    Water (17) maps to ocean (1), snow/ice (15) to snow (3), and
    barren/sparsely vegetated (16) to desert (4). All other classes,
    including wetlands (11) and tundra (18), map to land (2).
    Class definitions: https://ceres.larc.nasa.gov/data/general-product-info/

    The dominant-class IGBP map does not provide land/ocean fractions,
    so it cannot identify coastal ERBE scenes (5). Input shape is preserved.
    """
    classes = np.asarray(igbp_classes)
    if (not np.all(np.isfinite(classes))
            or not np.all(classes == np.rint(classes))
            or np.any((classes < 1) | (classes > 18))):
        raise ValueError("CERES IGBP classes must be integers in [1, 18].")

    mapping = np.full(19, 2, dtype=np.intp)
    mapping[15] = 3
    mapping[16] = 4
    mapping[17] = 1
    return mapping[classes.astype(np.intp)]


def get_erbe_scene_types(jd_array, toa_grid: Grid, rad_config: dict):
    """ from the readme files
    ==========================
    ERBE Scene Type Definition
    ==========================

    Scene Type Bin                      Scene                   Cloud Fraction Range
        1              Clear Ocean                                0.00 - 0.05
        2              Clear Land                                 0.00 - 0.05
        3              Clear Snow                                 0.00 - 0.05
        4              Clear Desert                               0.00 - 0.05
        5              Clear Land-Ocean Mix (Coastal)             0.00 - 0.05
        6              Partly Cloudy Over Ocean                   0.05 - 0.50
        7              Partly Cloudy Over Land or Desert          0.05 - 0.50 
        8              Partly Cloudy Over Land-Ocean Mix          0.05 - 0.50
        9              Mostly Cloudy Over Ocean                   0.50 - 0.95
        10              Mostly Cloudy Over Land or Desert          0.50 - 0.95
        11              Mostly Cloudy Over Land-Ocean Mix          0.50 - 0.95
        12              Overcast                                   0.95 - 1.00

    Returns integer scene IDs with shape (toa_grid.n_points, len(jd_array)).
    Exact cloud-fraction boundaries belong to the lower-cloud bin: clear
    through 0.05, partly cloudy through 0.50, mostly cloudy through 0.95.
    Cloudy snow uses the cloudy land bins, and overcast applies to all surfaces.
    The CERES cloud loader returns percentages; spline overshoot is clipped
    to [0, 100] before conversion to fractions. Nonfinite values are rejected.
    """

    jd_array = np.atleast_1d(np.asarray(jd_array, dtype=float))
    if jd_array.ndim != 1 or not np.all(np.isfinite(jd_array)):
        raise ValueError("jd_array must be a one-dimensional array of finite dates.")
    if jd_array.size == 0:
        return np.empty((toa_grid.n_points, 0), dtype=np.intp)

    # Preserve categorical IGBP values when mapping onto the TOA grid.
    igbp_classes = REG_1DEG_GRID.map_field_to_different_grid(IGBP_CLASSES, toa_grid,
                                                             method='griddata_nearest')
    surface_types = map_igbp_to_erbe(igbp_classes)
    if surface_types.shape != (toa_grid.n_points,):
        raise ValueError("Mapped IGBP classes must have shape (n_toa,).")

    cloud_fractions = np.asarray(
        get_ceres_cloud_fractions(jd_array, toa_grid, rad_config), dtype=float
    )
    if cloud_fractions.shape != (toa_grid.n_points, jd_array.size):
        raise ValueError("CERES cloud percentages must have shape (n_toa, nt).")
    if not np.all(np.isfinite(cloud_fractions)):
        raise ValueError("CERES cloud percentages must be finite.")
    #cloud_fractions = np.clip(cloud_percentages, 0.0, 100.0) / 100.0
    cloud_bins = np.searchsorted([0.05, 0.50, 0.95], cloud_fractions, side='left')
    
    # Rows: ocean, land, snow, desert, coast. Columns: increasing cloud cover.
    scene_ids = np.array([
        [1, 6, 9, 12],
        [2, 7, 10, 12],
        [3, 7, 10, 12],
        [4, 7, 10, 12],
        [5, 8, 11, 12],
    ], dtype=np.intp)

    return scene_ids[surface_types[:, None] - 1, cloud_bins]
    

def get_ceres_cloud_fractions(out_jd_array, toa_grid: Grid, rad_config: dict):
    # TODO: check that this works

    n = 2 # number of days to load to built interpolator
    jd_interp = get_jd_to_build_interp(n, out_jd_array, rad_config["jd_interval"])

    all_cloud_fracs = [None] * len(jd_interp)
    for i, jd in enumerate(jd_interp):
        datestr = jd_to_mmddyyyy(jd)
        all_cloud_fracs[i] = np.load(os.path.join(CONFIG_DIR, 'earth', 
                                    'ADMs', 'CERES_cloud_fractions', datestr + '.npy'))
    spline_cloud_fracs = make_interp_spline(jd_interp, np.stack(all_cloud_fracs), k=1)

    cloud_fracs_reg_grid = spline_cloud_fracs(out_jd_array, extrapolate=False)
    cloud_fracs = REG_1DEG_GRID.map_field_to_different_grid(cloud_fracs_reg_grid, toa_grid)

    return (cloud_fracs/100)
