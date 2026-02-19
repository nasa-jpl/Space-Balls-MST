"""Settings for plots  - placeholder
"""
from hvt.plots import *
from mpylab import plt as plt
import mpy.units as units
import numpy as np

# figure size - for most plots
FIGURE_SIZE = (16/1.5, 9/1.5)

# figure size - for figures that look better as a square (AzEl)
SQUARE_FIGURE_SIZE = (FIGURE_SIZE[1], FIGURE_SIZE[1])

# date formating
# TODO: currently not working; not sure why
PLOT_DATE_FORMAT = "$YYYY-$MM-$DD$UTC"

# interval data epoch
# what epoch to use for data in interval plots
# options 'rise', 'tca', 'set'
INTERVAL_DATA_EPOCH = "tca"

# This is the section to add/remove plots you want to make
# Each entry in this list should have the following arguments
# (<plot class>, <positional arguments to plot class>, <keyword args>)
#
# Mostly, users should plan on adding PlotScatter plots (scatter plots)
# the input form for this is:
#   (PlotScatter, (<x variable>, <y variable>))
# optionally, you can have a color variable:
#   (PlotScatter, (<x variable>, <y variable>), {"cvar": <color variable>})
#
# It is possible to add other plot types than PlotScatter, but that is for the
# advanced user
USERPLOTS = {}
USERPLOTS["TerrestrialTarget"] = [
    # TEA vs SEA
    (PlotScatter, {"xvar": "sea", "yvar": "tea", "cvar": "time_since_epoch"}),
    
    # TEA vs Time
    (PlotScatter, {"xvar": "epoch", "yvar": "tea"}),

    # Azimuth-Elevation
    (PlotAzEl, {"tvar": "taa", "rvar": "tea", "cvar": "sea", "edgecolor": "k",
                "split_interval": 1 * units.day}),

    # LMST vs. Time
    (PlotLmstTime, {"cvar": "tea"}),

    # Perspective Map
    (PlotNadirMap, {"cvar": "tea", "label_target": True, "target_color": "k"}),

    # Ground Track
    (PlotGroundTrack, {"cvar": "tea", "label_target": True, "target_color": "k"}),

    # TEA and SEA vs Time
    (PlotScatterMulti, {"xvar": "epoch", "yvars": ["tea", "sea"]}),

    # Duration vs Time
    (PlotIntervalScatter, {"xvar": "epoch", "yvar": "duration", "cvar": "tea"})
    ]

USERPLOTS["InertialTarget"] = [
     # Duration vs Time
    (PlotIntervalScatter, {"xvar": "epoch", "yvar": "duration"})
]
USERPLOTS["EphemerisTarget"] = [
     # Duration vs Time
    (PlotIntervalScatter, {"xvar": "epoch", "yvar": "duration"})
]


# colorbar options per variable
OPTS = {
    "default": {"range": None,
                "ticks": None,
                "vmin": None,
                "vmax": None,
                "cmap": "plasma"},
    "tea": {"range": (0 * units.deg, 90 * units.deg),
            "ticks": np.arange(0, 91, 15) * units.deg,
            "cmap": "plasma"},
    "sea": {"range": (-90 * units.deg, 90 * units.deg),
            "ticks": np.arange(-90, 91, 15) * units.deg,
            "cmap": "PuOr_r"},

    # these are variables that wrap, so I use hsv colormap
    "lmst": {"range": (0 * units.hour, 24 * units.hour),
             "ticks": np.arange(0, 25, 2) * units.hour,
             "cmap": "hsv"},
    "taa": {"range": (0 * units.deg, 360 * units.deg),
            "ticks": np.arange(0, 360, 30) * units.deg,
            "cmap": "hsv"},
    "saa": {"range": (0 * units.deg, 360 * units.deg),
            "ticks": np.arange(0, 360, 30) * units.deg,
            "cmap": "hsv"},
}