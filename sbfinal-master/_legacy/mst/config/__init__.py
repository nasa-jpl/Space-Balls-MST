import os

# imports from below
#import config.earth.gravity.ggm02c_monte_grv as GGM02C
from .earth.gravity import ggm02c_monte_grv as GGM02C


# dir imports
from .earth import *
#from .coordsys import *
#from .forces import *

# file imports
from .config_default import *
from . import boas
from . import files
from . import time_handling
from . import frames
from . import directions

#from .boas import set_boa


'''
# configuration file
# user can put their own configuration file as "user_config.py" to override
# defaults
USER_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "user_config.py")
if os.path.exists(USER_CONFIG_FILE):
    from .user_config import *
else:
    from .config_default import *

# plot settings file
# user can put their own configuration file as "user_config.py" to override
# defaults
USER_PLOT_FILE = os.path.join(os.path.dirname(__file__),
                              "user_plot_settings.py")
if os.path.exists(USER_PLOT_FILE):
    from .user_plot_settings import *
else:
    from .plot_settings_default import *

# we also need to load the defaults anyways for tests

#from . import plot_settings_default
'''