from astropy.coordinates import get_body, ITRS, SkyCoord, CartesianRepresentation, SphericalRepresentation
from astropy.time import Time as Time_astropy
from astropy import units as AP_u
import numpy as np

def find_sun_ecef(monte_epoch):

    epochstring_astropy = monte_epoch_to_astropy_str(monte_epoch)

    # Define UTC time
    time = Time_astropy(epochstring_astropy, scale="utc")

    # Get GCRS (inertial) position of the Sun at given time
    sun_gcrs = get_body("sun", time)

    # Convert to Earth-fixed (ECEF/ITRS)
    sun_ecef = sun_gcrs.transform_to(ITRS(obstime=time))

    sun_ecef.cartesian
    sun_ecef.spherical
    sun_lon = sun_ecef.spherical.lon.value
    sun_lon = sun_lon-360 if sun_lon>180 else sun_lon
    sun_lat = sun_ecef.spherical.lat.value

    r_to_Sun = np.array([sun_ecef.cartesian.x.to(AP_u.km).value, 
                         sun_ecef.cartesian.y.to(AP_u.km).value, 
                         sun_ecef.cartesian.z.to(AP_u.km).value])

    return sun_lon, sun_lat, r_to_Sun


def monte_epoch_to_astropy_str(epoch):
    
    epoch_str = monte_epoch_to_str(epoch)
    epoch_str = epoch_str.replace(' ', 'T')

    return epoch_str


def monte_epoch_to_str(epoch): # TODO: should not be in this file
    epoch_str = str(epoch.date("UTC"))
    return epoch_str


