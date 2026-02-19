
# import Monte as M
# import mpy.units as units

def earth_radius(model='WGS84', units='km', monte_units=False):
    if model=="WGS84":
        radius_km = 6378.137   # [km]
    else:
        print(f"Model {model} not acknowledged")

    if units=="m":
        radius = radius_km * 1e3
    elif units=="km":
        radius = radius_km
    
    if monte_units:
        import mpy.units as units
        radius = radius_km * units.km

    return radius
    

def earth_flattening(model='WGS84'):
    if model=="WGS84":
        f = 0.00335281066475   # []
    else:
        print(f"Model {model} not acknowledged")
    
    return f


def earth_SOI(units='km', monte_units=False):
    earth_soi_km = 9.24e5

    if units=="m":
        earth_SOI = earth_soi_km * 1e3
    elif units=="km":
        earth_SOI = earth_soi_km
    
    if monte_units:
        import mpy.units as units
        earth_SOI = earth_soi_km * units.km

    return earth_SOI

def light_speed(units='m/s', monte_units=False):
    light_speed_ms = 2.998E8

    if units=="kms" or units=="km/s":
        light_speed = light_speed_ms * 1e-3
    elif units=="ms" or units=="m/s":
        light_speed = light_speed_ms

    if monte_units:
        import mpy.units as units
        light_speed = light_speed_ms * units.meter / units.sec

    return light_speed


def astronomical_unit(units='km', monte_units=False):
    au_km = 149597870.691 

    if units=="m":
        au = au_km * 1e3
    elif units=="km":
        au = au_km

    if monte_units:
        import mpy.units as units
        au = au_km * units.km
    
    return au
