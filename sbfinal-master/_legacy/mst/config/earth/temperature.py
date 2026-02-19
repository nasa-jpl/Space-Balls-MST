# Defines temperature profiles for Earth


def find_mean_temp(lat): # given the latitude, find temperature in K. Uses a mean temp profile (doy independent)
    # ref: https://www.abeqas.com/global-warming-nothing-to-worry-about/

    if lat <= -83:
        m = 1.857
        b = 394.14

    elif lat <= -78:
        m = 0.
        b = 240.

    elif lat <= -66:
        m = 2.333
        b = 422.

    elif lat <= -25:
        m = 0.634
        b = 309.85

    elif lat <= -6:
        m = 0.316
        b = 301.89

    elif lat <= 17:
        m = 0.
        b = 300.

    elif lat <= 80:
        m = -0.651
        b = 311.06

    else:
        m = 0.
        b = 259.

    T = m * lat + b

    #print("For latitude: ", lat, " the temp in K is: ", T )
    return T


def find_irflux(temp, emissivity): #given temp in K, find IR flux in W/m^2

    sbc = 5.67051e-8
    Wb = emissivity * sbc * temp**4
    #print("IR flux, W/m^2: ", Wb)
    return Wb



if __name__ == "__main__":
    # test functions
    e = 0.98

    lat = -85.
    temp = find_mean_temp(lat)
    Wb = find_irflux(temp,e)

    lat = -80.
    temp = find_mean_temp(lat)
    Wb = find_irflux(temp,e)

    lat = -70.
    temp = find_mean_temp(lat)
    Wb = find_irflux(temp,e)

    lat = -40.
    temp = find_mean_temp(lat)
    Wb = find_irflux(temp,e)

    lat = -10.
    temp = find_mean_temp(lat)
    Wb = find_irflux(temp,e)

    lat = 10.
    temp = find_mean_temp(lat)
    Wb = find_irflux(temp,e)

    lat = 40.
    temp = find_mean_temp(lat)
    Wb = find_irflux(temp,e)

    lat = 85.
    temp = find_mean_temp(lat)
    Wb = find_irflux(temp,e)





