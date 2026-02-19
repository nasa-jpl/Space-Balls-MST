import Monte as M
import mpy.units as units
import math

from .sensordef import Sensor


def set_sensor( sensorname, boa, frame, scname="Spaceball"):

    #frame = M.frame(nadir)
    fov = 180.0 *units.deg
    apRadius = 3.0*units.m
    print('Circular Sensor: name, vehicle name, frame, fov, apRadius:', sensorname, scname, frame, fov, apRadius)

    sen = Sensor(sensorname, frame, scname)
    circsensor = sen.init_circSensor(boa, fov, apRadius)

    return circsensor

