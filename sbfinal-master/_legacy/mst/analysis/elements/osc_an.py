# osculating elemtents scripts, AN related - from original Ted Sweetser SphereX lifetime script

import Monte as M
import mpy.units as units
#from analysis import headers

fixedFrameName = 'IAU Earth Fixed' 
framePoleName = 'IAU Earth Pole'


def find_antimes_period(boa, sc, startEpoch, stopEpoch ): # finds the AN times and avg period for 10 revs

    print ('getting ascNodeTimes')
    # query = TrajQuery(boa,'SphereX','Earth','IAU Earth Pole')
    tqOsc = M.TrajQuery( boa , sc.name ,'Earth' , fixedFrameName )
    ascNode = M.CoordinateEvent(tqOsc, M.Spherical.latitude(),  0.*units.deg ,'INCREASING' ,0.001*units.s )
    ascNodeTimes = ascNode.search( M.TimeInterval( startEpoch, stopEpoch ), 15.*units.sec )
    print("ascNodeTimes: ", ascNodeTimes)
    numrevs = len(ascNodeTimes) - 1
    period = ((ascNodeTimes[numrevs].time()-ascNodeTimes[0].time())/numrevs).seconds()
    if numrevs > 0:
        print ('The avg nodal period over ', numrevs, ' revs is %.3f s.' %( period.value() ) )
        print ('')
    else:
        print('Not enough data to determine the period using averaging methods.')
        
    return ascNodeTimes, period


def osc_elements_ANs_PV( boa, sc, ascNodeTimes, initTime ): # finds states at the ascending nodes

    query = M.TrajQuery( boa, sc.name, 'Earth', framePoleName ) 

    print ('')
    print ('Printing osculating elements at each Ascending Node, Position - Velocity')
    print ('')
    print ( '      Epoch                      Days     X        Y        Z           Vx        Vy        Vz')
    print ('')

    for ascNodeEvent in ascNodeTimes:
        oe = ascNodeEvent.time()
        oscState = query.state(oe,'IAU Earth Pole')
        print( '%s , %6.3f , %.3f , %.3f , %.3f    , %.6f , %.6f  , %.6f' %(oe.format()
                    ,(oe-initTime).convert('days')
                    ,M.Cartesian.x(oscState).convert('km')
                    ,M.Cartesian.y(oscState).convert('km')
                    ,M.Cartesian.z(oscState).convert('km')
                    ,M.Cartesian.dx(oscState).convert('km/s')
                    ,M.Cartesian.dy(oscState).convert('km/s')
                    ,M.Cartesian.dz(oscState).convert('km/s')
                    ) )
        print ('')


def filter_someANtimes( ascNodeTimes, period, dayInterval = 5, daySeconds = 86400 ):  # filters ascending node times to sample near an integral number of days for sampling freq.
    # note: dayInterval needs to be > 0  and periond non-zero
    if period.value() > 0:
        orbsPerDay = round( daySeconds / period.value() ) # 15 is typical for Earth LEO sc
        #print('orbits per day: ', orbsPerDay)
        trajQuasiDays = len(ascNodeTimes)/orbsPerDay   # a QuasiDay is 15 orbits (a bit more than a day above 560 km altitude)
        ascNodeSomeTimes = [ascNodeTimes[dayInterval*orbsPerDay*i] for i in range(int(trajQuasiDays/dayInterval))]
    else:
        ascNodeSomeTimes = [ascNodeTimes[0]]
        print('Only one ascending node time found')
    return ascNodeSomeTimes


def find_someDailyANtimes( boa, sc, startEpoch, stopEpoch, dayInterval ):  # finds a set of AN times for every # of days spec. by dayInterval

    ascNodeTimes, period = find_antimes_period(boa, sc, startEpoch, stopEpoch )
    ascNodeSomeTimes = filter_someANtimes( ascNodeTimes, period )

    return ascNodeSomeTimes


def someAN_osc_elements_latlongV( boa, sc, th, ascNodeSomeTimes, top=4):  #output osc elements, long lat, V, Filtered by ascNodeSomeTimes

    query = M.TrajQuery( boa, sc.name, 'Earth', framePoleName ) 

    print ('')
    print ('In IAU Earth coordinates:' )
    print ('      Epoch                     Days     SMA        Ecc      ArgPeri	  Incl         ArgLat     Lat       Lon      Velocity    MLTAN    LongOfNode')
    # oscEpochsInit = [ initTime ,initTime + 0.25 *s ]
    # oscEpochs = oscEpochsInit + Epoch.range( initTime, endTime, 30.*hour )[1:]
    # for oe in oscEpochs:

    for ascNodeEvent in ascNodeSomeTimes[:top]:
        oe = ascNodeEvent.time()
        oscState = query.state(oe,'IAU Earth Pole')
        oscStateEarth = query.state(oe,'IAU Earth Fixed')
        print( '%s  %6.1f  %.3f  %.6f  %8.2f     %8.5f  %9.4f   %9.4f  %9.4f  %11.4f %.3f %.3f' %(oe.format()
                    ,(oe-th.start_epoch).convert('days')
                    ,M.Conic.semiMajorAxis(oscState).convert('km')
                    ,M.Conic.eccentricity(oscState)
                    ,M.Conic.argumentOfPeriapsis(oscState).convert('deg')
                    ,M.Conic.inclination(oscState).convert('deg')
                    ,M.Conic.argumentOfLatitude(oscState).convert('deg')
                    ,M.Spherical.latitude(oscStateEarth).convert('deg')
                    ,M.Spherical.longitude(oscStateEarth).convert('deg')
                    ,M.Spherical.velocity(oscState).convert('m/s')
                    ,M.Conic.meanLocalTimeAtNode(oscState).convert('hours')
                    ,M.Conic.longitudeOfNode(oscState).convert('deg')
                    ) )
    print ('')


def someAN_PV(boa, sc, th, ascNodeSomeTimes, top=4):

    query = M.TrajQuery( boa, sc.name, 'Earth', framePoleName ) 

    print ('Printing osculating elements at some acending nodes: (EME2000)')
    print ( '      Epoch                      Days     X        Y        Z           Vx        Vy        Vz')
    count = 0
    for ascNodeEvent in ascNodeSomeTimes:
        oe = ascNodeEvent.time()
        oscState = query.state(oe,'EME2000') #IAU Earth Pole')
        #oscStateEarth = query.state(oe,'IAU Earth Fixed')
        print( '%s  %6.1f  %.3f  %.3f  %.3f     %.6f  %.6f   %.6f' %(oe.format()
                    ,(oe-th.start_epoch).convert('days')
                    ,M.Cartesian.x(oscState).convert('km')
                    ,M.Cartesian.y(oscState).convert('km')
                    ,M.Cartesian.z(oscState).convert('km')
                    ,M.Cartesian.dx(oscState).convert('km/s')
                    ,M.Cartesian.dy(oscState).convert('km/s')
                    ,M.Cartesian.dz(oscState).convert('km/s')
                    ) )
        count = count+1             
        if count == top:
            break
    print ('')
    print ('')


