# checks albedo values given SH files - single long - lat

import Monte as M
import mpy.units as units
#from ... import set_boa

import math
sqrt=math.sqrt
sin=math.sin
cos=math.cos
pi=math.pi

def set_SunDirECI(boa): # finds the sun direction and creates a direction object relative to ECI (EME2000) - moves with time
    coordFrame = "EME2000"
    suntqEME2000 = M.TrajQuery(boa ,'Sun' ,'Earth' , coordFrame)
    sundir = M.PositionDir( suntqEME2000 )
    return sundir   

def Earth_Surface_Radius( lon, L, H ):  # for Earth, finds the radius in km on the elipsoid given a longitude or latitude; add height factor H
    ae = 6378.137 #equatorial radius, km
    ec = 0.081819301 # earth eccentricity, not flattening = 0.00335281066475 
    rz=abs(ae/sqrt(1-(ec*sin(L*pi/180))**2)+H)*cos(L*pi/180)
    zq=abs(ae*(1-ec**2)/sqrt(1-(ec*sin(L*pi/180))**2)+H)*sin(L*pi/180)
    Rx = rz*cos(lon*pi/180)
    Ry = rz*sin(lon*pi/180)
    Rz = zq
    print("Position at Earth surface for lon: ", lon, " lat:  ", L, " Rx, Ry, Rz: ", Rx, Ry, Rz)
    return Rx, Ry, Rz


if __name__ == "__main__":

    # test load albedo

    #from mst import config # .boas as boaset
    #from mst.config.boas import set_boa
    #from ... import set_boa

    #boa = config.boas.set_boa() #boainit)
    #boa = "filename.boa"
    bfn = '../../../Users/reynerso/Documents/Python/mst/mst/config/default.boa' 
    boa = M.BoaLoad(bfn) #(os.path.dirname(__file__) +'/'+ config.DEFAULT_BOA)
    filepath = '../../../Users/reynerso/Documents/Python/Spaceball/sb_23Jul22/sb/mst/trajectory/propagate/'

    # construct Albedo

    # SH files from David

    import AlbedoCosine_20210321_12_Nmax10 as albedocos  #albedocos
    import AlbedoSine_20210321_12_Nmax10 as albedosin  
    import ThermalCosine_20210321_12_Nmax10 as thermalcos  
    import ThermalSine_20210321_12_Nmax10 as thermalsin  

    #import AlbedoCosine_20210321_01_6x6 as albedocos  #albedocos
    #import AlbedoSine_20210321_01_6x6 as albedosin  
    #import ThermalCosine_20210321_01_6x6 as thermalcos  
    #import ThermalSine_20210321_01_6x6 as thermalsin  

    print("**********************************************************  albedoCosineCof[0][0] = ",   albedocos.albedoCosineCof[0][0] ,"*************************")
    print("**********************************************************  albedoSineCof[0][0] = ",   albedosin.albedoSineCof[0][0] ,"*************************")
    print("**********************************************************  thermalCosineCof[0][0] = ",   thermalcos.thermalCosineCof[0][0] ,"*************************")
    print("**********************************************************  thermalSineCof[0][0] = ",   thermalsin.thermalSineCof[0][0] ,"*************************")

    print("**********************************************************  albedoCosineCof[0][1] = ",   albedocos.albedoCosineCof[0][1] ,"*************************")
    print("**********************************************************  albedoSineCof[0][1] = ",   albedosin.albedoSineCof[0][1] ,"*************************")
    print("**********************************************************  thermalCosineCof[0][1] = ",   thermalcos.thermalCosineCof[0][1] ,"*************************")
    print("**********************************************************  thermalSineCof[0][1] = ",   thermalsin.thermalSineCof[0][1] ,"*************************") 

    #uncomment the approp lines to use SH files.  Comment out the approp 3x1 matrix below or it will be used
        
    albedoCosineCof = albedocos.albedoCosineCof
    albedoSineCof = albedosin.albedoSineCof
    thermalCosineCof = thermalcos.thermalCosineCof
    thermalSineCof = thermalsin.thermalSineCof
    

    # Set values to zero to remove effect (default accepted values for Earth shown below)
    '''
    albedoCosineCof = M.Matrix( 3, 1 )
    albedoCosineCof[0][0] = 0.34
    albedoCosineCof[1][0] = 0.10
    albedoCosineCof[2][0] = 0.29
    
    albedoSineCof = M.Matrix( 3, 1 )
    albedoSineCof[0][0] = 0.0
    albedoSineCof[1][0] = 0.0
    albedoSineCof[2][0] = 0.0

    thermalCosineCof = M.Matrix( 3, 1 )
    thermalCosineCof[0][0] = 0.68
    thermalCosineCof[1][0] = -0.07
    thermalCosineCof[2][0] = -0.18

    or

    thermalCosineCof = M.Matrix( 3, 1 )
    thermalCosineCof[0][0] = 0.0 #0.68
    thermalCosineCof[1][0] = 0.0 #-0.07
    thermalCosineCof[2][0] = 0.0 #-0.18

    thermalSineCof = M.Matrix( 3, 1 )
    thermalSineCof[0][0] = 0.0
    thermalSineCof[1][0] = 0.0
    thermalSineCof[2][0] = 0.0
    '''

    #vehBody = sc.name
    #vehLength = 1.0 * units.m
    albedoBody = "Earth"
    earthRadius = 6378.1363 * units.km
    maxVehDist = 1000000.0 * units.km
    
    # make M.Albedo object for Earth
    earthAlbedo = M.Albedo( boa, albedoBody, earthRadius, albedoCosineCof, albedoSineCof, thermalCosineCof, thermalSineCof )

    earthAlbedo.setModelSphere( maxVehDist ) # sets the model on/off radius

    print("Earth Albedo loaded")

    #Albedo value calc

    '''
    # range of file values:
    minlon = 0
    maxlon = 50
    steplon = 5
    minlat = 0
    maxlat = 50
    steplat = 5
    
    alboutfile = open( filepath+"albedoout.csv", 'w' )
    alboutfile.write('Albedo Map' + '\n' + '\n')

    thmoutfile = open( filepath+"thermalout.csv", 'w' )
    thmoutfile.write('Thermal Map' + '\n' + '\n')

    # write first line header with longitude values

    alboutfile.write('lat\lon,')
    thmoutfile.write('lat\lon,')

    for lon in range(minlon, maxlon+steplon, steplon):
        alboutfile.write( str(lon) + ',' )
        thmoutfile.write( str(lon) + ',' )
        if lon == maxlon:
            alboutfile.write('\n')
            thmoutfile.write('\n')


    for lat in range(minlat, maxlat+steplat, steplat):

        thmoutfile.write( str(lat) + ',' )
        alboutfile.write( str(lat) + ',' )

        for lon in range(minlon, maxlon+steplon, steplon):
    '''

    lon = 23.4162
    lat = 25.6628

    rx, ry, rz = Earth_Surface_Radius( lon, lat, 0.0 )

    #need surface point, a Unit3Vec (only uses the direction)
    surfpoint = M.Unit3Vec( rx *units.km, ry *units.km, rz *units.km )  #arbitrary for ex. calc only

    #need a Monte Epoch, arbitrary...  
    epoch = M.Epoch(('2021-Mar-21 00:00:00 UTC'))
    epoch2 = M.Epoch(('2021-Sep-21 00:00:00 UTC'))

    '''
    timeint = M.TimeInterval( epoch, epoch2)  #all needed for make_Dir_Frame; needs th object
    dur = timeint.duration()
    th = TimeHandler( epoch, dur.days() )  
    '''
    
    # test making a DirectionFrame using a fixed direction and the Sun direction:

    sundirPosDir = set_SunDirECI(boa)  #function from directions.py
    sundir = sundirPosDir.unit(epoch , "EME2000") #convert to a Dbl3Vec

    #need the sun direction - a Dbl3Vec
    #sundir = M.Dbl3Vec(1.0,0.0,0.0)   #arbitrary for ex. calc only

    #always need some frame... string, use a predefined, inertial one:
    frame = "Earth Space True Equator"

    alval = earthAlbedo.albedo( epoch, surfpoint, sundir, frame)
    print("The albedo value is: ", alval)
    #alboutfile.write( str(alval) + ',' )

    thval = earthAlbedo.thermal( epoch, surfpoint, frame)
    print("The thermal value is: ", thval)
    #thmoutfile.write( str(thval) + ',' )

    #if lon == maxlon:
    #    alboutfile.write('\n')
    #    thmoutfile.write('\n')

    #alboutfile.close()
    #thmoutfile.close()