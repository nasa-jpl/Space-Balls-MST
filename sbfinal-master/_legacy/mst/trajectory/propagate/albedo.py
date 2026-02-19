import Monte as M
import mpy.units as units
#from ... import set_boa
from mst import config
from mst.config.time_handling import convert_Time_to_DateStr, convert_Epoch_to_DateStr, find_UT_hour, Time
import importlib
import importlib.util
import os
from pathlib import Path

from mst.sb_inputs import TIMEFINISH, TIMESTART


def make_filelist(th):
    #given a th object, create a date of the form: 2018-01-01, creates a filelist
    datestring = convert_Epoch_to_DateStr(th.start_epoch)
    print("")
    print("**********************************************************************************************")
    print("datestring in make_filelist = " , datestring)
    print("**********************************************************************************************")
    print("")
    #datestring = "2018-01-01"
    #temp files for testing
    alco = "AlbedoCosine_"+datestring+"_Nmax50"   #"AlbedoCosine_20210321_12_Nmax30"
    alsi = "AlbedoSine_"+datestring+"_Nmax50"   #"AlbedoSine_20210321_12_Nmax30"
    tcos = "ThermalCosine_"+datestring+"_Nmax50" #ThermalCosine_20210321_12_Nmax30"
    tsin = "ThermalSine_"+datestring+"_Nmax50"  #"ThermalSine_20210321_12_Nmax30"

    albedofiles = [alco, alsi, tcos, tsin]  # pass this as a function input param
    return albedofiles


def load_albedo(boa, sc, forces, th, userfile): # run both Albedo and Thermal coefs

    # forces: Earth Albedo
    forces.append(M.AlbedoForce)

    # construct Albedo
    '''
    #temp files for testing
    alco = "AlbedoCosine_2018-01-01_Nmax50"   #"AlbedoCosine_20210321_12_Nmax30"
    alsi = "AlbedoSine_2018-01-01_Nmax50" #"AlbedoSine_20210321_12_Nmax30"
    tcos = "ThermalCosine_2018-01-01_Nmax50" #ThermalCosine_20210321_12_Nmax30"
    tsin = "ThermalSine_2018-01-01_Nmax50"  #"ThermalSine_20210321_12_Nmax30"

    albedofiles = [alco, alsi, tcos, tsin]  # pass this as a function input param
    '''
    #start = Time(TIMESTART)
    #startepoch = config.time_handling.convert_Time_to_Epoch(start)
    #finish = Time(TIMEFINISH)
    #finishepoch = config.time_handling.convert_Time_to_Epoch(finish)

    print("")
    th.display()
    albedofiles = make_filelist(th)

    alco = albedofiles[0]
    alsi = albedofiles[1]
    tcos = albedofiles[2]
    tsin = albedofiles[3]

    # prior method of importing by hardcoding 4 filenames:
    #from . import albedofiles as albedocos  
    #from . import AlbedoSine_20210321_12_Nmax30 as albedosin  
    #from . import ThermalCosine_20210321_12_Nmax30 as thermalcos  
    #from . import ThermalSine_20210321_12_Nmax30 as thermalsin  

    #print("os.getcwd(): ", os.getcwd() )
    #print("os.curdir: ", os.curdir )
    #print("os.path.relpath('shfiles', start= os.curdir): ", os.path.relpath('shfiles', start= os.curdir))
    #print("Path('./shfiles').relative_to('.'):", Path('./shfiles').relative_to('.'))

    installpath = userfile.INPUT['mstinstalldir']
    file_path = installpath + 'mst/trajectory/propagate/shfiles/'
    #"/home/monte/Documents/Python/Spaceball/sb_final/sbfinal/mst/trajectory/propagate/shfiles/" # "/Users/reynerso/Documents/Python/Spaceball/sb_23Jul22/sb/mst/trajectory/propagate"
    
    print("file path is: ", file_path)

    if os.path.exists(file_path):
        print("Path exists: ",file_path)
    
    spec = importlib.util.spec_from_file_location(alco, file_path+alco+".py")
    spec2 = importlib.util.spec_from_file_location(alsi, file_path+alsi+".py")
    spec3 = importlib.util.spec_from_file_location(tcos, file_path+tcos+".py")
    spec4 = importlib.util.spec_from_file_location(tsin, file_path+tsin+".py")

    if spec is None:
        raise ImportError(
            'Cannot get module spec from file location', file_path)
    
    albedocos = importlib.util.module_from_spec(spec)
    albedosin = importlib.util.module_from_spec(spec2)
    thermalcos = importlib.util.module_from_spec(spec3)
    thermalsin = importlib.util.module_from_spec(spec4)
    
    spec.loader.exec_module( albedocos )
    spec2.loader.exec_module( albedosin )
    spec3.loader.exec_module( thermalcos )
    spec4.loader.exec_module( thermalsin )

    # failed import tests ... see correct working ver. above
    #albedocos = importlib.import_module( alco+".py", package = "..propagate")
    #albedocos = importlib.import_module( file_path+alco+".py", package = None ) #  "propagate")
    #albedocos = __import__(alco)
    #albedocos = importlib.import_module(alco)

    '''
    from . import AlbedoCosine_20210321_12_Nmax30 as albedocos  
    from . import AlbedoSine_20210321_12_Nmax30 as albedosin  
    from . import ThermalCosine_20210321_12_Nmax30 as thermalcos  
    from . import ThermalSine_20210321_12_Nmax30 as thermalsin  

    print("**********************************************************  albedoCosineCof[0][0] = ",   albedocos.albedoCosineCof[0][0] ,"*************************")
    print("**********************************************************  albedoSineCof[0][0] = ",   albedosin.albedoSineCof[0][0] ,"*************************")
    print("**********************************************************  thermalCosineCof[0][0] = ",   thermalcos.thermalCosineCof[0][0] ,"*************************")
    print("**********************************************************  thermalSineCof[0][0] = ",   thermalsin.thermalSineCof[0][0] ,"*************************")

    print("**********************************************************  albedoCosineCof[0][1] = ",   albedocos.albedoCosineCof[0][1] ,"*************************")
    print("**********************************************************  albedoSineCof[0][1] = ",   albedosin.albedoSineCof[0][1] ,"*************************")
    print("**********************************************************  thermalCosineCof[0][1] = ",   thermalcos.thermalCosineCof[0][1] ,"*************************")
    print("**********************************************************  thermalSineCof[0][1] = ",   thermalsin.thermalSineCof[0][1] ,"*************************") 
    '''

    albedoCosineCof = albedocos.albedoCosineCof
    albedoSineCof = albedosin.albedoSineCof
    thermalCosineCof = thermalcos.thermalCosineCof
    thermalSineCof = thermalsin.thermalSineCof

    '''
    # Use these 4 sets for Earth and the standard coefs, longitude independent
    # and comment out above thru the 'construct Albedo' comment line

    albedoCosineCof = M.Matrix( 3, 1 )
    albedoCosineCof[0][0] = 0.34
    albedoCosineCof[1][0] = 0.10
    albedoCosineCof[2][0] = 0.29
    
    albedoSineCof = M.Matrix( 3, 1 )
    albedoSineCof[0][0] = 1.0
    albedoSineCof[1][0] = 1.0
    albedoSineCof[2][0] = 1.0
    
    thermalCosineCof = M.Matrix( 3, 1 )
    thermalCosineCof[0][0] = 0.68
    thermalCosineCof[1][0] = -0.07
    thermalCosineCof[2][0] = -0.18
    
    thermalSineCof = M.Matrix( 3, 1 )
    thermalSineCof[0][0] = 1.0
    thermalSineCof[1][0] = 1.0
    thermalSineCof[2][0] = 1.0
    '''
    
    vehBody = sc.name
    vehLength = 1.0 * units.m
    albedoBody = "Earth"
    earthRadius = 6378.1363 * units.km
    maxVehDist = 1000000.0 * units.km

    # make M.Albedo object for Earth
    earthAlbedo = M.Albedo( boa, albedoBody, earthRadius, albedoCosineCof, albedoSineCof, thermalCosineCof, thermalSineCof )

    earthAlbedo.setModelSphere( maxVehDist ) # sets the model on/off radius

    # Future Construct for Albedo Mapping:
    # map albedo on the Earth:  use method: earthAlbedo.albedo( Epoch t, Unit3Vec< length > surfacePoint, Dbl3Vec sunDir, str frame ).
    # make map by varying the surfacePoint and the sunDir

    # map thermal emissivity on the Earth:  use method: earthAlbedo.thermal( Epoch t, Unit3Vec< length > surfacePoint, str frame ). 
    # make map by varying the surfacePoint  


    # make M.AlbedoPressure object for vehicle
    threshAngle = 180.0 * units.deg  # if you want to limit the arc angle set smaller
    earthAlbedoPress = M.AlbedoPressure( boa, vehBody, sc.name + ' Shape',  sc.name + ' Shape', ["Earth"], threshAngle )

    print( "Albedo + Thermal loaded" )
    print(earthAlbedoPress)

    return earthAlbedoPress



def load_albedo_only(boa, sc, forces):  # Use when we want Albedo only and no Thermal

    # forces: Earth Albedo
    forces.append(M.AlbedoForce)

    # construct Albedo
    #from . import AlbedoCosine_20210321_12_Nmax30 as albedocos  #albedocos
    #from . import AlbedoSine_20210321_12_Nmax30 as albedosin 

    from shfiles import AlbedoCosine_2018_01_01_Nmax50 as albedocos
    from shfiles import AlbedoSine_2018_01_01_Nmax50 as albedosin

    albedoCosineCof = albedocos.albedoCosineCof
    albedoSineCof = albedosin.albedoSineCof

    '''    
    albedoCosineCof = M.Matrix( 3, 1 )
    albedoCosineCof[0][0] = 0.34
    albedoCosineCof[1][0] = 0.10
    albedoCosineCof[2][0] = 0.29

    albedoSineCof = M.Matrix( 3, 1 )
    albedoSineCof[0][0] = 1.0
    albedoSineCof[1][0] = 1.0
    albedoSineCof[2][0] = 1.0
    '''

    thermalCosineCof = M.Matrix( 3, 1 )
    thermalCosineCof[0][0] = 0.0
    thermalCosineCof[1][0] = 0.0
    thermalCosineCof[2][0] = 0.0

    thermalSineCof = M.Matrix( 3, 1 )
    thermalSineCof[0][0] = 0.0
    thermalSineCof[1][0] = 0.0
    thermalSineCof[2][0] = 0.0

    vehBody = sc.name
    vehLength = 1.0 * units.m
    albedoBody = "Earth"
    earthRadius = 6378.1363 * units.km
    maxVehDist = 1000000.0  * units.km

    # make M.Albedo object for Earth
    earthAlbedo = M.Albedo( boa, albedoBody, earthRadius, albedoCosineCof, albedoSineCof, thermalCosineCof, thermalSineCof )

    earthAlbedo.setModelSphere( maxVehDist ) # sets the model on/off radius

    # Future Construct for Albedo Mapping:
    # map albedo on the Earth:  use method: earthAlbedo.albedo( Epoch t, Unit3Vec< length > surfacePoint, Dbl3Vec sunDir, str frame ).
    # make map by varying the surfacePoint and the sunDir

    # map thermal emissivity on the Earth:  use method: earthAlbedo.thermal( Epoch t, Unit3Vec< length > surfacePoint, str frame ). 
    # make map by varying the surfacePoint  


    # make M.AlbedoPressure object for vehicle
    threshAngle = 180.0 * units.deg  # if you want to limit the arc angle set smaller
    earthAlbedoPress = M.AlbedoPressure( boa, vehBody, sc.name + ' Shape', sc.name + ' Shape', ["Earth"], threshAngle )

    print( "Albedo Only loaded" )
    print(earthAlbedoPress)

    return earthAlbedoPress



def load_albedo_thermalonly(boa, sc, forces):  # used for thermal only and no albedo

    # forces: Earth Albedo
    forces.append(M.AlbedoForce)

    # construct Albedo

    #from . import ThermalCosine_20210321_12_Nmax30 as thermalcos  
    #from . import ThermalSine_20210321_12_Nmax30 as thermalsin

    from .shfiles import ThermalCosine_2018_01_01_Nmax50 as thermalcos
    from .shfiles import ThermalSine_2018_01_01_Nmax50 as thermalsin

    thermalCosineCof = thermalcos.thermalCosineCof
    thermalSineCof = thermalsin.thermalSineCof

    albedoCosineCof = M.Matrix( 3, 1 )
    albedoCosineCof[0][0] = 0.0
    albedoCosineCof[1][0] = 0.0
    albedoCosineCof[2][0] = 0.0

    albedoSineCof = M.Matrix( 3, 1 )
    albedoSineCof[0][0] = 0.0
    albedoSineCof[1][0] = 0.0
    albedoSineCof[2][0] = 0.0

    '''
    thermalCosineCof = M.Matrix( 3, 1 )
    thermalCosineCof[0][0] = 0.68
    thermalCosineCof[1][0] = -0.07
    thermalCosineCof[2][0] = -0.18

    thermalSineCof = M.Matrix( 3, 1 )
    thermalSineCof[0][0] = 1.0
    thermalSineCof[1][0] = 1.0
    thermalSineCof[2][0] = 1.0
    '''

    vehBody = sc.name
    vehLength = 1.0 * units.m
    albedoBody = "Earth"
    earthRadius = 6378.1363 * units.km
    maxVehDist = 1000000.0 * units.km

    # make M.Albedo object for Earth
    earthAlbedo = M.Albedo( boa, albedoBody, earthRadius, albedoCosineCof, albedoSineCof, thermalCosineCof, thermalSineCof )

    earthAlbedo.setModelSphere( maxVehDist ) # sets the model on/off radius

    # Future Construct for Albedo Mapping:
    # map albedo on the Earth:  use method: earthAlbedo.albedo( Epoch t, Unit3Vec< length > surfacePoint, Dbl3Vec sunDir, str frame ).
    # make map by varying the surfacePoint and the sunDir

    # map thermal emissivity on the Earth:  use method: earthAlbedo.thermal( Epoch t, Unit3Vec< length > surfacePoint, str frame ). 
    # make map by varying the surfacePoint  


    # make M.AlbedoPressure object for vehicle
    threshAngle = 180.0 * units.deg  # if you want to limit the arc angle set smaller
    earthAlbedoPress = M.AlbedoPressure( boa, vehBody, sc.name + ' Shape', sc.name + ' Shape', ["Earth"], threshAngle )

    print( "Albedo: Thermal Only loaded" )
    print(earthAlbedoPress)

    return earthAlbedoPress



if __name__ == "__main__":
    # test load albedo

    #from mst import config # .boas as boaset
    #from mst.config.boas import set_boa
    #from ... import set_boa

    #boa = config.boas.set_boa() #boainit)
    #boa = "filename.boa"
    bfn = '../../../Users/reynerso/Documents/Python/mst/mst/config/default.boa' 
    boa = M.BoaLoad(bfn) #(os.path.dirname(__file__) +'/'+ config.DEFAULT_BOA)

    # construct Albedo

    # SH files from David

    import AlbedoCosine_20210321_12_Nmax30 as albedocos  #albedocos
    import AlbedoSine_20210321_12_Nmax30 as albedosin  
    import ThermalCosine_20210321_12_Nmax30 as thermalcos  
    import ThermalSine_20210321_12_Nmax30 as thermalsin  

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
    albedoSineCof[0][0] = 1.0
    albedoSineCof[1][0] = 1.0
    albedoSineCof[2][0] = 1.0

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
    thermalSineCof[0][0] = 1.0
    thermalSineCof[1][0] = 1.0
    thermalSineCof[2][0] = 1.0
    '''

    #vehBody = sc.name
    #vehLength = 1.0 * units.m
    albedoBody = "Earth"
    earthRadius = (6378.1363 + 1000) * units.km
    maxVehDist = 1000000.0 * units.km
    
    # make M.Albedo object for Earth
    earthAlbedo = M.Albedo( boa, albedoBody, earthRadius, albedoCosineCof, albedoSineCof, thermalCosineCof, thermalSineCof )

    earthAlbedo.setModelSphere( maxVehDist ) # sets the model on/off radius

    print("Earth Albedo loaded")

    #Albedo value calc

    #need surface point, a Unit3Vec (only uses the direction)
    surfpoint = M.Unit3Vec( 10000.0 *units.km, 10000.0 *units.km, 10000.0 *units.km )  #arbitrary for ex. calc only

    #need the sun direction - a Dbl3Vec
    sundir = M.Dbl3Vec(1.0,0.0,0.0)   #arbitrary for ex. calc only

    #need a Monte Epoch, again arbitrary...  
    epoch = M.Epoch(('2021-Mar-21 00:00:00 UTC'))

    #always need some frame... string, use a predefined, inertial one:
    frame = "Earth Space True Equator"

    earthAlbedo.setRadius(6378.1363 * units.km)

    alval = earthAlbedo.albedo( epoch, surfpoint, sundir, frame)
    print("The albedo value is: ", alval)

    thval = earthAlbedo.thermal( epoch, surfpoint, frame)
    print("The thermal value is: ", thval)

    radius = earthAlbedo.radius()
    print("The radius is: ",radius)






