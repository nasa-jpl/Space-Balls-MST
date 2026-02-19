import Monte as M
import mpy.units as units
from .albedoAccel import albedo_accel_csv, find_albedo_accel
from .srpAccel import srp_accel_csv, find_srp_accel
from .aeroAccel import aero_accel_csv, find_aero_accel
from .gravAccel import grav_accel_csv, find_grav_accel
from .albedocompute import albedo_press_csv, find_albedo_press, find_iralbedo_press, iralbedo_press_csv, find_total_albedo_press, totalbedo_press_csv
from .comboAccel import find_combo_accel, combo_accel_csv


def gen_accel_data(boa, sc, pressForces, th, outFrameName, outpath, sampletime, userfile, initstartEpoch): #creates data files for each force in pressForces
    
    print("Generating Accel data files ... ")

    settings = userfile.SETTINGS

    outsets = [] # holds outset lists of accels for combining later
    
    for pf in pressForces:
        print("pf.__class__  :" , pf.__class__ )

        if pf.__class__ == M.AlbedoPressure :
            print ("Albedo Pressure Force generation...")
            albedooutset = find_albedo_accel(boa, sc, pf, th.start_epoch, th.stop_epoch, outFrameName, sampletime, userfile )
            albedo_accel_csv(albedooutset, outpath, th.start_epoch, sampletime, userfile, initstartEpoch )
            outsets.append(albedooutset)

        if pf.__class__ == M.SolarPressure :
            print ("\nSolar Radiation Pressure Force generation...")
            srpoutset = find_srp_accel(boa, sc, pf, th.start_epoch, th.stop_epoch, outFrameName, sampletime, userfile )
            srp_accel_csv(srpoutset, outpath, th.start_epoch, sampletime, userfile )
            outsets.append(srpoutset)

        if pf.__class__ == M.AtmDrag :
            print ("\nAerodynamic Pressure Force generation...")
            aerooutset = find_aero_accel(boa, sc, pf, th.start_epoch, th.stop_epoch, outFrameName, sampletime, userfile )
            aero_accel_csv(aerooutset, outpath, th.start_epoch, sampletime, userfile )
            outsets.append(aerooutset)       

        if pf.__class__ == M.Gravity :
            print ("\nGravity Acceleration generation...")
            outset = find_grav_accel(boa, sc, pf, th.start_epoch, th.stop_epoch, outFrameName, sampletime, userfile )
            grav_accel_csv(outset, outpath, th.start_epoch, sampletime, userfile )  

    #if (user select flag - coverage albedo):  then execute coverage albedo calcs and generate the csv
    if settings[ 'covalbedoforce' ]:

        print("\nAlbedo Pressure Vector generation using Coverage...")

        '''
        outset = find_albedo_press(boa, sc, th.start_epoch, th.stop_epoch, outFrameName, sampletime, userfile)
        albedo_press_csv(outset, outpath, th.start_epoch, sampletime, userfile)       # currently stands alone - not put in as a Monte pressure force yet

        outset2 = find_iralbedo_press(boa, sc, th.start_epoch, th.stop_epoch, outFrameName, sampletime, userfile)
        iralbedo_press_csv(outset2, outpath, th.start_epoch, sampletime, userfile)       # currently stands alone - not put in as a Monte pressure force yet
        '''

        covalboutset = find_total_albedo_press(boa, sc, th.start_epoch, th.stop_epoch, outFrameName, sampletime, userfile, outpath)
        totalbedo_press_csv(covalboutset, outpath, th.start_epoch, sampletime, userfile)       # currently stands alone - not put in as a Monte pressure force yet


    if settings[ 'comboforce']:

        #outset = find_combo_accel(boa, sc, pf, th.start_epoch, th.stop_epoch, outFrameName, sampletime, userfile )
        combooutset = find_combo_accel(outsets)
        combo_accel_csv( combooutset, outpath, th.start_epoch, sampletime, userfile )

