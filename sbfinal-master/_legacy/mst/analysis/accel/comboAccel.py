import Monte as M
import mpy.units as units
#from analysis import headers
from .. import headers
from mst.config.time_handling import convert_Epoch_to_DateStr
import pandas as pd
import matplotlib.pyplot as plt

#fixedFrameName = 'IAU Earth Fixed' 
#framePoleName = 'IAU Earth Pole'

def addlistvecs(vecs1, vecs2): #adds 2 lists of vectors

    vcount = 0
    vecs3 = []

    for vec1 in vecs1:
        vec2 = vecs2[vcount]
        vec3 = vec1 + vec2  #adding 2 Monte vectors (need to check)
        vecs3.append(vec3)
        vcount = vcount + 1

    return vecs3



def find_combo_accel(outsets): # finds total acceleration given an array of accel lists

    size = len(outsets)

    if size < 2:
        print('No accelerations to combine!!')

    if size >= 2: # combine two 
        outset1 = outsets[0]  # first accel set
        outset2 = outsets[1]  # second accel set

        vecs1 = outset1[2] # vecs in first set
        vecs2 = outset2[2] # vecs in second set

        vecs3 = addlistvecs( vecs1 , vecs2 )   # add 2 vectors but it may need a loop

        outsetcombo = [ outset1[0], outset1[1], vecs3 ] # format is: outset = [ times, longlat, vecs ]

    if size >= 3: # combine the prior addition with the 3rd vector list
        outset3 = outsetcombo # combo of accels 1 & 2
        outset4 = outsets[2]  # third accel set

        vecs4 = outset3[2] # vecs in the combo set
        vecs5 = outset4[2] # vecs in the third set

        vecs6 = addlistvecs( vecs4 , vecs5 )   # add 2 vectors but it may need a loop

        outsetcombo = [ outset3[0], outset3[1], vecs6 ] # format is: outset = [ times, longlat, vecs ]

    return outsetcombo


def combo_accel_csv(outset, outpath, startEpoch, sampletime, userfile):  #csv output

    datestring = convert_Epoch_to_DateStr(startEpoch)
    fileName = outpath + '/csvoutput/' + 'combo_accel_'+datestring+'.csv'

    outfile = open( fileName, 'w' )

    header = headers.make_header( userfile )
    print("header: ", header)
    outfile.write( header + '\n')

    headerline1='Printing combined acceleration every ' +str(sampletime)+ ' secs (m/s**2):\n\
    Epoch   %s \n' % (startEpoch.format()) + \
    'Secs_from_Epoch, Long, Lat, Ar, Ai, Ac, AMAG\n' # ric = radial, intrack, crosstrack
    print(headerline1)
    outfile.write( headerline1 + '\n')

    times = outset[0]
    longlats = outset[1]
    longs = []
    lats = []
    vecs = outset[2]
    xvals = []
    yvals = []
    zvals = []
    vmag = []


    count = 0

    for time in times:
        longitude = longlats[count][0]
        latitude = longlats[count][1]
        x = vecs[count][0]#.value()
        y = vecs[count][1]#.value()
        z = vecs[count][2]#.value()  
        mag = vecs[count].mag()  
        longs.append(longitude)
        lats.append(latitude)
        xvals.append(x)
        yvals.append(y)
        zvals.append(z)
        vmag.append(mag) #derived value for the df only - using loop to create

        outstring = '%6.2f, %3.5f, %3.5f, %.15f, %.15f, %.15f, %.15f' %( time
            ,longitude
            ,latitude
            ,x 
            ,y 
            ,z 
            ,mag
            )   

        #print( outstring )
        outfile.write( outstring + '\n')
        count = count + 1
    
    outfile.close()

    #datalen = [len(times),len(longlats),len(vecs),len(longs),len(lats),len(xvals),len(yvals),len(zvals), len(vmag) ]# check vector lengths
    #print('datalen: ',datalen)
    
    if userfile.PLOT[ 'ploton' ] &  userfile.PLOT[ 'combo_plot' ]: 
        data = {
            #'time': times,
            #'longitude': longs,
            #'latitude': lats,
            'combo_accel_r': xvals,
            'combo_accel_i': yvals,
            'combo_accel_c': zvals,
            'combo_accel': vmag,       
        }
        row_labels = times 
        df = pd.DataFrame(data = data, index = row_labels)

        df.plot()
        plt.savefig( outpath + '/pngoutput/' + 'combo_accel'+datestring+'.png' )
        plt.show()
        plt.close()

        #longlatdf = df.loc[:,['longitude','latitude']]

        #longlatdf.plot()
        #plt.show()

        print('Combo Accel df: \n', df)