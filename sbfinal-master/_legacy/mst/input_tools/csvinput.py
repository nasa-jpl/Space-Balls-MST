#import Monte as M
#import mpy.units as units
#from analysis import headers
import pandas as pd
#import matplotlib.pyplot as plt
import math


def find_indices(lat, lon): # finds correct lat and long indices in csv file, specific to the 1x1 deg format

    if lon == 0.0:
        longval = 0
    else:    
        longval =  math.ceil(lon) - 1

    if lat > 0:
        latval = math.ceil(lat) + 89
    else:
        latval = math.floor(lat) + 90    

    #print('lat/long:', latval, longval)    
    return latval, longval


def find_albedo_from_csv( filename, lat, lon): # returns the correct albedo value from the csv given a lat and long - deprecated
    latind, lonind = find_indices(lat,lon)
    df = pd.read_csv( filename )
    #print("df:")
    #print(df)
    alval = df.iat[ latind, lonind ]
    #print("CSV value, lat, long: ", alval, lat, lon)
    return alval
 

def read_albedo_from_csv(filename): # reads the csv file and creates a data frame
    #print("")
    #print("Reading Albedo from CSV, filename: ",filename)
    #print("")    
    df = pd.read_csv( filename )
    df = df.fillna(0) # turns NaN values into 0
    # using numpy: df.replace(np.nan,0)
    return df


def find_albedo_from_df( df, lat, lon):  # returns the correct albedo value from the data frame given a lat and long
    latind, lonind = find_indices(lat,lon)
    alval = df.iat[ latind, lonind ]
    #print("CSV value, lat, long, latind, lonind: ", alval, lat, lon, latind, lonind)
    return alval


def convert_albedo(alval, solarval=1361): # converts W/m^2 data into albedo fraction value (albedo flux/sun flux)
    alratio = alval/solarval
    return alratio

def find_swmean_from_df( df, lat, lon):  # returns the correct albedo value from the data frame given a lat and long
    latind, lonind = find_indices(lat,lon)
    alval = df.iat[ latind, lonind ]
    #print("CSV value, lat, long, latind, lonind: ", alval, lat, lon, latind, lonind)
    return alval

def find_cloudfrac_from_df( df, lat, lon):  # returns the correct albedo value from the data frame given a lat and long
    latind, lonind = find_indices(lat,lon)
    alval = df.iat[ latind, lonind ]
    #print("CSV value, lat, long, latind, lonind: ", alval, lat, lon, latind, lonind)
    return alval

def find_surftype_from_df( df, lat, lon):  # returns the correct albedo value from the data frame given a lat and long
    latind, lonind = find_indices(lat,lon)
    alval = df.iat[ latind, lonind ]
    #print("CSV value, lat, long, latind, lonind: ", alval, lat, lon, latind, lonind)
    return alval


if __name__ == "__main__":
    #used for testing conversions to get the right data positions in file;
    # execute line: % ./mpython /Users/reynerso/Documents/Python/Spaceball/sb_26Apr22/sb/mst/input_tools/csvinput.py

    lat = -88.5
    lon = 4.5
    #filename = '~/Documents/Python/Spaceball/sb_26Apr22/sb/data/CERES_SYN1deg-Day_Terra-Aqua-MODIS_Ed4.1_Subset_20210314_SWreflect.csv'

    #filename = '~/Documents/SpaceBalls/Data/Anisotropy/Cloud_fraction_SYN_daily_mean_20210321.csv'
    filename = '~/Documents/SpaceBalls/Data/Anisotropy/SW_flux_SYN_daily_mean_20210321.csv'
    #filename = '~/Documents/SpaceBalls/Data/Anisotropy/Surface_type.csv'


    outval = find_albedo_from_csv( filename, lat, lon)
    print('for lat/lon = ', lat, lon ,'albedo is: ', outval)

    df = read_albedo_from_csv(filename)
    outval2 = find_albedo_from_df( df, lat, lon)
    print('Using data frame... for lat/lon = ', lat, lon ,'albedo is: ', outval2 )

