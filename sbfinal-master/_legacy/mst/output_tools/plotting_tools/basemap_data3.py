from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

import pandas as pd

import os

#to get the current working directory
directory = os.getcwd()

print(directory)

pydir = "../../../../../home/monte/Documents/Python/Spaceball/sb_23Jul22/sb/mst/output_tools/plotting_tools/"

cities = pd.read_csv(pydir + 'CERES_test.csv')

cities.head()

# Extract the data we're interested in
lat = cities['latd'].values
lon = cities['longd'].values
population = cities['population_total'].values
area = cities['area_total_km2'].values




