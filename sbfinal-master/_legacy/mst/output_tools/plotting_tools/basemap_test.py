
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

'''
m = Basemap(projection='mill',
            llcrnrlat = -90,
            llcrnrlon = -180,
            urcrnrlat = 90,
            urcrnrlon = 180)

m.drawcoastlines()
m.drawcountries(linewidth=2)
#m.drawstates(color='b')
#m.drawcounties(color='darkred')
#m.etopo()
m.bluemarble()

plt.title('Basemap Tutorial')
plt.show()

'''

from itertools import chain
import numpy as np

def draw_map(m, scale=0.2):
    # draw a shaded-relief image
    m.shadedrelief(scale=scale)
    
    # lats and longs are returned as a dictionary
    lats = m.drawparallels(np.linspace(-90, 90, 13))
    lons = m.drawmeridians(np.linspace(-180, 180, 13))

    # keys contain the plt.Line2D instances
    lat_lines = chain(*(tup[1][0] for tup in lats.items()))
    lon_lines = chain(*(tup[1][0] for tup in lons.items()))
    all_lines = chain(lat_lines, lon_lines)
    
    # cycle through these lines and set the desired style
    for line in all_lines:
        line.set(linestyle='-', alpha=0.3, color='w')


fig = plt.figure(figsize=(8, 6), edgecolor='w')

#cylindrical projection
m = Basemap(projection='cyl', resolution=None,
            llcrnrlat=-90, urcrnrlat=90,
            llcrnrlon=-180, urcrnrlon=180, )

'''
#Mollwiede
m = Basemap(projection='moll', resolution=None,
            lat_0=0, lon_0=0)

#
fig = plt.figure(figsize=(8, 8))
m = Basemap(projection='ortho', resolution=None,
            lat_0=50, lon_0=0)
'''

draw_map(m)

plt.title('Basemap Tutorial')
plt.show()