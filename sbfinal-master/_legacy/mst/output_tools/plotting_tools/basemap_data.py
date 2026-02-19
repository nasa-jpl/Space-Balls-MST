import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


# 3d blue ball - may need the pillow package
#plt.figure(figsize=(8, 8))
#m = Basemap(projection='ortho', resolution=None, lat_0=50, lon_0=-100)
#m.bluemarble(scale=0.5)

fig = plt.figure(figsize=(8, 8))
m = Basemap(projection='lcc', resolution=None,
            width=8E6, height=8E6, 
            lat_0=45, lon_0=-100,)
m.etopo(scale=0.5, alpha=0.5)

# Map (long, lat) to (x, y) for plotting
x, y = m(-122.3, 47.6)
plt.plot(x, y, 'ok', markersize=5)
plt.text(x, y, ' Seattle', fontsize=12);
