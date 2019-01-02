from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as mtri
import pandas as pd
from scipy.interpolate import griddata


x = np.linspace(0,10,50)
y = np.linspace(0,10,50)
z = 2*np.sin(x)+np.cos(y)

grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
grid_z = griddata((x, y), z, (grid_x, grid_y), method='nearest')

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_surface(grid_x, grid_y, grid_z)

plt.show()
