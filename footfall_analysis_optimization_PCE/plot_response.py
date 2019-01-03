import pickle
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
import pandas as pd
from matplotlib import cm
from scipy.interpolate import griddata
from math import factorial as fact
from pyDOE2 import lhs
import PCE_functions as fn
import timeit
from sklearn.linear_model import LinearRegression

### load data from PCE

file_name = 'D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/M2_p2_k2_actualThickness_logUniform.pkl'

M,p,k,bounds,P,n,ts_ED,ts,Y_ED,ts_scaled,ts_ED_scaled,index,Psi,reg,y_alpha,Y_rec = pickle.load(open(file_name,'rb'))

n_v = 16
n_r = 21
n_r_h = 4
t_base = 0.068

t_v = np.zeros(n)
t_r = np.zeros(n)

for i in range(n):
    t_v[i] = ts_scaled[i][0]
    t_r[i] = ts_scaled[i][n_v]

### generate more sampling points and predict response with polynomial approximation

k_new = 50
n_new = int(k_new*fact(M+p)/(fact(M)*fact(p)))

ts_ED_new = fn.sampling('log_uniform',bounds,M,n_new,t_base)

ts_new = np.zeros((n_v+n_r+n_r_h,n_new))

# TODO: not to differentiate between vault and ribs
ts_new[:n_v] = ts_ED_new[0].copy()      # vault thickness
ts_new[n_v:] = ts_ED_new[1].copy()      # ribs thickness
areas = fn.get_areas()
ts_ED_scaled_new = np.zeros((M,n_new))
ts_scaled_new = []

for i in range(n_new):
    t_scaled_new = fn.get_scaledThickness(areas, ts_new[:, i])
    ts_ED_scaled_new[0,i] = t_scaled_new[0]
    ts_ED_scaled_new[1,i] = t_scaled_new[n_v]

    ts_scaled_new.append(t_scaled_new)

# prediction
Psi_new = fn.create_Psi(index,ts_ED_scaled_new,'normal')
Y_new = reg.predict(Psi_new)


### t_r,t_r - R1 plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('$t_v,t_r-R_1$ plot (span=5m,l/d=10,M=2,p=2,k=2,log-uniform distribution sampling)')
ax.set_xlabel('$t_v$ [m]')
ax.set_ylabel('$t_r$ [m]')
ax.set_zlabel('R1 [-]')
ax.scatter(t_v, t_r, Y_ED, s=100)
ax.scatter(t_v, t_r, Y_rec, s=100)
ax.scatter(ts_ED_scaled_new[0], ts_ED_scaled_new[1], Y_new)

ax.legend(['experimental design (ED)','polynomial approximation (same sampling as ED)','polynomial approximation (different sampling from ED)'])

plt.show()