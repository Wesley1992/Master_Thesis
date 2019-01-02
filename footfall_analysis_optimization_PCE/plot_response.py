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
M = 2
p = 2
k = 2
bounds = [0.01,0.2]

n_v = 16
n_r = 21
n_r_h = 4

file_name = 'D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/M'+str(M)+'_p'+str(p)+'_k'+str(k)+'_actualThickness.pkl'
ts_ED,xi_ED,ts_scaled,Y_ED,Y_rec,y_alpha,Psi,index = pickle.load(open(file_name,'rb'))

n = len(ts_scaled)
t_v = np.zeros(n)
t_r = np.zeros(n)

for i in range(n):
    t_v[i] = ts_scaled[i][0]
    t_r[i] = ts_scaled[i][n_v]


### generate more sampling points and calculate response with polynomials

k_new = 50
n_new = int(k_new*fact(M+p)/(fact(M)*fact(p)))

U_new = np.transpose(lhs(M,samples=n_new))
ts_ED_new =  bounds[0]+(bounds[1]-bounds[0])*U_new
# t_vs_ED = ts_ED[0,:]
# t_rs_ED = ts_ED[1,:]

# xi_ED_new = (2*ts_ED_new-(bounds[0]+bounds[1]))/(bounds[1]-bounds[0])

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

# regression from experimental design
reg = LinearRegression()
reg.fit(Psi,Y_ED)
Y_rec = reg.predict(Psi)

Psi_new = fn.create_Psi(index,ts_ED_scaled_new,'normal')

Y_rec_new = reg.predict(Psi_new)

print('Y_ED=')
print(Y_ED)
print('Y_rec=')
print(Y_rec)
print('Y_rec_new=')
print(Y_rec_new)

# Y_rec_new = y_alpha @ np.transpose(Psi_new)

### t_r,t_r - R1 plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('$t_v,t_r-R1$ plot (span=5m,l/d=10,M=2,p=2,k=2,uniform distribution sampling)')
ax.set_xlabel('$t_v$ [m]')
ax.set_ylabel('$t_r$ [m]')
ax.set_zlabel('R1 [-]')
ax.scatter(t_v, t_r, Y_ED, s=100)
ax.scatter(t_v, t_r, Y_rec, s=100)
ax.scatter(ts_ED_scaled_new[0], ts_ED_scaled_new[1], Y_rec_new)

ax.legend(['experimental design (ED)','polynomial approximation (same sampling as ED)','polynomial approximation (different sampling from ED)'])

plt.show()


# gammas_PCE = t_r/t_r
#
# print(ts_scaled)