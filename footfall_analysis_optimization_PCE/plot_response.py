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

### load data from global analysis
spans = [5]
l2ds = [10]
gammas = [0.1,0.5,1,2,5,10]

for span in spans:
    for l2d in l2ds:
        for gamma in gammas:
            file_name='D:/Master_Thesis/code_data/footfall_analysis/data/data_mdl_0_4m/data_mdl_span'+str(span).replace('.','_')+'_l2d'+str(l2d).replace('.','_')+'_gamma'+str(gamma).replace('.','_')+'.pkl'

            f_n, m_n, node_lp, n_modes, dt, t, dis_modes_lp, vel_modes_lp, acc_modes_lp, acc_modes_lp_weight, rms_modes,rms_modes_weight, R, R_weight, Gamma_n = pickle.load(open(file_name,'rb'))

### load data from PCE
M = 2
p = 2
k = 2
bounds = [0.01,0.2]

n_v = 16
n_r = 21
n_r_h = 4

file_name = 'D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/M'+str(M)+'_p'+str(p)+'_k'+str(k)+'.pkl'
ts_ED,xi_ED,ts_scaled,Y_ED,Y_rec,y_alpha,Psi,index = pickle.load(open(file_name,'rb'))

n = len(ts_scaled)
t_v = np.zeros(n)
t_r = np.zeros(n)

for i in range(n):
    t_v[i] = ts_scaled[i][0]
    t_r[i] = ts_scaled[i][n_v]


### generate more sampling points and calculate response with polynomials

start = timeit.default_timer()

k_new = 100
n_new = int(k_new*fact(M+p)/(fact(M)*fact(p)))

U_new = np.transpose(lhs(M,samples=n_new))
ts_ED_new =  bounds[0]+(bounds[1]-bounds[0])*U_new
# t_vs_ED = ts_ED[0,:]
# t_rs_ED = ts_ED[1,:]

xi_ED_new = (2*ts_ED_new-(bounds[0]+bounds[1]))/(bounds[1]-bounds[0])

Psi_new = fn.create_Psi(index,xi_ED_new)

Y_rec_new = y_alpha @ np.transpose(Psi_new)

ts_new = np.zeros((n_v+n_r+n_r_h,n_new))
# TODO: not to differentiate between vault and ribs
ts_new[:n_v] = ts_ED_new[0].copy()      # vault thickness
ts_new[n_v:] = ts_ED_new[1].copy()      # ribs thickness

ts_scaled_new = []
areas = fn.get_areas()
for i in range(n_new):
    ts_scaled_new.append(fn.get_scaledThickness(areas,ts_new[:,i]))

t_v_new = np.zeros(n_new)
t_r_new = np.zeros(n_new)

for i in range(n_new):
    t_v_new[i] = ts_scaled_new[i][0]
    t_r_new[i] = ts_scaled_new[i][n_v]

stop = timeit.default_timer()

print('time for new data generation: '+str(stop-start)+' s')

### t_r,t_r - R1 plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('xi_v_ED')
ax.set_ylabel('xi_r_ED')
ax.set_zlabel('R1')
ax.scatter(xi_ED[0], xi_ED[1], Y_ED)
ax.scatter(xi_ED[0], xi_ED[1], Y_rec)
ax.scatter(xi_ED_new[0], xi_ED_new[1], Y_rec_new)

ax.legend(['experimental design (ED)','polynomial approximation (same sampling as ED)','polynomial approximation (different sampling from ED)'])

plt.show()


# gammas_PCE = t_r/t_r
#
# print(ts_scaled)