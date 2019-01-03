from devo_numpy import devo_numpy
import PCE_functions as fn
import numpy as np
import pickle
from time import time



def res_PCE(t):
    # t: array of dimension M
    # ts: array of dimension number of panels
    ts = np.zeros(n_v+n_r+n_r_h)
    ts[:n_v] = t[0].copy()  # vault thickness
    ts[n_v:] = t[1].copy()  # ribs thickness

    ts_scaled = fn.get_scaledThickness(areas, ts)

    t_scaled = np.zeros(M)
    t_scaled[0] = ts_scaled[0]
    t_scaled[1] = ts_scaled[n_v]

    Psi = fn.create_Psi(index,np.transpose([t_scaled]),'normal')

    res = reg.predict(Psi)

    return res

# load data from PCE
file_name = 'D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/M2_p2_k2_actualThickness_logUniform.pkl'
M,p,k,bounds,P,n,ts_ED,ts,Y_ED,ts_scaled,ts_ED_scaled,index,Psi,reg,y_alpha,Y_rec = pickle.load(open(file_name,'rb'))

n_v = 16
n_r = 21
n_r_h = 4

# GA optimization
areas = fn.get_areas()
bound = (0.01,0.2)
bounds_GA = []
for i in range(M):
    bounds_GA.append(bound)

start = time()

res_opt,t_opt = devo_numpy(fn=res_PCE,bounds=bounds_GA,population=100,generations=100,plot=0)

stop = time()

# scale the thickness
ts = np.zeros(n_v+n_r+n_r_h)
ts[:n_v] = t_opt[0].copy()  # vault thickness
ts[n_v:] = t_opt[1].copy()  # ribs thickness

ts_opt_scaled = fn.get_scaledThickness(areas, ts)

t_opt_scaled = np.zeros(M)
t_opt_scaled[0] = ts_opt_scaled[0]
t_opt_scaled[1] = ts_opt_scaled[n_v]

print('optimized response R1 = '+str(res_opt)+' with tv = '+str(t_opt_scaled[0])+', tr = '+str(t_opt_scaled[1])+'\n')
print('time: '+str(stop-start)+' s')