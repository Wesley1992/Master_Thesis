from devo_numpy import devo_numpy
import PCE_functions as fn
import numpy as np
import pickle
from time import time



def res_PCE(t):
    # t: array of dimension M
    # ts: array of dimension number of panels

    t_scaled = fn.get_scaledThickness(areas, np.array(t))

    Psi = fn.create_Psi(index,np.transpose([t_scaled]),'normal')

    res = reg.predict(Psi)

    return res

# load data from PCE
file_name = 'D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/M41_p1_k2_actualThickness_logUniform.pkl'
M, p, k, bounds, P, n, ts_samp, Y_ED, ts_scaled, index, Psi, reg, y_alpha, Y_rec, k_new, n_new, ts_samp_new, ts_scaled_new,Psi_new, Y_new = pickle.load(open(file_name,'rb'))

# GA optimization
areas = fn.get_areas()
bound = (0.05,0.2)
bounds_GA = []
for i in range(M):
    bounds_GA.append(bound)

start = time()

res_opt,t_opt = devo_numpy(fn=res_PCE,bounds=bounds_GA,population=100,generations=500,plot=0,limit=-1000)

stop = time()

print(t_opt)


# scale the thickness
t_opt_scaled = fn.get_scaledThickness(areas, np.array(t_opt))
print('optimized response R1 = '+str(res_opt)+' with t = ')
print(t_opt_scaled)
print('time: '+str(stop-start)+' s')