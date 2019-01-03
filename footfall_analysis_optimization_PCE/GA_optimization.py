from devo_numpy import devo_numpy
import PCE_functions as fn
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
import timeit



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


file_name = 'D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/M2_p2_k2_actualThickness_logUniform.pkl'
M,p,k,bounds,P,n,ts_ED,ts,Y_ED,ts_scaled,ts_ED_scaled,index,Psi,reg,y_alpha,Y_rec = pickle.load(open(file_name,'rb'))

n_v = 16
n_r = 21
n_r_h = 4

areas = fn.get_areas()
bound = (0.01,0.2)
bounds_GA = []
for i in range(M):
    bounds_GA.append(bound)

res_PCE(np.array([0.15,0.03]))


start = timeit.default_timer()

res = devo_numpy(fn=res_PCE,bounds=bounds_GA,population=200,generations=200,plot=False)

stop = timeit.default_timer()

print(res)
print('time: '+str(stop-start)+' s')