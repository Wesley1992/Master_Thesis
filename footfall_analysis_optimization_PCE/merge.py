import pickle
import numpy as np

file_name = 'D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/M41_p2_k2_temp_Mac_n50_3.pkl'
Y_ED1, ts_scaled1 = pickle.load(open(file_name,'rb'))

file_name = 'D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/M41_p2_k2_temp_Mac_n130_2.pkl'
Y_ED2, ts_scaled2 = pickle.load(open(file_name,'rb'))

file_name = 'D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/M41_p2_k2_temp_Mac_n230_1.pkl'
Y_ED3, ts_scaled3 = pickle.load(open(file_name,'rb'))

file_name = 'D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/M41_p2_k2_temp_Mac_n300.pkl'
Y_ED4, ts_scaled4 = pickle.load(open(file_name,'rb'))

file_name = 'D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/M41_p2_k2_temp_Windows_n120.pkl'
Y_ED5, ts_scaled5 = pickle.load(open(file_name,'rb'))

file_name = 'D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/M41_p2_k2_temp_Windows_n280_2.pkl'
Y_ED6, ts_scaled6 = pickle.load(open(file_name,'rb'))

file_name = 'D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/M41_p2_k2_temp_Windows_n720_1.pkl'
Y_ED7, ts_scaled7 = pickle.load(open(file_name,'rb'))

M = 41
p = 2
k = 2
n = 1806

Y_ED = np.append(Y_ED1,Y_ED2)
Y_ED = np.append(Y_ED,Y_ED3)
Y_ED = np.append(Y_ED,Y_ED4)
Y_ED = np.append(Y_ED,Y_ED5)
Y_ED = np.append(Y_ED,Y_ED6)
Y_ED = np.append(Y_ED,Y_ED7)
Y_ED = Y_ED[Y_ED!=0]


ts_scaled = np.append(ts_scaled1,ts_scaled2,axis=1)
ts_scaled = np.append(ts_scaled,ts_scaled3,axis=1)
ts_scaled = np.append(ts_scaled,ts_scaled4,axis=1)
ts_scaled = np.append(ts_scaled,ts_scaled5,axis=1)
ts_scaled = np.append(ts_scaled,ts_scaled6,axis=1)
ts_scaled = np.append(ts_scaled,ts_scaled7,axis=1)
ts_temp = np.zeros((M,1830))

for i in range(M):
    t = ts_scaled[i,:]
    ts_temp[i,:] = t[t!=0]

Y_ED_temp = np.zeros(n)
ts_scaled_temp = np.zeros((M,n))
for i in range(n):
    Y_ED_temp[i] = Y_ED[i]
    ts_scaled_temp[:,i] = ts_temp[:,i]

Y_ED = Y_ED_temp
ts_scaled = ts_scaled_temp

with open('D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/Y_t_M' + str(M) + '_p' + str(
        p) + '_k' + str(k) + '.pkl', 'wb') as data:
    pickle.dump([Y_ED, ts_scaled], data)
