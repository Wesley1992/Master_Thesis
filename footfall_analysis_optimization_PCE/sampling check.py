import numpy as np
import PCE_functions as fn
import matplotlib.pyplot as plt
import pickle
import matplotlib as mpl

mpl.rc('font', size=14)


# M = 41
# n = 100
# bounds = [0.01,0.2]
# M1 = 16
# ts_samp = fn.sampling('uniform',bounds,M,n,M1=M1)
# ts_scaled = np.zeros((M,n))
# areas = fn.get_areas()
# for i in range(n):
#     ts_scaled[:,i] = fn.get_scaledThickness(areas, ts_samp[:,i])


file_name = 'D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/PCE_M41_p1_k2_logUni_0.2_5.pkl'
# M, p, k, bounds, P, n, Y_ED, ts_scaled_logUni, index, Psi, reg, y_alpha, Y_rec = pickle.load(open(file_name,'rb'))

ts_scaled_logUni = pickle.load(open(file_name,'rb'))[8]

file_name = 'D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/Y_t_M41_p1_k2_uniformThickness_0.01_0.2_l2d15.pkl'
Y_ED, ts_scaled_uni = pickle.load(open(file_name,'rb'))

M=41
n=84


fig  = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Input samples for experimental design (span=5m,l/d=10)')
ax.set_xlabel('Thickness [m]')
ax.set_xlim(xmin=-0.01,xmax=0.40)
# ax.set_xscale('log')
ax.set_ylabel('Dimension [-]')
ax.set_yticks(ticks=[1]+[5*(i+1) for i in range(8)])

for i in range(M):
    ax.scatter(ts_scaled_logUni[i, :], np.ones(n)*(i+1),color='C0',s=10)

fig  = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Input samples for experimental design (span=5m,l/d=15)')
ax.set_xlabel('Thickness [m]')
ax.set_xlim(xmin=-0.01,xmax=0.40)
ax.set_ylabel('Dimension [-]')
ax.set_yticks(ticks=[1]+[5*(i+1) for i in range(8)])
for i in range(M):
    ax.scatter(ts_scaled_uni[i, :], np.ones(n)*(i+1),color='C0',s=10)

plt.show()