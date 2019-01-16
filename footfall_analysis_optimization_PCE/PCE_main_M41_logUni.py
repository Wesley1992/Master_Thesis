from math import factorial as fact
import numpy as np
import pickle
import timeit
import PCE_functions as fn
from sklearn.linear_model import LinearRegression
import os
from time import time
np.set_printoptions(threshold=np.nan)


### define input parameters
# number of input variables (model dimension), tv and tr for test
M = 41
# max total degree
p = 2
# oversampling rate
k = 2
# bounds for log-uniform distribution
bounds = [1/5,5]

# cardinality (number of coefficients)
P = fact(M+p)/(fact(M)*fact(p))
# total runs of experimental design (input number of each variable)
n = int(k*P)
# # sampling (initial thickness, to be scaled later)
# ts_samp = fn.sampling('log_uniform', bounds, M, n,M1=1) # shape=M*n
#
# ### evaluate experimental design
# # scaled thickness of all panels for all experimental designs
# ts_scaled = np.zeros((M,n))
# # start evaluating
# Y_ED = np.zeros(n)
# for i in range(n):
#     print('\n********** start evaluating the ' + str(i + 1) + 'th of ' + str(n) + ' experimental designs **********')
#     start = timeit.default_timer()
#
#     Y_ED[i],t_scaled = fn.evaluate_response(ts_samp[:,i])
#     ts_scaled[:,i] = t_scaled
#
#     if i%10 == 9:
#         with open('D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/M' + str(M) + '_p' + str(
#                 p) + '_k' + str(k) + '_temp_n'+str(i+1)+'_3.pkl', 'wb') as data:
#             pickle.dump([Y_ED, ts_scaled], data)
#         try:
#             os.remove(
#                 'D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/M' + str(
#                     M) + '_p' + str(
#                     p) + '_k' + str(k) + '_temp_n' + str(i - 9) + '_3.pkl')
#         except:
#             pass
#
#     stop = timeit.default_timer()
#     print('********** evaluation of the ' + str(i + 1) + 'th experimental design finished, time = '+str(stop-start)+' s\n')

### approximation with polynomials
## p=2
# load Y_ED and t_scaled
file_name = 'D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/Y_t_M' + str(M) + '_p' + str(
        p) + '_k' + str(k) + '.pkl'
Y_ED, ts_scaled = pickle.load(open(file_name,'rb'))

## p=1
# file_name = 'D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/M'+str(M)+'_p'+str(p)+'_k'+str(k)+'_actualThickness_logUniform.pkl'
# Y_ED = pickle.load(open(file_name,'rb'))[7]
# ts_scaled = pickle.load(open(file_name,'rb'))[8]

# # generate random data for test
# bounds_X = [0.01,0.3]
# bounds_Y = [1,30]
# Y_ED = bounds_Y[0]+(bounds_Y[1]-bounds_Y[0])*np.random.rand(n)
# ts_scaled = bounds_X[0]+(bounds_X[1]-bounds_X[0])*np.random.rand(M,n)

# obtain degree index
index = fn.create_degreeIndex(M,p)

# create Psi matrix
Psi = fn.create_Psi(index,ts_scaled,'normal')

# calculate polynomial coefficients with LR and recalculate
reg = LinearRegression()
reg.fit(Psi,Y_ED)
y_alpha_LR = reg.coef_
Y_rec_LR = reg.predict(Psi)

# calculate polynomial coefficients with lest square and recalculate
y_alpha_LS = np.linalg.inv(np.transpose(Psi)@Psi)@np.transpose(Psi)@Y_ED
Y_rec_LS = y_alpha_LS @ np.transpose(Psi)


### calculate response with more sampling data
# k_new = 10
# n_new = int(k_new*fact(M+p)/(fact(M)*fact(p)))
#
# ts_samp_new = fn.sampling('log_uniform', bounds, M, n_new,M1=1)
#
# areas = fn.get_areas()
# ts_scaled_new = np.zeros((M, n_new))
# for i in range(n_new):
#     ts_scaled_new[:,i] = fn.get_scaledThickness(areas, ts_samp_new[:, i])
#
# # prediction
# Psi_new = fn.create_Psi(index, ts_scaled_new, 'normal')
# Y_new = reg.predict(Psi_new)


print('Y_ED=')
print(Y_ED)
print('y_alpha_LR=')
print(y_alpha_LR)
print('Y_rec_LR=')
print(Y_rec_LR)
print('y_alpha_LS=')
print(y_alpha_LS)
print('Y_rec_LS=')
print(Y_rec_LS)

# print('Y_new=')
# print(Y_new)

# with open ('D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/M'+str(M)+'_p'+str(p)+'_k'+str(k)+'_actualThickness_logUniform.pkl','wb') as data:
#             # pickle.dump([M, p, k, bounds, P, n, Y_ED, ts_scaled, index, Psi, reg, y_alpha, Y_rec, k_new, n_new, ts_samp_new, ts_scaled_new,Psi_new, Y_new ], data)
#     pickle.dump([M, p, k, bounds, P, n, Y_ED, ts_scaled, index, Psi, reg, y_alpha, Y_rec], data)

print()