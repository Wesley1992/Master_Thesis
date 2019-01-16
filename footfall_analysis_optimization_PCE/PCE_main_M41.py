from math import factorial as fact
import numpy as np
import pickle
import timeit
import PCE_functions as fn
from sklearn.linear_model import LinearRegression
import os
from time import time

### define input parameters
# number of input variables (model dimension), tv and tr for test
M = 41
# max total degree
p = 1
# oversampling rate
k = 2
# the dimension as basis
M1 = 16
# bounds for distribution
bound_uni = [0.01, 0.2]

# cardinality (number of coefficients)
P = fact(M+p)/(fact(M)*fact(p))
# total runs of experimental design (input number of each variable)
n = int(k*P)
# sampling (initial thickness, to be scaled later)
ts_samp = fn.sampling('uniform', bound_uni, M, n, M1=M1) # shape=M*n

# ### evaluate experimental design with ts_samp (to create PCE model)
# # scaled thickness of all panels for all experimental designs
# ts_scaled = np.zeros((M,n))
# # start evaluating
# Y_ED = np.zeros(n)
# for i in range(n):
#     print('\n********** start evaluating the ' + str(i + 1) + 'th of ' + str(n) + ' experimental designs **********')
#     start = timeit.default_timer()
#
#     Y_ED[i],t_scaled = fn.evaluate_response(ts_samp[:,i],scale=False,l2d=15)
#     ts_scaled[:,i] = t_scaled
#
#     if i%10 == 9:
#         with open('D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/M' + str(M) + '_p' + str(
#                 p) + '_k' + str(k) + '_temp_n'+str(i+1)+'_samp_uniform_thickness.pkl', 'wb') as data:
#             pickle.dump([Y_ED, ts_scaled], data)
#         try:
#             os.remove(
#                 'D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/M' + str(
#                     M) + '_p' + str(
#                     p) + '_k' + str(k) + '_temp_n' + str(i - 9) + '_samp_uniform_thickness.pkl')
#         except:
#             pass
#
#     stop = timeit.default_timer()
#     print('********** evaluation of the ' + str(i + 1) + 'th experimental design finished, time = '+str(stop-start)+' s\n')
#
# with open('D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/Y_t_M' + str(M) + '_p' + str(
#         p) + '_k' + str(k) + '_uniformThickness_0.01_0.2_l2d15.pkl', 'wb') as data:
#     pickle.dump([Y_ED, ts_scaled], data)

# ### approximation with polynomials
# ## load Y_ED and t_scaled if not generated from aboce
# # p=2
# # file_name = 'D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/Y_t_M' + str(M) + '_p' + str(
# #         p) + '_k' + str(k) + '.pkl'
# # Y_ED, ts_scaled = pickle.load(open(file_name,'rb'))
#
# ## p=1
# # file_name = 'D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/M'+str(M)+'_p'+str(p)+'_k'+str(k)+'_actualThickness_logUniform.pkl'
# # Y_ED = pickle.load(open(file_name,'rb'))[7]
# # ts_scaled = pickle.load(open(file_name,'rb'))[8]
#
# ## p=1, uniform sampling
# file_name = 'D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/M41_p1_k2_uniformThickness.pkl'
# Y_ED, ts_scaled = pickle.load(open(file_name,'rb'))

# ## p=1,k=3 uniform sampling
# file_name = 'D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/Y_t_M' + str(M) + '_p' + str(
#         p) + '_k' + str(k) + '_uniformThickness.pkl'
# Y_ED, ts_scaled = pickle.load(open(file_name,'rb'))

# ## p=1,k=2 uniform sampling,[0.01,0.3]
# file_name = 'D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/Y_t_M' + str(M) + '_p' + str(
#         p) + '_k' + str(k) + '_uniformThickness_0.01_0.3.pkl'
# Y_ED, ts_scaled = pickle.load(open(file_name,'rb'))

## p=1,k=2 uniform sampling,l/d=15,[0.01,0.2]
file_name = 'D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/Y_t_M' + str(M) + '_p' + str(
        p) + '_k' + str(k) + '_uniformThickness_0.01_0.2_l2d15.pkl'
Y_ED, ts_scaled = pickle.load(open(file_name,'rb'))
#
# obtain degree index
index = fn.create_degreeIndex(M,p)

### PCE in standard space
# create Psi matrix, input is transferred to [-1,1]
xi = (2 * ts_scaled - (bound_uni[0] + bound_uni[1])) / (bound_uni[1] - bound_uni[0])
start = time()
Psi_std = fn.create_Psi(index, xi, 'legendre')
stop = time()
print('time for Psi_std = '+str(stop-start)+' s')

# calculate polynomial coefficients with lest square and recalculate
y_alpha_LS_std = np.linalg.inv(np.transpose(Psi_std) @ Psi_std) @ np.transpose(Psi_std) @ Y_ED
Y_rec_LS_std = y_alpha_LS_std @ np.transpose(Psi_std)

# # calculate polynomial coefficients with linear regression and recalculate
# reg = LinearRegression()
# reg.fit(Psi,Y_ED)
# y_alpha_LR = reg.coef_
# Y_rec_LR = reg.predict(Psi)

### PCE in actual space
# keep actual thickness
start = time()
Psi_act = fn.create_Psi(index,ts_scaled,'normal')
stop = time()
print('time for Psi_act = '+str(stop-start)+' s')
# calculate polynomial coefficients with lest square and recalculate
y_alpha_LS_act = np.linalg.inv(np.transpose(Psi_act)@Psi_act)@np.transpose(Psi_act)@Y_ED
Y_rec_LS_act = y_alpha_LS_act @ np.transpose(Psi_act)

# # calculate polynomial coefficients with linear regression and recalculate
# reg_act = LinearRegression()
# reg_act.fit(Psi_act,Y_ED)
# y_alpha_LR_act = reg_act.coef_
# Y_rec_LR_act = reg_act.predict(Psi_act)


### calculate response with new scaled data
bound_log = [1/5,5]
ts_samp_new_k2 = fn.sampling('log_uniform', bound_log, M, n,M1=M1)
areas = fn.get_areas()
ts_scaled_new_k2 = np.zeros((M, n))
for i in range(n):
    ts_scaled_new_k2[:,i] = fn.get_scaledThickness(areas, ts_samp_new_k2[:, i])

# in standard space
xi_new_k2 = (2 * ts_scaled_new_k2 - (bound_uni[0] + bound_uni[1])) / (bound_uni[1] - bound_uni[0])
Psi_new_k2 = fn.create_Psi(index, xi_new_k2, 'legendre')
Y_new_k2_LS = y_alpha_LS_std @ np.transpose(Psi_new_k2)

# in actual space
Psi_new_k2_act = fn.create_Psi(index, ts_scaled_new_k2, 'normal')
Y_new_k2_LS_act = y_alpha_LS_act @ np.transpose(Psi_new_k2_act)
# predict with coefficients from least square


# predict with coefficients from linear regression




# ### calculate response with more sampling data
# k_new = 10
# n_new_k10 = int(k_new*fact(M+p)/(fact(M)*fact(p)))
#
# ts_samp_new_k10 = fn.sampling('log_uniform', bound_log, M, n_new_k10,M1=M1)
#
# ts_scaled_new_k10 = np.zeros((M, n_new_k10))
# for i in range(n_new_k10):
#     ts_scaled_new_k10[:,i] = fn.get_scaledThickness(areas, ts_samp_new_k10[:, i])
#
# xi_new_k10 = (2 * ts_scaled_new_k10 - (bound_uni[0] + bound_uni[1])) / (bound_uni[1] - bound_uni[0])
# Psi_new_k10 = fn.create_Psi(index, xi_new_k10, 'legendre')
#
# # predict with coefficients from least square
# Y_new_k10_LS = y_alpha_LS @ np.transpose(Psi_new_k10)
#
# # predict with coefficients from linear regression
# Y_new_k10_LR = reg.predict(Psi_new_k10)

print('Y_ED=')
print(Y_ED)
print('y_alpha_LS_std=')
print(y_alpha_LS_std)
print('Y_rec_LS_std=')
print(Y_rec_LS_std)
print('y_alpha_LS_act=')
print(y_alpha_LS_act)
print('Y_rec_LS_act=')
print(Y_rec_LS_act)

print('Y_new_k2_LS=')
print(Y_new_k2_LS)
print('Y_new_k2_LS_act=')
print(Y_new_k2_LS_act)



with open ('D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/PCE_M'+str(M)+'_p'+str(p)+'_k'+str(k)+'_uni_0.01_0.2_l2d15.pkl','wb') as data:
            pickle.dump([y_alpha_LS_std, y_alpha_LS_act, index, bound_uni], data)
    # pickle.dump([M, p, k, bounds, P, n, Y_ED, ts_scaled, index, Psi, reg, y_alpha, Y_rec], data)
