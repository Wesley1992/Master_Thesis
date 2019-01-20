from devo_numpy import devo_numpy
import PCE_functions as fn
import numpy as np
import pickle
from time import time
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)


def res_PCE(t):
    # t: array of dimension M
    # ts: array of dimension number of panels

    t_scaled = fn.get_scaledThickness(areas, np.array(t))

    ## in standard space
    # xi = (2 * t_scaled - (bound_uni[0] + bound_uni[1])) / (bound_uni[1] - bound_uni[0])
    # Psi = fn.create_Psi(index,np.transpose([xi]),'legendre')
    # res = y_alpha_std @ np.transpose(Psi)

    ## in actual space
    Psi = fn.create_Psi(index,np.transpose([t_scaled]),'normal')
    res = y_alpha_act @ np.transpose(Psi)


    return res

# load PCE model

# p = 1, k=2, uniform thickness [0.01,0.3]
file_name = 'D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/PCE_M41_p1_k2_uni_0.01_0.2_l2d15.pkl'
y_alpha_std,y_alpha_act,index,bound_uni= pickle.load(open(file_name,'rb'))

# # p = 1, k=3, uniform thickness [0.01,0.3]
# file_name = 'D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/PCE_std_actual_M41_p1_k3.pkl'
# y_alpha_std,y_alpha_act,index,bound_uni= pickle.load(open(file_name,'rb'))

# # p = 1, k=2, uniform thickness [0.01,0.2]
# file_name = 'D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/PCE_std_actual_M41_p1_k2.pkl'
# y_alpha_std,y_alpha_act,index,bound_uni= pickle.load(open(file_name,'rb'))

# p = 1, M1=16
# file_name = 'D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/M41_p1_k2_logUniform_M1_16.pkl'
# M, p, k, bounds, P, n, ts_samp, Y_ED, ts_scaled, index, Psi, reg, y_alpha, Y_rec, k_new, n_new, ts_samp_new, ts_scaled_new,Psi_new, Y_new = pickle.load(open(file_name,'rb'))

# p = 2
# file_name = 'D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/M41_p2_k2_actualThickness_logUniform.pkl'
# M, p, k, bounds, P, n, Y_ED, ts_scaled, index, Psi, reg, y_alpha, Y_rec = pickle.load(open(file_name,'rb'))


# iterate for given bound
M=41
p=1
k=2

# range_max = 10
# bounds_iter = [i+1 for i in range(range_max)]
#
# res_opt_GA_all = []
# t_opt_GA_all = []
# t_opt_scaled_all = []
# t_opt_sort_all = []
# index_sort_all = []
# res_opt_rec_all = []

# for i in bounds_iter:
#     print('\n'+str(i) + 'th of ' + str(range_max) + ' iteration starts')



# GA optimization
M = 41
areas = fn.get_areas()
bound = (1,50)
# bound = (1, i)
bounds_GA = []
for j in range(M):
    bounds_GA.append(bound)

start = time()

res_opt_GA, t_opt_GA = devo_numpy(fn=res_PCE, bounds=bounds_GA, population=100, generations=500, plot=0, limit=-100000)

stop = time()

# scale the thickness
t_opt_scaled = fn.get_scaledThickness(areas, np.array(t_opt_GA))

t_opt_sort = sorted(t_opt_scaled,reverse=True)
index_sort = np.flip(np.argsort(t_opt_scaled),axis=0)+1

print('time: '+str(stop-start)+' s\n')
print('optimized response R1_GA = ')
print(res_opt_GA)
print('\nt_opt = ')
print(t_opt_scaled)
print('\nt_opt_sort = ')
print(t_opt_sort)
print('\nindex_sort = ')
print(index_sort)
print('\ny_alpha_act = ')
print(y_alpha_act)



#     # calculate the optimized response
#     res_opt_rec,t_opt_scaled_rec = fn.evaluate_response(t_opt_scaled)
#     print('res_opt_rec='+str(res_opt_rec))
# #
#     res_opt_GA_all.append(res_opt_GA)
#     t_opt_GA_all.append(t_opt_GA)
#     t_opt_scaled_all.append(t_opt_scaled)
#     t_opt_sort_all.append(t_opt_sort)
#     index_sort_all.append(index_sort)
#     res_opt_rec_all.append(res_opt_rec)
#
#     print(str(i)+'th of '+str(range_max)+' iteration finished\n')
#     #
#     # if i%10 == 0:
#     #     with open(
#     #             'D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/GA_iterations_results_range' + str(
#     #                     range_max) + '_M' + str(M) + '_p' + str(
#     #                     p) + '_k' + str(k) + '_logUniform_M1_' + str(M1) + '_temp_n'+str(i)+'.pkl', 'wb') as data:
#     #         pickle.dump(
#     #             [res_opt_GA_all, t_opt_GA_all, t_opt_scaled_all, t_opt_sort_all, index_sort_all, res_opt_rec_all], data)
#
# with open('D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/GA_iterations_results_range'+str(range_max)+'_M' + str(M) + '_p' + str(
#         p) + '_k' + str(k) + '_uniform_l2d15.pkl', 'wb') as data:
#     pickle.dump([res_opt_GA_all,t_opt_GA_all,t_opt_scaled_all,t_opt_sort_all,index_sort_all,res_opt_rec_all], data)
