from devo_numpy import devo_numpy
import PCE_functions as fn
import numpy as np
import pickle
from time import time
import matplotlib.pyplot as plt


### max. thickness ratio - R1 plot
range_max = 10
M = 41
p = 1
k = 2
M1 = 1

# # load data for M=41,p=1,k=2,loguniform, l/d=10
# file_name = 'D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/GA_iterations_results_range'+str(range_max)+'_M' + str(M) + '_p' + str(
#         p) + '_k' + str(k) + '_logUniform_M1_' + str(M1) + '.pkl'
# res_opt_GA_all,t_opt_GA_all,t_opt_scaled_all,t_opt_sort_all,index_sort_all,res_opt_rec_all = pickle.load(open(file_name,'rb'))

# file_name = 'D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/response_original_plate_gamma_range100_temp_n50.pkl'
# res_original,ts_original = pickle.load(open(file_name,'rb'))


# load data for M=41,p=1,k=2,uniform, l/d=15
file_name = 'D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/GA_iterations_results_range'+str(range_max)+'_M' + str(M) + '_p' + str(
        p) + '_k' + str(k) + '_uniform_l2d15.pkl'
res_opt_GA_all,t_opt_GA_all,t_opt_scaled_all,t_opt_sort_all,index_sort_all,res_opt_rec_all = pickle.load(open(file_name,'rb'))

file_name = 'D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/response_original_plate_gammaRange10_l2d15.pkl'
res_original,ts_original,gamma_max = pickle.load(open(file_name,'rb'))


print(res_original)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Max. thickness ratio - R1 plot')
ax.set_xlabel('Max. thickness ratio')
ax.set_ylabel('R1')
ax.set_xscale('log')
ax.minorticks_on()
ax.grid()
ax.grid(which='minor',axis='x',visible=True)
ax.grid(which='minor',axis='y',visible=True)
ax.plot([i+1 for i in range(range_max)], res_opt_rec_all,'r')
ax.plot([i+1 for i in range(range_max)],res_original,'--r')
ax.legend(['R1_opt','R1_uniform thickness'],loc=2)

# plot res_opt_GA
ax2 = ax.twinx()
ax2.set_ylabel('R1_PE', fontsize=12)
ax2.tick_params('y')
ax2.minorticks_on()
ax2.plot([i+1 for i in range(range_max)], res_opt_GA_all,'b')
ax2.legend(['R1_PCE'], loc=1)
# ax2.plot([1, n_modes + 1], [0, 0], '--', color=[0.7, 0.7, 0.7])

plt.show()

# for i in range(14,21):
#     print('\nindex_sort_all['+str(i+1)+'] = ')
#     print(index_sort_all[i])
#     print('t_opt_sort_all[' + str(i + 1) + '] = ')
#     print(t_opt_sort_all[i])
#     print('res['+str(i+1)+'] = ')
#     print(res_opt_rec_all[i])