from devo_numpy import devo_numpy
import PCE_functions as fn
import numpy as np
import pickle
from time import time
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rc('font', size=14)


# # load data for M=41,p=1,k=2,loguniform, l/d=10
# file_name = 'D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/GA_iterations_results_range100_M41_p1_k2_logUniform_M1_1_l2d10.pkl'
# res_opt_GA_all,t_opt_GA_all,t_opt_scaled_all,t_opt_sort_all,index_sort_all,res_opt_rec_all = pickle.load(open(file_name,'rb'))
#
# file_name = 'D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/response_original_plate_gamma_range100_temp_n50.pkl'
# res_original,ts_original = pickle.load(open(file_name,'rb'))


# load data for M=41,p=1,k=2,uniform, l/d=15
file_name = 'D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/GA_iterations_results_range10_M41_p1_k2_uniform_l2d15.pkl'
res_opt_GA_all,t_opt_GA_all,t_opt_scaled_all,t_opt_sort_all,index_sort_all,res_opt_rec_all = pickle.load(open(file_name,'rb'))

file_name = 'D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/response_original_plate_gammaRange10_l2d15.pkl'
res_original,ts_original,gamma_max = pickle.load(open(file_name,'rb'))

### max. thickness ratio - R1 plot
range_max = 10
M = 41
p = 1
k = 2
M1 = 1

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('$t_{max}/t_{min}-R_1$ plot (span=5m,l/d=15)')
ax.set_xlabel('$t_{max}/t_{min}$ [-]')
ax.set_ylabel('$R_1$ [-]')
ax.set_ylim(ymin=4,ymax=22)
ax.set_xlim(xmin=0.6,xmax=10.4)
ax.set_xticks(ticks=[i+1 for i in range(10)])

# ax.set_xscale('log')
# ax.minorticks_on()
# ax.grid()
# ax.grid(which='minor',axis='x',visible=True)
# ax.grid(which='minor',axis='y',visible=True)
ax.plot([i+1 for i in range(range_max)], res_opt_rec_all[:range_max],'r')
ax.plot([i+1 for i in range(range_max)],res_original[:range_max],'--r')
ax.legend(['$R_{1,optimized}$','$R_{1,uniform}$'],loc=2)

# plot res_opt_GA
ax2 = ax.twinx()
ax2.set_ylabel('Response reduction [%]')
ax2.set_ylim(ymin=-2,ymax=42)
ax2.tick_params('y')
# ax2.minorticks_on()
ax2.plot([i+1 for i in range(range_max)], 100*(res_original[:range_max]-res_opt_rec_all[:range_max])/res_original[:range_max],'b')
ax2.legend(['response reduction'], loc=1)
# ax2.plot([1, n_modes + 1], [0, 0], '--', color=[0.7, 0.7, 0.7])

plt.show()

# for i in range(14,21):
#     print('\nindex_sort_all['+str(i+1)+'] = ')
#     print(index_sort_all[i])
#     print('t_opt_sort_all[' + str(i + 1) + '] = ')
#     print(t_opt_sort_all[i])
#     print('res['+str(i+1)+'] = ')
#     print(res_opt_rec_all[i])