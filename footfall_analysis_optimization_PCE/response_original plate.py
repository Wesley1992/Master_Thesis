from math import factorial as fact
import numpy as np
import pickle
import timeit
import PCE_functions as fn
from sklearn.linear_model import LinearRegression
import os

### clarification: the goal of this script is to calculate the response of original plate with identical vaul thickness
### and ribs thickness, just for comparison with optimized one. The thickness ratio varies from 1 to 100

### define input parameters
gamma_max = 10
M = 2
n = gamma_max

# generate fake sample thickness
ts_samp1 = [i+1 for i in range(gamma_max)]
ts_samp2 = [1 for i in range(gamma_max)]
ts_samp = np.array([ts_samp1,ts_samp2])

# number of vault and rib panels !!! only for test
n_v = 16
n_r = 21
n_r_h = 4


### evaluate experimental design
# TODO: not to differentiate between vault and ribs later !!! only for test
# original thickness of all panels for all experimental designs
ts = np.zeros((n_v+n_r+n_r_h,n))
ts[:n_v] = ts_samp[0].copy()      # vault thickness
ts[n_v:] = ts_samp[1].copy()      # ribs thickness

Y_ED = np.zeros(n)
# scaled thickness of all panels for all experimental designs
ts_scaled = []
# scaled thickness of vault and ribs in this particular case !!! only for test
ts_vr_scaled = np.zeros((M, n))
# start evaluating
for i in range(n):
    print('\n********** start evaluating the ' + str(i + 1) + 'th of ' + str(n) + ' experimental designs **********')
    start = timeit.default_timer()

    Y_ED[i],t_scaled = fn.evaluate_response(ts[:,i])
    ts_scaled.append(t_scaled)
    # !!! only for test
    ts_vr_scaled[0][i] = t_scaled[0]
    ts_vr_scaled[1][i] = t_scaled[n_v]

    stop = timeit.default_timer()
    print('********** evaluation of the ' + str(i + 1) + 'th experimental design finished, time = '+str(stop-start)+' s\n')

    # if i % 10 == 9:
    #     with open('D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/response_original_plate_gamma_range100_temp_n'+str(i+1)+'_1.pkl', 'wb') as data:
    #         pickle.dump([Y_ED, ts_scaled], data)
    #     try:
    #         os.remove(
    #             'D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/response_original_plate_gamma_range100_temp_n'+str(i-9)+'_1.pkl')
    #     except:
    #         pass

print('Y_ED=')
print(Y_ED)

with open ('D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/response_original_plate_gammaRange'+str(gamma_max)+'_l2d15.pkl','wb') as data:
            # pickle.dump([ts_ED,xi_ED,ts_scaled,Y_ED,Y_rec,y_alpha,Psi,index],data)
            pickle.dump([Y_ED, ts_scaled,gamma_max], data)