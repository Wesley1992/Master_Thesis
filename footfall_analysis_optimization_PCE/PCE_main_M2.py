from math import factorial as fact
import numpy as np
import pickle
import timeit
import PCE_functions as fn
from sklearn.linear_model import LinearRegression

### define input parameters
# number of input variables (model dimension), tv and tr for test
M = 2
# max total degree
p = 2
# oversampling rate
k = 2
# bounds for log-uniform distribution
bound_log = [1/5,5]
# bounds for uniform distribution
bound_uni = [0.01,0.2]
# number of vault and rib panels !!! only for test
n_v = 16
n_r = 21
n_r_h = 4

# cardinality (number of coefficients)
P = fact(M+p)/(fact(M)*fact(p))
# total runs of experimental design (input number of each variable)
n = int(k*P)
# sampling
ts_samp_log = fn.sampling('log_uniform', bound_log, M, n,1) # shape=M*n
ts_samp_uni = fn.sampling('uniform', bound_uni, M, n)

### evaluate experimental design
# TODO: not to differentiate between vault and ribs later !!! only for test
# original thickness of all panels for all experimental designs
ts_log = np.zeros((n_v+n_r+n_r_h,n))
ts_uni = np.zeros((n_v+n_r+n_r_h,n))

ts_log[:n_v] = ts_samp_log[0].copy()      # vault thickness
ts_log[n_v:] = ts_samp_log[1].copy()      # ribs thickness

ts_uni[:n_v] = ts_samp_uni[0].copy()      # vault thickness
ts_uni[n_v:] = ts_samp_uni[1].copy()      # ribs thickness

Y_ED_log = np.zeros(n)
Y_ED_uni = np.zeros(n)
# scaled thickness of all panels for all experimental designs
ts_scaled_log = []
ts_scaled_uni = []

# scaled thickness of vault and ribs in this particular case !!! only for test
ts_vr_scaled_log = np.zeros((M, n))
ts_vr_scaled_uni = np.zeros((M, n))

# start evaluating
for i in range(n):
    print('\n********** start evaluating the ' + str(i + 1) + 'th of ' + str(n) + ' experimental designs **********')
    start = timeit.default_timer()

    Y_ED_log[i],t_scaled_log = fn.evaluate_response(ts_log[:,i])
    ts_scaled_log.append(t_scaled_log)
    # !!! only for test
    ts_vr_scaled_log[0][i] = t_scaled_log[0]
    ts_vr_scaled_log[1][i] = t_scaled_log[n_v]

    Y_ED_uni[i], t_scaled_uni = fn.evaluate_response(ts_uni[:, i])
    ts_scaled_uni.append(t_scaled_uni)
    # !!! only for test
    ts_vr_scaled_uni[0][i] = t_scaled_uni[0]
    ts_vr_scaled_uni[1][i] = t_scaled_uni[n_v]

    stop = timeit.default_timer()
    print('********** evaluation of the ' + str(i + 1) + 'th experimental design finished, time = '+str(stop-start)+' s\n')

# obtain degree index
index = fn.create_degreeIndex(M,p)

# create Psi matrix
# !!! ts_ED_scaled only for this case
Psi_log = fn.create_Psi(index, ts_vr_scaled_log, 'normal')
Psi_uni = fn.create_Psi(index, ts_vr_scaled_uni, 'normal')


# calculate polynomial coefficients
reg = LinearRegression()
reg.fit(Psi,Y_ED)
y_alpha = reg.coef_
# y_alpha = np.linalg.inv(np.transpose(Psi)@Psi)@np.transpose(Psi)@Y_ED   # somehow doesn't work properly


# recalculate response based on obtained coefficients
Y_rec = reg.predict(Psi)
# Y_rec = y_alpha @ np.transpose(Psi)   # somehow doesn't work properly
print('Y_ED=')
print(Y_ED)
print('y_alpha=')
print(y_alpha)
print('Y_rec=')
print(Y_rec)

with open ('D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/M'+str(M)+'_p'+str(p)+'_k'+str(k)+'_actualThickness_logUniform.pkl','wb') as data:
            # pickle.dump([ts_ED,xi_ED,ts_scaled,Y_ED,Y_rec,y_alpha,Psi,index],data)
            pickle.dump([M, p, k, bounds, P, n, ts_samp, ts, Y_ED, ts_scaled, ts_vr_scaled, index, Psi, reg, y_alpha, Y_rec], data)