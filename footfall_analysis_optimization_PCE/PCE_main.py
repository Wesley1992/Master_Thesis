from math import factorial as fact
from pyDOE2 import lhs
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
bounds = [-np.log10(5),np.log10(5)]
# number of vault and rib panels !!! only for test
n_v = 16
n_r = 21
n_r_h = 4

# cardinality (number of coefficients)
P = fact(M+p)/(fact(M)*fact(p))
# total runs of experimental design (input number of each variable)
n = int(k*P)
# sampling (initial thickness, to be scaled later)
t_base = 0.068      # thickness when tv=tr  !!! only for test
ts_ED = fn.sampling('log_uniform',bounds,M,n,t_base) # shape=M*n

### evaluate experimental design
# TODO: not to differentiate between vault and ribs later !!! only for test
# original thickness of all panels for all experimental designs
ts = np.zeros((n_v+n_r+n_r_h,n))
ts[:n_v] = ts_ED[0].copy()      # vault thickness
ts[n_v:] = ts_ED[1].copy()      # ribs thickness

Y_ED = np.zeros(n)
# scaled thickness of all panels for all experimental designs
ts_scaled = []
# scaled thickness of vault and ribs in this particular case !!! only for test
ts_ED_scaled = np.zeros((M,n))
# start evaluating
for i in range(n):
    print('\n********** start evaluating the ' + str(i + 1) + 'th of ' + str(n) + ' experimental designs **********')
    start = timeit.default_timer()

    Y_ED[i],t_scaled = fn.evaluate_response(ts[:,i])
    ts_scaled.append(t_scaled)
    # !!! only for test
    ts_ED_scaled[0][i] = t_scaled[0]
    ts_ED_scaled[1][i] = t_scaled[n_v]

    stop = timeit.default_timer()
    print('********** evaluation of the ' + str(i + 1) + 'th experimental design finished, time = '+str(stop-start)+' s\n')

# obtain degree index
index = fn.create_degreeIndex(M,p)

# create Psi matrix
# !!! ts_ED_scaled only for this case
Psi = fn.create_Psi(index,ts_ED_scaled,'normal')

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
            pickle.dump([M,p,k,bounds,P,n,ts_ED,ts,Y_ED,ts_scaled,ts_ED_scaled,index,Psi,reg,y_alpha,Y_rec], data)