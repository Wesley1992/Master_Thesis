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
# bounds for uniform distribution
bounds = [0.01,0.2]
# number of vault and rib panels
n_v = 16
n_r = 21
n_r_h = 4

# cardinality (number of coefficients)
P = fact(M+p)/(fact(M)*fact(p))
# total runs of experimental design (input number of each variable)
n = int(k*P)
# Latin Hypercube sampling (standard uniform distribution U([0,1]))
U = np.transpose(lhs(M,samples=n))
# transform standard uniform distribution to experimental design input: X = a+(b-a)U
ts_ED =  bounds[0]+(bounds[1]-bounds[0])*U # shape=M*n
print('ts_ED=')
print(ts_ED)

# # evaluate experimental design
ts = np.zeros((n_v+n_r+n_r_h,n))
# TODO: not to differentiate between vault and ribs
ts[:n_v] = ts_ED[0].copy()      # vault thickness
ts[n_v:] = ts_ED[1].copy()      # ribs thickness

Y_ED = np.zeros(n)
ts_scaled = []
ts_ED_scaled = np.zeros((M,n))

for i in range(n):
    print('\n### evalutate the ' + str(i + 1) + 'th of ' + str(n) + ' experimental designs')
    start = timeit.default_timer()

    Y_ED[i],t_scaled = fn.evaluate_response(ts[:,i])
    ts_scaled.append(t_scaled)
    ts_ED_scaled[0][i] = t_scaled[0]
    ts_ED_scaled[1][i] = t_scaled[n_v]

    stop = timeit.default_timer()
    print('time for evaluating the '+str(i+1)+' th experimental design: '+str(stop-start)+' s\n')


# # t_vs_ED = ts_ED[0,:]
# # t_rs_ED = ts_ED[1,:]
# # transform experimental design input to Legendre polynomials input U([-1,1])
# xi_ED = (2*ts_ED-(bounds[0]+bounds[1]))/(bounds[1]-bounds[0])
# print('xi_ED=')
# print(xi_ED)

# obtain degree index
index = fn.create_degreeIndex(M,p)
print('degree_index=')
print(index)

# create Psi matrix
Psi = fn.create_Psi(index,ts_ED_scaled,'normal')

print('ts_ED_scaled=')
print(ts_ED_scaled)
print('Psi_matrix=')
print(Psi)

# print('ts_scaled=')
# print(ts_scaled)

# calculate polynomial coefficients
reg = LinearRegression()
reg.fit(Psi,Y_ED)
y_alpha = reg.coef_
# y_alpha = np.linalg.inv(np.transpose(Psi)@Psi)@np.transpose(Psi)@Y_ED
print('y_alpha=')
print(y_alpha)

# recalculate response based on obtained coefficients
# Y_rec = y_alpha @ np.transpose(Psi)
Y_rec = reg.predict(Psi)
print('Y_ED=')
print(Y_ED)
print('Y_rec=')
print(Y_rec)

with open ('D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/M'+str(M)+'_p'+str(p)+'_k'+str(k)+'_actualThickness.pkl','wb') as data:
            # pickle.dump([ts_ED,xi_ED,ts_scaled,Y_ED,Y_rec,y_alpha,Psi,index],data)
            pickle.dump([ts_ED, ts_ED_scaled, ts_scaled, Y_ED, Y_rec, y_alpha, Psi, index,reg], data)
print()