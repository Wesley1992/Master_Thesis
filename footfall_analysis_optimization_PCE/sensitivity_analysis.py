import numpy as np
import pickle
import PCE_functions as fn
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# number of input variables (model dimension), tv and tr for test
M = 41
# max total degree
p = 1
# oversampling rate
k = 2
# delta for morris sensitivity analysis
d = 0.001
# sampling number
n_morris = 10

# load data file
file_name = 'D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/M'+str(M)+'_p'+str(p)+'_k'+str(k)+'_actualThickness_logUniform.pkl'
M, p, k, bounds, P, n, ts_samp, Y_ED, ts_scaled, index, Psi, reg, y_alpha, Y_rec, k_new, n_new, ts_samp_new, ts_scaled_new, Psi_new, Y_new = pickle.load(open(file_name,'rb'))

# Morris sensitivity analysis
ts_samp_morris = fn.sampling('log_uniform', bounds, M, n_morris) # shape=M*n
ts_scaled_morris = np.zeros((M,n_morris))
areas = fn.get_areas()
for i in range(n_morris):
    ts_scaled_morris[:,i] = fn.get_scaledThickness(areas, ts_samp_morris[:,i])

Psi_morris = fn.create_Psi(index, ts_scaled_morris, 'normal')
Y_morris = reg.predict(Psi_morris)

# print('reg.coef_')
# print(reg.coef_)

Y_delta_morris = np.zeros((M, n_morris))
for i in range(M):
    ts_scaled_morris_delta = ts_scaled_morris.copy()
    ts_scaled_morris_delta[i,:] += d
    Psi_morris_delta = fn.create_Psi(index, ts_scaled_morris_delta, 'normal')
    Y_delta_morris[i, :] = reg.predict(Psi_morris_delta)

    # if i == 0:
        # print('Psi_morris_delta-Psi_morris')
        # print(Psi_morris_delta-Psi_morris)
        # print('Y_morris')
        # print(Y_morris)
        # print('Y_morris_rec')
        # print(reg.coef_@np.transpose(Psi_morris))
        # print('Y_delta_morris')
        # print(Y_delta_morris)
        # print('Y_morris_delta_rec')
        # print(reg.coef_@np.transpose(Psi_morris_delta))

# gradient
D = np.zeros((M,n_morris))    # shape M*n_morris
for i in range(M):
    D[i,:] = (Y_delta_morris[i,:] - Y_morris)/d

# mean and std deviation
mu_D = np.zeros(M)
sigma_D = np.zeros(M)

for i in range(M):
    mu_D[i] = np.mean(D[i])
    sigma_D[i] = np.std(D[i])

print('mean_D')
print(mu_D)
print('std_D')
print(sigma_D)

### Corelation analysis
# input/output correlation
IOC = np.zeros(M)
for i in range(M):
    IOC[i]=pearsonr(ts_scaled[i,:],Y_ED)[0]
print('IOC')
print(IOC)


### plot
# # mu_D-sigma_D plot
# fig  = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_title('Morris sensitivity analysis', fontsize=12)
# ax.set_xlabel('Mean ($\mu$)', fontsize=12)
# ax.set_ylabel('Std deviation ($\sigma$)', fontsize=12)
# ax.scatter(mu_D, sigma_D, )
# text = [i for i in range(1,M+1)]
# i=0
# for x,y in zip(mu_D,sigma_D):
#     ax.annotate(str(i), xy=(x, y))
#     i+=1

# input/output plot
for i in range(M):
    fig  = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Input/Output plot for dimension '+str(i+1), fontsize=12)
    ax.set_xlabel('Thickness', fontsize=12)
    ax.set_ylabel('Response factor', fontsize=12)
    ax.scatter(ts_scaled[i,:],Y_ED)

plt.show()

print()


