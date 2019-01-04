import numpy as np
import PCE_functions as fn
import matplotlib.pyplot as plt

M = 41
n = 100
bounds = [1/5,5]
M1 = 16
ts_samp = fn.sampling('log_uniform',bounds,M,n,M1=M1)
ts_scaled = np.zeros((M,n))
areas = fn.get_areas()
for i in range(n):
    ts_scaled[:,i] = fn.get_scaledThickness(areas, ts_samp[:,i])

fig  = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Original sampling with M='+str(M)+',n='+str(n)+',bounds='+str(bounds)+',M1='+str(M1), fontsize=12)
ax.set_xlabel('value', fontsize=12)
# ax.set_xscale('log')
ax.set_ylabel('dimentioin', fontsize=12)
for i in range(M):
    ax.scatter(ts_samp[i, :], np.ones(n)*(i+1))

fig  = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Scaled thickness with M='+str(M)+',n='+str(n)+',bounds='+str(bounds)+',M1='+str(M1), fontsize=12)
ax.set_xlabel('value', fontsize=12)
ax.set_ylabel('dimentioin', fontsize=12)
for i in range(M):
    ax.scatter(ts_scaled[i, :], np.ones(n)*(i+1))

plt.show()