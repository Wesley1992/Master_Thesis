import numpy as np
import random
import pickle

file_name = 'D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/GA_iterations_results_range100_M41_p1_k2_logUniform_M1_1_l2d10.pkl'
ts = pickle.load(open(file_name,'rb'))[2]
# n = len(ts)
for i in range(len(ts)):
    ts[i] = ts[i].tolist()

file_name = 'D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/ts_opt_l2d10.pkl'
with open (file_name,'wb') as data:
    pickle.dump(ts,data,protocol=2)


file_name = 'D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/ts_opt_l2d10.pkl'
ts_all = pickle.load(open(file_name,'rb'))
t_max = max(ts_all[9])
ts = ts_all[0]

print(t_max)
print(ts)
