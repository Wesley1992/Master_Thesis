import numpy as np
import scipy.io
import pickle

file_name = 'D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/Y_t_M41_p1_k2_uniformThickness_0.01_0.3.pkl'

Y_ED =pickle.load( open( file_name, "rb" ) )[0]
ts_uniform = pickle.load( open( file_name, "rb" ) )[1]

scipy.io.savemat('D:/Master_Thesis/code_data/footfall_analysis_optimization_PCE/data/M41_p1_k3_matlab.mat'
, mdict={'Y_ED': Y_ED,'ts':ts_uniform})