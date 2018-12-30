from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy.interpolate import griddata
import math


### input for loading file
spans = [5,6,7,8,9,10]
l2ds = [10,12.5,15,17.5,20]
gammas = [0.1,0.5,1,2,5,10]


### load file
data_file = 'D:/Master_Thesis/code_data/footfall_analysis/data/other/data_m1_f1_R1.pkl'


m1,f1,R1_weight = pickle.load(open(data_file,'rb'))

m1_grid_mergeSpan = m1.reshape(len(spans)*len(l2ds),len(gammas))
f1_grid_mergeSpan = f1.reshape(len(spans)*len(l2ds),len(gammas))
R1_weight_grid_mergeSpan = R1_weight.reshape(len(spans)*len(l2ds),len(gammas))

m1_scatter = m1.reshape(len(spans)*len(l2ds)*len(gammas))
f1_scatter = f1.reshape(len(spans)*len(l2ds)*len(gammas))
R1_weight_scatter = R1_weight.reshape(len(spans)*len(l2ds)*len(gammas))

### m1,f1-R1 plot
# m1,f1-R1 plot for each span
fig = plt.figure()
ax = fig.gca(projection='3d')

for i in range(len(spans)):
    surf = ax.plot_surface(m1[i], f1[i], R1_weight[i])

ax.set_title('$m_1,f_1-R_1 plot$')
ax.set_xlabel('$m_1$ [kg]')
ax.set_ylabel('$f_1$ [Hz]')
ax.set_zlabel('$R_1$ [-]')

ax.set_zlim(zmin=0)

# m1,f1-R1 plot for all spans
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_surface(m1_grid_mergeSpan, f1_grid_mergeSpan, R1_weight_grid_mergeSpan)

ax.set_title('$m_1,f_1-R_1 plot$')
ax.set_xlabel('$m_1$ [kg]')
ax.set_ylabel('$f_1$ [Hz]')
ax.set_zlabel('$R_1$ [-]')

ax.set_zlim(zmin=0)


### m1,f1-R1 contour plot
# m1,f1-R1 contour plot for each span separately
fig, ax = plt.subplots()
ax.set_title('$m_1,f_1-R_1$ contour')
ax.set_xlabel('$m_1$ [kg]')
ax.set_ylabel('$f_1$ [Hz]')

colors = ['b', 'g', 'r', 'c', 'm', 'k']

for i in range(len(spans)):
    contour = ax.contour(m1[i], f1[i], R1_weight[i],[1,2,4,6,8,12,16,20,28,36,48,60,72],colors=colors[i])
    ax.clabel(contour,fontsize=10,fmt='%1.1f')
    contour.collections[i].set_label('span='+str(spans[i])+'m')
ax.legend()

# m1,f1-R1 contour plot for all spans
fig, ax = plt.subplots()
ax.set_title('$m_1,f_1-R_1$ contour')
ax.set_xlabel('$m_1$ [kg]')
ax.set_ylabel('$f_1$ [Hz]')

contour = ax.contour(m1_grid_mergeSpan, f1_grid_mergeSpan, R1_weight_grid_mergeSpan,[1,2,4,6,8,12,16,20,28,36,48,60,72])
ax.clabel(contour,fontsize=10,fmt='%1.1f')



### m1,f1-R1 scatter plot
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.set_title('$m_1,f_1-R_1$ scatter plot')
ax.set_xlabel('$m_1$ [kg]')
ax.set_ylabel('$f_1$ [Hz]')
ax.set_zlabel('$R_1$ [-]')

ax.scatter(m1_scatter,f1_scatter,R1_weight_scatter)

### interpolate
m1_intp = [300]
f1_intp = [100]

m1_f1_intp_grid_m1, m1_f1_intp_grid_f1 = np.meshgrid(m1_intp,f1_intp)


R1_weight_intp = griddata(np.vstack((m1_scatter,f1_scatter)).T, R1_weight_scatter, (m1_f1_intp_grid_m1, m1_f1_intp_grid_f1), method='linear')
if math.isnan(float(R1_weight_intp)):
    print('R1 is nan')
print(m1_f1_intp_grid_m1)
print(m1_f1_intp_grid_f1)
print(R1_weight_intp)
print(R1_weight[0,0,2])

plt.show()