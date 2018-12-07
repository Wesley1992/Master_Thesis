import pickle
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.ticker as mticker

def log_tick_formatter(val, pos=None):
    return "{:.2e}".format(10**val)

### input for loading file
spans = [5,8,10]
l2ds = [10,15,20]
# gammas = [0.1,1,10]

R_weight = []
f_n = []
m_n = []

for i in range(len(spans)):
    R_weight.append([])
    f_n.append([])
    m_n.append([])
    for j in range(len(l2ds)):
        R_weight[i].append([])
        f_n[i].append([])
        m_n[i].append([])
        # for k in range(len(gammas)):
        #     R_weight[i][j].append([])
        #     f_n[i][j].append([])
        #     m_n[i][j].append([])

            # model_file = 'D:/Master_Thesis/footfall_analysis/data/data_mdl_t0_01span/data_mdl_span' + str(spans[i]).replace('.', '_') + '_l2d' + str(l2ds[j]).replace('.','_') + '_gamma' + str(gammas[k]).replace('.', '_') + '.pkl'
        model_file = 'D:/Master_Thesis/footfall_analysis_plate/data_mdl_plate_span' + str(spans[i]).replace('.', '_') + '_l2d' + str(l2ds[j]).replace('.','_')  + '.pkl'

        R_weight[i][j] = pickle.load(open(model_file,'rb'))[-2][-1]
        f_n[i][j] = pickle.load(open(model_file,'rb'))[4]
        m_n[i][j] = pickle.load(open(model_file,'rb'))[5]

### generate the meshgrid for plot
# l2ds_meshgrid,gammas_meshgrid = np.meshgrid(l2ds,gammas)
R_weight = np.reshape(np.array([R_weight[i][j] for i in range(len(spans)) for j in range(len(l2ds))]),(len(spans),len(l2ds)))
f1 = np.reshape(np.array([f_n[i][j][0] for i in range(len(spans)) for j in range(len(l2ds))]),(len(spans),len(l2ds)))
m1 = np.reshape(np.array([m_n[i][j][0] for i in range(len(spans)) for j in range(len(l2ds))]),(len(spans),len(l2ds)))
# f1_meshgrid, m1_meshgrid = np.meshgrid(f1[:9],m1[:9])


# ### l2d,gamma-R plot with span=5m
# fig = plt.figure()
# ax = fig.gca(projection='3d')
#
# surf = ax.plot_surface(l2ds_meshgrid, np.log10(gammas_meshgrid), np.transpose(R_weight[0]))
# cset = ax.contour(l2ds_meshgrid, np.log10(gammas_meshgrid), np.transpose(R_weight[0]),offset=0,cmap=cm.coolwarm)
#
# ax.set_title('l/d,$t_v$/$t_r$ - R plot with span=5m')
# ax.set_xlabel('l/d')
# ax.set_ylabel('$t_v$/$t_r$')
# ax.set_zlabel('R [-]')
#
# ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
# ax.set_zlim(zmin=0)
#
# fig.colorbar(cset)
#
# ### l2d,gamma-R contour plot
# for i in range(len(spans)):
#     fig, ax = plt.subplots()
#     ax.set_title('l/d,$t_v/t_r$ - R contour (span=' + str(spans[i]) + 'm)')
#     ax.set_xlabel('l/d')
#     ax.set_ylabel('$t_v/t_r$')
#     contour = ax.contour(l2ds_meshgrid, gammas_meshgrid, np.transpose(R_weight[i]),[1,8,16,24,32,40,48,56,64,72,80])
#     ax.clabel(contour,fontsize=10,fmt='%1.1f')
#     ax.set_yscale('log')

### f1,m1-R plot
fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(m1, f1, R_weight)
cset = ax.contour(m1, f1, R_weight,offset=0,cmap=cm.coolwarm)

ax.set_title('$f_1$,$m_1$-R plot')
ax.set_xlabel('$m_1$ [kg]')
ax.set_ylabel('$f_1$ [Hz]')
ax.set_zlabel('R [-]')

ax.set_zlim(zmin=0)

fig.colorbar(cset)

### m1,f1-R contour plot

# for i in range(len(spans)):
fig, ax = plt.subplots()
ax.set_title('m1,f1-R contour')
ax.set_xlabel('$m_1$ [kg]')
ax.set_ylabel('$f_1$ [Hz]')
contour = ax.contour(m1, f1, R_weight,[0.25,0.5,1,2,4,8,16])
ax.clabel(contour,fontsize=10,fmt='%1.1f')

### span-R plot with varying gammas, l/d=15
# fig, ax = plt.subplots()
# ax.set_title('Response factor in relation to span (l/d=15)')
# ax.plot(spans,R_weight[:,1,0])
# ax.plot(spans,R_weight[:,1,1])
# ax.plot(spans,R_weight[:,1,2])
# ax.set_xlabel('Span [m]')
# ax.set_ylabel('R [-]')
# ax.legend(['$t_v/t_r=0.1$','$t_v/t_r=1$','$t_v/t_r=10$'])

### span-R plot with varying l/d
fig, ax = plt.subplots()
ax.set_title('Response factor in relation to span')
ax.plot(spans,R_weight[:,0])
ax.plot(spans,R_weight[:,1])
ax.plot(spans,R_weight[:,2])
ax.set_xlabel('Span [m]')
ax.set_ylabel('R [-]')
ax.legend(['l/d=10','l/d=15','l/d=20'])

### span-m1 plot with varying l/d
fig, ax = plt.subplots()
ax.set_title('$m_1$ in relation to span')
ax.plot(spans,m1[:,0])
ax.plot(spans,m1[:,1])
ax.plot(spans,m1[:,2])
ax.set_xlabel('Span [m]')
ax.set_ylabel('$m_1$ [kg]')
ax.legend(['l/d=10','l/d=15','l/d=20'])

### span-f1 plot with varying l/d
fig, ax = plt.subplots()
ax.set_title('$f_1$ in relation to span')
ax.plot(spans,f1[:,0])
ax.plot(spans,f1[:,1])
ax.plot(spans,f1[:,2])
ax.set_xlabel('Span [m]')
ax.set_ylabel('$f_1$ [Hz]')
ax.legend(['l/d=10','l/d=15','l/d=20'])


plt.show()
