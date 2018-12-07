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
gammas = [0.1,1,10]

R_weight = []
R1_weight = []
f_n = []
m_n = []

for i in range(len(spans)):
    R_weight.append([])
    R1_weight.append([])
    f_n.append([])
    m_n.append([])
    for j in range(len(l2ds)):
        R_weight[i].append([])
        R1_weight[i].append([])
        f_n[i].append([])
        m_n[i].append([])
        for k in range(len(gammas)):
            R_weight[i][j].append([])
            R1_weight[i][j].append([])
            f_n[i][j].append([])
            m_n[i][j].append([])

            # model_file = 'D:/Master_Thesis/footfall_analysis/data/data_mdl_t0_01span/data_mdl_span' + str(spans[i]).replace('.', '_') + '_l2d' + str(l2ds[j]).replace('.','_') + '_gamma' + str(gammas[k]).replace('.', '_') + '.pkl'
            model_file = 'D:/Master_Thesis/footfall_analysis/data_mdl_span' + str(spans[i]).replace('.', '_') + '_l2d' + str(l2ds[j]).replace('.','_') + '_gamma' + str(gammas[k]).replace('.', '_') + '.pkl'

            R_weight[i][j][k] = pickle.load(open(model_file,'rb'))[-1][-1]
            R1_weight[i][j][k] = pickle.load(open(model_file,'rb'))[-1][0]
            f_n[i][j][k] = pickle.load(open(model_file,'rb'))[4]
            m_n[i][j][k] = pickle.load(open(model_file,'rb'))[5]

### generate the meshgrid for plot
l2ds_meshgrid,gammas_meshgrid = np.meshgrid(l2ds,gammas)
R_weight = np.reshape(np.array([R_weight[i][j][k] for i in range(len(spans)) for j in range(len(l2ds)) for k in range(len(gammas))]),(len(spans),len(l2ds),len(gammas)))
R1_weight = np.reshape(np.array([R1_weight[i][j][k] for i in range(len(spans)) for j in range(len(l2ds)) for k in range(len(gammas))]),(len(spans),len(l2ds),len(gammas)))
f1 = np.reshape(np.array([f_n[i][j][k][0] for i in range(len(spans)) for j in range(len(l2ds)) for k in range(len(gammas))]),(len(spans),len(l2ds),len(gammas)))
m1 = np.reshape(np.array([m_n[i][j][k][0] for i in range(len(spans)) for j in range(len(l2ds)) for k in range(len(gammas))]),(len(spans),len(l2ds),len(gammas)))
# f1_meshgrid, m1_meshgrid = np.meshgrid(f1[:9],m1[:9])


### l2d,gamma-R plot with span=5m
fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(l2ds_meshgrid, np.log10(gammas_meshgrid), np.transpose(R_weight[0]))
cset = ax.contour(l2ds_meshgrid, np.log10(gammas_meshgrid), np.transpose(R_weight[0]),offset=0,cmap=cm.coolwarm)

ax.set_title('l/d,$t_v$/$t_r$ - R plot with span=5m')
ax.set_xlabel('l/d')
ax.set_ylabel('$t_v$/$t_r$')
ax.set_zlabel('R [-]')

ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
ax.set_zlim(zmin=0)

fig.colorbar(cset)

### l2d,gamma-R contour plot
for i in range(len(spans)):
    fig, ax = plt.subplots()
    ax.set_title('l/d,$t_v/t_r$ - R contour (span=' + str(spans[i]) + 'm)')
    ax.set_xlabel('l/d')
    ax.set_ylabel('$t_v/t_r$')
    contour = ax.contour(l2ds_meshgrid, gammas_meshgrid, np.transpose(R_weight[i]),[1,8,16,24,32,40,48,56,64,72,80])
    ax.clabel(contour,fontsize=10,fmt='%1.1f')
    ax.set_yscale('log')

### f1,m1-R1 plot with span=5m
fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(m1[0], f1[0], R1_weight[0])
cset = ax.contour(m1[0], f1[0], R1_weight[0],offset=0,cmap=cm.coolwarm)

ax.set_title('$f_1$,$m_1$-$R_1$ plot with span=5m')
ax.set_xlabel('$m_1$ [kg]')
ax.set_ylabel('$f_1$ [Hz]')
ax.set_zlabel('$R_1$ [-]')

ax.set_zlim(zmin=0)

fig.colorbar(cset)

### m1,f1-R1 contour plot

for i in range(len(spans)):
    fig, ax = plt.subplots()
    ax.set_title('$m_1,f_1-R_1$ contour (span=' + str(spans[i]) + 'm)')
    ax.set_xlabel('$m_1$ [kg]')
    ax.set_ylabel('$f_1$ [Hz]')
    contour = ax.contour(m1[i], f1[i], R1_weight[i],[1,2,4,8,16,32,48,64,80])
    ax.clabel(contour,fontsize=10,fmt='%1.1f')

# ### span-R plot with varying gammas, l/d=15
# fig, ax = plt.subplots()
# ax.set_title('Response factor in relation to span (l/d=15)')
# ax.plot(spans,R_weight[:,1,0])
# ax.plot(spans,R_weight[:,1,1])
# ax.plot(spans,R_weight[:,1,2])
# ax.set_xlabel('Span [m]')
# ax.set_ylabel('R [-]')
# ax.legend(['$t_v/t_r=0.1$','$t_v/t_r=1$','$t_v/t_r=10$'])
#
# ### span-R plot with varying l/d, gamma=1
# fig, ax = plt.subplots()
# ax.set_title('Response factor in relation to span ($t_v/t_r=1$)')
# ax.plot(spans,R_weight[:,0,1])
# ax.plot(spans,R_weight[:,1,1])
# ax.plot(spans,R_weight[:,2,1])
# ax.set_xlabel('Span [m]')
# ax.set_ylabel('R [-]')
# ax.legend(['l/d=10','l/d=15','l/d=20'])
#
# ### span-R plot with varying l/d, gamma=10
# fig, ax = plt.subplots()
# ax.set_title('Response factor in relation to span ($t_v/t_r=10$)')
# ax.plot(spans,R_weight[:,0,2])
# ax.plot(spans,R_weight[:,1,2])
# ax.plot(spans,R_weight[:,2,2])
# ax.set_xlabel('Span [m]')
# ax.set_ylabel('R [-]')
# ax.legend(['l/d=10','l/d=15','l/d=20'])
#
# ### span-m1 plot with varying gammas, l/d=15
# fig, ax = plt.subplots()
# ax.set_title('$m_1$ in relation to span (l/d=15)')
# ax.plot(spans,m1[:,1,0])
# ax.plot(spans,m1[:,1,1])
# ax.plot(spans,m1[:,1,2])
# ax.set_xlabel('Span [m]')
# ax.set_ylabel('$m_1$ [kg]')
# ax.legend(['$t_v/t_r=0.1$','$t_v/t_r=1$','$t_v/t_r=10$'])
#
# ### span-m1 plot with varying l/d, gamma=10
# fig, ax = plt.subplots()
# ax.set_title('$m_1$ in relation to span (gamma=10)')
# ax.plot(spans,m1[:,0,2])
# ax.plot(spans,m1[:,1,2])
# ax.plot(spans,m1[:,2,2])
# ax.set_xlabel('Span [m]')
# ax.set_ylabel('$m_1$ [kg]')
# ax.legend(['l/d=10','l/d=15','l/d=20'])
#
# ### span-f1 plot with varying gammas, l/d=15
# fig, ax = plt.subplots()
# ax.set_title('$f_1$ in relation to span (l/d=15)')
# ax.plot(spans,f1[:,1,0])
# ax.plot(spans,f1[:,1,1])
# ax.plot(spans,f1[:,1,2])
# ax.set_xlabel('Span [m]')
# ax.set_ylabel('$f_1$ [Hz]')
# ax.legend(['$t_v/t_r=0.1$','$t_v/t_r=1$','$t_v/t_r=10$'])
#
# ### span-f1 plot with varying l/d, gamma=10
# fig, ax = plt.subplots()
# ax.set_title('$f_1$ in relation to span (gamma=10)')
# ax.plot(spans,f1[:,0,2])
# ax.plot(spans,f1[:,1,2])
# ax.plot(spans,f1[:,2,2])
# ax.set_xlabel('Span [m]')
# ax.set_ylabel('$f_1$ [Hz]')
# ax.legend(['l/d=10','l/d=15','l/d=20'])

### l/d-R plot with varying gamma groups (diff. spans in each group)
fig, ax = plt.subplots()
ax.set_title('Response factor in relation to span/depth ratio')
legend = []
for i in range(len(gammas)):
    for j in range(len(spans)):
        ax.plot(l2ds,R_weight[j,:,i])
        legend.append('span='+str(spans[j])+'m, $t_v/t_r$='+str(gammas[i]))
ax.set_xlabel('l/d')
ax.set_ylabel('R [-]')
ax.legend(legend)

### gamma-R plot with varying l/d groups (diff. spans in each group)
fig, ax = plt.subplots()
ax.set_title('Response factor in relation to span/depth ratio')
legend = []
for i in range(len(l2ds)):
    for j in range(len(spans)):
        ax.plot(gammas,R_weight[j,:,i])
        legend.append('span='+str(spans[j])+'m, $l/d$='+str(l2ds[i]))
ax.set_xlabel('$t_v/t_r$')
ax.set_ylabel('R [-]')
ax.set_xscale('log')
ax.legend(legend)




plt.show()
