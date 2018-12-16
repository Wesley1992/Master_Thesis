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
spans = [5,6,7,8,9,10]
l2ds = [10,12.5,15,17.5,20]
gammas = [0.1,0.5,1,2,5,10]

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
            model_file = 'D:/Master_Thesis/code_data/footfall_analysis/data/data_mdl_0_4m/data_mdl_span' + str(spans[i]).replace('.', '_') + '_l2d' + str(l2ds[j]).replace('.','_') + '_gamma' + str(gammas[k]).replace('.', '_') + '.pkl'

            R_weight[i][j][k] = pickle.load(open(model_file,'rb'))[-2][-1]
            R1_weight[i][j][k] = pickle.load(open(model_file,'rb'))[-2][0]
            f_n[i][j][k] = pickle.load(open(model_file,'rb'))[0]
            m_n[i][j][k] = pickle.load(open(model_file,'rb'))[1]

### generate the meshgrid for plot
l2d_gamma_mesh_l2d, l2d_gamma_mesh_gamma = np.meshgrid(l2ds,gammas)
l_gamma_mesh_l, l_gamma_mesh_gamma = np.meshgrid(spans,gammas)
l_l2d_mesh_l, l_l2d_mesh_l2d = np.meshgrid(spans,l2ds)

R_weight = np.reshape(np.array([R_weight[i][j][k] for i in range(len(spans)) for j in range(len(l2ds)) for k in range(len(gammas))]),(len(spans),len(l2ds),len(gammas)))
R1_weight = np.reshape(np.array([R1_weight[i][j][k] for i in range(len(spans)) for j in range(len(l2ds)) for k in range(len(gammas))]),(len(spans),len(l2ds),len(gammas)))
f1 = np.reshape(np.array([f_n[i][j][k][0] for i in range(len(spans)) for j in range(len(l2ds)) for k in range(len(gammas))]),(len(spans),len(l2ds),len(gammas)))
m1 = np.reshape(np.array([m_n[i][j][k][0] for i in range(len(spans)) for j in range(len(l2ds)) for k in range(len(gammas))]),(len(spans),len(l2ds),len(gammas)))


### l2d,gamma-R plot with span=5m
# fig = plt.figure()
# ax = fig.gca(projection='3d')
#
# surf = ax.plot_surface(12d_gamma_mesh_l2d, np.log10(12d_gamma_mesh_gamma), np.transpose(R_weight[5]))
# cset = ax.contour(12d_gamma_mesh_l2d, np.log10(12d_gamma_mesh_gamma), np.transpose(R_weight[5]),offset=0,cmap=cm.coolwarm)
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

# ### l2d,gamma-R contour plot
# for i in range(len(spans)):
#     fig, ax = plt.subplots()
#     ax.set_title('l/d,$t_v/t_r$ - R contour (span=' + str(spans[i]) + 'm)')
#     ax.set_xlabel('l/d')
#     ax.set_ylabel('$t_v/t_r$')
#     contour = ax.contour(l2d_gamma_mesh_l2d, l2d_gamma_mesh_gamma, np.transpose(R_weight[i]),[1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120])
#     ax.clabel(contour,fontsize=10,fmt='%1.1f')
#     ax.set_yscale('log')
#
# plt.show()

# ### f1,m1-R1 plot with span=5m
# fig = plt.figure()
# ax = fig.gca(projection='3d')
#
# surf = ax.plot_surface(m1[0], f1[0], R1_weight[0])
# cset = ax.contour(m1[0], f1[0], R1_weight[0],offset=0,cmap=cm.coolwarm)
#
# ax.set_title('$f_1$,$m_1$-$R_1$ plot with span=5m')
# ax.set_xlabel('$m_1$ [kg]')
# ax.set_ylabel('$f_1$ [Hz]')
# ax.set_zlabel('$R_1$ [-]')
#
# ax.set_zlim(zmin=0)
# fig.colorbar(cset)

### m1,f1-R1 contour plot
fig, ax = plt.subplots()
ax.set_title('$m_1,f_1-R_1$ contour')
ax.set_xlabel('$m_1$ [kg]')
ax.set_ylabel('$f_1$ [Hz]')

colors = ['b', 'g', 'r', 'c', 'm', 'k']

for i in range(len(spans)):
    contour = ax.contour(m1[i], f1[i], R1_weight[i],[1,2,4,6,8,12,16,20,28,36,48,60],colors=colors[i])
    ax.clabel(contour,fontsize=10,fmt='%1.1f')
    contour.collections[i].set_label('span='+str(spans[i])+'m')
ax.legend()
#
# plt.show()
#
### m1,f1-R contour plot
fig, ax = plt.subplots()
ax.set_title('$m_1,f_1-R$ contour')
ax.set_xlabel('$m_1$ [kg]')
ax.set_ylabel('$f_1$ [Hz]')

colors = ['b', 'g', 'r', 'c', 'm', 'k']

for i in range(len(spans)):
    contour = ax.contour(m1[i], f1[i], R_weight[i],[1,2,4,6,8,12,20,28,36,48,60,80,100,120],colors=colors[i])
    ax.clabel(contour,fontsize=10,fmt='%1.1f')
    contour.collections[i].set_label('span='+str(spans[i])+'m')

ax.legend()

plt.show()

#
#
# ### span,gamma-m1 contour plot
# fig, ax = plt.subplots()
# ax.set_title('span,$t_v/t_r-m_1$ contour')
# ax.set_xlabel('span [m]')
# ax.set_ylabel('$t_v/t_r$')
# ax.set_yscale('log')
# colors = ['b', 'g', 'r', 'c', 'm']
#
# for i in range(len(l2ds)):
#     contour = ax.contour(l_gamma_mesh_l, l_gamma_mesh_gamma, np.transpose(m1[:,i,:]),[200,300,400,500,750,1000,1500,2000,3000,4000,5000,6000],colors=colors[i])
#     ax.clabel(contour,fontsize=10,fmt='%1.1f')
#     contour.collections[i].set_label('l/d='+str(l2ds[i]))
# ax.legend()
#
# ### l/d,gamma-m1 contour plot
# fig, ax = plt.subplots()
# ax.set_title('l/d,$t_v/t_r-m_1$ contour')
# ax.set_xlabel('l/d')
# ax.set_ylabel('$t_v/t_r$')
# ax.set_yscale('log')
# colors = ['b', 'g', 'r', 'c', 'm', 'k']
#
# for i in range(len(spans)):
#     contour = ax.contour(l2d_gamma_mesh_l2d, l2d_gamma_mesh_gamma, np.transpose(m1[i,:,:]),[200,300,400,500,750,1000,1500,2000,3000,4000,5000,6000],colors=colors[i])
#     ax.clabel(contour,fontsize=10,fmt='%1.1f')
#     contour.collections[i].set_label('span='+str(spans[i]))
# ax.legend()
#
# ### span,gamma-f1 contour plot
# fig, ax = plt.subplots()
# ax.set_title('span,$t_v/t_r-f_1$ contour')
# ax.set_xlabel('span [m]')
# ax.set_ylabel('$t_v/t_r$')
# ax.set_yscale('log')
# colors = ['b', 'g', 'r', 'c', 'm']
#
# for i in range(len(l2ds)):
#     contour = ax.contour(l_gamma_mesh_l, l_gamma_mesh_gamma, np.transpose(f1[:,i,:]),colors=colors[i])
#     ax.clabel(contour,fontsize=10,fmt='%1.1f')
#     contour.collections[i].set_label('l/d='+str(l2ds[i]))
# ax.legend()
#
# ### l/d,gamma-f1 contour plot
# fig, ax = plt.subplots()
# ax.set_title('l/d,$t_v/t_r-f_1$ contour')
# ax.set_xlabel('l/d')
# ax.set_ylabel('$t_v/t_r$')
# ax.set_yscale('log')
# colors = ['b', 'g', 'r', 'c', 'm','k']
#
# for i in range(len(spans)):
#     contour = ax.contour(l2d_gamma_mesh_l2d, l2d_gamma_mesh_gamma, np.transpose(f1[i,:,:]),colors=colors[i])
#     ax.clabel(contour,fontsize=10,fmt='%1.1f')
#     contour.collections[i].set_label('span='+str(spans[i]))
# ax.legend()
#
# ### span,l/d-f1 contour plot
# fig, ax = plt.subplots()
# ax.set_title('span,l/d-$f_1$ contour')
# ax.set_xlabel('span')
# ax.set_ylabel('l/d')
# colors = ['b', 'g', 'r', 'c', 'm','k']
#
# for i in range(len(gammas)):
#     contour = ax.contour(l_l2d_mesh_l, l_l2d_mesh_l2d, np.transpose(f1[:,:,i]),colors=colors[i])
#     ax.clabel(contour,fontsize=10,fmt='%1.1f')
#     contour.collections[i].set_label('$t_v/t_r=$'+str(gammas[i]))
# ax.legend()
#
# ### l/d-R plot with varying gamma groups (diff. spans in each group)
# fig, ax = plt.subplots()
# ax.set_title('Response factor in relation to l/d')
# legend = []
# colors = ['b', 'g', 'r', 'c', 'm', 'k']
# markers = ['o','^','s','x','2','d']
# for i in range(len(gammas)):
#     for j in range(len(spans)):
#         ax.plot(l2ds,R_weight[j,:,i],color=colors[i],marker=markers[j])
#         legend.append('span='+str(spans[j])+'m, $t_v/t_r$='+str(gammas[i]))
# ax.set_xlabel('l/d')
# ax.set_ylabel('R [-]')
# ax.legend(legend)
#
# ### gamma-R plot with varying l/d groups (diff. spans in each group)
# fig, ax = plt.subplots()
# ax.set_title('Response factor in relation to span/depth ratio')
# legend = []
# colors = ['b', 'g', 'r', 'c', 'm']
# markers = ['o','^','s','x','2','d']
# for i in range(len(l2ds)):
#     for j in range(len(spans)):
#         ax.plot(gammas,R_weight[j,i,:],color=colors[i],marker=markers[j])
#         legend.append('span='+str(spans[j])+'m, $l/d$='+str(l2ds[i]))
# ax.set_xlabel('$t_v/t_r$')
# ax.set_ylabel('R [-]')
# ax.set_xscale('log')
# ax.legend(legend)
#
# #
# ### gamma-f1,m1,R plot (span=5m,l/d=20)
# spans_cmp = [5]
# l2ds_cmp = [20]
# gammas_cmp = [0.1,0.5,1,2,5,10]
#
# fig, ax = plt.subplots()
# ax.set_title('Change of $f_1, m_1, R_1$ in relation to $t_v/t_r$')
# ax.set_xlabel('$t_v/t_r$')
# ax.set_ylabel('Normalized valued')
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.tick_params('y')
# ax.minorticks_on()
# ax.plot(gammas_cmp,f1[0,-1,:]/f1[0,-1,0])
# ax.plot(gammas_cmp,m1[0,-1,:]/m1[0,-1,0])
# ax.plot(gammas_cmp,R1_weight[0,-1,:]/R1_weight[0,-1,0])
# ax.legend(['$f_1/f_{1,gamma=0.1}$','$m_1/m_{1,gamma=0.1}$','$R_1/R_{1,gamma=0.1}$'])
#
# plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
#
# # archive
# ### span-R plot with varying gammas, l/d=15
# # fig, ax = plt.subplots()
# # ax.set_title('Response factor in relation to span (l/d=15)')
# # ax.plot(spans,R_weight[:,1,0])
# # ax.plot(spans,R_weight[:,1,1])
# # ax.plot(spans,R_weight[:,1,2])
# # ax.set_xlabel('Span [m]')
# # ax.set_ylabel('R [-]')
# # ax.legend(['$t_v/t_r=0.1$','$t_v/t_r=1$','$t_v/t_r=10$'])
# #
# # plt.show()
# #
# # ### span-R plot with varying l/d, gamma=1
# # fig, ax = plt.subplots()
# # ax.set_title('Response factor in relation to span ($t_v/t_r=1$)')
# # ax.plot(spans,R_weight[:,0,1])
# # ax.plot(spans,R_weight[:,1,1])
# # ax.plot(spans,R_weight[:,2,1])
# # ax.set_xlabel('Span [m]')
# # ax.set_ylabel('R [-]')
# # ax.legend(['l/d=10','l/d=15','l/d=20'])
# #
# # ### span-R plot with varying l/d, gamma=10
# # fig, ax = plt.subplots()
# # ax.set_title('Response factor in relation to span ($t_v/t_r=10$)')
# # ax.plot(spans,R_weight[:,0,2])
# # ax.plot(spans,R_weight[:,1,2])
# # ax.plot(spans,R_weight[:,2,2])
# # ax.set_xlabel('Span [m]')
# # ax.set_ylabel('R [-]')
# # ax.legend(['l/d=10','l/d=15','l/d=20'])
# #
# # ### span-m1 plot with varying gammas, l/d=15
# # fig, ax = plt.subplots()
# # ax.set_title('$m_1$ in relation to span (l/d=15)')
# # ax.plot(spans,m1[:,1,0])
# # ax.plot(spans,m1[:,1,1])
# # ax.plot(spans,m1[:,1,2])
# # ax.set_xlabel('Span [m]')
# # ax.set_ylabel('$m_1$ [kg]')
# # ax.legend(['$t_v/t_r=0.1$','$t_v/t_r=1$','$t_v/t_r=10$'])