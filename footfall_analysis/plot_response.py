import pickle
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import pearsonr
import matplotlib as mpl
# import statistics.median as median
from sklearn.metrics import mean_squared_error as mse

mpl.rc('font', size=14)


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

R_weight_plate_m = []
R1_weight_plate_m = []
f_n_plate_m = []
m_n_plate_m = []


for i in range(len(spans)):
    R_weight.append([])
    R1_weight.append([])
    f_n.append([])
    m_n.append([])

    R_weight_plate_m.append([])
    R1_weight_plate_m.append([])
    f_n_plate_m.append([])
    m_n_plate_m.append([])

    for j in range(len(l2ds)):
        R_weight[i].append([])
        R1_weight[i].append([])
        f_n[i].append([])
        m_n[i].append([])

        R_weight_plate_m[i].append([])
        R1_weight_plate_m[i].append([])
        f_n_plate_m[i].append([])
        m_n_plate_m[i].append([])


        model_file_plate_m = 'D:/Master_Thesis/code_data/footfall_analysis_plate_0_4depth/data/data_mdl_plate_span' + str(spans[i]).replace('.', '_') + '_l2d' + str(l2ds[j]).replace('.', '_')+ '.pkl'
        R_weight_plate_m[i][j] = pickle.load(open(model_file_plate_m, 'rb'))[-2][-1]
        R1_weight_plate_m[i][j] = pickle.load(open(model_file_plate_m, 'rb'))[-2][0]
        f_n_plate_m[i][j] = pickle.load(open(model_file_plate_m, 'rb'))[0]
        m_n_plate_m[i][j] = pickle.load(open(model_file_plate_m, 'rb'))[1]

        for k in range(len(gammas)):
            R_weight[i][j].append([])
            R1_weight[i][j].append([])
            f_n[i][j].append([])
            m_n[i][j].append([])

            model_file = 'D:/Master_Thesis/code_data/footfall_analysis/data/data_mdl_0_4m/data_mdl_span' + str(spans[i]).replace('.', '_') + '_l2d' + str(l2ds[j]).replace('.','_') + '_gamma' + str(gammas[k]).replace('.', '_') + '.pkl'

            R_weight[i][j][k] = pickle.load(open(model_file,'rb'))[-2][-1]
            R1_weight[i][j][k] = pickle.load(open(model_file,'rb'))[-2][0]
            f_n[i][j][k] = pickle.load(open(model_file,'rb'))[0]
            m_n[i][j][k] = pickle.load(open(model_file,'rb'))[1]


### generate the meshgrid for plot
l2d_gamma_mesh_l2d, l2d_gamma_mesh_gamma = np.meshgrid(l2ds,gammas)
l_gamma_mesh_l, l_gamma_mesh_gamma = np.meshgrid(spans,gammas)
l_l2d_mesh_l, l_l2d_mesh_l2d = np.meshgrid(spans,l2ds)

### reshape in array
R_weight = np.reshape(np.array([R_weight[i][j][k] for i in range(len(spans)) for j in range(len(l2ds)) for k in range(len(gammas))]),(len(spans),len(l2ds),len(gammas)))
R1_weight = np.reshape(np.array([R1_weight[i][j][k] for i in range(len(spans)) for j in range(len(l2ds)) for k in range(len(gammas))]),(len(spans),len(l2ds),len(gammas)))
f1 = np.reshape(np.array([f_n[i][j][k][0] for i in range(len(spans)) for j in range(len(l2ds)) for k in range(len(gammas))]),(len(spans),len(l2ds),len(gammas)))
m1 = np.reshape(np.array([m_n[i][j][k][0] for i in range(len(spans)) for j in range(len(l2ds)) for k in range(len(gammas))]),(len(spans),len(l2ds),len(gammas)))

# R_weight_plate_g = np.reshape(np.array([R_weight_plate_g[i][j] for i in range(len(spans)) for j in range(len(l2ds))]),(len(spans),len(l2ds)))
# R1_weight_plate_g = np.reshape(np.array([R1_weight_plate_g[i][j] for i in range(len(spans)) for j in range(len(l2ds))]),(len(spans),len(l2ds)))
# f1_plate_g = np.reshape(np.array([f_n_plate_g[i][j][0] for i in range(len(spans)) for j in range(len(l2ds))]),(len(spans),len(l2ds)))
# m1_plate_g = np.reshape(np.array([m_n_plate_g[i][j][0] for i in range(len(spans)) for j in range(len(l2ds))]),(len(spans),len(l2ds)))

R_weight_plate_m = np.reshape(np.array([R_weight_plate_m[i][j] for i in range(len(spans)) for j in range(len(l2ds))]),(len(spans),len(l2ds)))
R1_weight_plate_m = np.reshape(np.array([R1_weight_plate_m[i][j] for i in range(len(spans)) for j in range(len(l2ds))]),(len(spans),len(l2ds)))
f1_plate_m = np.reshape(np.array([f_n_plate_m[i][j][0] for i in range(len(spans)) for j in range(len(l2ds))]),(len(spans),len(l2ds)))
m1_plate_m = np.reshape(np.array([m_n_plate_m[i][j][0] for i in range(len(spans)) for j in range(len(l2ds))]),(len(spans),len(l2ds)))

# m1/m
m12m = np.zeros((len(spans),len(l2ds),len(gammas)))
for i in range(len(spans)):
    for j in range(len(l2ds)):
        for k in range(len(gammas)):
            m12m[i,j,k] = m1[i,j,k]/(spans[i]**3/l2ds[j]*2400*0.4)



# ### l2d,gamma-R plot with span=5m
# # fig = plt.figure()
# # ax = fig.gca(projection='3d')
# #
# # surf = ax.plot_surface(l2d_gamma_mesh_l2d, np.log10(l2d_gamma_mesh_gamma), np.transpose(R_weight[5]))
# # cset = ax.contour(l2d_gamma_mesh_l2d, np.log10(l2d_gamma_mesh_gamma), np.transpose(R_weight[5]),offset=0,cmap=cm.coolwarm)
# #
# # ax.set_title('l/d,$t_v$/$t_r$ - R plot with span=5m')
# # ax.set_xlabel('l/d')
# # ax.set_ylabel('$t_v$/$t_r$')
# # ax.set_zlabel('R [-]')
# #
# # ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
# # ax.set_zlim(zmin=0)
# #
# # fig.colorbar(cset)
#

# ## l2d,gamma-R contour plot (for each span)
# for i in range(len(spans)):
#     fig, ax = plt.subplots()
#     ax.set_title('l/d,$t_v/t_r$ - R contour (span=' + str(spans[i]) + 'm)')
#     ax.set_xlabel('l/d [-]')
#     ax.set_ylabel('$t_v/t_r$ [-]')
#     contour = ax.contour(l2d_gamma_mesh_l2d, l2d_gamma_mesh_gamma, np.transpose(R_weight[i]),[1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120],cmap=cm.jet)
#     ax.clabel(contour,fontsize=12,fmt='%1.1f')
#     ax.set_yscale('log')
# #
#
# ### span,l2d-R plot of solid plate with the same mass as vaulted slab
# # fig = plt.figure()
# # ax = fig.gca(projection='3d')
# #
# # surf = ax.plot_surface(l_l2d_mesh_l, l_l2d_mesh_l2d, np.transpose(R1_weight_plate_m))
# # cset = ax.contour(l_l2d_mesh_l, l_l2d_mesh_l2d, np.transpose(R1_weight_plate_m),offset=0,cmap=cm.coolwarm)
# #
# # ax.set_title('span,l/d - R plot of solid plate with the same mass as vaulted slab')
# # ax.set_xlabel('span [m]')
# # ax.set_ylabel('l/d')
# # ax.set_zlabel('R [-]')
# #
# # ax.set_zlim(zmin=0)
# #
# # fig.colorbar(cset)
#
#
# ### span,l2d-R contour plot of solid plate with the same mass as vaulted slab
# # fig, ax = plt.subplots()
# # ax.set_title('span,l/d - R contour plot of solid plate with the same mass as vaulted slab')
# # ax.set_xlabel('span [m]')
# # ax.set_ylabel('l/d')
# # contour = ax.contour(l_l2d_mesh_l, l_l2d_mesh_l2d, np.transpose(R1_weight_plate_m),[1,2,4,6,8,10,12,16,20,24,28,32,40,48,56,64,72])
# # ax.clabel(contour,fontsize=10,fmt='%1.1f')
#
#
# ### f1,m1-R1 plot with span=5m
# # fig = plt.figure()
# # ax = fig.gca(projection='3d')
# #
# # surf = ax.plot_surface(m1[0], f1[0], R1_weight[0])
# # cset = ax.contour(m1[0], f1[0], R1_weight[0],offset=0,cmap=cm.coolwarm)
# #
# # ax.set_title('$f_1$,$m_1$-$R_1$ plot with span=5m')
# # ax.set_xlabel('$m_1$ [kg]')
# # ax.set_ylabel('$f_1$ [Hz]')
# # ax.set_zlabel('$R_1$ [-]')
# #
# # ax.set_zlim(zmin=0)
# # fig.colorbar(cset)
#
#
# ## m1,f1-R1 plot
# fig = plt.figure()
# ax = fig.gca(projection='3d')
#
# for i in range(len(spans)):
#     surf = ax.plot_surface(m1[i], f1[i], R1_weight[i])
#
# ax.set_title('$m_1,f_1-R_1 plot$')
# ax.set_xlabel('$m_1$ [kg]')
# ax.set_ylabel('$f_1$ [Hz]')
# ax.set_zlabel('$R_1$ [-]')
#
# # ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
# ax.set_zlim(zmin=0)
#
# # fig.colorbar(cset)
#
#
## m1,f1-R1 contour plot
fig, ax = plt.subplots()
ax.set_title('$m_1,f_1-R_1$ contour')
ax.set_xlabel('$m_1$ [kg]')
ax.set_ylabel('$f_1$ [Hz]')

colors = ['b', 'g', 'r', 'c', 'm', 'k']

for i in range(len(spans)):
    contour = ax.contour(m1[i], f1[i], R1_weight[i],[1,2,4,6,8,12,16,20,28,36,48,60,72],colors=colors[i])
    ax.clabel(contour,fontsize=12,fmt='%1.1f')
    contour.collections[i].set_label('span='+str(spans[i])+'m')

ax.scatter([420,626,868,1023,1564],[61,60,59,56,52],color='r',s=10)
ax.set_xlim(xmin=0,xmax=2000)
ax.set_ylim(ymin=45,ymax=70)
ax.grid()
ax.minorticks_on()
ax.grid(which='minor',axis='x',visible=True)
ax.grid(which='minor',axis='y',visible=True)
ax.legend()

#
# ### f1,m1-R1 plot of solid plate with the same geometry as vaulted slab
# # fig = plt.figure()
# # ax = fig.gca(projection='3d')
# #
# # surf = ax.plot_surface(m1_plate_g, f1_plate_g, R1_weight_plate_g)
# # cset = ax.contour(m1_plate_g, f1_plate_g, R1_weight_plate_g,offset=0,cmap=cm.coolwarm)
# #
# # ax.set_title('$f_1$,$m_1$-$R_1$ plot of solid plate with the same geometry as vaulted slab')
# # ax.set_xlabel('$m_1$ [kg]')
# # ax.set_ylabel('$f_1$ [Hz]')
# # ax.set_zlabel('$R_1$ [-]')
# #
# # ax.set_zlim(zmin=0)
# # fig.colorbar(cset)
#
#
# ### m1,f1-R1 contour plot for plate with the same geometry
# # fig, ax = plt.subplots()
# # ax.set_title('$m_1,f_1-R_1$ contour of solid plate with the same outer geometry as vaulted slab')
# # ax.set_xlabel('$m_1$ [kg]')
# # ax.set_ylabel('$f_1$ [Hz]')
# #
# # contour = ax.contour(m1_plate_g, f1_plate_g, R1_weight_plate_g,[0.25,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5])
# # ax.clabel(contour,fontsize=10,fmt='%1.1f')
#
#
# ### f1,m1-R1 plot of solid plate with the same mass as vaulted slab
# # fig = plt.figure()
# # ax = fig.gca(projection='3d')
# #
# # surf = ax.plot_surface(m1_plate_m, f1_plate_m, R1_weight_plate_m)
# # cset = ax.contour(m1_plate_m, f1_plate_m, R1_weight_plate_m,offset=0,cmap=cm.coolwarm)
# #
# # ax.set_title('$f_1$,$m_1$-$R_1$ plot of solid plate with the same mass as vaulted slab')
# # ax.set_xlabel('$m_1$ [kg]')
# # ax.set_ylabel('$f_1$ [Hz]')
# # ax.set_zlabel('$R_1$ [-]')
# #
# # ax.set_zlim(zmin=0)
# # fig.colorbar(cset)
#
#
# ### m1,f1-R1 contour plot for plate with the same mass
# # fig, ax = plt.subplots()
# # ax.set_title('$m_1,f_1-R_1$ contour of solid plate with the same mass as vaulted slab')
# # ax.set_xlabel('$m_1$ [kg]')
# # ax.set_ylabel('$f_1$ [Hz]')
# #
# # contour = ax.contour(m1_plate_m, f1_plate_m, R1_weight_plate_m,[1,2,3,4,5,6,7,8,10,12,14,16,20,24,32,40,48])
# # ax.clabel(contour,fontsize=10,fmt='%1.1f')
#
#
# ### m1,f1-R contour plot
# # fig, ax = plt.subplots()
# # ax.set_title('$m_1,f_1-R$ contour')
# # ax.set_xlabel('$m_1$ [kg]')
# # ax.set_ylabel('$f_1$ [Hz]')
# #
# # colors = ['b', 'g', 'r', 'c', 'm', 'k']
# #
# # for i in range(len(spans)):
# #     contour = ax.contour(m1[i], f1[i], R_weight[i],[1,2,4,6,8,12,20,28,36,48,60,80,100,120],colors=colors[i])
# #     ax.clabel(contour,fontsize=10,fmt='%1.1f')
# #     contour.collections[i].set_label('span='+str(spans[i])+'m')
# #
# # ax.legend()
#
#
# ### span,gamma-m1 contour plot
# fig, ax = plt.subplots()
# ax.set_title('span,$t_v/t_r-m_1$ contour')
# ax.set_xlabel('span [m]')
# ax.set_ylabel('$t_v/t_r$')
# ax.set_yscale('log')
# colors = ['C0', 'C1', 'C2', 'C3', 'C4']
# # colors = ['b', 'g', 'r', 'c', 'm']
#
# for i in range(len(l2ds)):
#     contour = ax.contour(l_gamma_mesh_l, l_gamma_mesh_gamma, np.transpose(m1[:,i,:]),[200,300,400,500,750,1000,1500,2000,3000,4000,5000,6000],colors=colors[i])
#     ax.clabel(contour,fontsize=10,fmt='%1.1f')
#     contour.collections[i].set_label('l/d='+str(l2ds[i]))
# ax.legend()
#
#
# ### l/d,gamma-m1 contour plot
# fig, ax = plt.subplots()
# ax.set_title('l/d,$t_v/t_r-m_1$ contour')
# ax.set_xlabel('l/d')
# ax.set_ylabel('$t_v/t_r$')
# ax.set_yscale('log')
# colors = ['C0', 'C1', 'C2', 'C3', 'C4','C5']
#
# for i in range(len(spans)):
#     contour = ax.contour(l2d_gamma_mesh_l2d, l2d_gamma_mesh_gamma, np.transpose(m1[i,:,:]),[200,300,400,500,750,1000,1500,2000,3000,4000,5000,6000],colors=colors[i])
#     ax.clabel(contour,fontsize=10,fmt='%1.1f')
#     contour.collections[i].set_label('span='+str(spans[i]))
# ax.legend()
#
# ### span,l/d-m1 contour plot
# fig, ax = plt.subplots()
# ax.set_title('span, l/d -$m_1$ contour')
# ax.set_xlabel('span [m]')
# ax.set_ylabel('l/d')
# colors = ['C0', 'C1', 'C2', 'C3', 'C4','C5']
#
# for i in range(len(gammas)):
#     contour = ax.contour(l_l2d_mesh_l, l_l2d_mesh_l2d, np.transpose(m1[:,:,i]),[200,300,400,500,750,1000,1500,2000,3000,4000,5000,6000],colors=colors[i])
#     ax.clabel(contour,fontsize=10,fmt='%1.1f')
#     contour.collections[i].set_label('$t_v/t_r$='+str(gammas[i]))
# ax.legend()
#
#
#
# ## sensitivity analysis, l,l/d,gamma-m1 correlation
# # dimension of model
# M = 3
# IOC = np.zeros(M)
#
# x_span = []
# y_span = []
# for i in range(len(spans)):
#     n_points = (m1[i,:,:].flatten()).shape[0]
#     x_span.append(spans[i]*np.ones(n_points))
#     y_span.append(m1[i,:,:].flatten())
# x_span = np.array(x_span).flatten()
# y_span = np.array(y_span).flatten()
# IOC[0] = pearsonr(x_span,y_span)[0]
#
# x_l2d = []
# y_l2d = []
# for i in range(len(l2ds)):
#     n_points = (m1[:,i,:].flatten()).shape[0]
#     x_l2d.append(l2ds[i]*np.ones(n_points))
#     y_l2d.append(m1[:,i,:].flatten())
# x_l2d = np.array(x_l2d).flatten()
# y_l2d = np.array(y_l2d).flatten()
# IOC[1] = pearsonr(x_l2d,y_l2d)[0]
#
# x_gamma = []
# y_gamma = []
# for i in range(len(gammas)):
#     n_points = (m1[:,:,i].flatten()).shape[0]
#     x_gamma.append(gammas[i]*np.ones(n_points))
#     y_gamma.append(m1[:,:,i].flatten())
# x_gamma = np.array(x_gamma).flatten()
# y_gamma = np.array(y_gamma).flatten()
# IOC[2] = pearsonr(np.log10(x_gamma),y_gamma)[0]
#
#
# # input/output plot
# fig  = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_title('Span-$m_1$ scatter plot with IOC = {:.3f}'.format(IOC[0]))
# ax.set_xlabel('Span [m]')
# ax.set_ylabel('$m_1$ [kg]')
# for i in range(len(spans)):
#     n_points = (m1[i,:,:].flatten()).shape[0]
#     ax.scatter(spans[i]*np.ones(n_points),m1[i,:,:].flatten(),color='#1f77b4',s=10)
#
# fig  = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_title('l/d-$m_1$ scatter plot with IOC = {:.3f}'.format(IOC[1]))
# ax.set_xlabel('l/d [-]')
# ax.set_ylabel('$m_1$ [kg]')
# for i in range(len(l2ds)):
#     n_points = (m1[:,i,:].flatten()).shape[0]
#     ax.scatter(l2ds[i]*np.ones(n_points),m1[:,i,:].flatten(),color='#1f77b4', s=10)
#     ax.set_xticks(ticks=l2ds)
#
# fig  = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_title('$t_v/t_r$-$m_1$ scatter plot with IOC = {:.3f}'.format(IOC[2]))
# ax.set_xlabel('$t_v/t_r$ [-]')
# ax.set_xscale('log')
# ax.set_ylabel('$m_1$ [kg]')
# for i in range(len(gammas)):
#     n_points = (m1[:,:,i].flatten()).shape[0]
#     ax.scatter(gammas[i]*np.ones(n_points),m1[:,:,i].flatten(),color='#1f77b4', s=10)

# ## sensitivity analysis, l,l/d,gamma-m1/m correlation
# # dimension of model
# M = 3
# IOC = np.zeros(M)
#
# x_span = []
# y_span = []
# for i in range(len(spans)):
#     n_points = (m12m[i,:,:].flatten()).shape[0]
#     x_span.append(spans[i]*np.ones(n_points))
#     y_span.append(m12m[i,:,:].flatten())
# x = np.array(x_span).flatten()
# y = np.array(y_span).flatten()
# IOC[0] = pearsonr(x,y)[0]
#
# x_l2d = []
# y_l2d = []
# for i in range(len(l2ds)):
#     n_points = (m12m[:,i,:].flatten()).shape[0]
#     x_l2d.append(l2ds[i]*np.ones(n_points))
#     y_l2d.append(m12m[:,i,:].flatten())
# x = np.array(x_l2d).flatten()
# y = np.array(y_l2d).flatten()
# IOC[1] = pearsonr(x,y)[0]
#
# x_gamma = []
# y_gamma = []
# for i in range(len(gammas)):
#     n_points = (m12m[:,:,i].flatten()).shape[0]
#     x_gamma.append(gammas[i]*np.ones(n_points))
#     y_gamma.append(m12m[:,:,i].flatten())
# x = np.array(x_gamma).flatten()
# y = np.array(y_gamma).flatten()
# IOC[2] = pearsonr(np.log(x),y)[0]
#
#
# # input/output plot
# fig  = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_title('Span-$m_1/m$ scatter plot with IOC = {:.3f}'.format(IOC[0]))
# ax.set_xlabel('Span [m]')
# ax.set_ylabel('$m_1/m$ [-]')
# for i in range(len(spans)):
#     n_points = (m12m[i, :, :].flatten()).shape[0]
#     ax.scatter(spans[i]*np.ones(n_points),m12m[i,:,:].flatten(),color='#1f77b4',s=10)
#
# fig  = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_title('l/d-$m_1/m$ scatter plot with IOC = {:.3f}'.format(IOC[1]))
# ax.set_xlabel('l/d [-]')
# ax.set_ylabel('$m_1/m$ [-]')
# for i in range(len(l2ds)):
#     n_points = (m12m[:, i, :].flatten()).shape[0]
#     ax.scatter(l2ds[i]*np.ones(n_points),m12m[:,i,:].flatten(),color='#1f77b4',s=10)
#     ax.set_xticks(ticks=l2ds)
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_title('$t_v/t_r$-$m_1/m$ scatter plot with IOC = {:.3f}'.format(IOC[2]))
# ax.set_xlabel('$t_v/t_r$ [-]')
# ax.set_xscale('log')
# ax.set_ylabel('$m_1/m$ [-]')
# for i in range(len(gammas)):
#     n_points = (m12m[:, :, i].flatten()).shape[0]
#     ax.scatter(gammas[i] * np.ones(n_points), m12m[:, :, i].flatten(), color='#1f77b4', s=10)
#
# # l/d-m1/m plot
# colors = ['C0', 'C1', 'C2', 'C3', 'C4','C5']
# markers = ['o','^','s','x','*','d']
#
# fig, ax = plt.subplots()
# ax.set_title('Relative change of $m_1/m$ (normalized by l/d=10) in relation to l/d for $t_v/t_r=1$')
# ax.set_xlabel('l/d [-]')
# # ax.set_xlim(xmin=gammas[0]*0.91,xmax=gammas[-1]*1.09)
# ax.set_ylabel('relative change of $m_1/m$ [-]')
# ax.set_xticks(ticks=l2ds)
#
# for j in range(len(spans)):
#     ax.plot(l2ds,m12m[j,:,2]/m12m[j,0,2],color=colors[0],marker=markers[j],markersize=8,label='span='+str(spans[j])+'m')
# ax.legend()


# ## span,gamma-f1 contour plot
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
#
# ## l/d,gamma-f1 contour plot
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
#
# ## span,l/d-f1 contour plot
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
#
# ## sensitivity analysis, l,l/d,gamma-f1 correlation
# # dimension of model
# M = 3
# IOC = np.zeros(M)
#
# x_span = []
# y_span = []
# for i in range(len(spans)):
#     n_points = (f1[i,:,:].flatten()).shape[0]
#     x_span.append(spans[i]*np.ones(n_points))
#     y_span.append(f1[i,:,:].flatten())
# x_span = np.array(x_span).flatten()
# y_span = np.array(y_span).flatten()
# IOC[0] = pearsonr(x_span,y_span)[0]
#
# x_l2d = []
# y_l2d = []
# for i in range(len(l2ds)):
#     n_points = (f1[:,i,:].flatten()).shape[0]
#     x_l2d.append(l2ds[i]*np.ones(n_points))
#     y_l2d.append(f1[:,i,:].flatten())
# x_l2d = np.array(x_l2d).flatten()
# y_l2d = np.array(y_l2d).flatten()
# IOC[1] = pearsonr(x_l2d,y_l2d)[0]
#
# x_gamma = []
# y_gamma = []
# for i in range(len(gammas)):
#     n_points = (f1[:,:,i].flatten()).shape[0]
#     x_gamma.append(gammas[i]*np.ones(n_points))
#     y_gamma.append(f1[:,:,i].flatten())
# x_gamma = np.array(x_gamma).flatten()
# y_gamma = np.array(y_gamma).flatten()
# IOC[2] = pearsonr(np.log10(x_gamma),y_gamma)[0]
#
#
# # input/output plot
# fig  = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_title('span-$f_1$ scatter plot with IOC = {:.3f}'.format(IOC[0]))
# ax.set_xlabel('span [m]')
# ax.set_ylabel('$f_1$ [Hz]')
# for i in range(len(spans)):
#     n_points = (f1[i,:,:].flatten()).shape[0]
#     ax.scatter(spans[i]*np.ones(n_points),f1[i,:,:].flatten(),color='#1f77b4',s=10)
#
# fig  = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_title('l/d-$f_1$ scatter plot with IOC = {:.3f}'.format(IOC[1]))
# ax.set_xlabel('l/d [-]')
# ax.set_ylabel('$f_1$ [Hz]')
# for i in range(len(l2ds)):
#     n_points = (f1[:,i,:].flatten()).shape[0]
#     ax.scatter(l2ds[i]*np.ones(n_points),f1[:,i,:].flatten(),color='#1f77b4', s=10)
#     ax.set_xticks(ticks=l2ds)
#
# fig  = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_title('$t_v/t_r$-$f_1$ scatter plot with IOC = {:.3f}'.format(IOC[2]))
# ax.set_xlabel('$t_v/t_r$ [-]')
# ax.set_xscale('log')
# ax.set_ylabel('$f_1$ [Hz]')\
# for i in range(len(gammas)):
#     n_points = (f1[:,:,i].flatten()).shape[0]
#     ax.scatter(gammas[i]*np.ones(n_points),f1[:,:,i].flatten(),color='#1f77b4', s=10)


## sensitivity analysis, m1-f1 correlation
# # dimension of model
# IOC = pearsonr(m1.flatten(),f1.flatten())[0]
#
# fig  = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_title('$m_1$-$f_1$ scatter plot with IOC = {:.3f}'.format(IOC))
# ax.set_xlabel('$m_1$ [kg]')
# ax.set_ylabel('$f_1$ [Hz]')
#
# ax.scatter(m1.flatten(),f1.flatten(),color='#1f77b4', s=10)


#
# ## l/d-R plot with varying gamma groups (diff. span in each group)
# fig, ax = plt.subplots()
# ax.set_title('Response factor in relation to l/d')
# legend = []
# colors = ['b', 'g', 'r', 'c', 'm', 'k']
# markers = ['o','^','s','x','*','d']
# for i in range(len(gammas)):
#     for j in range(len(spans)):
#         ax.plot(l2ds,R_weight[j,:,i],color=colors[i],marker=markers[j])
#         legend.append('$t_v/t_r$='+str(gammas[i])+', span='+str(spans[j])+'m')
# ax.set_xlabel('l/d')
# ax.set_ylabel('R [-]')
# ax.legend(legend)
#
#
# ## gamma-R plot with varying l/d groups (diff. spans in each group)
# fig, ax = plt.subplots()
# ax.set_title('Response factor in relation to span/depth ratio')
# legend = []
# colors = ['b', 'g', 'r', 'c', 'm']
# markers = ['o','^','s','x','*','d']
# for i in range(len(l2ds)):
#     for j in range(len(spans)):
#         ax.plot(gammas,R_weight[j,i,:],color=colors[i],marker=markers[j])
#         legend.append('span='+str(spans[j])+'m, $l/d$='+str(l2ds[i]))
# ax.set_xlabel('$t_v/t_r$')
# ax.set_ylabel('R [-]')
# ax.set_xscale('log')
# ax.legend(legend)


# ## gamma-normalized f1,m1,R1 by tv/tr=0.1
#
# line_styles = ['-.','--','-']
# colors = ['b', 'g', 'r', 'c', 'm','k']
# markers = ['o','^','s','x','*']
#
# label_linestyle = ['$m_1/m_{1,t_v/t_r=0.1}$','$f_1/f_{1,t_v/t_r=0.1}$','$R_1/R_{1,t_v/t_r=0.1}$']
#
# fig, ax = plt.subplots()
# ax.set_title('Normalized $m_1, f_1, R_1$ in relation to $t_v/t_r$ (normalized by $t_v/t_r$=0.1)')
# ax.set_xlabel('$t_v/t_r$ [-]')
# ax.set_xlim(xmin=gammas[0]*0.91,xmax=gammas[-1]*1.09)
# ax.set_ylabel('Normalized Value [-]')
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.grid()
# ax.grid(which='minor',axis='x',visible=True)
# ax.grid(which='minor',axis='y',visible=True)
#
# for axis in [ax.xaxis,ax.yaxis]:
#     formatter = ScalarFormatter()
#     formatter.set_scientific(False)
#     axis.set_major_formatter(formatter)
#     if axis == ax.xaxis:
#         axis.set_major_formatter(FormatStrFormatter("%.1f"))
#         axis.set_minor_formatter(FormatStrFormatter("%.1f"))
#     else:
#         axis.set_major_formatter(FormatStrFormatter("%.2f"))
#         axis.set_minor_formatter(FormatStrFormatter("%.2f"))
#
# for i in range(len(spans)):
#     for j in range(len(l2ds)):
#         ax.plot(gammas,m1[i,j,:]/m1[i,j,0],linestyle=line_styles[0],color=colors[i],marker=markers[j])
#         ax.plot(gammas,f1[i,j,:]/f1[i,j,0],linestyle=line_styles[1],color=colors[i],marker=markers[j])
#         ax.plot(gammas,R1_weight[i,j,:]/R1_weight[i,j,0],linestyle=line_styles[2],color=colors[i],marker=markers[j])
#         # legend.append('span='+str(spans[i])+'m')
#
# ax.plot([gammas[0],gammas[-1]],[1,1],'--',color=[0.7,0.7,0.7])
#
#
# legend_line=ax.legend(handles=[Line2D([0],[0],color=[0.4,0.4,0.4],lw=1.5,linestyle=line_styles[i],label=label_linestyle[i]) for i in range(len(label_linestyle))],loc='upper left',bbox_to_anchor=(0,1))
# legend_color=ax.legend(handles=[Line2D([0],[0],color=colors[i],lw=4,label='span='+str(spans[i])+'m') for i in range(len(spans))],loc='upper left',bbox_to_anchor=(0,0.90))
# legend_marker=ax.legend(handles=[Line2D([0],[0],color=[0.5,0.5,0.5],marker=markers[i],label='l/d='+str(l2ds[i])) for i in range(len(l2ds))],loc='upper left',bbox_to_anchor=(0,0.73))
#
# plt.gca().add_artist(legend_line)
# plt.gca().add_artist(legend_color)
# plt.gca().add_artist(legend_marker)


# ## span,l/d-R contour plot comparison
# fig, ax = plt.subplots()
# ax.set_title('span,l/d- R contour comparison between vaulted slab and solid plate with the same mass')
# ax.set_xlabel('span [m]')
# ax.set_ylabel('l/d')
#
# colors = ['b', 'g', 'r', 'c', 'm', 'y']
#
# for i in range(len(gammas)):
#     contour = ax.contour(l_l2d_mesh_l, l_l2d_mesh_l2d, np.transpose(R_weight[:,:,i]),[1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120],colors = colors[i])
#     ax.clabel(contour,fontsize=10,fmt='%1.1f')
#     contour.collections[0].set_label('$t_v/t_r$=' + str(gammas[i]))
#
# contour = ax.contour(l_l2d_mesh_l, l_l2d_mesh_l2d, np.transpose(R_weight_plate_m),[1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120],colors = 'k')
# ax.clabel(contour,fontsize=10,fmt='%1.1f')
# contour.collections[0].set_label('solid plate')
#
# ax.legend()
#
# ### span-R plot comprison to solid plate_m with varying gamma groups (diff. l/d in each group)
# # fig, ax = plt.subplots()
# # ax.set_title('Normalized response by solid plate_m in relation to span')
# # legend = []
# # colors = ['b', 'g', 'r', 'c', 'm', 'k']
# # markers = ['o','^','s','x','*','d']
# # for i in range(2,len(gammas)):
# #     for j in range(len(l2ds)):
# #         ax.plot( spans, R_weight[:,j,i]/R_weight_plate_m[:,j],color=colors[i],marker=markers[j])
# #
# # legend_color=ax.legend(handles=[Line2D([0],[0],color=colors[i],lw=1,label='$t_v/t_r$='+str(gammas[i])) for i in range(2,len(gammas))],loc='upper left',bbox_to_anchor=(0,1))
# # legend_marker=ax.legend(handles=[Line2D([0],[0],color=[0.5,0.5,0.5],marker=markers[i],label='l/d='+str(l2ds[i])) for i in range(len(l2ds))],loc='upper left',bbox_to_anchor=(0,0.88))
# #
# # plt.gca().add_artist(legend_color)
# # plt.gca().add_artist(legend_marker)
# #
# #
# # ax.plot([spans[0],spans[-1]],[1,1],'--',color=[0.7,0.7,0.7])
# # ax.set_xlabel('span [m]')
# # ax.set_ylabel('$R/R_{solid plate}$ [-]')
# # ax.set_yscale('log')
# # ax.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
# #
# # formatter = ScalarFormatter()
# # formatter.set_scientific(False)
# # ax.yaxis.set_major_formatter(formatter)
# # ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
#
#
# ### gamma-normalized f1,m1,R1 by solid plate
# # line_styles = ['-.','--','-']
# # colors = ['b', 'g', 'r', 'c', 'm','k']
# # markers = ['o','^','s','x','*']
# #
# # label_linestyle = ['$f_1/f_{1,solid plate}$','$m_1/m_{1,solid plate}$','$R_1/R_{1,solid plate}$']
# #
# # fig, ax = plt.subplots()
# # ax.set_title('Relative change of $f_1, m_1, R_1$ normalized by solid plate in relation to $t_v/t_r$')
# # ax.set_xlabel('$t_v/t_r$')
# # ax.set_xlim(xmin=gammas[0]*0.91,xmax=gammas[-1]*1.09)
# # ax.set_ylabel('Relative change')
# # ax.set_xscale('log')
# # ax.set_yscale('log')
# # ax.grid(True)
# # ax.grid(which='minor',axis='y',visible=True)
# #
# # for axis in [ax.xaxis,ax.yaxis]:
# #     formatter = ScalarFormatter()
# #     formatter.set_scientific(False)
# #     axis.set_major_formatter(formatter)
# #     if axis == ax.xaxis:
# #         axis.set_major_formatter(FormatStrFormatter("%.1f"))
# #         axis.set_minor_formatter(FormatStrFormatter("%.1f"))
# #     else:
# #         axis.set_major_formatter(FormatStrFormatter("%.2f"))
# #         axis.set_minor_formatter(FormatStrFormatter("%.2f"))
# #
# # for i in range(len(spans)):
# #     for j in range(len(l2ds)):
# #         ax.plot(gammas,f1[i,j,:]/f1_plate_m[i,j],linestyle=line_styles[0],color=colors[i],marker=markers[j])
# #         ax.plot(gammas,m1[i,j,:]/m1_plate_m[i,j],linestyle=line_styles[1],color=colors[i],marker=markers[j])
# #         ax.plot(gammas,R1_weight[i,j,:]/R1_weight_plate_m[i,j],linestyle=line_styles[2],color=colors[i],marker=markers[j])
# #         # legend.append('span='+str(spans[i])+'m')
# #
# # ax.plot([gammas[0],gammas[-1]],[1,1],'--',color=[0.7,0.7,0.7])
# #
# #
# # legend_line=ax.legend(handles=[Line2D([0],[0],color=[0.4,0.4,0.4],lw=1,linestyle=line_styles[i],label=label_linestyle[i]) for i in range(len(label_linestyle))],loc='upper left',bbox_to_anchor=(0,1))
# # legend_color=ax.legend(handles=[Line2D([0],[0],color=colors[i],lw=4,label='span='+str(spans[i])+'m') for i in range(len(spans))],loc='upper left',bbox_to_anchor=(0,0.90))
# # legend_marker=ax.legend(handles=[Line2D([0],[0],color=[0.5,0.5,0.5],marker=markers[i],label='l/d='+str(l2ds[i])) for i in range(len(l2ds))],loc='upper left',bbox_to_anchor=(0,0.75))
# #
# # plt.gca().add_artist(legend_line)
# # plt.gca().add_artist(legend_color)
# # plt.gca().add_artist(legend_marker)
#
# ## mass increase-normalized f1,m1,R1 by initial mass
# spans_opt = [5]
# l2ds_opt = [10,15,20]
# gammas_opt = [0.1,1,10]
#
# # mass_incs = [0.1,0.2,0.3]
# mass_incs = [0.05,0.1,0.15,0.2,0.25,0.3]
#
# # spans_opt = [5]
# # l2ds_opt = [20]
# # gammas_opt = [5]
# #
# # mass_incs = [0.05,0.1,0.15,0.2,0.25,0.3]
#
# R_weight_opt = []
# R1_weight_opt = []
# f_n_opt = []
# m_n_opt = []
#
# for i in range(len(spans_opt)):
#     R_weight_opt.append([])
#     R1_weight_opt.append([])
#     f_n_opt.append([])
#     m_n_opt.append([])
#
#     for j in range(len(l2ds_opt)):
#         R_weight_opt[i].append([])
#         R1_weight_opt[i].append([])
#         f_n_opt[i].append([])
#         m_n_opt[i].append([])
#
#         for k in range(len(gammas_opt)):
#             R_weight_opt[i][j].append([])
#             R1_weight_opt[i][j].append([])
#             f_n_opt[i][j].append([])
#             m_n_opt[i][j].append([])
#
#             for q in range(len(mass_incs)):
#                 R_weight_opt[i][j][k].append([])
#                 R1_weight_opt[i][j][k].append([])
#                 f_n_opt[i][j][k].append([])
#                 m_n_opt[i][j][k].append([])
#
#                 model_file = 'D:/Master_Thesis/code_data/footfall_analysis_optimization/data/data_mdl_span' + str(spans_opt[i]).replace('.', '_') + '_l2d' + str(l2ds_opt[j]).replace('.', '_') + '_gamma' + str(
#                         gammas_opt[k]).replace('.', '_')+'_massInc'+str(mass_incs[q]).replace('.','_')+'_thickness_middle1.pkl'
#
#                 R_weight_opt[i][j][k][q] = pickle.load(open(model_file,'rb'))[-2][-1]
#                 R1_weight_opt[i][j][k][q] = pickle.load(open(model_file,'rb'))[-2][0]
#                 f_n_opt[i][j][k][q] = pickle.load(open(model_file,'rb'))[0]
#                 m_n_opt[i][j][k][q] = pickle.load(open(model_file,'rb'))[1]
#
# ### reshape in array
# R_weight_opt = np.reshape(np.array([R_weight_opt[i][j][k][q] for i in range(len(spans_opt)) for j in range(len(l2ds_opt)) for k in range(len(gammas_opt)) for q in range(len(mass_incs))]),(len(spans_opt),len(l2ds_opt),len(gammas_opt),len(mass_incs)))
# R1_weight_opt = np.reshape(np.array([R1_weight_opt[i][j][k][q] for i in range(len(spans_opt)) for j in range(len(l2ds_opt)) for k in range(len(gammas_opt)) for q in range(len(mass_incs))]),(len(spans_opt),len(l2ds_opt),len(gammas_opt),len(mass_incs)))
# f1_opt = np.reshape(np.array([f_n_opt[i][j][k][q][0] for i in range(len(spans_opt)) for j in range(len(l2ds_opt)) for k in range(len(gammas_opt)) for q in range(len(mass_incs))]),(len(spans_opt),len(l2ds_opt),len(gammas_opt),len(mass_incs)))
# m1_opt = np.reshape(np.array([m_n_opt[i][j][k][q][0] for i in range(len(spans_opt)) for j in range(len(l2ds_opt)) for k in range(len(gammas_opt)) for q in range(len(mass_incs))]),(len(spans_opt),len(l2ds_opt),len(gammas_opt),len(mass_incs)))
#
#
# line_styles = ['-.','--','-']
# colors = ['b', 'g', 'r', 'c', 'm','k']
# markers = ['o','^','s','x','P','d']
#
# label_linestyle = ['$f_{1,optimized}/f_{1,initial}$','$m_{1,optimized}/m_{1,initial}$','$R_{1,optimized}/R_{1,initial}$']
#
# fig,(ax1,ax2,ax3)= plt.subplots(3,1,gridspec_kw={'height_ratios':[2,2,2]})
# fig.suptitle('Normalized $m_1,f_1,R_1$ of optimized floor in relation to mass increase (span=5m,thickness change, middle 1)')
#
# ax1.set_xlim(xmin=-0.01*100,xmax=mass_incs[-1]*1.01*100)
# ax1.set_ylabel('$m_{1,opt}/m_{1,init}$')
# ax1.set_yscale('log')
# ax1.grid(True)
# ax1.grid(which='minor',axis='y',visible=True)
#
# ax2.set_xlim(xmin=-0.01*100,xmax=mass_incs[-1]*1.01*100)
# ax2.set_ylabel('$f_{1,opt}/f_{1,init}$')
# ax2.set_yscale('log')
# ax2.grid(True)
# ax2.grid(which='minor',axis='y',visible=True)
#
# ax3.set_xlabel('mass increase [%]')
# ax3.set_xlim(xmin=-0.01*100,xmax=mass_incs[-1]*1.01*100)
# ax3.set_ylabel('$R_{1,opt}/R_{1,init}$')
# ax3.set_yscale('log')
# ax3.grid(True)
# ax3.grid(which='minor',axis='y',visible=True)
#
# plt.setp(ax1.get_xticklabels(),visible=False)
# plt.setp(ax2.get_xticklabels(),visible=False)
# plt.subplots_adjust(hspace=0.1,top=0.935,bottom=0.05)
#
#
# for axis in [ax1.yaxis,ax2.yaxis,ax3.yaxis]:
#     formatter = ScalarFormatter()
#     formatter.set_scientific(False)
#     axis.set_major_formatter(formatter)
#
#     axis.set_major_formatter(FormatStrFormatter("%.1f"))
#     axis.set_minor_formatter(FormatStrFormatter("%.1f"))
#
# R1_norm = np.zeros((1,len(l2ds_opt),len(gammas_opt),len(mass_incs)+1))
#
# # for i in range(len(spans)):
# i=0
# for j in range(len(l2ds_opt)):
#     for k in range(len(gammas_opt)):
#
#         ax1.plot(100*np.array([0]+mass_incs),np.concatenate([[1],m1_opt[i,j,k,:]/m1[i,l2ds.index(l2ds_opt[j]),gammas.index(gammas_opt[k])]]),color=colors[j],marker=markers[k])
#         ax2.plot(100*np.array([0]+mass_incs),np.concatenate([[1],f1_opt[i,j,k,:]/f1[i,l2ds.index(l2ds_opt[j]),gammas.index(gammas_opt[k])]]),color=colors[j],marker=markers[k])
#         ax3.plot(100*np.array([0]+mass_incs),np.concatenate([[1],R1_weight_opt[i,j,k,:]/R1_weight[i,l2ds.index(l2ds_opt[j]),gammas.index(gammas_opt[k])]]),color=colors[j],marker=markers[k])
#         R1_norm[0,j,k,:] = np.concatenate([[1],R1_weight_opt[i,j,k,:]/R1_weight[i,l2ds.index(l2ds_opt[j]),gammas.index(gammas_opt[k])]])
#
# ax1.plot([0]+[mass_incs[-1]],[1,1],'--',color=[0.7,0.7,0.7])
# ax2.plot([0]+[mass_incs[-1]],[1,1],'--',color=[0.7,0.7,0.7])
# ax3.plot([0]+[mass_incs[-1]],[1,1],'--',color=[0.7,0.7,0.7])
#
# # legend_line=ax.legend(handles=[Line2D([0],[0],color=[0.4,0.4,0.4],lw=1,linestyle=line_styles[i],label=label_linestyle[i]) for i in range(len(label_linestyle))],loc='upper left',bbox_to_anchor=(0,1))
# legend_color1=ax1.legend(handles=[Line2D([0],[0],color=colors[i],lw=4,label='l/d='+str(l2ds_opt[i])) for i in range(len(l2ds_opt))],loc='upper left',bbox_to_anchor=(0,1))
# legend_marker1=ax1.legend(handles=[Line2D([0],[0],color=[0.5,0.5,0.5],marker=markers[i],label='$t_v/t_r$='+str(gammas_opt[i])) for i in range(len(gammas_opt))],loc='upper left',bbox_to_anchor=(0,0.75))
#
# ax1.add_artist(legend_color1)
# ax1.add_artist(legend_marker1)
#
# print('min(R1_opt/R1),5% = ')
# print(np.min(R1_norm[0,:,:,1]))
# print('max(R1_opt/R1),5% = ')
# print(np.max(R1_norm[0,:,:,1]))
# print('min(R1_opt/R1),10% = ')
# print(np.min(R1_norm[0,:,:,2]))
# print('max(R1_opt/R1),10% = ')
# print(np.max(R1_norm[0,:,:,2]))
#
# print('R1_opt/R1,5%,floor = ')
# print(np.min(R1_norm[0,1,1,1]))
# print('R1_opt/R1,10%,floor = ')
# print(np.min(R1_norm[0,1,1,2]))
#
# print('R1,floor = ')
# print(R1_weight[0,2,2])
# print('R1_opt,5%,floor = ')
# print(R1_weight_opt[0,1,1,0])
# print('m1_opt,5%,floor = ')
# print(m1_opt[0,1,1,0])
# print('f1_opt,5%,floor = ')
# print(f1_opt[0,1,1,0])
# print('R1_opt,10%,floor = ')
# print(R1_weight_opt[0,1,1,1])
# print('m1_opt,10%,floor = ')
# print(m1_opt[0,1,1,1])
# print('f1_opt,10%,floor = ')
# print(f1_opt[0,1,1,1])


# #
##
### sensitivity analysis, l,l/d,gamma-R correlation
# # dimension of model
# M = 3
# IOC = np.zeros(M)
#
# x_span = []
# y_span = []
# for i in range(len(spans)):
#     n_points = (R_weight[i,:,:].flatten()).shape[0]
#     x_span.append(spans[i]*np.ones(n_points))
#     y_span.append(R_weight[i,:,:].flatten())
# x_span = np.array(x_span).flatten()
# y_span = np.array(y_span).flatten()
# IOC[0] = pearsonr(x_span,y_span)[0]
#
# x_l2d = []
# y_l2d = []
# for i in range(len(l2ds)):
#     n_points = (R_weight[:,i,:].flatten()).shape[0]
#     x_l2d.append(l2ds[i]*np.ones(n_points))
#     y_l2d.append(R_weight[:,i,:].flatten())
# x_l2d = np.array(x_l2d).flatten()
# y_l2d = np.array(y_l2d).flatten()
# IOC[1] = pearsonr(x_l2d,y_l2d)[0]
#
# x_gamma = []
# y_gamma = []
# for i in range(len(spans)):
#     n_points = (R_weight[:,:,i].flatten()).shape[0]
#     x_gamma.append(gammas[i]*np.ones(n_points))
#     y_gamma.append(R_weight[:,:,i].flatten())
# x_gamma = np.array(x_gamma).flatten()
# y_gamma = np.array(y_gamma).flatten()
# IOC[2] = pearsonr(np.log10(x_gamma),y_gamma)[0]
#
# print('IOC = ')
# print(IOC)
#
# # input/output plot
# fig  = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_title('span-R scatter plot with IOC = {:.3f}'.format(IOC[0]))
# ax.set_xlabel('span [m]')
# ax.set_ylabel('R [-]')
# for i in range(len(spans)):
#     n_points = (R_weight[i,:,:].flatten()).shape[0]
#     ax.scatter(spans[i]*np.ones(n_points),R_weight[i,:,:].flatten(),color='#1f77b4',s=10)
#
# fig  = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_title('l/d-R scatter plot with IOC = {:.3f}'.format(IOC[1]))
# ax.set_xlabel('l/d [-]')
# ax.set_xticks(ticks=l2ds)
# ax.set_ylabel('R [-]')
# for i in range(len(l2ds)):
#     n_points = (R_weight[:,i,:].flatten()).shape[0]
#     ax.scatter(l2ds[i]*np.ones(n_points),R_weight[:,i,:].flatten(),color='#1f77b4',s=10)
#
# fig  = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_title('$t_v/t_r$-R scatter plot with IOC = {:.3f}'.format(IOC[2]))
# ax.set_xlabel('$t_v/t_r [-]$')
# ax.set_xscale('log')
# ax.set_ylabel('R [-]')
# for i in range(len(gammas)):
#     n_points = (R_weight[:,:,i].flatten()).shape[0]
#     ax.scatter(gammas[i]*np.ones(n_points),R_weight[:,:,i].flatten(),color='#1f77b4',s=10)
#
# ## sensitivity analysis, f1,m1-R1 correlation
# # dimension of model
# M = 2
# IOC = np.zeros(M)
#
# IOC[0] = pearsonr(f1.flatten(),R1_weight.flatten())[0]
# IOC[1] = pearsonr(m1.flatten(),R1_weight.flatten())[0]
#
# # input/output plot
# fig  = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_title('$f_1-R_1$ scatter plot with IOC = {:.3f}'.format(IOC[0]))
# ax.set_xlabel('$f_1$ [Hz]')
# ax.set_ylabel('$R_1$ [-]')
# ax.scatter(f1.flatten(),R1_weight.flatten(),s=10)
#
# fig  = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_title('$m_1-R_1$ scatter plot with IOC = {:.3f}'.format(IOC[1]))
# ax.set_xlabel('$m_1$ [kg]')
# ax.set_ylabel('$R_1$ [-]')
# ax.scatter(m1.flatten(),R1_weight.flatten(),s=10)


# ### points-normalized m1,f1,R1 (sensitivity analysis for f1)
# # m1 that should keep constant
# m1_const = 850
# # number of such points
# n_const = 0
# # +-deviation from m1_check
# dev = 0.1
#
# f1_rec = []
# m1_rec = []
# R1_rec = []
#
# for i in range(len(spans)):
#     for j in range(len(l2ds)):
#         for k in range (len(gammas)):
#
#             if m1_const * (1 - dev) < m1[i,j,k] < m1_const * (1 + dev):
#                 f1_rec.append(f1[i,j,k])
#                 m1_rec.append(m1[i, j, k])
#                 R1_rec.append(R1_weight[i, j, k])
#                 n_const += 1
#
# m1_rec.reverse()
# f1_rec.reverse()
# R1_rec.reverse()
#
# # m1_med = median(m1_rec)
#
# colors = ['b', 'g', 'r', 'c', 'm','k']
# markers = ['o','^','s','x','P','d']
#
# fig, ax = plt.subplots()
# ax.set_title('Normalized $m_1,f_1,R_1$ of points with roughly constant $m_1$')
# ax.set_xlabel('points number [-]')
# # ax.set_xlim(xmin=-0.02*100,xmax=mass_incs[-1]*1.09*100)
# ax.set_ylabel('Normalized Value [-]')
# ax.set_yscale('log')
# ax.grid()
# ax.grid(which='minor',axis='y',visible=True)
# # ax.grid(which='major',axis='y',visible=True)
#
# ax.set_xticks(ticks=[i for i in range(n_const)])
#
# ax.plot([i for i in range(n_const)], np.array(m1_rec)/m1_rec[0], marker='o',color=colors[0])
# ax.plot([i for i in range(n_const)], (np.array(f1_rec)/f1_rec[0])**1, marker='o',color=colors[1])
# ax.plot([i for i in range(n_const)], np.array(R1_rec)/R1_rec[0], marker='o',color=colors[2])
#
# ax.legend(['$m_1/m_{1,0}$','$f_1/f_{1,0}$','$R_1/R_{1,0}$'])
#
# for axis in [ax.yaxis]:
#     formatter = ScalarFormatter()
#     formatter.set_scientific(False)
#     axis.set_major_formatter(formatter)
#
#     axis.set_major_formatter(FormatStrFormatter("%.1f"))
#     axis.set_minor_formatter(FormatStrFormatter("%.1f"))


# ### m1,f1-R1 prediction
# # constant
# n = 180
# C = np.mean(R1_weight.flatten()*m1.flatten()*f1.flatten()**1.5)
# R1_pred = C/(m1.flatten()*f1.flatten()**1.5)
# MSE = mse(R1_weight.flatten(),R1_pred)
#
# print(C)
#
# # input/output plot
# fig  = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_title('$f_1-R_{{1,true}},R_{{1,pred}}$ scatter plot with MSE = {:.3f}'.format(MSE*100)+'%')
# ax.set_xlabel('$f_1$ [Hz]')
# ax.set_ylabel('$R_1$ [-]')
# ax.scatter(f1.flatten(),R1_weight.flatten(),s=10,color='C0')
# ax.scatter(f1.flatten(),R1_pred,s=10,color='r')
# ax.legend(['true','predict'])
#
# fig  = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_title('$m_1-R_{{1,true}},R_{{1,pred}}$ scatter plot with MSE = {:.3f}'.format(MSE*100)+'%')
# ax.set_xlabel('$m_1$ [kg]')
# ax.set_ylabel('$R_1$ [-]')
# ax.scatter(m1.flatten(),R1_weight.flatten(),s=10,color='C0')
# ax.scatter(m1.flatten(),R1_pred,s=10,color='r')
# ax.legend(['true','predict'])

# ### count acceptable floors in m1-f1 scatter plot
# # dimension of model
# f1_OK = []
# m1_OK = []
# R_OK = []
# n_OK = 0
#
# f1_fail = []
# m1_fail = []
# R_fail = []
#
# for i in range(len(spans)):
#     for j in range(len(l2ds)):
#         for k in range (len(gammas)):
#
#             if R_weight[i,j,k]<8:
#                 f1_OK.append(f1[i,j,k])
#                 m1_OK.append(m1[i, j, k])
#                 R_OK.append(R_weight[i, j, k])
#                 n_OK += 1
#             else:
#                 f1_fail.append(f1[i,j,k])
#                 m1_fail.append(m1[i, j, k])
#                 R_fail.append(R_weight[i, j, k])
#
# fig  = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_title('$m_1$-$f_1$ scatter plot')
# ax.set_xlabel('$m_1$ [kg]')
# ax.set_ylabel('$f_1$ [Hz]')
# ax.scatter(m1_OK,f1_OK,color='#1f77b4', s=10)
# ax.scatter(m1_fail,f1_fail,color='r', s=10)
# ax.legend(['$R\leq8$ (27.2%)','$R>8$ (72.7%)'])
#
# print(n_OK)


plt.show()
