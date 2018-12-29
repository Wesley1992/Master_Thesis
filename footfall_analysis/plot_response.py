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



### l2d,gamma-R plot with span=5m
# fig = plt.figure()
# ax = fig.gca(projection='3d')
#
# surf = ax.plot_surface(l2d_gamma_mesh_l2d, np.log10(l2d_gamma_mesh_gamma), np.transpose(R_weight[5]))
# cset = ax.contour(l2d_gamma_mesh_l2d, np.log10(l2d_gamma_mesh_gamma), np.transpose(R_weight[5]),offset=0,cmap=cm.coolwarm)
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


### l2d,gamma-R contour plot (for each span)
# for i in range(len(spans)):
#     fig, ax = plt.subplots()
#     ax.set_title('l/d,$t_v/t_r$ - R contour (span=' + str(spans[i]) + 'm)')
#     ax.set_xlabel('l/d')
#     ax.set_ylabel('$t_v/t_r$')
#     contour = ax.contour(l2d_gamma_mesh_l2d, l2d_gamma_mesh_gamma, np.transpose(R_weight[i]),[1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120])
#     ax.clabel(contour,fontsize=10,fmt='%1.1f')
#     ax.set_yscale('log')


### span,l2d-R plot of solid plate with the same mass as vaulted slab
# fig = plt.figure()
# ax = fig.gca(projection='3d')
#
# surf = ax.plot_surface(l_l2d_mesh_l, l_l2d_mesh_l2d, np.transpose(R1_weight_plate_m))
# cset = ax.contour(l_l2d_mesh_l, l_l2d_mesh_l2d, np.transpose(R1_weight_plate_m),offset=0,cmap=cm.coolwarm)
#
# ax.set_title('span,l/d - R plot of solid plate with the same mass as vaulted slab')
# ax.set_xlabel('span [m]')
# ax.set_ylabel('l/d')
# ax.set_zlabel('R [-]')
#
# ax.set_zlim(zmin=0)
#
# fig.colorbar(cset)


### span,l2d-R contour plot of solid plate with the same mass as vaulted slab
# fig, ax = plt.subplots()
# ax.set_title('span,l/d - R contour plot of solid plate with the same mass as vaulted slab')
# ax.set_xlabel('span [m]')
# ax.set_ylabel('l/d')
# contour = ax.contour(l_l2d_mesh_l, l_l2d_mesh_l2d, np.transpose(R1_weight_plate_m),[1,2,4,6,8,10,12,16,20,24,28,32,40,48,56,64,72])
# ax.clabel(contour,fontsize=10,fmt='%1.1f')


### f1,m1-R1 plot with span=5m
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


### m1,f1-R1 plot
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


### m1,f1-R1 contour plot
# fig, ax = plt.subplots()
# ax.set_title('$m_1,f_1-R_1$ contour')
# ax.set_xlabel('$m_1$ [kg]')
# ax.set_ylabel('$f_1$ [Hz]')
#
# colors = ['b', 'g', 'r', 'c', 'm', 'k']
#
# for i in range(len(spans)):
#     contour = ax.contour(m1[i], f1[i], R1_weight[i],[1,2,4,6,8,12,16,20,28,36,48,60,72],colors=colors[i])
#     ax.clabel(contour,fontsize=10,fmt='%1.1f')
#     contour.collections[i].set_label('span='+str(spans[i])+'m')
# ax.legend()


### f1,m1-R1 plot of solid plate with the same geometry as vaulted slab
# fig = plt.figure()
# ax = fig.gca(projection='3d')
#
# surf = ax.plot_surface(m1_plate_g, f1_plate_g, R1_weight_plate_g)
# cset = ax.contour(m1_plate_g, f1_plate_g, R1_weight_plate_g,offset=0,cmap=cm.coolwarm)
#
# ax.set_title('$f_1$,$m_1$-$R_1$ plot of solid plate with the same geometry as vaulted slab')
# ax.set_xlabel('$m_1$ [kg]')
# ax.set_ylabel('$f_1$ [Hz]')
# ax.set_zlabel('$R_1$ [-]')
#
# ax.set_zlim(zmin=0)
# fig.colorbar(cset)


### m1,f1-R1 contour plot for plate with the same geometry
# fig, ax = plt.subplots()
# ax.set_title('$m_1,f_1-R_1$ contour of solid plate with the same outer geometry as vaulted slab')
# ax.set_xlabel('$m_1$ [kg]')
# ax.set_ylabel('$f_1$ [Hz]')
#
# contour = ax.contour(m1_plate_g, f1_plate_g, R1_weight_plate_g,[0.25,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5])
# ax.clabel(contour,fontsize=10,fmt='%1.1f')


### f1,m1-R1 plot of solid plate with the same mass as vaulted slab
# fig = plt.figure()
# ax = fig.gca(projection='3d')
#
# surf = ax.plot_surface(m1_plate_m, f1_plate_m, R1_weight_plate_m)
# cset = ax.contour(m1_plate_m, f1_plate_m, R1_weight_plate_m,offset=0,cmap=cm.coolwarm)
#
# ax.set_title('$f_1$,$m_1$-$R_1$ plot of solid plate with the same mass as vaulted slab')
# ax.set_xlabel('$m_1$ [kg]')
# ax.set_ylabel('$f_1$ [Hz]')
# ax.set_zlabel('$R_1$ [-]')
#
# ax.set_zlim(zmin=0)
# fig.colorbar(cset)


### m1,f1-R1 contour plot for plate with the same mass
# fig, ax = plt.subplots()
# ax.set_title('$m_1,f_1-R_1$ contour of solid plate with the same mass as vaulted slab')
# ax.set_xlabel('$m_1$ [kg]')
# ax.set_ylabel('$f_1$ [Hz]')
#
# contour = ax.contour(m1_plate_m, f1_plate_m, R1_weight_plate_m,[1,2,3,4,5,6,7,8,10,12,14,16,20,24,32,40,48])
# ax.clabel(contour,fontsize=10,fmt='%1.1f')


### m1,f1-R contour plot
# fig, ax = plt.subplots()
# ax.set_title('$m_1,f_1-R$ contour')
# ax.set_xlabel('$m_1$ [kg]')
# ax.set_ylabel('$f_1$ [Hz]')
#
# colors = ['b', 'g', 'r', 'c', 'm', 'k']
#
# for i in range(len(spans)):
#     contour = ax.contour(m1[i], f1[i], R_weight[i],[1,2,4,6,8,12,20,28,36,48,60,80,100,120],colors=colors[i])
#     ax.clabel(contour,fontsize=10,fmt='%1.1f')
#     contour.collections[i].set_label('span='+str(spans[i])+'m')
#
# ax.legend()


### span,gamma-m1 contour plot
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


### l/d,gamma-m1 contour plot
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


### span,gamma-f1 contour plot
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


### l/d,gamma-f1 contour plot
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


### span,l/d-f1 contour plot
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


### l/d-R plot with varying gamma groups (diff. span in each group)
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


### gamma-R plot with varying l/d groups (diff. spans in each group)
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


### gamma-normalized f1,m1,R1 by tv/tr=0.1

# line_styles = ['-.','--','-']
# colors = ['b', 'g', 'r', 'c', 'm','k']
# markers = ['o','^','s','x','*']
#
# label_linestyle = ['$f_1/f_{1,t_v/t_r=0.1}$','$m_1/m_{1,t_v/t_r=0.1}$','$R_1/R_{1,t_v/t_r=0.1}$']
#
# fig, ax = plt.subplots()
# ax.set_title('Relative change of $f_1, m_1, R_1$ in relation to $t_v/t_r$')
# ax.set_xlabel('$t_v/t_r$')
# ax.set_xlim(xmin=gammas[0]*0.91,xmax=gammas[-1]*1.09)
# ax.set_ylabel('Relative change')
# ax.set_xscale('log')
# ax.set_yscale('log')
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
#         ax.plot(gammas,f1[i,j,:]/f1[i,j,0],linestyle=line_styles[0],color=colors[i],marker=markers[j])
#         ax.plot(gammas,m1[i,j,:]/m1[i,j,0],linestyle=line_styles[1],color=colors[i],marker=markers[j])
#         ax.plot(gammas,R1_weight[i,j,:]/R1_weight[i,j,0],linestyle=line_styles[2],color=colors[i],marker=markers[j])
#         # legend.append('span='+str(spans[i])+'m')
#
# ax.plot([gammas[0],gammas[-1]],[1,1],'--',color=[0.7,0.7,0.7])
#
# legend_line=ax.legend(handles=[Line2D([0],[0],color=[0.4,0.4,0.4],lw=1.5,linestyle=line_styles[i],label=label_linestyle[i]) for i in range(len(label_linestyle))],loc='upper left',bbox_to_anchor=(0,1))
# legend_color=ax.legend(handles=[Line2D([0],[0],color=colors[i],lw=4,label='span='+str(spans[i])+'m') for i in range(len(spans))],loc='upper left',bbox_to_anchor=(0,0.90))
# legend_marker=ax.legend(handles=[Line2D([0],[0],color=[0.5,0.5,0.5],marker=markers[i],label='l/d='+str(l2ds[i])) for i in range(len(l2ds))],loc='upper left',bbox_to_anchor=(0,0.75))
#
# plt.gca().add_artist(legend_line)
# plt.gca().add_artist(legend_color)
# plt.gca().add_artist(legend_marker)

### span,l/d-R contour plot comparison
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

### span-R plot comprison to solid plate_m with varying gamma groups (diff. l/d in each group)
# fig, ax = plt.subplots()
# ax.set_title('Normalized response by solid plate_m in relation to span')
# legend = []
# colors = ['b', 'g', 'r', 'c', 'm', 'k']
# markers = ['o','^','s','x','*','d']
# for i in range(2,len(gammas)):
#     for j in range(len(l2ds)):
#         ax.plot( spans, R_weight[:,j,i]/R_weight_plate_m[:,j],color=colors[i],marker=markers[j])
#
# legend_color=ax.legend(handles=[Line2D([0],[0],color=colors[i],lw=1,label='$t_v/t_r$='+str(gammas[i])) for i in range(2,len(gammas))],loc='upper left',bbox_to_anchor=(0,1))
# legend_marker=ax.legend(handles=[Line2D([0],[0],color=[0.5,0.5,0.5],marker=markers[i],label='l/d='+str(l2ds[i])) for i in range(len(l2ds))],loc='upper left',bbox_to_anchor=(0,0.88))
#
# plt.gca().add_artist(legend_color)
# plt.gca().add_artist(legend_marker)
#
#
# ax.plot([spans[0],spans[-1]],[1,1],'--',color=[0.7,0.7,0.7])
# ax.set_xlabel('span [m]')
# ax.set_ylabel('$R/R_{solid plate}$ [-]')
# ax.set_yscale('log')
# ax.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
#
# formatter = ScalarFormatter()
# formatter.set_scientific(False)
# ax.yaxis.set_major_formatter(formatter)
# ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))


### gamma-normalized f1,m1,R1 by solid plate
# line_styles = ['-.','--','-']
# colors = ['b', 'g', 'r', 'c', 'm','k']
# markers = ['o','^','s','x','*']
#
# label_linestyle = ['$f_1/f_{1,solid plate}$','$m_1/m_{1,solid plate}$','$R_1/R_{1,solid plate}$']
#
# fig, ax = plt.subplots()
# ax.set_title('Relative change of $f_1, m_1, R_1$ normalized by solid plate in relation to $t_v/t_r$')
# ax.set_xlabel('$t_v/t_r$')
# ax.set_xlim(xmin=gammas[0]*0.91,xmax=gammas[-1]*1.09)
# ax.set_ylabel('Relative change')
# ax.set_xscale('log')
# ax.set_yscale('log')
#
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
#         ax.plot(gammas,f1[i,j,:]/f1_plate_m[i,j],linestyle=line_styles[0],color=colors[i],marker=markers[j])
#         ax.plot(gammas,m1[i,j,:]/m1_plate_m[i,j],linestyle=line_styles[1],color=colors[i],marker=markers[j])
#         ax.plot(gammas,R1_weight[i,j,:]/R1_weight_plate_m[i,j],linestyle=line_styles[2],color=colors[i],marker=markers[j])
#         # legend.append('span='+str(spans[i])+'m')
#
# ax.plot([gammas[0],gammas[-1]],[1,1],'--',color=[0.7,0.7,0.7])
#
# legend_line=ax.legend(handles=[Line2D([0],[0],color=[0.4,0.4,0.4],lw=1,linestyle=line_styles[i],label=label_linestyle[i]) for i in range(len(label_linestyle))],loc='upper left',bbox_to_anchor=(0,1))
# legend_color=ax.legend(handles=[Line2D([0],[0],color=colors[i],lw=4,label='span='+str(spans[i])+'m') for i in range(len(spans))],loc='upper left',bbox_to_anchor=(0,0.90))
# legend_marker=ax.legend(handles=[Line2D([0],[0],color=[0.5,0.5,0.5],marker=markers[i],label='l/d='+str(l2ds[i])) for i in range(len(l2ds))],loc='upper left',bbox_to_anchor=(0,0.75))
#
# plt.gca().add_artist(legend_line)
# plt.gca().add_artist(legend_color)
# plt.gca().add_artist(legend_marker)

### mass increase-normalized f1,m1,R1 by initial mass
spans_opt = [5]
l2ds_opt = [10,15,20]
gammas_opt = [0.1,1,10]

mass_incs = [0.05,0.1,0.15,0.2,0.25,0.3]

# spans_opt = [5]
# l2ds_opt = [20]
# gammas_opt = [5]
#
# mass_incs = [0.05,0.1,0.15,0.2,0.25,0.3]

R_weight_opt = []
R1_weight_opt = []
f_n_opt = []
m_n_opt = []

for i in range(len(spans_opt)):
    R_weight_opt.append([])
    R1_weight_opt.append([])
    f_n_opt.append([])
    m_n_opt.append([])

    for j in range(len(l2ds_opt)):
        R_weight_opt[i].append([])
        R1_weight_opt[i].append([])
        f_n_opt[i].append([])
        m_n_opt[i].append([])

        for k in range(len(gammas_opt)):
            R_weight_opt[i][j].append([])
            R1_weight_opt[i][j].append([])
            f_n_opt[i][j].append([])
            m_n_opt[i][j].append([])

            for q in range(len(mass_incs)):
                R_weight_opt[i][j][k].append([])
                R1_weight_opt[i][j][k].append([])
                f_n_opt[i][j][k].append([])
                m_n_opt[i][j][k].append([])

                model_file = 'D:/Master_Thesis/code_data/footfall_analysis_optimization/data/data_mdl_span' + str(spans_opt[i]).replace('.', '_') + '_l2d' + str(l2ds_opt[j]).replace('.', '_') + '_gamma' + str(
                        gammas_opt[k]).replace('.', '_')+'_massInc'+str(mass_incs[q]).replace('.','_')+'_thickness_middle1.pkl'

                R_weight_opt[i][j][k][q] = pickle.load(open(model_file,'rb'))[-2][-1]
                R1_weight_opt[i][j][k][q] = pickle.load(open(model_file,'rb'))[-2][0]
                f_n_opt[i][j][k][q] = pickle.load(open(model_file,'rb'))[0]
                m_n_opt[i][j][k][q] = pickle.load(open(model_file,'rb'))[1]

### reshape in array
R_weight_opt = np.reshape(np.array([R_weight_opt[i][j][k][q] for i in range(len(spans_opt)) for j in range(len(l2ds_opt)) for k in range(len(gammas_opt)) for q in range(len(mass_incs))]),(len(spans_opt),len(l2ds_opt),len(gammas_opt),len(mass_incs)))
R1_weight_opt = np.reshape(np.array([R1_weight_opt[i][j][k][q] for i in range(len(spans_opt)) for j in range(len(l2ds_opt)) for k in range(len(gammas_opt)) for q in range(len(mass_incs))]),(len(spans_opt),len(l2ds_opt),len(gammas_opt),len(mass_incs)))
f1_opt = np.reshape(np.array([f_n_opt[i][j][k][q][0] for i in range(len(spans_opt)) for j in range(len(l2ds_opt)) for k in range(len(gammas_opt)) for q in range(len(mass_incs))]),(len(spans_opt),len(l2ds_opt),len(gammas_opt),len(mass_incs)))
m1_opt = np.reshape(np.array([m_n_opt[i][j][k][q][0] for i in range(len(spans_opt)) for j in range(len(l2ds_opt)) for k in range(len(gammas_opt)) for q in range(len(mass_incs))]),(len(spans_opt),len(l2ds_opt),len(gammas_opt),len(mass_incs)))


line_styles = ['-.','--','-']
colors = ['b', 'g', 'r', 'c', 'm','k']
markers = ['o','^','s','x','P','d']

label_linestyle = ['$f_{1,optimized}/f_{1,initial}$','$m_{1,optimized}/m_{1,initial}$','$R_{1,optimized}/R_{1,initial}$']

fig, ax = plt.subplots()
ax.set_title('$f_1, m_1, R_1$ of optimized slab(span=5m) normalized by initial slab in relation to mass increase(thickness change,middle 1)')
ax.set_xlabel('mass increase [%]')
ax.set_xlim(xmin=-0.02*100,xmax=mass_incs[-1]*1.09*100)
ax.set_ylabel('Relative values')
ax.set_yscale('log')
ax.grid(True)
ax.grid(which='minor',axis='y',visible=True)

for axis in [ax.yaxis]:
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    axis.set_major_formatter(formatter)

    axis.set_major_formatter(FormatStrFormatter("%.2f"))
    axis.set_minor_formatter(FormatStrFormatter("%.2f"))

# for i in range(len(spans)):
i=0
for j in range(len(l2ds_opt)):
    for k in range(len(gammas_opt)):
        print(f1_opt[i,j,k,:])
        print('\n')

        ax.plot(100*np.array([0]+mass_incs),np.concatenate([[1],f1_opt[i,j,k,:]/f1[i,l2ds.index(l2ds_opt[j]),gammas.index(gammas_opt[k])]]),linestyle=line_styles[0],color=colors[j],marker=markers[k])
        ax.plot(100*np.array([0]+mass_incs),np.concatenate([[1],m1_opt[i,j,k,:]/m1[i,l2ds.index(l2ds_opt[j]),gammas.index(gammas_opt[k])]]),linestyle=line_styles[1],color=colors[j],marker=markers[k])
        ax.plot(100*np.array([0]+mass_incs),np.concatenate([[1],R1_weight_opt[i,j,k,:]/R1_weight[i,l2ds.index(l2ds_opt[j]),gammas.index(gammas_opt[k])]]),linestyle=line_styles[2],color=colors[j],marker=markers[k])
    # legend.append('span='+str(spans[i])+'m')

ax.plot([0]+[mass_incs[-1]],[1,1],'--',color=[0.7,0.7,0.7])

legend_line=ax.legend(handles=[Line2D([0],[0],color=[0.4,0.4,0.4],lw=1,linestyle=line_styles[i],label=label_linestyle[i]) for i in range(len(label_linestyle))],loc='upper left',bbox_to_anchor=(0,1))
legend_color=ax.legend(handles=[Line2D([0],[0],color=colors[i],lw=4,label='l/d='+str(l2ds_opt[i])) for i in range(len(l2ds_opt))],loc='upper left',bbox_to_anchor=(0,0.875))
legend_marker=ax.legend(handles=[Line2D([0],[0],color=[0.5,0.5,0.5],marker=markers[i],label='$t_v/t_r$='+str(gammas_opt[i])) for i in range(len(gammas_opt))],loc='upper left',bbox_to_anchor=(0,0.75))

plt.gca().add_artist(legend_line)
plt.gca().add_artist(legend_color)
plt.gca().add_artist(legend_marker)


plt.show()
