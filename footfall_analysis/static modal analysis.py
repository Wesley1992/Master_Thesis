
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import os

from compas_fea.structure import Structure


__author__    = ['Andrew Liew <liew@arch.ethz.ch>']
__copyright__ = 'Copyright 2018, BLOCK Research Group - ETH Zurich'
__license__   = 'MIT License'
__email__     = 'liew@arch.ethz.ch'


spans = [5]
l2ds = [20]
gammas = [1]

n_modes = 100

### extract modal shapes and reaction forces
for span in spans:
    for l2d in l2ds:
        for gamma in gammas:

            # Input obj file
            mdl = Structure(name='mdl_span'+str(span).replace('.','_')+'_l2d'+str(l2d).replace('.','_')+'_gamma'+str(gamma).replace('.','_')+'_100modes', path=os.path.dirname(os.path.abspath(__file__)))
            file_obj = 'D:/Master_Thesis/modal_span_depth_thickness/'+mdl.name+'.obj'
            mdl = Structure.load_from_obj(file_obj)
            f_n = np.array(mdl.results['step_modal']['frequencies'])
            m_n = np.array(mdl.results['step_modal']['masses'])
            n_modes_max = f_n.shape[0]
            node_lp = mdl.sets['nset_loadPoint']['selection']

            n_nodes = mdl.nodes.__len__()
            s = np.zeros(n_nodes) # spatial distribution of load
            s[node_lp] = 1

            # extract mode shape
            uz_ms = []  # mode shape z direction

            for i in range(n_modes):
                uz_ms.append([])
                for node in mdl.nodes:
                        uz_ms[i].append(mdl.results['step_modal']['nodal']['uz{0}'.format(i+1)][node])
            uz_ms = np.array(uz_ms)

            xyz_nodes = mdl.nodes_xyz()
            xyz_nodes = np.array(xyz_nodes)

# # extract uz under point load from abaqus
#
# uz_abq = []  # disp in z direction
# ux_abq = []
# uy_abq = []
# um_pl = []
# for node in mdl.nodes:
#     uz_abq.append(mdl.results['step_load']['nodal']['uz'][node])
#     ux_abq.append(mdl.results['step_load']['nodal']['ux'][node])
#     uy_abq.append(mdl.results['step_load']['nodal']['uy'][node])
#     um_pl.append(mdl.results['step_load']['nodal']['um'][node])
#
# uz_abq = np.array(uz_abq)
# ux_abq = np.array(ux_abq)
# uy_abq = np.array(uy_abq)
# um_pl = np.array(um_pl)


### extract rz under self weight
            mdl_rf = Structure(name='mdl_span'+str(span).replace('.','_')+'_l2d'+str(l2d).replace('.','_')+'_gamma'+str(gamma).replace('.','_')+'_reactionForces', path=os.path.dirname(os.path.abspath(__file__)))
            file_obj = 'D:/Master_Thesis/modal_span_depth_thickness/'+mdl_rf.name+'.obj'
            mdl_rf = Structure.load_from_obj(file_obj)
            rz = []
            for node in mdl_rf.nodes:
                rz.append(mdl_rf.results['step_loads']['nodal']['rfz'][node])

            m = np.array(rz)/9.81


### static modal analysis

# m_total = 5627.3
# m_assume = np.ones(n_nodes)*m_total/n_nodes
# s = np.zeros(n_nodes)
# s[node_lp] = 1
            W = -76 * 9.81
#
# n_modes = 100
#             Gamma = np.zeros(n_modes)
#             L = np.zeros(n_modes)
#             m_n_eff = np.zeros(n_modes)
#             s_n = np.zeros((n_nodes,n_modes))
#             s_sum = np.zeros(n_nodes)
#
# q_n = np.zeros(n_modes)
# uz = np.zeros(n_nodes)

# for i in range(n_modes):
#     L[i] = uz_ms[i]@s
#     Gamma[i] = L[i]/m_n[i]
#
#     q_n[i] = Gamma[i]*W/(2*np.pi*f_n[i])**2
#     uz += q_n[i]*uz_ms[i]
#
# ### plot displacement under point load
# scale = 50000
# fig  = plt.figure()
#
# axes = fig.add_subplot(111, projection = '3d')
# axes.set_title('Displacement under point load from modal analysis and abaqus', fontsize=12)
# axes.set_xlabel('x', fontsize=12)
# axes.set_ylabel('y', fontsize=12)
# axes.set_zlabel('z', fontsize=12)
#
# # from modal analysis
# X = xyz_nodes[:,0]
# Y = xyz_nodes[:,1]
# Z = xyz_nodes[:,2] + uz*scale
#
# max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
# Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
# Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
# Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
#
# axes.minorticks_on()
# axes.scatter(X, Y, Z, s=0.3, c=np.abs(uz*scale), cmap=cm.coolwarm, label='modal analysis')
#
# # from abaqus
# X = xyz_nodes[:,0] + ux_abq*scale
# Y = xyz_nodes[:,1] + uy_abq*scale
# Z = xyz_nodes[:,2] + uz_abq*scale
#
# max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
# Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
# Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
# Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
#
# axes.minorticks_on()
# axes.scatter(X, Y, Z, s=0.2, c='gray', alpha = 0.3, label='abaqus')
#
# axes.legend()
#
# for xb, yb, zb in zip(Xb, Yb, Zb):
#     axes.plot([xb], [yb], [zb], 'w')
#
# plt.grid()


# examine how many modes should be adopted
            Gamma = np.zeros(n_modes)
            m_n_eff = np.zeros(n_modes)
            s_n = np.zeros((n_nodes,n_modes))
            s_sum = np.zeros((n_nodes,n_modes))
            # q_n = np.zeros(n_modes)
# uz = np.zeros(n_nodes)
# uz_modes = np.zeros((n_modes, n_nodes))
# rms_s = np.zeros(n_modes)

            for n_mode in range(1, n_modes+1):

                L = np.zeros(n_mode)
                Gamma = np.zeros(n_mode)
                # q_n = np.zeros(n_mode)

                for i in range(n_mode):
                    L[i] = uz_ms[i] @ s
                    Gamma[i] = L[i] / m_n[i]

                    # q_n[i] = Gamma[i] * W / (2 * np.pi * f_n[i]) ** 2
                    # uz_modes[n_mode-1] += q_n[i] * uz_ms[i]

                    s_n[:, i] = Gamma[i] * m * uz_ms[i]
                    s_sum[:,n_mode-1] += s_n[:, i]
                    m_n_eff[n_mode-1] += Gamma[i] * L[i]

                    # rms_s[n_mode-1] = np.sqrt(np.mean((s_sum[:, n_mode-1] - s) ** 2))  # rms of difference between s_sum and s

            # r = uz_modes[:,node_lp]/uz_abq[node_lp]  # ratio of displacement from modal analysis and abaqus
            r_s = np.sum(s_sum,0)/np.sum(s)  # ratio of total modal force distribution to load magnitude
            r_s_lp = s_sum[node_lp]/s[node_lp]  # ratio of modal force distribution on load point to load magnitude on load point

# # plot the ratio uz/uz_abq to number of modes involved
# fig  = plt.figure()
#
# axes = fig.add_subplot(111)
# axes.set_title('Ratio of displacements from modal analysis and abaqus in relation to number of modes involved', fontsize=12)
# axes.set_xlabel('Number of modes involved in modal analysis', fontsize=12)
# axes.set_ylabel('Ratio $r=u_{z,modal}/u_{z,abaqus}$', fontsize=12)
#
# axes.minorticks_on()
# axes.plot(range(1,n_modes+1), r, '-', color=[0, 0, 0])


# plot the ratio of total effective modal force to load magnitude
            fig  = plt.figure()

            axes = fig.add_subplot(111)
            axes.set_title('Ratio of sum of effective modal mass to total load magnitude', fontsize=12)
            axes.set_xlabel('Number of modes involved in modal analysis', fontsize=12)
            axes.set_ylabel('Ratio [-]', fontsize=12)
            axes.minorticks_on()
            axes.plot(range(1,n_modes+1), m_n_eff/np.sum(s), '-', color=[0, 0, 0])

            # plot the ratio of modal force distribution on load point to load magnitude on load point
            fig  = plt.figure()

            axes = fig.add_subplot(111)
            axes.set_title('Ratio of calculated modal force on load point to real load', fontsize=12)
            axes.set_xlabel('Number of modes involved in modal analysis', fontsize=12)
            axes.set_ylabel('Ratio $r=s_{modal,load point}/s_{real,load point}$', fontsize=12)

            axes.minorticks_on()
            axes.plot(range(1,n_modes+1), np.transpose(r_s_lp), '-', color=[0, 0, 0])


# plot the ratio of total modal force distribution to load magnitude
            fig  = plt.figure()

            axes = fig.add_subplot(111)
            axes.set_title('Ratio of total spatial modal force to real load magnitude', fontsize=12)
            axes.set_xlabel('Number of modes involved in modal analysis', fontsize=12)
            axes.set_ylabel('Ratio $r=sum(s_n)/sum(s)$', fontsize=12)

            axes.minorticks_on()
            axes.plot(range(1,n_modes+1), np.transpose(r_s), '-', color=[0, 0, 0])
            axes.plot([1, n_modes+1], [1, 1], '--', color=[0.7, 0.7, 0.7])



# # plot the rms of s_sum-s in ration of modes involved
# fig  = plt.figure()
#
# axes = fig.add_subplot(111)
# axes.set_title('rms of s_sum-s in ration of modes involved', fontsize=12)
# axes.set_xlabel('Number of modes involved in modal analysis', fontsize=12)
# axes.set_ylabel('rms', fontsize=12)
#
# axes.minorticks_on()
# axes.plot(range(1,n_modes+1), np.transpose(rms_s), '-', color=[0, 0, 0])


### plot modal force distribution
            fig  = plt.figure()

            axes = fig.add_subplot(111, projection = '3d')
            axes.set_title('Spatial modal force field', fontsize=12)
            axes.set_xlabel('x', fontsize=12)
            axes.set_ylabel('y', fontsize=12)
            axes.set_zlabel('z', fontsize=12)

            # from modal analysis
            X = xyz_nodes[:,0]
            Y = xyz_nodes[:,1]
            Z = s_sum[:,-1]

            max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
            Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
            Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
            Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())

            axes.minorticks_on()
            axes.scatter(X, Y, Z, s=1)

plt.show()
# plt.show(block=False)
# print()