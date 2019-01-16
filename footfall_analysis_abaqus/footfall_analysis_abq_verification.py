from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import timeit

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from compas_fea.structure import Structure
from scipy.integrate import odeint
from scipy.interpolate import interp1d

mpl.rc('font', size=15)

__author__    = ['Hao Wu <wuhao@student.ethz.ch>']
__copyright__ = 'Copyright 2018, BLOCK Research Group - ETH Zurich'
__license__   = 'MIT License'
__email__     = 'wuhao@student.ethz.ch'

### structure

span = 5
l2d = 20
gamma = 1


file_name='D:/Master_Thesis/code_data/footfall_analysis/data/data_mdl_0_4m/data_mdl_span'+str(span).replace('.','_')+'_l2d'+str(l2d).replace('.','_')+'_gamma'+str(gamma).replace('.','_')+'_100modes.pkl'

f_n, m_n, node_lp, n_modes, dt, t, dis_modes_lp, vel_modes_lp, acc_modes_lp, acc_modes_lp_weight, rms_modes,rms_modes_weight, R, R_weight, Gamma_n = pickle.load(open(file_name,'rb'))
# f_n,m_n,node_lp,n_modes,dt,t,dis_modes_lp,vel_modes_lp,acc_modes_lp,acc_modes_lp_weight,rms_modes,rms_modes_weight,R,R_weight,Gamma_n

### read results from abaqus calculation

n = 2001
t_cut = 0
T_rms = 1001

file = open('D:/Master_Thesis/code_data/footfall_analysis_abaqus/response_span5_l2d20_gamma1_abq.txt','rb')
lines = file.readlines()
t_abq = []
dis_abq = []
rms_abq = np.zeros(n)

for line in lines:
    t_abq.append(line.split()[0])
    dis_abq.append(line.split()[2])
file.close()


t_abq = np.array([float(i) for i in t_abq])
dis_abq = np.array([float(i) for i in dis_abq])*1000
acc_abq = np.gradient(np.gradient(dis_abq/1000,dt),dt)

n_abq = t_abq.shape[0]
rms_abq = np.zeros(n_abq)

for i in range(T_rms, n_abq):
    rms_abq[i] = np.sqrt(np.mean(acc_abq[i-T_rms:i]**2))


### plot response under walking load

### plot

# plot footfall loading
# fig  = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_title('Footfall loading')
# ax.set_xlabel('Time $t$ [s]')
# ax.set_ylabel('Force $F$ [N]')
# ax.minorticks_on()
# ax.plot(t, F, '-', color=[1, 0, 0])
# ax.plot([0, te], [W, W], ':', color=[0.7, 0.7, 0.7])

# # plot response
# fig = plt.figure()
# fig.suptitle(
#     'Footfall Response (span=' + str(span) + 'm, l/d=' + str(l2d) + ', $t_v/t_r$='+str(gamma)+', dt=' + str(
#         dt) + ', ' + str(n_modes) + ' modes)')
# ax1 = fig.add_subplot(311)
# ax1.set_xlabel('Time $t$ [s]')
# ax1.set_ylabel('Displacement $u$ [mm]')
# ax1.minorticks_on()
# ax1.plot(t, np.transpose(dis_modes_lp[-1, :]), color='r')
# ax1.plot(t, np.transpose(dis_modes_lp[0, :]), '--', color='r')
# ax1.plot(t_abq, dis_abq,color='b')
# ax1.plot([0, t[-1]], [0, 0], '--', color=[0.7, 0.7, 0.7])
# ax1.legend([str(n_modes) + ' modes', 'first mode','abaqus'],loc=1)
# ax1.set_xlim(xmin=-0.01,xmax=1.01)
#
# ax2 = fig.add_subplot(312)
# ax2.set_xlabel('Time $t$ [s]')
# ax2.set_ylabel('Acceleration $a$ [$m/s^2$]')
# ax2.minorticks_on()
# ax2.plot(t, np.transpose(acc_modes_lp[-1, :]), 'r')
# ax2.plot(t, np.transpose(acc_modes_lp[0, :]), '--', color='r')
# ax2.plot(t_abq, acc_abq,color='b')
# # ax2.plot(t_abq, acc_abq_original,'--',color='k')
# ax2.plot([0, t[-1]], [0, 0], '--', color=[0.7, 0.7, 0.7])
# ax2.legend([str(n_modes) + ' modes', 'first mode','abaqus'],loc=1)
# ax2.set_xlim(xmin=-0.01,xmax=1.01)
# ax2.set_ylim(ymin=-5,ymax=5)
#
# ax3 = fig.add_subplot(313)
# ax3.set_xlabel('Time $t$ [s]')
# ax3.set_ylabel('Acceleration (RMS) $a_\mathrm{rms}$ [$m/s^2$]')
# ax3.minorticks_on()
# ax3.plot(t, rms_modes[-1,:], color='r')
# ax3.plot(t, rms_modes[0,:], '--', color='r')
# ax3.plot(t_abq, rms_abq, color='b')
# ax3.legend([str(n_modes)+' modes', 'first mode', 'abaqus'])
# ax3.plot([0, t[-1]], [0, 0], '--', color=[0.7, 0.7, 0.7])
# ax3.set_xlim(xmin=-0.01,xmax=1.01)

# plot response factor in relation to number of modes involved
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title(
    'Response factor and participation factor in relation of modes involved in modal superposition (span=' + str(span) + 'm, l/d=' + str(l2d) + ', $t_v/t_r$='+str(gamma)+', dt=' + str(
                     dt) + ', ' + str(n_modes) + ' modes)')
ax.set_xlabel('Modes')
ax.set_ylabel('Response Factor [-]')
ax.minorticks_on()
ax.plot(range(1, n_modes + 1), R, 'r')
ax.plot(range(1, n_modes + 1), R_weight, 'b')
ax.set_ylim(ymin=0,ymax=170)
ax.set_xlim(xmin=0,xmax=101)
ax.legend(['response factor not weighted', 'response factor weighted'], loc=2)
xticks = [5*(i+1) for i in range(20)]
xticks.insert(0,1)
ax.set_xticks(ticks=xticks)
# plot participation factor
ax2 = ax.twinx()
ax2.set_ylabel('Participation Factor [-]')
ax2.set_ylim(ymin=-0.002,ymax=0.0045)
ax2.tick_params('y')
ax2.minorticks_on()
ax2.bar(range(1, n_modes + 1), Gamma_n, width=0.3, color='m')
ax2.legend(['participation factor'], loc=1)
ax2.plot([1, n_modes + 1], [0, 0], '--', color=[0.7, 0.7, 0.7])


# plot contribution factor in relation to modes involved

contr_orig = np.insert(np.diff(R),0,R[0])/R[-1]
contr_weight = np.insert(np.diff(R_weight),0,R_weight[0])/R_weight[-1]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title(
    'Contribution factor in relation of modes involved in modal superposition (span=' + str(span) + 'm, l/d=' + str(l2d) + ', $t_v/t_r$='+str(gamma)+', dt=' + str(
                     dt) + ', ' + str(n_modes) + ' modes)')
ax.set_xlabel('Modes')
ax.set_ylabel('Contribution Factor [-]')
ax.minorticks_on()
ax.bar(np.array([i+1 for i in range(100)])-0.15, contr_orig, width=0.3, color='r',align='center')
ax.bar(np.array([i+1 for i in range(100)])+0.15, contr_weight, width=0.3, color='b',align='center')
ax.set_xlim(xmin=0,xmax=101)
ax.legend(['not weighted', 'weighted'], loc=1)
ax.set_xticks(ticks=xticks)

plt.show()



