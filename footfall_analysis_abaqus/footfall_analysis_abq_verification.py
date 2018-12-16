from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import timeit

import matplotlib.pyplot as plt
import numpy as np
from compas_fea.structure import Structure
from scipy.integrate import odeint
from scipy.interpolate import interp1d

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
# axes = fig.add_subplot(111)
# axes.set_title('Footfall loading', fontsize=12)
# axes.set_xlabel('Time $t$ [s]', fontsize=12)
# axes.set_ylabel('Force $F$ [N]', fontsize=12)
# axes.minorticks_on()
# axes.plot(t, F, '-', color=[1, 0, 0])
# axes.plot([0, te], [W, W], ':', color=[0.7, 0.7, 0.7])

# plot response
fig = plt.figure()
fig.suptitle(
    'Footfall Response (span=' + str(span) + 'm, l/d=' + str(l2d) + ', gamma='+str(gamma)+', dt=' + str(
        dt) + ', ' + str(n_modes) + ' modes)')
axes1 = fig.add_subplot(311)
axes1.set_xlabel('Time $t$ [s]', fontsize=12)
axes1.set_ylabel('Displacement $u$ [mm]', fontsize=12)
axes1.minorticks_on()
axes1.plot(t, np.transpose(dis_modes_lp[-1, :]), color='r')
axes1.plot(t, np.transpose(dis_modes_lp[0, :]), '--', color='r')
axes1.plot(t_abq, dis_abq,color='b')
axes1.plot([0, t[-1]], [0, 0], '--', color=[0.7, 0.7, 0.7])
axes1.legend([str(n_modes) + ' modes', 'first mode','abaqus'],loc=1)

axes2 = fig.add_subplot(312)
axes2.set_xlabel('Time $t$ [s]', fontsize=12)
axes2.set_ylabel('Acceleration $a$ [$m/s^2$]', fontsize=12)
axes2.minorticks_on()
axes2.plot(t, np.transpose(acc_modes_lp[-1, :]), 'r')
axes2.plot(t, np.transpose(acc_modes_lp[0, :]), '--', color='r')
axes2.plot(t_abq, acc_abq,color='b')
# axes2.plot(t_abq, acc_abq_original,'--',color='k')
axes2.plot([0, t[-1]], [0, 0], '--', color=[0.7, 0.7, 0.7])
axes2.legend([str(n_modes) + ' modes', 'first mode','abaqus'])

axes3 = fig.add_subplot(313)
axes3.set_xlabel('Time $t$ [s]', fontsize=12)
axes3.set_ylabel('Acceleration (RMS) $a_\mathrm{rms}$ [$m/s^2$]', fontsize=12)
axes3.minorticks_on()
axes3.plot(t, rms_modes[-1,:], color='r')
axes3.plot(t, rms_modes[0,:], '--', color='r')
axes3.plot(t_abq, rms_abq, color='b')
axes3.legend([str(n_modes)+' modes', 'first mode', 'abaqus'])
axes3.plot([0, t[-1]], [0, 0], '--', color=[0.7, 0.7, 0.7])


plt.show()



