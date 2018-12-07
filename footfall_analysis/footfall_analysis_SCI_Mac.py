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

# weight function for acceleration based on the frequency
def Wb(f):
    if 1<f<2:
        Wb = 0.4
    elif 2<=f<5:
        Wb = f/5
    elif 5<=f<=16:
        Wb = 1.0
    elif f>16:
        Wb = 16/f
    return Wb


### structures

spans = [9]
l2ds = [12.5]
gammas = [0.5,1,2]

### parameters often changed
n_modes = 50  # number of modes used shall not exceed that extracted from abaqus modal analysis
dt = 0.0005  # s
t_cut = 0

### Loading
W = -76 * 9.81    # load from walking people
f_load = 2.0  # hz, frequency of walking
ksi = 0.03  # -, damping ratio

te = 1  # time span for history analysis
n  = int(te / dt)+1
t  = np.linspace(0, te, n)  # s
a  = [0.436 * (1 * f_load - 0.95),
      0.006 * (2 * f_load + 12.3),
      0.007 * (3 * f_load + 5.20),
      0.007 * (4 * f_load + 2.00)]
p  = [0, -0.5 * np.pi, np.pi, 0.5 * np.pi]
F  = np.zeros(n) + W  # N

for i in range(len(a)):
    F += W * a[i] * np.sin(2 * np.pi * (i + 1) * f_load * t + p[i])  # N

Fn = interp1d(t, F, fill_value="extrapolate")

### rms and response factor parameters
T_rms = int(1/f_load//dt)+2  # time period for calculating rms
acc_base = 0.005

### solving
for span in spans:
    for l2d in l2ds:
        for gamma in gammas:

            # Input obj file
            mdl = Structure(name='mdl_span'+str(span).replace('.','_')+'_l2d'+str(l2d).replace('.','_')+'_gamma'+str(gamma).replace('.','_'), path=os.path.dirname(os.path.abspath(__file__)))
            # file_obj = 'D:/Master_Thesis/modal_span_depth_thickness/mdl_t_0_01span/'+mdl.name+'.obj'
            file_obj = 'D:/Master_Thesis/modal_span_depth_thickness/'+mdl.name+'.obj'
            mdl = Structure.load_from_obj(file_obj)
            f_n = np.array(mdl.results['step_modal']['frequencies'])
            m_n = np.array(mdl.results['step_modal']['masses'])
            n_modes_max = f_n.shape[0]
            try:
                node_lp = mdl.sets['nset_loadPoint']['selection']
            except:
                node_lp = mdl.sets['nset_loadPoint'].selection
            #

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

            # ODE solving and rms calculation
            start = timeit.default_timer()

            dis_mode = np.zeros((n_nodes, n))
            vel_mode = np.zeros((n_nodes, n))
            acc_mode = np.zeros((n_nodes, n))
            rms_mode = np.zeros(n)
            rms_mode_weight = np.zeros(n)
            rms_modes = np.zeros(n_modes)
            rms_modes_weight = np.zeros(n_modes)

            dis_modal = np.zeros((n_nodes, n))
            vel_modal = np.zeros((n_nodes, n))
            acc_modal = np.zeros((n_nodes, n))
            rms_modal = np.zeros(n)  # only for load point, rms from superposition of weighted mode rms
            rms_modal_weight = np.zeros(n)

            # response from the first mode
            dis_modal_1 = np.zeros((n_nodes, n))
            vel_modal_1 = np.zeros((n_nodes, n))
            acc_modal_1 = np.zeros((n_nodes, n))
            rms_modal_1 = np.zeros(n)  # only for load point, rms from superposition of weighted mode rms
            rms_modal_weight_1 = np.zeros(n)

            rms_acc_modal = np.zeros(n)
            rms_acc_modes = np.zeros(n_modes)

            R = np.zeros(n_modes)
            R_weight = np.zeros(n_modes)
            R_acc = np.zeros(n_modes)

            # modal participation factor
            Gamma_n = np.zeros(n_modes)

            for i in range(n_modes):
                # Solve with multi-DOF
                def fn_modal(u, t, m, w, s, uz):
                    return [u[1], Fn(t) * s @ uz / m - u[0] * w ** 2 - 2 * ksi * w * u[1]]

                sol = odeint(fn_modal, [0, 0], t[:], args=(m_n[i], 2 * np.pi * f_n[i], s, uz_ms[i]))

                Gamma_n[i] = s @ uz_ms[i] / m_n[i]

                for j in range(n):
                    dis_mode[:, j] = sol[j, 0] * uz_ms[i] * 1000
                    vel_mode[:, j] = sol[j, 1] * uz_ms[i]

                for j in range(n_nodes):
                    acc_mode[j, int(t_cut / dt):] = np.gradient(vel_mode[j, int(t_cut / dt):], dt)

                # calculate rms in each mode
                for j in range(T_rms, n):
                    rms_mode[j] = np.sqrt(np.mean(acc_mode[node_lp, j - T_rms:j] ** 2))

                rms_mode_weight[:] = rms_mode[:] * Wb(f_n[i])

                # superpose the responses
                dis_modal += dis_mode
                vel_modal += vel_mode
                acc_modal += acc_mode
                # calculate acc-rms from sum till nth mode
                for j in range(T_rms, n):
                    rms_acc_modal[j] = np.sqrt(np.mean(acc_modal[node_lp, j - T_rms:j] ** 2))

                rms_modal += rms_mode
                rms_modal_weight += rms_mode_weight

                rms_modes[i] = np.max(rms_modal)
                rms_modes_weight[i] = np.max(rms_modal_weight)
                rms_acc_modes[i] = np.max(rms_acc_modal)

                R[i] = rms_modes[i] / acc_base
                R_weight[i] = rms_modes_weight[i] / acc_base
                R_acc[i] = rms_acc_modes[i] / acc_base

                if i == 0:
                    dis_modal_1[:] = dis_modal[:]
                    vel_modal_1[:] = vel_modal[:]
                    acc_modal_1[:] = acc_modal[:]
                    rms_modal_1[:] = rms_modal[:]  # only for load point, rms from superposition of weighted mode rms
                    rms_modal_weight_1[:] = rms_modal_weight[:]

                print('ODE: mode', str(i + 1))

            stop = timeit.default_timer() #

            ### plot

            # # plot footfall loading
            # fig  = plt.figure()
            # axes = fig.add_subplot(111)
            # axes.set_title('Footfall loading', fontsize=12)
            # axes.set_xlabel('Time $t$ [s]', fontsize=12)
            # axes.set_ylabel('Force $F$ [N]', fontsize=12)
            # axes.minorticks_on()
            # axes.plot(t, F, '-', color=[1, 0, 0])
            # axes.plot([0, te], [W, W], ':', color=[0.7, 0.7, 0.7])

            # # plot response
            # fig = plt.figure()
            # fig.suptitle(
            #     'Footfall Response with Modal Analysis (span=' + str(span) + 'm, l/d=' + str(l2d) + ', gamma='+str(gamma)+', dt=' + str(
            #         dt) + ', ' + str(n_modes) + ' modes)')
            # axes1 = fig.add_subplot(311)
            # axes1.set_xlabel('Time $t$ [s]', fontsize=12)
            # axes1.set_ylabel('Displacement $u$ [mm]', fontsize=12)
            # axes1.minorticks_on()
            # axes1.plot(t, np.transpose(dis_modal[node_lp, :]), color='r')
            # axes1.plot(t, np.transpose(dis_modal_1[node_lp, :]), '--', color='r')
            # axes1.legend([str(n_modes) + ' modes', 'first mode'])
            #
            # axes2 = fig.add_subplot(312)
            # axes2.set_xlabel('Time $t$ [s]', fontsize=12)
            # axes2.set_ylabel('Acceleration $a$ [$m/s^2$]', fontsize=12)
            # axes2.minorticks_on()
            # axes2.plot(t, np.transpose(acc_modal[node_lp, :]), 'r')
            # axes2.plot(t, np.transpose(acc_modal_1[node_lp, :]), '--', color='r')
            # axes2.plot([0, t[-1]], [0, 0], '--', color=[0.7, 0.7, 0.7])
            # axes2.legend([str(n_modes) + ' modes', 'first mode'])
            #
            # axes3 = fig.add_subplot(313)
            # axes3.set_xlabel('Time $t$ [s]', fontsize=12)
            # axes3.set_ylabel('Acceleration (RMS) $a_\mathrm{rms}$ [$m/s^2$]', fontsize=12)
            # axes3.minorticks_on()
            # axes3.plot(t, rms_modal, color='r')
            # axes3.plot(t, rms_modal_1, '--', color='r')
            # axes3.plot(t, rms_modal_weight, color='b')
            # axes3.plot(t, rms_modal_weight_1, '--', color='b')
            # axes3.legend(
            #     ['50 modes not weighted', 'first mode not weighted', '50 modes weighted', 'first mode weighted'])
            # axes3.plot([0, t[-1]], [0, 0], '--', color=[0.7, 0.7, 0.7])
            #
            # # plot response factor in relation to number of modes involved
            # fig = plt.figure()
            # axes = fig.add_subplot(111)
            # axes.set_title(
            #     'Rsponse factor in relation to number of modes involved (span=' + str(span) + 'm, l/d=' + str(l2d) + ', gamma='+str(gamma)+', dt=' + str(
            #                      dt) + ', ' + str(n_modes) + ' modes)', fontsize=12)
            # axes.set_xlabel('Number of modes involved in modal analysis', fontsize=12)
            # axes.set_ylabel('Response Factor', fontsize=12)
            # axes.minorticks_on()
            # axes.plot(range(1, n_modes + 1), R, 'r')
            # axes.plot(range(1, n_modes + 1), R_weight, 'b')
            # axes.set_ylim(ymin=0)
            # axes.set_xlim(xmin=0)
            # axes.legend(['not weighted response factor', 'weighted response factor'], loc=2)
            # # plot participation factor
            # axes2 = axes.twinx()
            # axes2.set_ylabel('Participation factor [-]', fontsize=12)
            # axes2.tick_params('y')
            # axes2.minorticks_on()
            # axes2.bar(range(1, n_modes + 1), Gamma_n, width=0.2, color='m')
            # axes2.legend(['participation factor'], loc=1)
            # axes2.plot([1, n_modes + 1], [0, 0], '--', color=[0.7, 0.7, 0.7])

            ### save important variables
            with open ('D:/Master_Thesis/footfall_analysis/data/data_mdl_funicular_0_4m/data_'+mdl.name+'.pkl','wb') as data:
                pickle.dump([W,te,t,F,f_n,m_n,node_lp,n_modes,dt,dis_modal[node_lp],acc_modal[node_lp],rms_modal,rms_modal_weight,dis_modal_1[node_lp],acc_modal_1[node_lp],rms_modal_1,rms_modal_weight_1,rms_acc_modal,rms_modes,rms_modes_weight,rms_acc_modes,R,R_weight,R_acc,Gamma_n],data)

            print('Time for solving ' + str(n_modes) + ' modes of ' + mdl.name + ': ' + str(stop - start) + 's')

plt.show()
# plt.show(block=False)

# print()