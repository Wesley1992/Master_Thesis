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

spans = [10]
l2ds = [17.5,20]

# spans = [5,8,10]
# l2ds = [10,15,20]

### parameters often changed
n_modes = 50  # number of modes used shall not exceed that extracted from abaqus modal analysis
dt = 0.001  # s
t_cut = 0

### Loading
W = -76 * 9.81    # load from walking people
f_load = 2.0  # hz, frequency of walking
ksi = 0.03  # -, damping ratio

te = 5  # time span for history analysis
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

        start_mdl = timeit.default_timer()

        # Input obj file
        mdl = Structure(name='mdl_plate_span'+str(span).replace('.','_')+'_l2d'+str(l2d).replace('.','_'), path=os.path.dirname(os.path.abspath(__file__)))
        file_obj = 'D:/Master_Thesis/modal/modal_plate_span_0_4depth/'+mdl.name+'.obj'
        mdl = Structure.load_from_obj(file_obj)
        f_n = np.array(mdl.results['step_modal']['frequencies'])
        m_n = np.array(mdl.results['step_modal']['masses'])
        n_modes_max = f_n.shape[0]
        try:
            node_lp = mdl.sets['nset_loadPoint']['selection']
        except:
            node_lp = mdl.sets['nset_loadPoint'].selection
        n_nodes = mdl.nodes.__len__()
        s = np.zeros(n_nodes) # spatial distribution of load
        s[node_lp] = 1

        # extract mode shape
        uz_ms = []  # mode shape z direction

        for i in range(n_modes):
            uz_ms.append([])
            for node in mdl.nodes:
                uz_ms[i].append(mdl.results['step_modal']['nodal']['uz{0}'.format(i + 1)][node])
        uz_ms = np.array(uz_ms)

        # ODE solving and rms calculation
        dis_mode = np.zeros((n_nodes, n))
        vel_mode = np.zeros((n_nodes, n))
        acc_mode = np.zeros((n_nodes, n))

        dis = np.zeros((n_nodes, n))
        vel = np.zeros((n_nodes, n))
        acc = np.zeros((n_nodes, n))

        acc_weight = np.zeros((n_nodes, n))

        dis_modes_lp = np.zeros((n_modes, n))
        vel_modes_lp = np.zeros((n_modes, n))
        acc_modes_lp = np.zeros((n_modes, n))
        acc_modes_lp_weight = np.zeros((n_modes, n))

        rms_modes = np.zeros((n_modes, n))
        rms_modes_weight = np.zeros((n_modes, n))

        R = np.zeros(n_modes)
        R_weight = np.zeros(n_modes)

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

            dis += dis_mode
            vel += vel_mode
            acc += acc_mode
            acc_weight += acc_mode * Wb(f_n[i])

            dis_modes_lp[i, :] = dis[node_lp, :]
            vel_modes_lp[i, :] = vel[node_lp, :]
            acc_modes_lp[i, :] = acc[node_lp, :]
            acc_modes_lp_weight[i, :] = acc_weight[node_lp, :]

            for j in range(T_rms, n):
                rms_modes[i, j] = np.sqrt(np.mean(acc_modes_lp[i, j - T_rms:j] ** 2))
                rms_modes_weight[i, j] = np.sqrt(np.mean(acc_modes_lp_weight[i, j - T_rms:j] ** 2))

            R[i] = np.max(rms_modes[i, :]) / acc_base
            R_weight[i] = np.max(rms_modes_weight[i, :]) / acc_base

            print('ODE: mode', str(i + 1))


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

        # plot response
        fig = plt.figure()
        fig.suptitle(
            'Footfall Response with Modal Analysis (span=' + str(span) + 'm, l/d=' + str(l2d) + ' dt=' + str(
                dt) + ', ' + str(n_modes) + ' modes)')
        axes1 = fig.add_subplot(311)
        axes1.set_xlabel('Time $t$ [s]', fontsize=12)
        axes1.set_ylabel('Displacement $u$ [mm]', fontsize=12)
        axes1.minorticks_on()
        axes1.plot(t, np.transpose(dis_modes_lp[-1, :]), color=[1, 0, 0])

        axes2 = fig.add_subplot(312)
        axes2.set_xlabel('Time $t$ [s]', fontsize=12)
        axes2.set_ylabel('Acceleration $a$ [$m/s^2$]', fontsize=12)
        axes2.minorticks_on()
        axes2.plot(t, np.transpose(acc_modes_lp[-1, :]), color=[1, 0, 0])
        axes2.plot([0, t[-1]], [0, 0], '--', color=[0.7, 0.7, 0.7])

        axes3 = fig.add_subplot(313)
        axes3.set_xlabel('Time $t$ [s]', fontsize=12)
        axes3.set_ylabel('Acceleration (RMS) $a_\mathrm{rms}$ [$m/s^2$]', fontsize=12)
        axes3.minorticks_on()
        axes3.plot(t, rms_modes[-1, :], color=[1, 0, 0])
        axes3.plot(t, rms_modes_weight[-1, :], color=[0, 0, 1])
        axes3.legend(['not weighted', 'weighted'])
        axes3.plot([0, t[-1]], [0, 0], '--', color=[0.7, 0.7, 0.7])

        # plot response factor in relation to number of modes involved
        fig = plt.figure()
        axes = fig.add_subplot(111)
        axes.set_title(
            'Rsponse factor in relation to number of modes involved (span=' + str(span) + 'm, l/d=' + str(
                l2d) + ', dt=' + str(dt) + ')', fontsize=12)
        axes.set_xlabel('Number of modes involved in modal analysis', fontsize=12)
        axes.set_ylabel('Response Factor', fontsize=12)
        axes.set_xticks(range(1, n_modes + 1))
        axes.minorticks_on()
        axes.tick_params(axis='x', which='minor', bottom=False)
        axes.plot(range(1, n_modes + 1), R, color=[1, 0, 0])
        axes.plot(range(1, n_modes + 1), R_weight, color=[0, 0, 1])
        axes.set_ylim(ymin=0)
        axes.set_xlim(xmin=0)
        axes.legend(['not weighted', 'weighted'])
        axes.plot([1, n_modes + 1], [0, 0], '--', color=[0.7, 0.7, 0.7])

        # plot participation factor
        axes2 = axes.twinx()
        axes2.set_ylabel('Participation factor [-]', fontsize=12)
        axes2.minorticks_on()
        axes2.tick_params(axis='x', which='minor', bottom=False)
        axes2.bar(range(1, n_modes + 1), Gamma_n, width=0.2, color='m')
        axes2.legend(['participation factor'], loc=1)
        axes2.plot([1, n_modes + 1], [0, 0], '--', color=[0.7, 0.7, 0.7])

        ### save important variables
        # with open('D:/Master_Thesis/code_data/footfall_analysis_plate_0_4depth/data/data_' + mdl.name + '.pkl',
        #           'wb') as data:
        #     pickle.dump(
        #         [f_n, m_n, node_lp, n_modes, dt, t, dis_modes_lp, vel_modes_lp, acc_modes_lp, acc_modes_lp_weight,
        #          rms_modes, rms_modes_weight, R, R_weight, Gamma_n], data)

        stop_mdl = timeit.default_timer()

        print('Time for solving ' + str(n_modes) + ' modes of ' + mdl.name + ': ' + str(stop_mdl - start_mdl) + 's')

plt.show()

# plt.show(block=False)

# print()