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



# parameters often changed
span = 5
l2d = 20
gamma = 1

n_modes = 100  # number of modes used shall not exceed that extracted from abaqus modal analysis
dt = 0.0005  # s
# t_cut = 0
t_cut = 0.0005

# Input obj file
mdl = Structure(name='abq_mdl_span'+str(span).replace('.','_')+'_l2d'+str(l2d).replace('.','_')+'_gamma'+str(gamma).replace('.','_'), path=os.path.dirname(os.path.abspath(__file__)))
file_obj = 'D:/Master_Thesis/modal_span_depth_thickness/'+mdl.name+'.obj'  # need to be changed if rus on another computer
mdl = Structure.load_from_obj(file_obj)
f_n = np.array(mdl.results['step_modal']['frequencies'])
m_n = np.array(mdl.results['step_modal']['masses'])

n_modes_max = f_n.shape[0]
node_lp = mdl.sets['nset_loadPoint']['selection']

xyz_nodes = mdl.nodes_xyz()
xyz_nodes = np.array(xyz_nodes)
n_nodes = xyz_nodes.shape[0]
# xyz_lp = xyz_nodes[node_lp]

# extract mode shape

uz_ms = []  # mode shape z direction
# ux_ms = []  # mode shape x direction
# uy_ms = []  # mode shape y direction
# um_ms = []  # norm of mode shape vector

for i in range(n_modes):
    uz_ms.append([])
    # ux_ms.append([]); uy_ms.append([]); um_ms.append([])
    for node in mdl.nodes:
            uz_ms[i].append(mdl.results['step_modal']['nodal']['uz{0}'.format(i+1)][node])
            # ux_ms[i].append(mdl.results['step_modal']['nodal']['ux{0}'.format(i+1)][node])
            # uy_ms[i].append(mdl.results['step_modal']['nodal']['uy{0}'.format(i+1)][node])
            # um_ms[i].append(mdl.results['step_modal']['nodal']['um{0}'.format(i+1)][node])

uz_ms = np.array(uz_ms)
# ux_ms = np.array(ux_ms)
# uy_ms = np.array(uy_ms)
# um_ms = np.array(um_ms)

# Loading and ODE parameters
W = -76 * 9.81    # load from walking people
f_load = 2.0  # hz, frequency of walking
ksi = 0.03  # -, damping ratio
s = np.zeros(n_nodes) # spatial distribution of load
s[node_lp] = 1

te = 1  # time span for history analysis
# te = 10 * 2 / f_load
T_rms = int(1/f_load//dt)+2  # time period for calculating rms
acc_base = 0.005

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

# F[n // 2:] = W

Fn = interp1d(t, F, fill_value="extrapolate")

# plot walk loading

fig  = plt.figure()

axes = fig.add_subplot(111)
axes.set_title('Footfall loading', fontsize=12)
axes.set_xlabel('Time $t$ [s]', fontsize=12)
axes.set_ylabel('Force $F$ [N]', fontsize=12)
axes.minorticks_on()
axes.plot(t, F, '-', color=[1, 0, 0])
axes.plot([0, te], [W, W], ':', color=[0.7, 0.7, 0.7])


# ODE solve

start = timeit.default_timer()

dis_mode = np.zeros((n_nodes, n))
vel_mode = np.zeros((n_nodes, n))
acc_mode = np.zeros((n_nodes, n))
rms_mode = np.zeros(n)
rms_mode_weight = np.zeros(n)
rms_modes_rms = np.zeros(n_modes)
rms_modes_rms_weight = np.zeros(n_modes)
rms_modes_acc = np.zeros(n_modes)
rms_modes_acc_weight = np.zeros(n_modes)


dis_modal = np.zeros((n_nodes, n))
vel_modal = np.zeros((n_nodes, n))
acc_modal = np.zeros((n_nodes, n))
rms_modal_rms = np.zeros(n)  # only for load point, rms from superposition of weighted mode rms
rms_modal_rms_weight = np.zeros(n)
rms_modal_acc = np.zeros((n_modes, n))  # only for load point, rms from weighted superposed acceleration(acc_modal_weight)
rms_modal_acc_weight = np.zeros((n_modes, n))
# rms_modal_1 = np.zeros(n)
# acc_lp_modes = np.zeros((n_modes, n)) # the acc of load point calculated by different numbers of modes

# dis_modal_1 = np.zeros((n_nodes, n))  # response from the 1st mode
# vel_modal_1 = np.zeros((n_nodes, n))
# acc_modal_1 = np.zeros((n_nodes, n))

acc_mode_weight = np.zeros((n_nodes, n))
acc_modal_weight = np.zeros((n_nodes, n))
# acc_lp_modes_weight = np.zeros((n_modes, n))
# rms_modal_weight = np.zeros(n)


for i in range(n_modes):

    #Solve with multi-DOF
    def fn_modal(u, t, m, w, s, uz):
        return [u[1], Fn(t)*s@uz / m - u[0] * w**2 - 2 * ksi * w * u[1]]
    sol = odeint(fn_modal, [0, 0], t[:], args=(m_n[i], 2 * np.pi * f_n[i], s, uz_ms[i]))

    for j in range(n):
        dis_mode[:,j] = sol[j, 0]*uz_ms[i]*1000
        vel_mode[:,j] = sol[j, 1]*uz_ms[i]

    for j in range(n_nodes):
        # acc_mode[j,:] = np.gradient(vel_mode[j,:], dt)
        # cut the first several miliseconds
        acc_mode[j,int(t_cut/dt):] = np.gradient(vel_mode[j,int(t_cut/dt):], dt)

    ### calculate and weight the rms in each mode
    for j in range(T_rms, n):
        rms_mode[j] = np.sqrt(np.mean(acc_mode[node_lp, j - T_rms:j] ** 2))
    rms_mode_weight[:] = rms_mode[:] * Wb(f_n[i])

    ### calculate and weight the acc in each mode
    acc_mode_weight[:] = acc_mode[:]*Wb(f_n[i])

    dis_modal += dis_mode
    vel_modal += vel_mode
    acc_modal += acc_mode
    acc_modal_weight += acc_mode_weight
    # acc_modal_weight += acc_mode_weight
    rms_modal_rms += rms_mode
    rms_modal_rms_weight += rms_mode_weight

    ### calculate the rms from acceleration
    for j in range(T_rms, n):
        rms_modal_acc[i,j] = np.sqrt(np.mean(acc_modal[node_lp, j - T_rms:j] ** 2))
        rms_modal_acc_weight[i, j] = np.sqrt(np.mean(acc_modal_weight[node_lp, j - T_rms:j] ** 2))

    rms_modes_rms[i] = np.max(rms_modal_rms)
    rms_modes_rms_weight[i] = np.max(rms_modal_rms_weight)
    rms_modes_acc[i] = np.max(rms_modal_acc[i,:])
    rms_modes_acc_weight[i] = np.max(rms_modal_acc_weight[i,:])

    # for j in range(i+1):
    #     rms_modes[j] += np.max(rms_mode[i,:])
    #     rms_modes[j] += np.max(rms_mode[i, :])

    # if i == 0:
    #     dis_modal_1[:] = dis_mode[:]
    #     vel_modal_1[:] = vel_mode[:]
    #     acc_modal_1[:] = acc_mode[:]

    print('ODE: mode',str(i+1))

# for i in range(T_rms, n - 1):
#     rms_modal[i] = np.sqrt(np.mean(acc_modal[node_lp,i-T_rms:i]**2))
#     rms_modal_1[i] = np.sqrt(np.mean(acc_modal_1[node_lp, i - T_rms:i] ** 2))
    # rms_modal_weight[i] = np.sqrt(np.mean(acc_modal_weight[node_lp,i-T_rms:i]**2))

stop = timeit.default_timer()


### read results from abaqus calculation

file = open('response_'+mdl.name+'.txt')
lines = file.readlines()
t_abq = []
acc_abq = []
dis_abq = []
rms_abq = np.zeros(n)

for line in lines:
    t_abq.append(line.split()[0])
    acc_abq.append(line.split()[1])
    dis_abq.append(line.split()[2])
file.close()


t_abq = np.array([float(i) for i in t_abq])
dis_abq = np.array([float(i) for i in dis_abq])*1000

acc_abq = np.gradient(np.gradient(dis_abq/1000,dt),dt)
acc_abq[:int(t_cut/dt)] = 0

# acc_abq = np.array([float(i) for i in acc_abq])

n_abq = t_abq.shape[0]
rms_abq = np.zeros(n_abq)

for i in range(T_rms, n_abq):
    rms_abq[i] = np.sqrt(np.mean(acc_abq[i-T_rms:i]**2))

### plot response under walking load

# Modal Analysis
fig  = plt.figure()
fig.suptitle('Comparison of Footfall Responses (span='+str(span)+'m, l/d='+str(l2d)+', gamma='+str(gamma)+', dt='+str(dt)+', '+str (n_modes)+' modes)')
axes1 = fig.add_subplot(311)
axes1.set_xlabel('Time $t$ [s]', fontsize=12)
axes1.set_ylabel('Displacement $u$ [mm]', fontsize=12)
axes1.minorticks_on()
axes1.plot(t, np.transpose(dis_modal[node_lp,:]),color=[1, 0, 0])
# axes1.plot(t[:-1], np.transpose(dis_modal_1[node_lp,:]),'--', color=[1, 0, 0])
axes1.plot(t_abq, dis_abq,color=[0, 0, 1])
axes1.legend(['modal analysis', 'abaqus'])
# axes1.plot([0, t[-1]], [dis_modal[node_lp,-1], dis_modal[node_lp,-1]], '--', color=[0.7, 0.7, 0.7])

axes2 = fig.add_subplot(312)
axes2.set_xlabel('Time $t$ [s]', fontsize=12)
axes2.set_ylabel('Acceleration $a$ [$m/s^2$]', fontsize=12)
axes2.minorticks_on()
axes2.plot(t, np.transpose(acc_modal[node_lp,:]),color=[1, 0, 0])
# axes2.plot(t[:-1], np.transpose(acc_modal_weight[node_lp,:]),color=[0, 0, 1])
# axes2.plot(t[:-1], np.transpose(acc_modal_1[node_lp,:]),'--', color=[1, 0, 0])
axes2.plot(t_abq, acc_abq,color=[0, 0, 1])
axes2.legend(['modal analysis', 'abaqus'])
axes2.plot([0, t[-1]], [0, 0], '--', color=[0.7, 0.7, 0.7])

axes3 = fig.add_subplot(313)
axes3.set_xlabel('Time $t$ [s]', fontsize=12)
axes3.set_ylabel('RMS Acceleration $a_\mathrm{rms}$ [$m/s^2$]', fontsize=12)
axes3.minorticks_on()
# axes3.plot(t, rms_modal_acc[n_modes-1],color=[1,0,0])
# axes3.plot(t, rms_modal_acc_weight[n_modes-1],'--',color=[1,0,0])
axes3.plot(t, rms_modal_rms,color=[1,0,0])
axes3.plot(t, rms_modal_rms_weight,'--',color=[1,0,0])
# axes3.plot(t, rms_modal_weight,color=[1,0,0])
# axes3.plot(t[:-1], rms_modal_1,'--',color=[1,0,0])
# axes3.plot(t_abq, rms_abq,color=[0,0,1])
# axes3.legend(['acc not weighted', 'acc weighted','rms not weighted', 'rms weighted', 'abaqus'])
axes3.legend(['not weighted', 'weighted'])
axes3.plot([0, t[-1]], [0, 0], '--', color=[0.7, 0.7, 0.7])


# plot response factor in relation to number of modes involved
fig  = plt.figure()

axes = fig.add_subplot(111)
axes.set_title('Rsponse factor in relation to number of modes involved (span='+str(span)+'m, l/d='+str(l2d)+', gamma='+str(gamma)+', dt='+str(dt)+')', fontsize=12)
axes.set_xlabel('Number of modes involved in modal analysis', fontsize=12)
axes.set_ylabel('Response Factor', fontsize=12)
axes.minorticks_on()
# axes.plot(range(1,n_modes+1), rms_modes_acc/acc_base, color=[1, 0, 0])
# axes.plot(range(1,n_modes+1), rms_modes_acc_weight/acc_base, '--', color=[1, 0, 0])
axes.plot(range(1,n_modes+1), rms_modes_rms/acc_base, color=[1, 0, 0])
axes.plot(range(1,n_modes+1), rms_modes_rms_weight/acc_base,'--', color=[1, 0, 0])
axes.legend(['not weighted', 'weighted'])
axes.plot([1, n_modes+1], [0, 0], '--', color=[0.7, 0.7, 0.7])

print('Time for solving '+str(n_modes)+' modes: '+str(stop-start)+'s')

plt.show()
