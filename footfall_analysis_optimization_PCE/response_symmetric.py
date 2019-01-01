
from compas_fea.structure import Structure

from compas.numerical import devo_numpy

from scipy.optimize import fmin_slsqp

from numpy import ones
import pickle
import numpy as np
from scipy.interpolate import griddata
import math
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import os
import glob
import timeit



__author__    = ['Andrew Liew <liew@arch.ethz.ch>']
__copyright__ = 'Copyright 2018, BLOCK Research Group - ETH Zurich'
__license__   = 'MIT License'
__email__     = 'liew@arch.ethz.ch'


def Wb(f):
    if 1 < f < 2:
        Wb = 0.4
    elif 2 <= f < 5:
        Wb = f / 5
    elif 5 <= f <= 16:
        Wb = 1.0
    elif f > 16:
        Wb = 16 / f
    return Wb


# function to be evaluated and minimized (R1 in this case)
def fn(ts):

    mdl = Structure.load_from_obj('D:/Master_Thesis/modal/modal_symmetric/mdl_span5_l2d10_gamma1_mesh_symmetric.obj', output=0)
    mdl.name = 'mdl_span5_l2d10_gamma1_mesh_symmetric_opt'

    v0 = 5 ** 3 / 10 * 0.4 / 4  # span**3/l2d*ratio/4, initial volume
    v  = 0

    for i in range(n_v):
        t_v = ts[i]
        a_v = mdl.areas['elset_vault_{0}'.format(i + 1)]
        v += t_v*a_v

    for i in range(n_r):
        t_r = ts[n_v+i]
        a_r = mdl.areas['elset_ribs_{0}'.format(i + 1)]
        v += t_r*a_r

    for i in range(n_r,n_r+n_r_h):
        t_r_h = ts[n_v+i]/2
        a_r_h = mdl.areas['elset_ribs_{0}'.format(i + 1)]
        v += t_r_h*a_r_h

    print('v0='+str(v0))
    print('v='+str(v))
    scale = v0 / v

    for i in range(n_v):
        t_v = ts[i] * scale
        mdl.sections['sec_vault_{0}'.format(i + 1)].geometry['t'] = t_v

    for i in range(n_r):
        t_r = ts[n_v + i] * scale
        mdl.sections['sec_ribs_{0}'.format(i + 1)].geometry['t'] = t_r

    for i in range(n_r,n_r+n_r_h):
        t_r_h = ts[n_v+i]/2*scale
        mdl.sections['sec_ribs_{0}'.format(i + 1)].geometry['t'] = t_r_h

    print('t_v='+str(t_v))
    print('t_r='+str(t_r))

    file_abaqus = 'D:/Master_Thesis/modal/modal_symmetric/' + mdl.name

    if os.path.exists(file_abaqus):
        os.chdir(file_abaqus)
        for file in glob.glob("*.lck"):
            os.remove(file)

    mdl.analyse_and_extract(software='abaqus', fields=['u'],components=['uz'])
    # mdl.save_to_obj()

    m1 = mdl.results['step_modal']['masses'][0]
    f1 = mdl.results['step_modal']['frequencies'][0]
    print('m1='+str(4*m1))
    print('f1='+str(f1))

    # interpolate R1 based on m1 and f1 from existing data
    m1_f1_intp_grid_m1, m1_f1_intp_grid_f1 = np.meshgrid(4*m1, f1)

    R1_weight= griddata(np.vstack((m1_scatter, f1_scatter)).T, R1_weight_scatter,
                              (m1_f1_intp_grid_m1, m1_f1_intp_grid_f1), method='linear')

    # if data not available, solve ODE
    if math.isnan(float(R1_weight)):
        print('out of interpolation range')
        start_ODE = timeit.default_timer()

        ### parameters often changed
        n_modes = 1  # number of modes used shall not exceed that extracted from abaqus modal analysis
        dt = 0.0005  # s
        t_cut = 0

        ### Loading
        W = -76 * 9.81/4  # load from walking people
        f_load = 2.0  # hz, frequency of walking
        ksi = 0.03  # -, damping ratio

        te = 1  # time span for history analysis
        n = int(te / dt) + 1
        t = np.linspace(0, te, n)  # s
        a = [0.436 * (1 * f_load - 0.95),
             0.006 * (2 * f_load + 12.3),
             0.007 * (3 * f_load + 5.20),
             0.007 * (4 * f_load + 2.00)]
        p = [0, -0.5 * np.pi, np.pi, 0.5 * np.pi]
        F = np.zeros(n) + W  # N

        for i in range(len(a)):
            F += W * a[i] * np.sin(2 * np.pi * (i + 1) * f_load * t + p[i])  # N

        Fn = interp1d(t, F, fill_value="extrapolate")

        ### rms and response factor parameters
        T_rms = int(1 / f_load // dt) + 2  # time period for calculating rms
        acc_base = 0.005

        ### solving
        # file_obj = 'D:/Master_Thesis/modal/modal_symmetric/mdl_span5_l2d10_gamma1_mesh_symmetric_opt.obj'
        # mdl = Structure.load_from_obj(file_obj)

        f_n = np.array(mdl.results['step_modal']['frequencies'])
        m_n = np.array(mdl.results['step_modal']['masses'])

        try:
            node_lp = mdl.sets['nset_loadPoint']['selection']
        except:
            node_lp = mdl.sets['nset_loadPoint'].selection

        n_nodes = mdl.nodes.__len__()
        s = np.zeros(n_nodes)  # spatial distribution of load
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

            stop_ODE = timeit.default_timer()
            print('Time for solving mode '+str(i+1)+': ' + str(stop_ODE - start_ODE) + 's')

        R1_weight = R_weight[0]

    print('R1='+str(R1_weight))

    return R1_weight


### data loading for reponse interpolation
spans = [5,6,7,8,9,10]
l2ds = [10,12.5,15,17.5,20]
gammas = [0.1,0.5,1,2,5,10]

data_file = 'D:/Master_Thesis/code_data/footfall_analysis/data/other/data_m1_f1_R1.pkl'

m1_data,f1_data,R1_weight_data = pickle.load(open(data_file,'rb'))

m1_scatter = m1_data.reshape(len(spans)*len(l2ds)*len(gammas))
f1_scatter = f1_data.reshape(len(spans)*len(l2ds)*len(gammas))
R1_weight_scatter = R1_weight_data.reshape(len(spans)*len(l2ds)*len(gammas))

### thickness input
# number of panels for each group
n_v = 16    # vault
n_r = 21    # ribs
n_r_h = 4   # ribs with half thickness

t_v = 0.01*ones(n_v)
t_r = 0.2*ones(n_r)
t_r_h = 0.2*ones(n_r_h)
ts = np.concatenate((t_v,t_r,t_r_h))

start = timeit.default_timer()

fn(ts)  # function for evaluation and minimization

stop = timeit.default_timer()

print('time for fn evaluation:'+str(stop-start)+' s')


#
# ts = ones(40) * 0.030
# print(fn(ts))

#bounds = [[0.010, 0.060] for i in range(40)]
#ans = devo_numpy(fn=fn, bounds=bounds, population=150, generations=300, printout=1)
#ans = fmin_slsqp(fn, ts, disp=2, bounds=bounds, full_output=1, iter=50)
