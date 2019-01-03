import numpy as np
import itertools as it
from sympy.utilities.iterables import multiset_permutations

from compas_fea.structure import Structure

import pickle
from scipy.interpolate import griddata
import math
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import os
import glob
import timeit
from pyDOE2 import lhs

n_v = 16
n_r = 21
n_r_h = 4

def create_legendrePoly(p,x):

    """Construct Legendre polynomials

    Parameters
    ----------
    p : int
        the max. degree of univariate polynomials.
    x : float
        input point to be evaluated.

    Returns
    -------
    lgd_poly
        an array of evaluated polynomials up to nth degree.

    """
    for i in range(p+1):
        if i == 0:
            lgd_poly = np.array([1],dtype='f')

        elif i == 1:
            lgd_poly = np.append(lgd_poly,x)

        elif i >= 2:
            lgd_poly = np.append(lgd_poly,1/i*((2*i-1)*x*lgd_poly[-1]-(i-1)*lgd_poly[-2]))

    for i in range(p+1):
        lgd_poly[i] = np.sqrt(2*i+1)*lgd_poly[i]

    return lgd_poly

def create_degreeIndex(M,p):
    """Construct degree index tuples indicating the degrees of univariate polynomials for multivariate polynomials basis by multiplication

    Parameters
    ----------
    M : int
        dimension of model (number of input variables).
    p : int
        the total degree of metal model.

    Returns
    -------
    index
        a list of arrays indicating the degrees of univariate polynomials

    """
    # Firstly generates all the tuples of integers a(i) that satisfy the following conditions:
    #     - a(i) <= P
    #     - sum(a) = P
    #     - a(i+1) <= a(i)
    # The additional constraints must be met: P >= J

    # Then calculate the permutation without repetition

    if p == 0:
        index = [0]*M
    else:
        index = []
        for pp in range(1,p+1):
            index.append([])
            # jj is the number of interaction terms
            for jj in range(1,np.min([pp,M])+1):
                index[pp-1].append([])
                # numbers are possible values whose sum can be pp
                numbers = [i+1 for i in range(pp)]
                # in each jj, combinations of numbers with sum=qq are selected
                combs_jj = [seq for seq in it.combinations_with_replacement(numbers,jj) if sum(seq)==pp]
                for k in range(len(combs_jj)):
                    index[pp-1][jj-1].append([])
                    index[pp-1][jj-1][k] = list(combs_jj[k])
                    index[pp - 1][jj - 1][k].sort(reverse=True)
                    # supplement remaining positions with zeros
                    for n_zeros in range(M-len(index[pp-1][jj-1][k])):
                        index[pp - 1][jj - 1][k].append(0)
                # calculate unique permutations without repetitions (can be very computationally intensive and memory demanding)
                for k in range(len(index[pp - 1][jj - 1])):
                    permut = multiset_permutations(index[pp - 1][jj - 1][k])
                    index[pp - 1][jj - 1][k] = permut
    # open the nested list
    for i in range(3):
        index = list(it.chain.from_iterable(index))
    # incorporate the scenario of zero degree for all variables
    index.insert(0,[0 for i in range(M)])
    return index

def create_multiVarPoly(degree_index,x,strategy):
    # x is the input of all variables in one experimental design
    P = len(degree_index)
    M = x.shape[0]
    multi_poly = np.ones(P)

    if strategy == 'legendre':
        xi = x
        for i in range(P):
            index = degree_index[i]
            for j in range(M):
                uni_poly = create_legendrePoly(index[j],xi[j])[-1]
                multi_poly[i] *= uni_poly

    elif strategy == 'normal':
        for i in range(P):
            index = degree_index[i]
            for j in range(M):
                uni_poly = x[j]**index[j]
                multi_poly[i] *= uni_poly

    return multi_poly

def create_Psi(degree_index,x_ED,strategy):
    # xi_ED(M,n)is the input of experimental design of all variables
    n = x_ED.shape[1]
    P = len(degree_index)
    Psi = np.zeros((n,P))
    for i in range(n):
        Psi[i] = create_multiVarPoly(degree_index,x_ED[:,i],strategy)
    return Psi

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

def evaluate_response(ts):

    global n_v,n_r,n_r_h

    ### data loading for reponse interpolation
    spans = [5, 6, 7, 8, 9, 10]
    l2ds = [10, 12.5, 15, 17.5, 20]
    gammas = [0.1, 0.5, 1, 2, 5, 10]

    data_file = 'D:/Master_Thesis/code_data/footfall_analysis/data/other/data_m1_f1_R1.pkl'

    m1_data, f1_data, R1_weight_data = pickle.load(open(data_file, 'rb'))

    m1_scatter = m1_data.reshape(len(spans) * len(l2ds) * len(gammas))
    f1_scatter = f1_data.reshape(len(spans) * len(l2ds) * len(gammas))
    R1_weight_scatter = R1_weight_data.reshape(len(spans) * len(l2ds) * len(gammas))

    mdl = Structure.load_from_obj('D:/Master_Thesis/modal/modal_symmetric/mdl_span5_l2d10_gamma1_mesh_symmetric.obj',
                                  output=0)
    mdl.name = 'mdl_span5_l2d10_gamma1_mesh_symmetric_opt'

    v0 = 5 ** 3 / 10 * 0.4 / 4  # span**3/l2d*ratio/4, initial volume
    v = 0

    for i in range(n_v):
        t_v = ts[i]
        a_v = mdl.areas['elset_vault_{0}'.format(i + 1)]
        v += t_v * a_v

    for i in range(n_r):
        t_r = ts[n_v + i]
        a_r = mdl.areas['elset_ribs_{0}'.format(i + 1)]
        v += t_r * a_r

    for i in range(n_r, n_r + n_r_h):
        t_r_h = ts[n_v + i] / 2
        a_r_h = mdl.areas['elset_ribs_{0}'.format(i + 1)]
        v += t_r_h * a_r_h

    scale = v0 / v

    ts_scaled = scale * ts

    for i in range(n_v):
        t_v = ts_scaled[i]
        mdl.sections['sec_vault_{0}'.format(i + 1)].geometry['t'] = t_v

    for i in range(n_r):
        t_r = ts_scaled[n_v+i]
        mdl.sections['sec_ribs_{0}'.format(i + 1)].geometry['t'] = t_r

    for i in range(n_r, n_r + n_r_h):
        t_r_h = ts_scaled[n_v + i] / 2
        mdl.sections['sec_ribs_{0}'.format(i + 1)].geometry['t'] = t_r_h
    # !!! only for test
    print('!!! t_v=' + str(t_v))
    print('!!! t_r=' + str(t_r))

    file_abaqus = 'D:/Master_Thesis/modal/modal_symmetric/' + mdl.name

    if os.path.exists(file_abaqus):
        os.chdir(file_abaqus)
        for file in glob.glob("*.lck"):
            os.remove(file)

    print('abaqus starts modal analysis')
    mdl.analyse_and_extract(software='abaqus', fields=['u'], components=['uz'],output=0)

    m1 = mdl.results['step_modal']['masses'][0]
    f1 = mdl.results['step_modal']['frequencies'][0]
    print('m1=' + str(4 * m1))
    print('f1=' + str(f1))

    # interpolate R1 based on m1 and f1 from existing data
    m1_f1_intp_grid_m1, m1_f1_intp_grid_f1 = np.meshgrid(4 * m1, f1)

    R1_weight = griddata(np.vstack((m1_scatter, f1_scatter)).T, R1_weight_scatter,
                         (m1_f1_intp_grid_m1, m1_f1_intp_grid_f1), method='linear')

    # if data not available, calculate response by solving ODE
    if math.isnan(float(R1_weight)):
        print('response out of interpolation range, solving ODE')
        start_ODE = timeit.default_timer()

        ### parameters often changed
        n_modes = 1  # number of modes used shall not exceed that extracted from abaqus modal analysis
        dt = 0.0005  # s
        t_cut = 0

        ### Loading
        W = -76 * 9.81 / 4  # load from walking people
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

            stop_ODE = timeit.default_timer()

            print('ODE solving of mode '+str(i + 1)+' finished, time = '+str(stop_ODE-start_ODE)+' s')

        R1_weight = R_weight[0]

    print('R1=' + str(R1_weight))

    return R1_weight, ts_scaled

def get_scaledThickness(areas,ts):
    # ts is the thickness of all variables for one experimental design

    v0 = 5 ** 3 / 10 * 0.4 / 4  # span**3/l2d*ratio/4, initial volume
    v = 0

    for i in range(n_v):
        t_v = ts[i]
        a_v = areas[i]
        v += t_v * a_v

    for i in range(n_r):
        t_r = ts[n_v + i]
        a_r = areas[n_v+i]
        v += t_r * a_r

    for i in range(n_r, n_r + n_r_h):
        t_r_h = ts[n_v + i] / 2
        a_r_h = areas[n_v+i]
        v += t_r_h * a_r_h

    scale = v0 / v
    t_scaled = ts * scale

    return t_scaled

def get_areas():
    mdl = Structure.load_from_obj('D:/Master_Thesis/modal/modal_symmetric/mdl_span5_l2d10_gamma1_mesh_symmetric.obj',
                                  output=0)
    global n_v,n_r,n_r_h

    areas = []
    for i in range(n_v):
        areas.append(mdl.areas['elset_vault_{0}'.format(i + 1)])

    for i in range(n_r+n_r_h):
        areas.append(mdl.areas['elset_ribs_{0}'.format(i + 1)])

    return areas

def sampling(strategy,bounds,M,n,base):
    """generate samples

    Parameters
    ----------
    strategy : str
        dict{'lhs','uniform','log_uniform'}
    bounds : float array
    M : int
        dimension of model
    n : int
        number of samples

    Returns
    -------
    samples
        an array of dimension M*n

    """
    if strategy == 'lhs':
        samples = np.transpose(lhs(M,samples=n))

    elif strategy == 'uniform':
        samples = np.transpose(lhs(M, samples=n))
        samples = bounds[0]+(bounds[1]-bounds[0])*samples

    elif strategy == 'log_uniform':
        samples = np.transpose(lhs(M, samples=n))
        samples = bounds[0] + (bounds[1] - bounds[0]) * samples
        samples = base*10**samples

    return samples




