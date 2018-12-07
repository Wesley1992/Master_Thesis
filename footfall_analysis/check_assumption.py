import pickle
import numpy as np
import matplotlib.pyplot as plt
from compas_fea.structure import Structure

spans = [5,8]
l2ds = [10,15,20]
gammas = [0.1,1,10]

# spans = [5,8]
# l2ds = [10,15,20]
# gammas = [0.1,1,10]

# file_name = 'D:/Master_Thesis/footfall_analysis/data/data_mdl_t0_01span/data_volume_area_thickness.pkl'
file_name = 'data_volume_area_thickness.pkl'
[v,a,t] =pickle.load(open(file_name,'rb'))

v = np.reshape(np.array(v),(len(spans),len(l2ds),len(gammas)))
a_v = np.reshape(np.array(a)[:,0],(len(spans),len(l2ds),len(gammas)))
a_r = np.reshape(np.array(a)[:,1],(len(spans),len(l2ds),len(gammas)))
t_v = np.reshape(np.array(t)[:,0],(len(spans),len(l2ds),len(gammas)))
t_r = np.reshape(np.array(t)[:,1],(len(spans),len(l2ds),len(gammas)))

### thickness
fig  = plt.figure()
axes = fig.add_subplot(111)
axes.set_title('Thickness ratio $t_{v,\gamma=1}$/(span/100) in relation to span', fontsize=12)
axes.set_xlabel('Span [m]', fontsize=12)
axes.set_ylabel('Ratio [-]', fontsize=12)
# axes.set_ylim(1,4)
axes.minorticks_on()
axes.plot(spans, t_v[:,0,1]/(np.array(spans)/100))
axes.plot(spans, t_v[:,1,1]/(np.array(spans)/100))
axes.plot(spans, t_v[:,2,1]/(np.array(spans)/100))
axes.legend(['l/d=10','l/d=15','l/d=20'])


fig  = plt.figure()
axes = fig.add_subplot(111)
axes.set_title('Thickness ratio $t_{v,\gamma=10}/t_{v,\gamma=1}$ in relation to span', fontsize=12)
axes.set_xlabel('Span [m]', fontsize=12)
axes.set_ylabel('Ratio [-]', fontsize=12)
axes.set_ylim(1,4)
axes.minorticks_on()
axes.plot(spans, t_v[:,0,2]/t_v[:,0,1])
axes.plot(spans, t_v[:,1,2]/t_v[:,1,1])
axes.plot(spans, t_v[:,2,2]/t_v[:,2,1])
axes.legend(['l/d=10','l/d=15','l/d=20'])

### mass proportion to solid plate
v0 = np.zeros((len(spans),len(l2ds)))
for i in range(len(spans)):
    for j in range(len(l2ds)):
        v0[i,j] = spans[i]**3/l2ds[j]

fig  = plt.figure()
axes = fig.add_subplot(111)
axes.set_title('Mass ratio $m_{funicular}/m_{solid}$ in relation to span', fontsize=12)
axes.set_xlabel('Span [m]', fontsize=12)
axes.set_ylabel('Ratio [-]', fontsize=12)
axes.minorticks_on()
axes.plot(spans, v[:,0,1]/v0[:,0])
axes.plot(spans, v[:,1,1]/v0[:,1])
axes.plot(spans, v[:,2,1]/v0[:,2])
axes.legend(['l/d=10','l/d=15','l/d=20'])

### time for solving ODE
# spans_t = [5,8,10]
# l2ds_t = [10]
# gammas_t = [1]
#
# t_ODE = np.array([[[70]],[[1225]],[[4400]]]) # for spans=[5,8,10], l/d=[10], gamma=[1]
# n_nodes = np.zeros((len(spans_t),len(l2ds_t),len(gammas_t)))
#
# for i in range(len(spans_t)):
#     for j in range(len(l2ds_t)):
#         for k in range(len(gammas_t)):
#             mdl_name = 'mdl_span'+str(spans_t[i]).replace('.','_')+'_l2d'+str(l2ds_t[j]).replace('.','_')+'_gamma'+str(gammas_t[k]).replace('.','_')
#             file_obj = 'D:/Master_Thesis/modal_span_depth_thickness/'+mdl_name+'.obj'  # need to be changed if rus on another computer
#             mdl = Structure.load_from_obj(file_obj)
#             n_nodes[i,j,k] = mdl.nodes.__len__()
#
# fig  = plt.figure()
# axes = fig.add_subplot(111)
# axes.set_title('Number of vertices and time for solving ODE (25 modes) in relation to span', fontsize=12)
# axes.set_xlabel('Span [m]', fontsize=12)
# axes.set_ylabel('Number of vertices', fontsize=12,color='b')
# axes.tick_params('y',colors='b')
# axes.minorticks_on()
# axes.plot(spans, n_nodes[:,0,0],'b')
#
# axes2 = axes.twinx()
# axes2.set_ylabel('Time for solving ODE [s]', fontsize=12,color='r')
# axes2.tick_params('y',colors='r')
# axes2.minorticks_on()
# axes2.plot(spans, t_ODE[:,0,0],'r')

plt.show()

###

# print()