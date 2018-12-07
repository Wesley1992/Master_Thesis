import pickle
import matplotlib.pyplot as plt
import numpy as np

# input for loading file

spans = [5,8,10]
l2ds = [10,15,20]
# gammas = [0.1,1,10]

for span in spans:
    for l2d in l2ds:
        # for gamma in gammas:
            # file_name='D:/Master_Thesis/footfall_analysis/data/data_mdl_t0_01span/data_mdl_span'+str(span).replace('.','_')+'_l2d'+str(l2d).replace('.','_')+'_gamma'+str(gamma).replace('.','_')+'.pkl'
        file_name = 'D:/Master_Thesis/footfall_analysis_plate_0_4depth/data_mdl_plate_span' + str(span).replace('.','_') + '_l2d' + str(l2d).replace('.', '_') + '.pkl'

        [W, te, t, F, f_n, m_n, node_lp, n_modes, dt, dis_modal, acc_modal, rms_modal,rms_modal_weight, rms_acc_modal, rms_modes, rms_modes_weight, rms_acc_modes, R, R_weight, R_acc]=pickle.load(open(file_name,'rb'))

        print(R_weight[0]/R_weight[-1])
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
#
# plot response
        fig  = plt.figure()
        fig.suptitle('Footfall Response with Modal Analysis (span='+str(span)+'m, l/d='+str(l2d)+' dt='+str(dt)+', '+str (n_modes)+' modes)')
        axes1 = fig.add_subplot(311)
        axes1.set_xlabel('Time $t$ [s]', fontsize=12)
        axes1.set_ylabel('Displacement $u$ [mm]', fontsize=12)
        axes1.minorticks_on()
        axes1.plot(t, np.transpose(dis_modal),color=[1, 0, 0])

        axes2 = fig.add_subplot(312)
        axes2.set_xlabel('Time $t$ [s]', fontsize=12)
        axes2.set_ylabel('Acceleration $a$ [$m/s^2$]', fontsize=12)
        axes2.minorticks_on()
        axes2.plot(t, np.transpose(acc_modal),color=[1, 0, 0])
        axes2.plot([0, t[-1]], [0, 0], '--', color=[0.7, 0.7, 0.7])

        axes3 = fig.add_subplot(313)
        axes3.set_xlabel('Time $t$ [s]', fontsize=12)
        axes3.set_ylabel('Acceleration (RMS) $a_\mathrm{rms}$ [$m/s^2$]', fontsize=12)
        axes3.minorticks_on()
        axes3.plot(t, rms_modal,color=[1,0,0])
        axes3.plot(t, rms_modal_weight,color=[0,0,1])
        axes3.legend(['not weighted', 'weighted'])
        axes3.plot([0, t[-1]], [0, 0], '--', color=[0.7, 0.7, 0.7])


# plot response factor in relation to number of modes involved
        fig  = plt.figure()
        axes = fig.add_subplot(111)
        axes.set_title('Rsponse factor in relation to number of modes involved (span='+str(span)+'m, l/d='+str(l2d)+', dt='+str(dt)+')', fontsize=12)
        axes.set_xlabel('Number of modes involved in modal analysis', fontsize=12)
        axes.set_ylabel('Response Factor', fontsize=12)
        axes.minorticks_on()
        axes.plot(range(1,n_modes+1), R, color=[1, 0, 0])
        axes.plot(range(1,n_modes+1), R_weight, color=[0, 0, 1])
        axes.legend(['not weighted', 'weighted'])
        axes.plot([1, n_modes+1], [0, 0], '--', color=[0.7, 0.7, 0.7])

plt.show()

