import pickle
import matplotlib.pyplot as plt
import numpy as np

# input for loading file

spans = [10]
l2ds = [10,12.5,15,17.5,20]
# gammas = [0.1,1,10]

for span in spans:
    for l2d in l2ds:
        # for gamma in gammas:
            # file_name='D:/Master_Thesis/footfall_analysis/data/data_mdl_t0_01span/data_mdl_span'+str(span).replace('.','_')+'_l2d'+str(l2d).replace('.','_')+'_gamma'+str(gamma).replace('.','_')+'.pkl'
        file_name = 'D:/Master_Thesis/code_data/footfall_analysis_plate_0_4depth/data/data_mdl_plate_span' + str(span).replace('.','_') + '_l2d' + str(l2d).replace('.', '_') + '.pkl'

        f_n, m_n, node_lp, n_modes, dt, t, dis_modes_lp, vel_modes_lp, acc_modes_lp, acc_modes_lp_weight, rms_modes, rms_modes_weight, R, R_weight, Gamma_n = pickle.load(
            open(file_name, 'rb'))

        print('f1: ',f_n[0])
        print('m1: ',m_n[0])
        print('R: ',R_weight[-1])
        print('\n')
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
        axes1.plot(t, np.transpose(dis_modes_lp[-1,:]),color=[1, 0, 0])

        axes2 = fig.add_subplot(312)
        axes2.set_xlabel('Time $t$ [s]', fontsize=12)
        axes2.set_ylabel('Acceleration $a$ [$m/s^2$]', fontsize=12)
        axes2.minorticks_on()
        axes2.plot(t, np.transpose(acc_modes_lp[-1,:]),color=[1, 0, 0])
        axes2.plot([0, t[-1]], [0, 0], '--', color=[0.7, 0.7, 0.7])

        axes3 = fig.add_subplot(313)
        axes3.set_xlabel('Time $t$ [s]', fontsize=12)
        axes3.set_ylabel('Acceleration (RMS) $a_\mathrm{rms}$ [$m/s^2$]', fontsize=12)
        axes3.minorticks_on()
        axes3.plot(t, rms_modes[-1,:],color=[1,0,0])
        axes3.plot(t, rms_modes_weight[-1,:],color=[0,0,1])
        axes3.legend(['not weighted', 'weighted'])
        axes3.plot([0, t[-1]], [0, 0], '--', color=[0.7, 0.7, 0.7])


# plot response factor in relation to number of modes involved
        fig  = plt.figure()
        axes = fig.add_subplot(111)
        axes.set_title('Rsponse factor in relation to number of modes involved (span='+str(span)+'m, l/d='+str(l2d)+', dt='+str(dt)+')', fontsize=12)
        axes.set_xlabel('Number of modes involved in modal analysis', fontsize=12)
        axes.set_ylabel('Response Factor', fontsize=12)
        axes.set_xticks(range(1,n_modes+1))
        axes.minorticks_on()
        axes.tick_params(axis='x', which='minor', bottom=False)
        axes.plot(range(1,n_modes+1), R, color=[1, 0, 0])
        axes.plot(range(1,n_modes+1), R_weight, color=[0, 0, 1])
        axes.set_ylim(ymin=0)
        axes.set_xlim(xmin=0)
        axes.legend(['not weighted', 'weighted'])
        axes.plot([1, n_modes+1], [0, 0], '--', color=[0.7, 0.7, 0.7])

        # plot participation factor
        axes2 = axes.twinx()
        axes2.set_ylabel('Participation factor [-]', fontsize=12)
        axes2.minorticks_on()
        axes2.tick_params(axis='x', which='minor', bottom=False)
        axes2.bar(range(1, n_modes + 1), Gamma_n, width=0.2, color='m')
        axes2.legend(['participation factor'], loc=1)
        axes2.plot([1, n_modes + 1], [0, 0], '--', color=[0.7, 0.7, 0.7])

plt.show()

