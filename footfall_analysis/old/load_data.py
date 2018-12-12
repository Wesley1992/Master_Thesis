import pickle
import matplotlib.pyplot as plt
import numpy as np

# input for loading file
spans = [5]
l2ds = [20]
gammas = [2]

# spans = [5,6,7,8,9,10]
# l2ds = [10,12.5,15,17.5,20]
# gammas = [0.1,0.5,1,2,5,10]

for span in spans:
    for l2d in l2ds:
        for gamma in gammas:
            # file_name='D:/Master_Thesis/footfall_analysis/data/data_mdl_t0_01span/data_mdl_span'+str(span).replace('.','_')+'_l2d'+str(l2d).replace('.','_')+'_gamma'+str(gamma).replace('.','_')+'.pkl'
            file_name='D:/Master_Thesis/code_data/footfall_analysis/data/data_mdl_0_4m_wrong/data_mdl_span'+str(span).replace('.','_')+'_l2d'+str(l2d).replace('.','_')+'_gamma'+str(gamma).replace('.','_')+'.pkl'

            [W, te, t, F, f_n, m_n, node_lp, n_modes, dt, dis_modal, acc_modal, rms_modal,rms_modal_weight, dis_modal_1, acc_modal_1, rms_modal_1, rms_modal_weight_1,rms_acc_modal, rms_modes, rms_modes_weight, rms_acc_modes, R, R_weight, R_acc, Gamma_n]=pickle.load(open(file_name,'rb'))

            # to change:
            f_load = 2
            T_rms = int(1 / f_load // dt) + 2
            n = len(t)
            rms_acc_modal_1 = np.zeros(n)
            acc_modal_1=np.squeeze(acc_modal_1)
            for j in range(T_rms, n):
                rms_acc_modal_1[j] = np.sqrt(np.mean(acc_modal_1[j - T_rms:j] ** 2))
            # print(rms_acc_modal_1)
            print(R_weight[-1])
            print(R_weight[0] / R_weight[-1])
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

            fig = plt.figure()
            fig.suptitle('Footfall Response with Modal Analysis (span=' + str(span) + 'm, l/d=' + str(l2d) + ', gamma='+str(gamma)+', dt=' + str(dt) + ', ' + str(n_modes) + ' modes)')
            axes1 = fig.add_subplot(311)
            axes1.set_xlabel('Time $t$ [s]', fontsize=12)
            axes1.set_ylabel('Displacement $u$ [mm]', fontsize=12)
            axes1.minorticks_on()
            axes1.plot(t, np.transpose(dis_modal), color='r')
            axes1.plot(t, np.transpose(dis_modal_1), '--', color='r')
            axes1.legend([str(n_modes) + ' modes', 'first mode'])

            axes2 = fig.add_subplot(312)
            axes2.set_xlabel('Time $t$ [s]', fontsize=12)
            axes2.set_ylabel('Acceleration $a$ [$m/s^2$]', fontsize=12)
            axes2.minorticks_on()
            axes2.plot(t, np.transpose(acc_modal), 'r')
            axes2.plot(t, np.transpose(acc_modal_1), '--', color='r')
            axes2.plot([0, t[-1]], [0, 0], '--', color=[0.7, 0.7, 0.7])
            axes2.legend([str(n_modes) + ' modes', 'first mode'])

            axes3 = fig.add_subplot(313)
            axes3.set_xlabel('Time $t$ [s]', fontsize=12)
            axes3.set_ylabel('Acceleration (RMS) $a_\mathrm{rms}$ [$m/s^2$]', fontsize=12)
            axes3.minorticks_on()
            axes3.plot(t, rms_modal, color='r')
            axes3.plot(t, rms_modal_1, '--', color='r')
            axes3.plot(t, rms_modal_weight, color='b')
            axes3.plot(t, rms_modal_weight_1, '--', color='b')
            axes3.plot(t, rms_acc_modal, color='g')
            axes3.plot(t, rms_acc_modal_1, '--', color='g')
            axes3.legend(
                ['50 modes not weighted', 'first mode not weighted', '50 modes weighted', 'first mode weighted','acc','acc 1 mode'])
            axes3.plot([0, t[-1]], [0, 0], '--', color=[0.7, 0.7, 0.7])

            # plot response factor in relation to number of modes involved
            fig = plt.figure()
            axes = fig.add_subplot(111)
            axes.set_title(
                'Rsponse factor in relation to number of modes involved (span=' + str(span) + 'm, l/d=' + str(l2d) + ', gamma='+str(gamma)+', dt=' + str(
                     dt) + ', ' + str(n_modes) + ' modes)', fontsize=12)
            axes.set_xlabel('Number of modes involved in modal analysis', fontsize=12)
            axes.set_ylabel('Response Factor', fontsize=12)
            axes.minorticks_on()
            axes.plot(range(1, n_modes + 1), R, 'r')
            axes.plot(range(1, n_modes + 1), R_weight, 'b')
            axes.set_ylim(ymin=0)
            axes.set_xlim(xmin=0)
            axes.legend(['not weighted response factor', 'weighted response factor'], loc=2)
            # plot participation factor
            axes2 = axes.twinx()
            axes2.set_ylabel('Participation factor [-]', fontsize=12)
            axes2.tick_params('y')
            axes2.minorticks_on()
            axes2.bar(range(1, n_modes + 1), Gamma_n, width=0.2, color='m')
            axes2.legend(['participation factor'], loc=1)
            axes2.plot([1, n_modes + 1], [0, 0], '--', color=[0.7, 0.7, 0.7])

plt.show()

