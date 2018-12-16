from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pickle


### input for loading file
spans = [5,6,7,8,9,10]
l2ds = [10,12.5,15,17.5,20]
gammas = [0.1,0.5,1,2,5,10]


### load file
data_file = 'D:/Master_Thesis/code_data/footfall_analysis/data/other/data_m1_f1_R1.pkl'


m1,f1,R1_weight = pickle.load(open(data_file,'rb'))

### m1,f1-R1 plot
fig = plt.figure()
ax = fig.gca(projection='3d')

for i in range(len(spans)):
    surf = ax.plot_surface(m1[i], f1[i], R1_weight[i])

ax.set_title('$m_1,f_1-R_1 plot$')
ax.set_xlabel('$m_1$ [kg]')
ax.set_ylabel('$f_1$ [Hz]')
ax.set_zlabel('$R_1$ [-]')

ax.set_zlim(zmin=0)


### m1,f1-R1 contour plot
fig, ax = plt.subplots()
ax.set_title('$m_1,f_1-R_1$ contour')
ax.set_xlabel('$m_1$ [kg]')
ax.set_ylabel('$f_1$ [Hz]')

colors = ['b', 'g', 'r', 'c', 'm', 'k']

for i in range(len(spans)):
    contour = ax.contour(m1[i], f1[i], R1_weight[i],[1,2,4,6,8,12,16,20,28,36,48,60,72],colors=colors[i])
    ax.clabel(contour,fontsize=10,fmt='%1.1f')
    contour.collections[i].set_label('span='+str(spans[i])+'m')
ax.legend()

plt.show()