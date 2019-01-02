import numpy as np

Y_ED=np.array([ 3.65967007, 14.04638478,  9.03802237])

Psi = np.array([[1.  ,       0.02829841 ,0.14058009],
 [1.   ,      0.07752549 ,0.04658574],
 [1.   ,      0.06196282 ,0.07630115]])



y_alpha = np.linalg.inv(np.transpose(Psi)@Psi)@np.transpose(Psi)@Y_ED
print('y_alpha=')
print(y_alpha)

# recalculate response based on obtained coefficients
Y_rec = y_alpha @ np.transpose(Psi)
print('Y_ED=')
print(Y_ED)
print('Y_rec=')
print(Y_rec)
print()

# y_alpha=
# [  1596204.217648   -15660647.45277882  -8201935.44120646]
# Y_ED=
# [3.79273849 8.59756241 8.84138227]
# Y_rec=
# [4.11669786 8.90649856 9.14195976]