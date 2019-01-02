import numpy as np
from sklearn.linear_model import LinearRegression
reg = LinearRegression()

Y_ED=np.array([ 3.65967007, 14.04638478,  9.03802237])

Psi = np.array([[1.  ,       0.02829841 ,0.14058009],
 [1.   ,      0.08752549 ,0.04658574],
 [1.   ,      0.06196283 ,0.07630115]])

reg.fit(Psi, Y_ED)

Y_rec = reg.predict(Psi)


print('y_alpha=')
print(reg.coef_)
print('Y_ED=')
print(Y_ED)
print('Y_rec=')
print(Y_rec)

