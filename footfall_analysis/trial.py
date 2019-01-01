import numpy as np
import itertools as it

index = [1,1,0]
print(list(it.permutations(index)))

iter = it.permutations(index)
print(type(iter))

for item in iter:
    print(item)
# print(iter)
# iter = list(set(iter))
# print(iter)

a=[1]
a.insert(0,0)
print(a)


a = np.append(np.array([1]),2)
print(a)
print(a[0])

poly = np.ones(3)
a=3
poly[0] *= a
a=4
poly[0] *= a
print(poly)

a=np.array([1,2])
a[0] = 1.2
print(type(a[0]))
print(a)

print(np.ones(3))

b=np.array([1,2,3])
c=b.copy()
print(c)