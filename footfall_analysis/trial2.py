import numpy as np
import itertools as it

a = [np.zeros(3)]
b = np.ones(3)
print(type(a))
print(b)
a.append(b)
print(a)

c = []
c.append([0]*2)
c[0][0]=1
print(c)

index = [[[]]]
print(index)
index[0][0].append([0]*3)
print(index)

index = []
index.append([1])
print(index)


import itertools
numbers = [1, 2, 3, 7, 7, 9, 10]
result = [seq for i in range(len(numbers), 0, -1)
          for seq in itertools.combinations(numbers, i) if sum(seq) == 10]
print(result)

pp = 5
jj = 2
numbers = [i+1 for i in range(pp)]
index = [seq for seq in it.combinations_with_replacement(numbers,jj) if sum(seq)==pp]
print(index)
print(list(index[0]))
print(type(list(index[0])))

a=[2,1,3]
a.sort(reverse=True)
print(a)