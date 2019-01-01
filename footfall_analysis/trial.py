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



