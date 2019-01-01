from math import factorial as fact
from pyDOE2 import *
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
from sympy.utilities.iterables import multiset_permutations

def legendre_poly(x,n):

    """Construct Legendre polynomials

    Parameters
    ----------
    x : float
        array of input points to be evaluated.
    n : int
        the max. degree of polynomials.

    Returns
    -------
    P
        an array of polynomials up to n degrees.

    """
    for i in range(n+1):
        if i == 0:
            P = np.ones(np.array(x).shape)
        elif i == 1:
            P = np.append([P],[np.array(x)],axis=0)
        elif i >= 2:
            P = np.append(P,[1/i*((2*i-1)*x*P[-1,:]-(i-1)*P[-2,:])],axis=0)

    for i in range(n+1):
        P[i,:] = np.sqrt(2*i+1)*P[i,:]

    return P

def degree_index(M,P):
    """Construct degree index tuples indicating the degrees of univariate polynomials for multivariate polynomials basis by multiplication

    Parameters
    ----------
    M : int
        dimension of model (number of input variables).
    P : int
        the total degree of metal model.

    Returns
    -------
    index
        a list of arrays indicating the degrees of univariate polynomials

    """
    # firstly generates all the tuples of integers a(i) that satisfy the following conditions:
    #     - a(i) <= P
    #     - sum(a) = P
    #     - a(i+1) <= a(i)
    # The additional constraints must be met: P >= J
    # then calculate the permutation without repetition

    if P == 0:
        index = [0]*M
    else:
        index = []
        for pp in range(1,P+1):
            index.append([])
            # jj is the number of interaction terms
            for jj in range(1,np.min([pp,M])+1):
                index[pp-1].append([])
                # numbers are possible values whose sum can be pp
                numbers = [i+1 for i in range(pp)]
                # in each jj, combinations of numbers with sum=qq are selected
                combs_jj = [seq for seq in it.combinations_with_replacement(numbers,jj) if sum(seq)==pp]
                for k in range(len(combs_jj)):
                    index[pp-1][jj-1].append([])
                    index[pp-1][jj-1][k] = list(combs_jj[k])
                    index[pp - 1][jj - 1][k].sort(reverse=True)
                    # supplement remaining positions with zeros
                    for n_zeros in range(M-len(index[pp-1][jj-1][k])):
                        index[pp - 1][jj - 1][k].append(0)
                # calculate unique permutations without repetitions (can be very computationally intensive and memory demanding)
                for k in range(len(index[pp - 1][jj - 1])):
                    permut = multiset_permutations(index[pp - 1][jj - 1][k])
                    index[pp - 1][jj - 1][k] = permut
    # open the nested list
    for i in range(3):
        index = list(it.chain.from_iterable(index))

    return index


M = 41
P = 2
index = degree_index(M,P)
print(index)
print(len(index)+1)
                # if jj == 0:
                #     index[pp][jj].append([0]*M)
                #     # i += 1
                #     index[pp][jj][0][0] = pp
                # else:
                #     index[pp][jj].append(index[pp][jj-1][0])
                #     index[pp][jj][0][0] -= 1
                #     index[i][0] -= 1
                #     index[1] =



# x = np.linspace(-1,1,100)
# P = LegendrePoly(x,5)
#
# fig  = plt.figure()
# axes = fig.add_subplot(111)
# # axes.set_title('Footfall loading', fontsize=12)
# axes.set_xlabel('x', fontsize=12)
# axes.set_ylabel('P', fontsize=12)
# # axes.minorticks_on()
# for i in range(P.shape[0]):
#     axes.plot(x, P[i,:],label='n='+str(i))
# axes.legend()
# plt.show()


### define input parameters
# number of input variables (model dimension), tv and tr for test
# M = 2
# # max total degree
# p = 2
# # oversampling rate
# k = 2
# # bounds for uniform distribution
# bounds = [0.01,0.2]
# # number of vault and rib panels
# n_v = 16
# n_r = 21
# n_r_h = 4
#
# # cardinality
# P = fact(M+p)/(fact(M)*fact(p))
# # total runs of experimental design
# n = k*P
# # Latin Hypercube sampling (standard uniform distribution [0,1])
# U = lhs(M,samples=int(n))
# # transform standard uniform distribution to actual distribution: X = a+(b-a)U
# t_vs = bounds[0]+(bounds[1]-bounds[0])*U[:,0]
# t_rs = bounds[0]+(bounds[1]-bounds[0])*U[:,1]

#

