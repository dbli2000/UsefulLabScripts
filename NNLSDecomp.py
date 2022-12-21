#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 11:53:53 2020

@author: dli
"""
#Import 
required packages
import numpy as np
import sympy
import math
import scipy
from scipy.optimize import nnls
import seaborn as sns

#Have final outcome vector
groundtruth = np.array([0, 7740461, 216476, 106118, 169158,
                        459763, 1269380, 416119, 171831,
                        84327,  46872,   28439,  18298,
                        12435,  9216,    8712,   24498]).T

#Set up matrix
d = 16
rows = 17
columns = 100
matrix = np.zeros((rows, columns))

#Define modified binomial coefficient
def binom(n, k):
    return math.factorial(n) // math.factorial(k) // math.factorial(n-k)

#Fill in PMF matrix
for i in range(rows):
    for k in range(columns):
        n = k
        j= i
        try:
            Stirling = sympy.functions.combinatorial.numbers.stirling(n, j)
            PMF = math.factorial(j)*Stirling*binom(d, j)/(d**(n))
            matrix[i, k] = PMF
        except:
            matrix[i, k] = 0

#Show heatmap of matrix
#sns.heatmap(matrix)

#Run NNLS on true results
outcome = nnls(matrix, groundtruth)
resultsvector = outcome[0]

#Plot results
test = np.matmul(matrix, np.array(resultsvector).T)
sns.heatmap([test,groundtruth])
print(resultsvector)