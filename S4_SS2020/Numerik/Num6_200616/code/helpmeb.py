import numpy as np
from scipy.linalg import lu
A = np.array([
    [1, 1, 1],
    [1, 2, 4],
    [1, 3, 9],
    [1, 4, 16],
    [1, 5, 25],
    [1, 6, 36],
    [1, 7, 49]
])
AT = A.transpose()
b = np.array([
    [2.31, 2.01, 1.80, 1.66, 1.55, 1.47, 1.41]
]).transpose()

pl, u = lu(np.c_[AT.dot(A),AT.dot(b)], permute_l=True)
print(u)