import math
import numpy as np
import runden
def r(A):
    precision = 3
    if isinstance(A, np.ndarray):
        return runden.roundMatrix(A,precision)
    else:
        return runden.roundNumber(A,precision)
A = np.array([
    [1.07, 1.10],
    [1.07, 1.11],
    [1.07, 1.15]
])
b = np.array([[1,-1,0]]).transpose()
v = A.transpose()[0].reshape(3,1)
v = r(v - r(np.linalg.norm(v)) * np.array([[1,0,0]]).transpose())
H_v = r(np.identity(len(v)) - r(r(2/r(np.dot(v.transpose(), v))) * r(np.dot(v, v.transpose().reshape(1,3)))))
print(r(H_v.dot(A)))