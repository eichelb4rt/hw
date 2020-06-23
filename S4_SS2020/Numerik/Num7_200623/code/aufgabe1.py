import math
import numpy as np
from runden import r, multiplyRounded as mult

A = np.array([
    [1.07, 1.10],
    [1.07, 1.11],
    [1.07, 1.15]
])
b = np.array([[1,-1,0]]).transpose()
v = A.transpose()[0].reshape(3,1)
v = r(v + r(np.linalg.norm(v)) * np.array([[1,0,0]]).transpose())
H_v = r(np.identity(len(v)) - r(r(2/r(np.dot(v.transpose(), v))) * mult(v, v.transpose())))
w = np.array([[-0.00165, -0.0383]]).transpose().reshape(2,1)
w = r(w - r(np.linalg.norm(w)) * np.array([[1,0]]).transpose())
H_w = r(np.identity(len(w)) - r(r(2/r(np.dot(w.transpose(), w))) * mult(w, w.transpose())))
A_2 = np.array([[-0.00165, -0.0383]]).transpose().reshape(2,1)
H_w_extra = np.array([
    [1.0,0.0,0.0],
    [0.0, 0.04, -1.0],
    [0.0, -1.0, 0.04]
]).reshape(3,3)
Q = mult(H_v, H_w_extra)
print(mult(Q.transpose(), b))