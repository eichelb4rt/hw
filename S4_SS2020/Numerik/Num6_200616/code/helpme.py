import numpy as np
A = np.array([
    [-2, 4],
    [-1, 1],
    [1, 1],
    [2, 4]
])
AT = A.transpose()
b = np.array([
    [9],
    [3],
    [0],
    [0]
])
print(AT.dot(b))