import numpy as np
A = np.array([
    [1.05, 1.02],
    [1.04, 1.02]
])

G = np.array([
    [1, 0],
    [-104/105, 1]
])

b = np.array([
    [1],
    [2]
])

z = np.array([
    [1],
    [1.01]
])

x = np.array([
    [-0.99],
    [2]
])

Delta = b - A.dot(x)

print(G.dot(np.c_[A,Delta]))