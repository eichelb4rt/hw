import numpy as np
import sys

def iter(W: np.array, x_0: np.array, k: int):
    x_k = x_0
    for i in range(0,k):
        x_k = W.dot(x_k)
    return x_k

W = np.array([
    [0.4, 0.3, 0.2], 
    [0.4, 0.2, 0.6], 
    [0.2, 0.5, 0.2]
])
x_0 = np.array([
    [0],
    [1],
    [0]
])
print(iter(W, x_0, int(sys.argv[1])))