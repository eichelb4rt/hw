import numpy as np
from runden import r, multiplyRounded as mult

def main():
    np.set_printoptions(precision = 53)
    A = np.array([
        [1, 4, 3],
        [4, 5, 6],
        [3, 6, 7]
    ], dtype = np.double)
    i = 0
    j = 1
    w = np.sqrt((A[i,i] - A[j,j])**2 + 4 * A[i,j]**2)
    tau = (A[i,i] - A[j,j]) / w
    for i in range(10):
        A = jacobi_step(A)
        print(A)
        print("")

def jacobi_step(A):
    # get dimensions of A
    if A.shape[0] != A.shape[1]:
        raise ArithmeticError("A is not symmetrical!")
    n = A.shape[0]  # with A as a n x n matrix
    # find the maximum non-diagonal value of A
    max_value = A[0,1]
    max_i = 0
    max_j = 1
    for i in range(n):
        for j in range(i+1,n):  # only look at the upper right triangle part since A is symmetrical and we don't want to look at the diagonal
            if max_value < A[i,j]:
                max_value = A[i,j]
                max_i = i
                max_j = j
    # makes it much more readable
    i = max_i
    j = max_j
    # now actually calculate
    w = np.sqrt((A[i,i] - A[j,j])**2 + 4 * A[i,j]**2)
    tau = (A[i,i] - A[j,j]) / w
    sigma = np.sign(A[i,j])
    c = np.sqrt((1 + tau) / 2)
    s = sigma * np.sqrt((1 - tau) / 2)
    # make G
    G = np.identity(n, dtype = np.double)
    G[i,i] = c
    G[j,j] = c
    G[i,j] = s
    G[j,i] = -s
    # calculate next A
    return (G.dot(A)).dot(G.transpose())

if __name__ == "__main__":
    main()