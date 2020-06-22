import math
import numpy as np

default_precision = 3

def roundMatrix(A, precision: int = default_precision):
    temp = np.zeros(shape=A.shape)  # create temporary matrix that we can change and return without changing A
    for i, zeile in enumerate(A):
        for j, a_ij in enumerate(zeile):
            temp[i,j] = roundNumber(a_ij, precision)
    return temp

def roundNumber(number, precision: int = default_precision):
    if number == 0:
        return 0
    pow = -math.floor(math.log(abs(number), 10)) - 1 # get the number right after the decimal point so round rounds the way i want it to
    return round(number * 10**pow, precision) * 10**(-pow)

def multiplyRounded(A, B, precision: int = default_precision):
    if A.shape[1] == B.shape[0]:
        temp = np.zeros(shape = (A.shape[0], B.shape[1]))
        for i in range(A.shape[0]):
            for j in range(B.shape[1]):
                sum = 0
                for k in range(A.shape[1]):
                    sum += A[i,k] * B[k,j]
                    sum = roundNumber(sum, precision)
                temp[i,j] = sum
        return temp
    else:
        print('ERROR THIS IS NOT A VALID MULTIPLICATION!!!')
        return 0

def r(A):
    if isinstance(A, np.ndarray):
        return roundMatrix(A)
    else:
        return roundNumber(A)

def mult(A, B):
    return multiplyRounded(A, B)