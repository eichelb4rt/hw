import math
import numpy as np
def roundMatrix(A, precision: int):
    temp = np.zeros(shape=A.shape)  # create temporary matrix that we can change and return without changing A
    for i, zeile in enumerate(A):
        for j, a_ij in enumerate(zeile):
            temp[i,j] = roundNumber(a_ij, 3)
    return temp

def roundNumber(number, precision: int):
    if number == 0:
        return 0
    pow = -math.floor(math.log(abs(number), 10)) - 1 # get the number right after the decimal point so round rounds the way i want it to
    return round(number * 10**pow, precision) * 10**(-pow)
