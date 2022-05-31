import numpy as np


def newton(F, x_0, error=0.1):
    # TODO: implement newton
    pass


def implicit_euler(f, y_0, t):
    # TODO: implement implicit euler
    y = np.empty((len(t), len(y_0)))
    for i, t_i in enumerate(t):
        if i == 0:
            y[0] = y_0
            continue
        # this is explicit euler
        y[i] = y[i - 1] + (t_i - t[i - 1]) * f(t[i - 1], y[i - 1])
    return y
