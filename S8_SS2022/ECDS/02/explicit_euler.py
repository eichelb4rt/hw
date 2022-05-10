import numpy as np


def explicit_euler(f, y_0, t):
    # make an array of the approximations (depending on whether it's a vector or a scalar)
    if hasattr(y_0, "__len__"):
        y = np.empty((len(t), len(y_0)))
    else:
        y = np.empty((len(t)))

    for i, t_i in enumerate(t):
        if i == 0:
            y[0] = y_0
            continue
        y[i] = y[i - 1] + (t_i - t[i - 1]) * f(t[i - 1], y[i - 1])
    return y
