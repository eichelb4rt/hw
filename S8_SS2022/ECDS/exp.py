import math
import numpy as np
import matplotlib.pyplot as plt
from explicit_euler import explicit_euler


def main():
    y_0 = 1
    t = np.arange(0, 5, 0.1)
    f = lambda t_i, y_i: y_i
    approximation = explicit_euler(f, y_0, t)
    actual_function = lambda x: math.exp(x)
    plt.plot(t, approximation, label="explicit")
    plt.plot(t, np.vectorize(actual_function)(t), label="exp")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
