import math
import numpy as np
import matplotlib.pyplot as plt
from explicit_euler import explicit_euler


def main():
    ALPHA = 0.5
    BETA = 0.01
    GAMMA = 0.04
    N = 1000

    y_0 = np.array([
        997,
        3,
        0,
        0
    ])
    t = np.arange(0, 100, 0.01)
    # S: 0, I: 1, R: 2, H: 3
    f = lambda t_i, y_i: np.array([
        - ALPHA * y_i[0] * y_i[1] / N,
        ALPHA * y_i[0] * y_i[1] / N - GAMMA * y_i[1] - BETA * y_i[1],
        BETA * y_i[1],
        GAMMA * y_i[1]
    ])
    approximation = explicit_euler(f, y_0, t)
    plt.plot(t, approximation[:, 0], label="S")
    plt.plot(t, approximation[:, 1], label="I")
    plt.plot(t, approximation[:, 2], label="R")
    plt.plot(t, approximation[:, 3], label="H")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
