import numpy as np
import matplotlib.pyplot as plt
from explicit_euler import explicit_euler


def main():
    EPS = [1.1, 0.1]
    GAMMA = [0.4, 0.4]

    y_0 = np.array([10, 10])
    t = np.arange(0, 100, 0.01)
    f = lambda t_i, y_i: np.array([
        y_i[0] * (EPS[0] - GAMMA[0] * y_i[1]),
        - y_i[1] * (EPS[1] - GAMMA[1] * y_i[0])
    ])
    approximation = explicit_euler(f, y_0, t)
    plt.plot(t, approximation[:, 0], label="prey")
    plt.plot(t, approximation[:, 1], label="hunter")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
