import numpy as np
import matplotlib.pyplot as plt


def lerp(A, B, t):
    return t * A + (1 - t) * B


def bezier_t(P1, P2, P3, P4, t):
    PP1 = lerp(P1, P2, t)
    PP2 = lerp(P2, P3, t)
    PP3 = lerp(P3, P4, t)
    PPP1 = lerp(PP1, PP2, t)
    PPP2 = lerp(PP2, PP3, t)
    return lerp(PPP1, PPP2, t)


def bezier(P1, P2, P3, P4, steps=101):
    points = np.empty((steps, len(P1)))
    # include endpoint 1
    ts = np.arange(0, 1 + 1 / (steps - 1), 1 / (steps - 1))
    for i, t in enumerate(ts):
        points[i] = bezier_t(P1, P2, P3, P4, t)
    return points


def main():
    P1 = np.array([0, 0])
    P2 = np.array([1, 0])
    P3 = np.array([1, 1])
    P4 = np.array([0, 1])
    points = bezier(P1, P2, P3, P4)
    plt.scatter(points[:, 0], points[:, 1])
    plt.show()


if __name__ == "__main__":
    main()
