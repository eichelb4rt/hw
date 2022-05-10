import numpy as np
import matplotlib.pyplot as plt


def lerp(a, b, t):
    return t * a + (1 - t) * b


def bezier_t(p1, p2, p3, p4, t):
    pp1 = lerp(p1, p2, t)
    pp2 = lerp(p2, p3, t)
    pp3 = lerp(p3, p4, t)
    ppp1 = lerp(pp1, pp2, t)
    ppp2 = lerp(pp2, pp3, t)
    return lerp(ppp1, ppp2, t)


def bezier(p1, p2, p3, p4, steps=101):
    points = np.empty((steps, len(p1)))
    ts = np.linspace(0, 1, steps)
    for i, t in enumerate(ts):
        points[i] = bezier_t(p1, p2, p3, p4, t)
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
