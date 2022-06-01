import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
from explicit_euler import explicit_euler
from bezier import bezier

N = 1000
Y_0 = np.array([
    997,
    3,
    0,
    0
])
T = np.arange(0, 100, 0.01)
# animation bezier stuff
P1 = np.array([
    0.5,
    0.01,
    0.3
])
P2 = np.array([
    1,
    0.01,
    0.005
])
P3 = np.array([
    0.5,
    0.5,
    0.005
])
P4 = P1
ANIMATION_STEPS = 50
PARAMS = bezier(P1, P2, P3, P4, steps=ANIMATION_STEPS)


def approximate(alpha, beta, gamma):
    # S: 0, I: 1, R: 2, H: 3
    f = lambda t_i, y_i: np.array([
        - alpha * y_i[0] * y_i[1] / N,
        alpha * y_i[0] * y_i[1] / N - gamma * y_i[1] - beta * y_i[1],
        beta * y_i[1],
        gamma * y_i[1]
    ])
    return explicit_euler(f, Y_0, T)


def main():
    fig, (ax0, ax1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 3]})
    max_param = np.max(PARAMS)

    def animate(i):
        ALPHA, BETA, GAMMA = PARAMS[i]
        # animate params
        ax0.clear()
        ax0.set_ylim(0, max_param * 1.1)
        ax0.bar(["alpha", "beta", "gamma"], [ALPHA, BETA, GAMMA], color=['orange', 'green', 'black'])
        ax0.tick_params(labelrotation=45)
        # animate approximation
        approximation = approximate(ALPHA, BETA, GAMMA)
        ax1.clear()
        ax1.plot(T, approximation[:, 0], label="S", color='blue')
        ax1.plot(T, approximation[:, 1], label="I", color='orange')
        ax1.plot(T, approximation[:, 2], label="R", color='green')
        ax1.plot(T, approximation[:, 3], label="H", color='black')
        ax1.legend()
    ani = animation.FuncAnimation(fig, animate, frames=ANIMATION_STEPS, interval=10)
    ani.save("animation.gif")
    # plt.show()


if __name__ == "__main__":
    main()
