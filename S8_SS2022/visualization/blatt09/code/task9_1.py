import math
import numpy as np
import matplotlib.pyplot as plt

# Given is a vector field v(x,y) = (-y, x)^T.
# Utility to sample a given position [x,y]:
def v(pos):
    # time is not needed in our field, but i added it for convenience
    return np.array([-pos[1], pos[0]])

# Show the vector field using a quiver plot
X, Y = np.meshgrid(np.arange(-8, 8), np.arange(-8, 8))
U = -Y
V = X

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot()
ax.set_title(r'$v(x,y) =  (-y \quad x)^T$')
ax.quiver(X, Y, U, V)


####################
# Task 1           #
####################

start = np.array([1, 0])
t_max = 2 * math.pi
delta_t = 0.7

ts = np.arange(0, t_max, delta_t)
if ts[-1] != t_max:
    ts = np.append(ts, t_max)

def generate(v, start, ts, step):
    x = [start]
    d_ts = ts[1:] - ts[:-1]
    for t, d_t in zip(ts[:-1], d_ts):
        x.append(step(v, x[-1], d_t))
    return np.array(x)

def euler_step(v, x, d_t):
    return x + d_t * v(x)

def midpoint_step(v, x, d_t):
    d_v = d_t * v(x)
    v_mid = v(x + d_v / 2)
    return x + d_t * v_mid

def rk_step(v, x, d_t):
    k_1 = d_t * v(x)
    k_2 = d_t * v(x + k_1 / 2)
    k_3 = d_t * v(x + k_2 / 2)
    k_4 = d_t * v(x + k_3)
    return x + (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6    

euler = generate(v, start, ts, euler_step)
midpoint = generate(v, start, ts, midpoint_step)
rk = generate(v, start, ts, rk_step)

plt.plot(euler[:, 0], euler[:, 1], label="euler")
plt.plot(midpoint[:, 0], midpoint[:, 1], label="midpoint")
plt.plot(rk[:, 0], rk[:, 1], label="rk")

plt.legend()
plt.show()