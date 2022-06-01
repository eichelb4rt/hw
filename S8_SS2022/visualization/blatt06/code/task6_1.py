import numpy as np # math functionality
import matplotlib.pyplot as plt # plotting

fig = plt.figure(figsize=(8,7))

ax = fig.add_subplot()
ax.set_title("Marching Squares")

# G holds some scalar values of a grid
G = np.array([[-1,-3,-3,-6,-9],
              [ 1, 1,-3,-6,-9],
              [ 2, 3, 3,-9,-9],
              [ 2, 3, 3,-9,-9],
              [ 9, 6, 6,-6,-9]])
G_rows = G.shape[0]
G_cols = G.shape[1]

# Build a grid 
x = np.linspace(0, G_cols-1, G_cols)
y = np.linspace(G_rows-1, 0, G_rows)
X, Y = np.meshgrid(x,y)
X /= G_cols-1
Y /= G_rows-1


####################
# Task 1           #
####################


# lambda * v_1 + (1 - lambda * v_2)
def find_lambda(v_1, v_2):
    return v_2 / (v_2 - v_1)

cells = np.array([[[G[i - 1, j - 1], G[i, j - 1], G[i - 1, j], G[i, j]] for j in range(1, G_cols)] for i in range(1, G_rows)])
# t = top, b = bottom, l = left, r = right
for i in range(0, G_rows - 1):
    for j in range(0, G_cols - 1):
        # tl, tr, bl, br = cells[i, j]
        print(cells[i, j] > 0)

# print(X)
# print(Y)











####################
# Plot the grid    #
####################
# set XY-ticks to resemble grid lines
ax.set_xticks(np.linspace(0, 1, G_cols))
ax.set_yticks(np.linspace(0, 1, G_rows))
ax.grid(True)
ax.set_axisbelow(True)
ax.get_xaxis().set_ticklabels([])
ax.get_yaxis().set_ticklabels([])

# annotate each gridpoint with the scalar value in G
ax.scatter(X, Y, s=200)
for i in range(G_rows):
    for j in range(G_cols):
        ax.annotate(G[i][j], xy=(X[i][j], Y[i][j]), ha='center', va='center', c='white')

# Always run show, to make sure everything is displayed.
plt.show()