import numpy as np # math functionality
import matplotlib.pyplot as plt # plotting

def phi(r):
    return np.exp(-r**2)

points = np.array([[-2,-2], [2,0], [0,-1], [-1,2]])
values = np.array( [0.2,     0.6,   0.3,    0.5])

fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(projection='3d')
ax.scatter(points[:,0], points[:,1], values, color='red')

R = np.array([[phi(np.linalg.norm(p_i - p_j)) for p_i in points] for p_j in points])
w = np.linalg.solve(R, values)
f = lambda x: np.array([phi(np.linalg.norm(p_i - x)) for p_i in points]) @ w
# ax.plot()

delta = 0.025
x = np.arange(-4.0, 4.0, delta)
y = np.arange(-4.0, 4.0, delta)
X, Y = np.meshgrid(x, y)
z = np.array([f(point) for point in zip(np.ravel(X), np.ravel(Y))])
Z = z.reshape(X.shape)
ax.plot_surface(X,Y,Z, cmap='viridis')



# Always run show, to make sure everything is displayed.
plt.show()