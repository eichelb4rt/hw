from matplotlib import pyplot as plt
import numpy as np

print("Hello?")
n_points = 1000
x_axis = [i / n_points for i in range(n_points)]
x_axis.append(1)
x_axis = np.array(x_axis)
gammacorrection = lambda x, gamma: x**gamma
for gamma in [0.5, 1, 2.5]:
    plt.plot(x_axis, gammacorrection(x_axis, gamma), label=f"gamma={gamma}")
plt.title("Transformation der Grauwerte bei Gammakorrektur")
plt.legend()
plt.savefig("gamma_correction.png")