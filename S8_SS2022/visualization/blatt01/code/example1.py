import numpy as np # math functionality
import matplotlib.pyplot as plt # plotting

# Create a figure (window)
fig = plt.figure(figsize=(15,4))

#############################
# Plot a function
#############################
# Here, some function of a is computed with numpy.
# Note that python does not need to know what type of object 'a' is.
def function(a):
    return np.exp(-a) * np.cos(2 * np.pi * a)

# A number sequence can be created with the arange function.
x = np.arange(0, 10, 0.1)

# If we want to apply a function to each object of this array,
# we can simply call numpy's included operations on the whole array.
y = function(x)

# To read out arrays/matrices directly, they can be printed.
print(y)

# Creating a subplot allows to plot the results.
# Here, a 1 x 4 grid of subplots is declared and position 1 is used.
axs1 = fig.add_subplot(1, 3, 1)
axs1.plot(x, y)


#############################
# Customize plots
#############################
# There are also plenty of customization options.
# For instance, we can plot our array
# - as a linear graph (red points)
# - as a quadratic function (blue dashes)
# - as a cubic function (green crosses)
# All within one window:
axs2 = fig.add_subplot(1, 3, 2)
axs2.plot(x, x, 'rp', label='linear')
axs2.plot(x, x**2, 'b--', label='quadratic')
axs2.plot(x, x**3, 'g+', label='cubic')

# Optionally define a title, labels, a grid, and a legend
axs2.set_title('Simple Plot')
axs2.set_xlabel('Label (x)')
axs2.set_ylabel('Label (y)')
axs2.grid(True)
axs2.legend()

# You can even use TeX math expressions inside your plots.
# Write the expression into a string r'$...$'. (r indicates a raw string)
axs3 = fig.add_subplot(1, 3, 3)
axs3.plot(x, y)
axs3.set_title(r'$e^{-x} \cdot \cos (2 \pi x)$')
axs3.grid(True)

#############################
# Always run plt.show() at the end of a file.
# This makes sure everything is displayed.
#############################
# In Spyder, all plots are directly shown in the 'plots' tab (per default).
# Still, always use plt.show(), otherwise the plots will not be shown when the
# script is called from console.
plt.show()