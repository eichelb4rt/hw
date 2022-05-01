import numpy as np  # math functionality
import matplotlib.pyplot as plt  # plotting
import matplotlib.image as mpimg  # loading images

# Using subplots allows several plots to be displayed at once.
# First, a figure is created. The figsize determines default size.
fig = plt.figure(figsize=(8, 7))


########################################################
# Plot 1
########################################################

# This loads an image as a numpy matrix.
# Actually, its three stacked matrices (or a 3D array).
# Each dimension corresponds with one component: red, green and blue.
img = mpimg.imread('circle.png')

# As this is a black and white image and rgb components are more or less equal,
# we can select the first component (red) to get a simple 2D matrix.
# This is done by using numpy's array slicing operations.
# [:, :, 0] means: Select all from dimensions 1 and 2 and only the first entry from 3.
bw_img = img[:, :, 0]

# Add a subplot to our image at position 1. The subplot-grid is 2x2
axs1 = fig.add_subplot(2, 2, 1)

# A 2D matrix can be displayed using 'imshow', rendering each entry as a pixel.
image1 = axs1.imshow(bw_img)

# Why is the image not black and white? -> Because the default colormap is applied.
# When reading in the image, all values were scaled to floats between [0,1].
# We can see this by adding a colorbar to the view:
axs1.set_title('Viridis Colormap (circle.png)')
width, height, _ = img.shape
axs1.annotate("", xy=(width / 2, height / 2), xytext=(0, 0),
              arrowprops={'arrowstyle': '->'})
fig.colorbar(image1, ax=axs1)


########################################################
# Plot 2
########################################################

# To display an image with gray levels, we can use the 'gray' colormap.
axs2 = fig.add_subplot(2, 2, 2)
axs2.set_title('Gray Colormap (circle.png)')
image2 = axs2.imshow(bw_img, cmap='gray')
step = 10
start = step
grid_points = np.array([[x, y] for x in range(start, width, step)
                       for y in range(start, height, step)])
axs2.scatter(grid_points[:, 0], grid_points[:, 1])
fig.colorbar(image2, ax=axs2)


########################################################
# Plot 3
########################################################

# This creates two arrays (x and y) with values between -3 and 3 in 0.025 increments.
delta = 0.025
x = y = np.arange(-3.0, 3.0, delta)

# Meshgrid builds a matrix from two given axes.
# X is a 2D array containing all row indices.
# Y is a 2D array containing all column indices.
X, Y = np.meshgrid(x, y)

# This way, the distance to the center (0,0) can be computed for each 'pixel'.
Radius = np.sqrt(X**2 + Y**2)

# From the radius, a 2D-Sin function could be derived.
# We use 'Z' to indicate a third coordinate (next to X and Y)
Z = np.sin(Radius)

# The result can be mapped to an image using a colormap
axs3 = fig.add_subplot(2, 2, 3)
axs3.set_title('2D sin')
image3 = axs3.imshow(Z, extent=[-3, 3, -3, 3], cmap='coolwarm')
contours = axs3.contour(X, Y, Z, 4, colors='black')
axs3.clabel(contours, inline=True, fontsize=8)
fig.colorbar(image3, ax=axs3)


########################################################
# Plot 4
########################################################

# The result can also be displayed in a 3D plot
axs4 = fig.add_subplot(2, 2, 4, projection='3d')
axs4.set_title('3D plot of 2D sin')
plot1 = axs4.plot_surface(X, Y, Z, cmap='coolwarm')  # shows a surface
# plot1 = axs4.plot_wireframe(X,Y,Z, cmap='coolwarm') # shows a wireframe
axs4.contour3D(X, Y, Z, 4, colors='black')
axs4.contour(X, Y, Z, 4, colors='black', offset=np.min(Z))
fig.colorbar(plot1, ax=axs4)


# Always run this, to make sure everything is displayed.
# Calling the program with the console allows you to rotate the 3D plot.
plt.show()
