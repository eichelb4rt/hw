import numpy as np
import matplotlib.pyplot as plt
import vtk
from vtk.util.numpy_support import vtk_to_numpy

# read a volume image
reader = vtk.vtkNrrdReader()
reader.SetFileName("MRHead.nrrd")
reader.Update()
imageData = reader.GetOutput()

# convert from vtkImageData to numpy array
x_dim, y_dim, z_dim = imageData.GetDimensions()
sc = imageData.GetPointData().GetScalars()
image = vtk_to_numpy(sc)
image = image.reshape(x_dim, y_dim, z_dim, order='F')
image = np.rot90(np.flip(image, axis=1))

# normalize image values
image = np.divide(image, float(np.max(image)))

# create a figure
fig = plt.figure(figsize=(16, 5))


####################
# Task 1a
####################

Z_AXIS = 2

# TODO: change axis?
projection_MIP = np.max(image, axis=Z_AXIS)
# TODO: plt.imshow(projection)
plt.subplot(131)
plt.imshow(projection_MIP, cmap='gray')

####################
# Task 1b
####################

INITIAL_LIGHT = 1
projection_xray = 1 - INITIAL_LIGHT * np.exp(-np.mean(image, axis=Z_AXIS))
plt.subplot(132)
plt.imshow(projection_xray, cmap='gray')


####################
# Task 1c
####################
gamma = 0.0
alpha = 0.06

I_max = np.zeros((x_dim, y_dim))
C_i = np.zeros((x_dim, y_dim))
alpha_i = 0
for i in range(z_dim):
    I_i = image[:,:,i]
    delta_i = (I_i - I_max) * (I_i > I_max)
    beta_i = 1 - delta_i * (1 + gamma) if gamma <= 0 else 1 - delta_i
    C_i = beta_i * C_i + (1 - beta_i * alpha_i) * I_i
    alpha_i = beta_i * alpha_i + (1 - beta_i * alpha_i) * alpha
    # update max
    I_max = np.maximum(I_max, I_i)

plt.subplot(133)
plt.imshow(C_i, cmap='gray')

# Always run show, to make sure everything is displayed.
# plt.savefig("plot.png")
plt.show()
