import vtk # the visualization toolkit

# Data structures for point field
gridSize = 16
points = vtk.vtkPoints()

# Create point geometry (the coordinates)
for i in range(gridSize):
    for j in range(gridSize):
        # calculate coordinates
        x = i / (gridSize - 1.0) * 3.0 - 1.0
        y = j / (gridSize - 1.0) * 3.0 - 2.0
        z = x * x**2 / 3.0 + y * y**2 / 3.0 - x * x / 2.0 + y * y / 2.0
        
        # insert the point (geometry)
        points.InsertNextPoint(x,y,z)