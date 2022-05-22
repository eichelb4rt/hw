import vtk # the visualizaton toolkit (python wrapper)

# This creates a polygonal cylinder model with eight circumferential facets
cylinder = vtk.vtkCylinderSource()
cylinder.SetResolution(8)

# The mapper is responsible for pushing the geometry into the graphics library.
# It may also do color mapping, if scalars or other attributes are defined.
cylinderMapper = vtk.vtkPolyDataMapper()
cylinderMapper.SetInputConnection(cylinder.GetOutputPort())

# The actor is a grouping mechanism: besides the geometry (mapper), it
# also has a property, transformation matrix, and/or texture map.
# Here we set its color and rotate it around the X axes.
cylinderActor = vtk.vtkActor()
cylinderActor.SetMapper(cylinderMapper)
cylinderActor.GetProperty().SetColor(1.0, 0.0, 0.0) # red (r, g, b)
#cylinderActor.GetProperty().SetRepresentationToWireframe()
cylinderActor.RotateX(45.0) # rotate by 45 degree around x-axis

# The renderer generates the image, which is then displayed on the render window.
# It can be thought of as a scene to which the actor is added.
renderer = vtk.vtkRenderer()
renderer.AddActor(cylinderActor)
renderer.SetBackground(0.8, 0.8, 0.8) # light gray color

# The render window is the actual window that appears on screen.
renderWindow = vtk.vtkRenderWindow()
renderWindow.SetSize(300, 300) # size in pixel
renderWindow.AddRenderer(renderer)

# The render window interactor captures mouse events
# and will perform appropriate camera manipulation.
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)
renderWindowInteractor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

# This starts the event loop.
renderWindowInteractor.Start()