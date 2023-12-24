import mujoco_py
import os
import matplotlib.pyplot as plt

# Load the model from the XML
model = mujoco_py.load_model_from_path('path_to_your_xml_file.xml')
sim = mujoco_py.MjSim(model)

# Setup the viewer for rendering
viewer = mujoco_py.MjViewer(sim)
viewer.cam.lookat[0] = 0  # x-coordinate
viewer.cam.lookat[1] = 0  # y-coordinate
viewer.cam.lookat[2] = 0  # z-coordinate
viewer.cam.distance = model.stat.extent * 0.5
viewer.cam.azimuth = 180
viewer.cam.elevation = -20

# Render the scene
viewer.render()

# Capture the image
image = viewer.read_pixels(width=500, height=500, depth=False)

# Display the image
plt.imshow(image)
plt.show()
