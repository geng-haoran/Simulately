import sapien.core as sapien
from PIL import Image
import numpy as np

# Create a SAPIEN Engine
engine = sapien.Engine()

# Create a rendering scene
renderer = sapien.VulkanRenderer()
engine.set_renderer(renderer)
scene = engine.create_scene()

# Add a camera
camera = scene.add_mounted_camera('camera', sapien.MountedCameraOption(), width=800, height=600)

# Add some objects
builder = scene.create_actor_builder()
builder.add_box_collision(half_size=[0.1, 0.1, 0.1])
builder.add_box_visual(half_size=[0.1, 0.1, 0.1], color=[1, 0, 0])
box = builder.build_static()
box.set_pose(sapien.Pose([0.5, 0, 0.5]))

# Step the simulation and render
scene.step()
scene.update_render()
camera.take_picture()
rgba = camera.get_color_rgba()

# Convert to an image and save
image = Image.fromarray((rgba[:, :, :3] * 255).astype(np.uint8))
image.save('/mnt/data/rendered_image.png')
