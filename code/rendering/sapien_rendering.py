"""Camera.

Concepts:
    - Create and mount cameras
    - Render RGB images, point clouds, segmentation masks
"""

import sapien.core as sapien
import numpy as np
from PIL import Image, ImageColor
from scipy.spatial.transform import Rotation as R

def main():
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)

    scene = engine.create_scene()
    scene.set_timestep(1 / 100.0)

    scene.add_ground(altitude=0)  # Add a ground
    actor_builder = scene.create_actor_builder()
    actor_builder.add_box_collision(half_size=[0.5, 0.5, 0.5])
    actor_builder.add_box_visual(half_size=[0.5, 0.5, 0.5], color=[1., 0., 0.])
    box = actor_builder.build(name='box')  # Add a box
    box.set_pose(sapien.Pose(p=[0, 0, 0.5]))

    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
    scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
    scene.add_point_light([1, -2, 2], [1, 1, 1], shadow=True)
    scene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=True)

    near, far = 0.05, 100
    width, height = 640, 480
    camera = scene.add_camera(
        name="camera",
        width=width,
        height=height,
        fovy=1,
        near=near,
        far=far,
    )
    q = R.from_euler('zyx', [0, np.arctan2(2, 4), 0]).as_quat()
    q_SAPIEN = [q[3],q[0],q[1],q[2]]
    camera.set_pose(sapien.Pose(
        p=[-4, 0, 2], 
        q = q_SAPIEN
        ))
    scene.step()  # make everything set
    scene.update_render()
    camera.take_picture()

    rgba = camera.get_float_texture('Color')  # [H, W, 4]
    rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
    rgba_pil = Image.fromarray(rgba_img)
    rgba_pil.save('color.png')

    position = camera.get_float_texture('Position')  # [H, W, 4]

    depth = -position[..., 2]
    depth_image = (depth * 1000.0).astype(np.uint16)
    depth_pil = Image.fromarray(depth_image)
    depth_pil.save('depth.png')

    seg_labels = camera.get_uint32_texture('Segmentation')  # [H, W, 4]
    colormap = sorted(set(ImageColor.colormap.values()))
    color_palette = np.array([ImageColor.getrgb(color) for color in colormap],
                             dtype=np.uint8)
    label0_image = seg_labels[..., 0].astype(np.uint8)  # mesh-level
    label1_image = seg_labels[..., 1].astype(np.uint8)  # actor-level
    label0_pil = Image.fromarray(color_palette[label0_image])
    label0_pil.save('label0.png')
    label1_pil = Image.fromarray(color_palette[label1_image])
    label1_pil.save('label1.png')

if __name__ == '__main__':
    main()
