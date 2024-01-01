import os
import time

import numpy as np
import sapien.core as sapien
from PIL import Image, ImageColor
from scipy.spatial.transform import Rotation as R
import torch

SAVE_IMG_AND_EXIT = False
MODE = "RGB"
# MODE = "DEPTH"
# MODE = "SEG"


def main():
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)

    if MODE == "DEPTH":
        sapien.VulkanRenderer.set_camera_shader_dir("trivial")
    else:
        sapien.VulkanRenderer.set_camera_shader_dir("ibl")

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
    q_SAPIEN = [q[3], q[0], q[1], q[2]]
    camera.set_pose(sapien.Pose(
        p=[-4, 0, 2],
        q=q_SAPIEN
    ))
    render = []
    rgb = []
    pos = []
    seg = []
    transfer = []
    for i in range(1000):
        # print("#" * 10, i, "#" * 10)
        scene.step()  # make everything set

        # Rendering
        s = time.time()
        scene.update_render()
        if MODE == "RGB":
            texture_name = "Color"
        elif MODE == "DEPTH":
            texture_name = "Position"
        elif MODE == "SEG":
            texture_name = "Segmentation"
        else:
            raise RuntimeError(f"Unknown rendering mode: {MODE}")

        await_dl_list = camera.take_picture_and_get_dl_tensors_async([texture_name])
        dl_list = await_dl_list.wait()
        dl_tensor = dl_list[0]
        gpu_torch_tensor = torch.from_dlpack(dl_tensor)  # get image torch GPU tensor
        t = time.time()

        # GPU to CPU
        t_s = time.time()
        shape = sapien.dlpack.dl_shape(dl_tensor)
        # if MODE == "SEG":
        #     output_array = torch.from_dlpack(dl_tensor).cpu().numpy()
        # else:
        output_array = np.zeros(shape, dtype=np.float32)
        sapien.dlpack.dl_to_numpy_cuda_async_unchecked(dl_tensor, output_array)
        sapien.dlpack.dl_cuda_sync()
        t_t = time.time()
        transfer.append(t_t - t_s)

        if MODE == "RGB":
            render.append(t - s)
            if SAVE_IMG_AND_EXIT:
                rgba_img = (output_array * 255).clip(0, 255).astype("uint8")
                img_pil = Image.fromarray(rgba_img)
                filename = "sapien_rgb.png"
        elif MODE == "DEPTH":
            render.append(t - s)
            if SAVE_IMG_AND_EXIT:
                depth = output_array[..., 2]
                depth_image = (depth * 1000.0).astype(np.uint16)
                img_pil = Image.fromarray(depth_image)
                filename = "sapien_depth"
        elif MODE == "SEG":
            render.append(t - s)
            if SAVE_IMG_AND_EXIT:
                colormap = sorted(set(ImageColor.colormap.values()))
                color_palette = np.array([ImageColor.getrgb(color) for color in colormap],
                                         dtype=np.uint8)
                mesh_seg = output_array[..., 0].astype(np.uint8)
                img_pil = Image.fromarray(color_palette[mesh_seg])
                filename = "sapien_seg"
        else:
            raise RuntimeError(f"Unknown rendering mode: {MODE}")

        # print("rendering:", t - s)
        if SAVE_IMG_AND_EXIT:
            os.makedirs("sapien", exist_ok=True)
            img_pil.save(f'sapien/{filename}')
            exit()

    # rgb = np.array(rgb)
    # pos = np.array(pos)
    # seg = np.array(seg)
    render = np.array(render)
    transfer = np.array(transfer)
    # print("rgb:", rgb.mean(), 1 / rgb.mean())  # 0.004276715517044068
    print(MODE, "render:", render.mean(), 1 / render.mean())  # 0.002738466024398804
    # print("seg:", seg.mean(), 1 / + seg.mean())  # 0.002738466024398804
    print("transfer:", transfer.mean(), 1 / transfer.mean())  # 0.002738466024398804


if __name__ == '__main__':
    main()
