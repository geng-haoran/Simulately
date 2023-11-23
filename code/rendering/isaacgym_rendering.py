"""
Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto.  Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

import numpy as np
import imageio
import math
import os

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

import torch


gym = gymapi.acquire_gym()

# Parse arguments
args = gymutil.parse_arguments(description="PyTorch tensor interop example",
                               custom_parameters=[
                                   {"name": "--headless", "action": "store_true", "help": ""}])

# configure sim
sim_params = gymapi.SimParams()
sim_params.gravity = gymapi.Vec3(0.0, -9.8, 0.0)
if args.physics_engine == gymapi.SIM_FLEX:
    sim_params.flex.solver_type = 5
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 8
    sim_params.flex.relaxation = 0.75
    sim_params.flex.warm_start = 0.4
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

# This determines whether physics tensors are on CPU or GPU
sim_params.use_gpu_pipeline = True
if not args.use_gpu_pipeline:
    print("Warning: Forcing GPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

options = gymapi.AssetOptions()
options.fix_base_link = True
box_asset = gym.create_box(sim, 1, 1, 1, options)

use_viewer = not args.headless

# create viewer
if use_viewer:
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        print("*** Failed to create viewer")
        quit()
else:
    viewer = None

# Add ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# set up env grid
num_envs = 1
envs_per_row = int(math.sqrt(num_envs))
spacing = 2.0
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

envs = []
cams = []
cam_tensors = []
dep_tensors = []
seg_tensors = []

# create envs
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
    envs.append(env)

    # add actor
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.5, 0.0)
    pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
    actor_handle = gym.create_actor(env, box_asset, pose, "box", i, 0, segmentationId = 1)

    gym.set_rigid_body_color(env, actor_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1, 0, 0))
    # add camera
    cam_props = gymapi.CameraProperties()
    cam_props.width = 640
    cam_props.height = 480
    cam_props.enable_tensors = True
    cam_handle = gym.create_camera_sensor(env, cam_props)
    gym.set_camera_location(cam_handle, env, gymapi.Vec3(4, 2, 0), gymapi.Vec3(0, 0, 0))
    cams.append(cam_handle)

    # obtain camera tensor
    cam_tensor = gym.get_camera_image_gpu_tensor(sim, env, cam_handle, gymapi.IMAGE_COLOR)
    torch_cam_tensor = gymtorch.wrap_tensor(cam_tensor)
    cam_tensors.append(torch_cam_tensor)
    dep_tensor = gym.get_camera_image_gpu_tensor(sim, env, cam_handle, gymapi.IMAGE_DEPTH)
    torch_dep_tensor = gymtorch.wrap_tensor(dep_tensor)
    dep_tensors.append(torch_dep_tensor)
    seg_tensor = gym.get_camera_image_gpu_tensor(sim, env, cam_handle, gymapi.IMAGE_SEGMENTATION)
    torch_seg_tensor = gymtorch.wrap_tensor(seg_tensor)
    seg_tensors.append(torch_seg_tensor)

# point camera at middle env
cam_pos = gymapi.Vec3(8, 2, 6)
cam_target = gymapi.Vec3(-8, 0, -6)
middle_env = envs[num_envs // 2 + envs_per_row // 2]
gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

# prepare tensor access
gym.prepare_sim(sim)

# create directory for saved images
img_dir = "interop_images"
if not os.path.exists(img_dir):
    os.mkdir(img_dir)

NEXT_REPORT = 10.0
frame_count = 0
next_fps_report = NEXT_REPORT
fpss = []
t1 = 0
while viewer is None or not gym.query_viewer_has_closed(viewer):
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # refresh state data in the tensor
    gym.refresh_rigid_body_state_tensor(sim)

    gym.step_graphics(sim)

    # render sensors and refresh camera tensors
    gym.render_all_camera_sensors(sim)
    gym.start_access_image_tensors(sim)

    frame_no = gym.get_frame_count(sim)
    # if frame_no % 10 == 0:
    #     for i in range(num_envs):
    #         # write tensor to image
    #         fname = os.path.join(img_dir, "cam-%04d-%04d.png" % (frame_no, i))
    #         cam_img = cam_tensors[i].cpu().numpy()
    #         dep_img = dep_tensors[i].cpu().numpy()
    #         seg_img = seg_tensors[i].cpu().numpy()
    #         import pdb; pdb.set_trace()
    #         imageio.imwrite(fname, cam_img)
    #         imageio.imwrite(fname, np.stack((dep_img, dep_img, dep_img)).astype(np.uint8).transpose(1, 2, 0))
    #         imageio.imwrite(fname, np.stack((seg_img, seg_img, seg_img)).astype(np.uint8).transpose(1, 2, 0))

    t = gym.get_elapsed_time(sim)
    if t >= next_fps_report:
        t2 = gym.get_elapsed_time(sim)
        fps = frame_count / (t2 - t1)
        print("FPS %.1f (%.1f)" % (fps, fps * num_envs))
        fpss.append(fps)
        frame_count = 0
        t1 = gym.get_elapsed_time(sim)
        next_fps_report = t1 + NEXT_REPORT

    gym.end_access_image_tensors(sim)

    if viewer is not None:
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    frame_count += 1
    # if frame_count == 100:
    #     import pdb; pdb.set_trace()
