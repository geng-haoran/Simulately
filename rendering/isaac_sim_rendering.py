# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": True})
import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt
import time
import numpy as np
import omni.isaac.core.utils.numpy.rotations as rot_utils
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.sensor import Camera
from omni.isaac.core import SimulationContext
from omni.kit.viewport.utility import get_active_viewport
from omni.syntheticdata.tests.utils import add_semantics

RGB = 0
DEPTH = 1
SEG = 2

test_mode = RGB

dt = 1/180.0
my_world = World(stage_units_in_meters=1.0, physics_dt=dt, rendering_dt=dt)

cube_2 = my_world.scene.add(
    DynamicCuboid(
        prim_path="/new_cube",
        name="cube_1",
        position=np.array([0.0, 0.0, 0.5]),
        scale=np.array([1.0, 1.0, 1.0]),
        size=1.0,
        color=np.array([255, 0, 0]),
    )
)
viewport_api = get_active_viewport()
render_product_path = viewport_api.get_render_product_path()
camera = Camera(
    prim_path="/World/camera",
    position=np.array([-10.0, 0.0, 15.0]),
    dt = dt,
    resolution=(640, 480),
    orientation=rot_utils.euler_angles_to_quats(np.array([0, 60, 0]), degrees=True),
    render_product_path=render_product_path,
)
stage = get_current_stage()
cubePrim = stage.GetPrimAtPath('/new_cube')
add_semantics(cubePrim, "cube")
my_world.scene.add_default_ground_plane()
my_world.reset()
camera.initialize()
if test_mode == DEPTH:
    camera.add_distance_to_image_plane_to_frame()
elif test_mode == SEG:
    camera.add_semantic_segmentation_to_frame()

# wait for a frame so the camera is initialized
my_world.step(render=True)

i = 0
frames = 0
depths = []
imgs = []
segs = []

for _ in range(2000):
    my_world.step(render=True)
    if i == 0:
        s = time.time()
    if i >= 0:
        data = camera.get_current_frame()
        # camera.get_current_frame()
        if test_mode == RGB:
            img = data["rgba"]
            imgs.append(img)
        elif test_mode == DEPTH:
            depth = data['distance_to_image_plane']
            depths.append(depth)
        elif test_mode == SEG:
            seg = data["semantic_segmentation"]['data']
            segs.append(seg)
        frames += 1

    i += 1
e = time.time()
print("FPS: ", frames/(e-s))

# assert all depth are equal
if len(depths)> 0 :
    for i in range(len(depths)-1):
        assert np.allclose(depths[i], depths[i+1])

# assert all segs are equal
if len(segs)> 0 :
    for i in range(len(segs)-1):
        assert np.allclose(segs[i], segs[i+1])

# FPS: 
# RGB: 175.9634264249718
# DEPTH: 167.12673659846513
# SEG: 139.164041078619
simulation_app.close()
