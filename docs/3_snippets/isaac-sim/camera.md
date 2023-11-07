# Camera Sensors

Creating camera sensors and read data from it.

## Create a Camera

```python
ROBOT_CAM_PRIM_PATH = "/robot/camera_link/robot_cam"

camera = Camera(
    prim_path=ROBOT_CAM_PRIM_PATH,
    frequency=15,
    resolution=(1920, 1080),
    translation=np.array([-0.04, 0.0, 0.05]),
    orientation=np.array([0.0, 0.0, 0.0, 1.0]),
)
```

In simulation, obtain camera data with:

```python
res = camera.get_current_frame()
```

However, `res` will be an empty list by default if you don't specify what you want by calling:

```python
camera.add_motion_vectors_to_frame()
camera.add_distance_to_image_plane_to_frame()
camera.add_bounding_box_2d_tight_to_frame()
...
```

To see what you can get here, refer to [this page](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/annotators_details.html#distance-to-image-plane). Note that if you want to obtain depth, you need `distance_to_image_plane` rather than `distance_to_camera`.

## Action Graph for Publishing Camera Data to ROS2

You need to first obtain the current stage (you can get it by calling `self.get_stage()` in derive classes of `BaseSample`) as `stage`. You also need to import:

```python
import omni.graph.core as og
```

To create the graph:

```python
CAM_GRAPH_PATH = "/robot/camera_graph"
ROBOT_CAM_PRIM_PATH = "/robot/camera_link/robot_cam"

keys = og.Controller.Keys
(camera_graph, _, _, _) = og.Controller.edit(
    {
        "graph_path": CAM_GRAPH_PATH,
        "evaluator_name": "push",
        "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
    },
    {
        keys.CREATE_NODES: [
            ("Context", "omni.isaac.ros2_bridge.ROS2Context"),
            ("OnImpulseEvent", "omni.graph.action.OnImpulseEvent"),
            ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
            ("createViewport", "omni.isaac.core_nodes.IsaacCreateViewport"),
            ("getRenderProduct", "omni.isaac.core_nodes.IsaacGetViewportRenderProduct"),
            ("setCamera", "omni.isaac.core_nodes.IsaacSetCameraOnRenderProduct"),
            ("cameraHelperRgb", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
            ("cameraHelperDepth", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
            ("cameraHelperInfo", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
        ],
        keys.CONNECT: [
            ("OnPlaybackTick.outputs:tick", "createViewport.inputs:execIn"),
            ("Context.outputs:context", "cameraHelperRgb.inputs:context"),
            ("Context.outputs:context", "cameraHelperDepth.inputs:context"),
            ("Context.outputs:context", "cameraHelperInfo.inputs:context"),
            ("createViewport.outputs:execOut", "getRenderProduct.inputs:execIn"),
            ("createViewport.outputs:viewport", "getRenderProduct.inputs:viewport"),
            ("getRenderProduct.outputs:execOut", "setCamera.inputs:execIn"),
            ("getRenderProduct.outputs:renderProductPath", "setCamera.inputs:renderProductPath"),
            ("setCamera.outputs:execOut", "cameraHelperRgb.inputs:execIn"),
            ("setCamera.outputs:execOut", "cameraHelperDepth.inputs:execIn"),
            ("setCamera.outputs:execOut", "cameraHelperInfo.inputs:execIn"),
            ("getRenderProduct.outputs:renderProductPath", "cameraHelperRgb.inputs:renderProductPath"),
            ("getRenderProduct.outputs:renderProductPath", "cameraHelperDepth.inputs:renderProductPath"),
            ("getRenderProduct.outputs:renderProductPath", "cameraHelperInfo.inputs:renderProductPath"),
        ],
        keys.SET_VALUES: [
            ("createViewport.inputs:viewportId", 0),
            ("createViewport.inputs:name", "realsense_cam"),
            ("cameraHelperRgb.inputs:frameId", "sim_camera"),
            ("cameraHelperRgb.inputs:topicName", "affordbot/grasp_rgb"),
            ("cameraHelperRgb.inputs:type", "rgb"),
            ("cameraHelperDepth.inputs:frameId", "sim_camera"),
            ("cameraHelperDepth.inputs:topicName", "affordbot/grasp_depth"),
            ("cameraHelperDepth.inputs:type", "depth"),
            ("cameraHelperInfo.inputs:frameId", "sim_camera"),
            ("cameraHelperInfo.inputs:topicName", "camera_info"),
            ("cameraHelperInfo.inputs:type", "camera_info"),
        ],
    },
)

set_targets(
    prim=stage.GetPrimAtPath(f"{CAM_GRAPH_PATH}/setCamera"),
    attribute="inputs:cameraPrim",
    target_prim_paths=[ROBOT_CAM_PRIM_PATH],
)
```

In version 2022.2.1 the point cloud utility is very slow, assuming known camarea parameters, you can use the following script to generate point cloud. Notice the convention difference for isaac sim cameras.

```python
import numpy as np
import cython
from cython.parallel import parallel, prange

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_points(int height, int width, float[:,:] depth, float fx, float fy, float cx, float cy):
    cdef points_cam = np.zeros((height, width, 3), dtype=np.double)
    cdef double[:, :, :] result_view = points_cam
    cdef int v, u, k
    cdef float d
    with nogil, parallel():
        for v in prange(height, schedule='static'):
            for u in range(width):
                d = depth[v,u]
                
                # https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/reference_conventions.html
                # rotate around x axis 180
                # point[2] = -d *100.0
                # point[0] = -( u - cx ) * point[2] / fx
                # point[1] = (v - cy) * point[2] / fy
                
                
                result_view[v, u, 2] = -d *100.0
                result_view[v, u, 0] = -( u - cx ) * result_view[v, u, 2] / fx
                result_view[v, u, 1] = (v - cy) * result_view[v, u, 2]/ fy

    return points_cam
```
