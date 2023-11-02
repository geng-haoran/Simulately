# Camera Sensors

This page provides instructions and snippets of deploying camera sensors.

## Creating Camera Sensors

Assume that you have a task class called `MyTask(Task)`, instantiated as `task`. It has a member `task.gym` as the Isaac Gym instance.

### Prepare the Camera Poses

You need to specify the positions and the looking positions of the cameras, in the environments' local frames. E.g. you want 2 (specified by `task.num_cameras`) cameras placed at `(0.5, 0.0, 0.3), (-0.5, 0.0, 0.3)` (`camera_eye_list`), and looking at `(0.0, 0.0, 0.1), (0.0, 0.0, 0.1)` (camera_lookat_list), respectively, you can set the member of the `task` (inside its `__init__` method):

```python
self.camera_eye_list = [ [0.5, 0.0, 0.3], [-0.5, 0.0, 0.3] ]
self.camera_lookat_list = [ [0.0, 0.0, 0.1], [0.0, 0.0, 0.1] ]
```

### Define Cameras

In the `__init__` function of the task class, add the code below to create.

```python
for i_env in range(num_envs):
    depth_tensors = []
    rgb_tensors = []
    seg_tensors = []
    vinv_mats = []
    proj_mats = []
    for i in range(self.num_cameras):
        camera_handle = self.gym.create_camera_sensor(env_ptr, self.camera_props)
        self.camera_handles.append(camera_handle)

        camera_eye = self.camera_eye_list[i]
        camera_lookat = self.camera_lookat_list[i]
        self.gym.set_camera_location(camera_handle, env_ptr, camera_eye, camera_lookat)

        camera_tensor_depth = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle, gymapi.IMAGE_DEPTH)
        camera_tensor_rgb = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle, gymapi.IMAGE_COLOR)
        camera_tensor_rgb_seg = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle, gymapi.IMAGE_SEGMENTATION)
        torch_cam_depth_tensor = gymtorch.wrap_tensor(camera_tensor_depth)
        torch_cam_rgb_tensor = gymtorch.wrap_tensor(camera_tensor_rgb)
        torch_cam_rgb_seg_tensor = gymtorch.wrap_tensor(camera_tensor_rgb_seg)

        cam_vinv = torch.inverse(torch.tensor(self.gym.get_camera_view_matrix(self.sim, env_ptr, camera_handle))).to(self.device)
        cam_proj = torch.tensor(self.gym.get_camera_proj_matrix(self.sim, env_ptr, camera_handle),device=self.device,)

        origin = self.gym.get_env_origin(env_ptr)
        self.env_origin[env_id][0] = origin.x
        self.env_origin[env_id][1] = origin.y
        self.env_origin[env_id][2] = origin.z

        depth_tensors.append(torch_cam_depth_tensor)
        rgb_tensors.append(torch_cam_rgb_tensor)
        seg_tensors.append(torch_cam_rgb_seg_tensor)
        vinv_mats.append(cam_vinv)
        proj_mats.append(cam_proj)

    depth_tensors.append(torch_cam_depth_tensor)
    rgb_tensors.append(torch_cam_rgb_tensor)
    seg_tensors.append(torch_cam_rgb_seg_tensor)
    vinv_mats.append(cam_vinv)
    proj_mats.append(cam_proj)
```
