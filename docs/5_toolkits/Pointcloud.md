---
sidebar_position: 2
---


# Pointcloud
> **Toolkits for point cloud processing**

### Point Cloud Visualizer
Visualizes a point cloud using Open3D. Supports N\*3 and N\*6 point clouds, and accepts both NumPy arrays and PyTorch tensors.
```python
import open3d as o3d
import numpy as np
import torch

def visualize_point_cloud(data):
    """
    Visualizes a point cloud using Open3D. Supports N*3 and N*6 point clouds,
    and accepts both NumPy arrays and PyTorch tensors.

    :param data: A NumPy array or PyTorch tensor of shape (N, 3) or (N, 6).
                 For (N, 3), it represents the (x, y, z) coordinates of the points.
                 For (N, 6), it represents the (x, y, z, r, g, b) coordinates and colors of the points.
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    if data.shape[1] not in [3, 6]:
        raise ValueError("The input data must have shape (N, 3) or (N, 6).")

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(data[:, :3])

    if data.shape[1] == 6:
        point_cloud.colors = o3d.utility.Vector3dVector(data[:, 3:])

    o3d.visualization.draw_geometries([point_cloud])
```

### Save Point Cloud to .OBJ files
Write a point cloud to .OBJ file. Supports N\*3 and N\*6 point clouds, but only save the xyz value to .OBJ file.
```python
def write_obj(points, file_name):
    with open(file_name, 'w') as f:
        for point in points:
            f.write(f"v {point[0]} {point[1]} {point[2]}\n")
        print("OBJ file created successfully!")
```