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


### Save Point Cloud to .PLY files
Write a point cloud to .PLY file. Supports N\*3 and N\*6 point clouds, numpy array and torch tensor.
```python
import numpy as np
import open3d as o3d
import torch

def convert_to_ply(points, filename):
    """
    Convert a point cloud (NumPy array or PyTorch tensor) to a PLY file using Open3D.
    
    :param points: NumPy array or PyTorch tensor of shape (N, 3) or (N, 6) 
                   where N is the number of points.
    :param filename: Name of the output PLY file.
    """

    # Convert PyTorch tensor to NumPy array if necessary
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()

    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()

    # Set the points. Assuming the first 3 columns are x, y, z coordinates
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    # If the points array has 6 columns, assume the last 3 are RGB values
    if points.shape[1] == 6:
        # Normalize color values to [0, 1] if they are not already
        colors = points[:, 3:6]
        if colors.max() > 1.0:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    # Write to a PLY file
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Point cloud saved to '{filename}'.")
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