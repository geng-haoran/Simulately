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


### Read Point Cloud from .PLY files
```python
import open3d as o3d
def read_pcs_from_ply(path):
    pcd = o3d.io.read_point_cloud(path)
    return np.array(pcd.points), np.array(pcd.colors)
```


### Save Point Cloud to .PLY files
Write a point cloud to .PLY file. Supports N\*3 and N\*6 point clouds, numpy array and torch tensor.
- A method using Open3D
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
- A method without using Open3D
    ```python
    def save_point_cloud_to_ply(points, colors, save_path='your_path_to_save.ply'):
        '''
        Save point cloud to ply file
        '''
        PLY_HEAD = f"ply\nformat ascii 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        file_sting = PLY_HEAD
        for i in range(len(points)):
            file_sting += f'{points[i][0]} {points[i][1]} {points[i][2]} {int(colors[i][0])} {int(colors[i][1])} {int(colors[i][2])}\n'
        f = open(save_path, 'w')
        f.write(file_sting)
        f.close()
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

### Point Cloud Sampling
Sample a point cloud to a fixed number of points. Supports N\*3 point clouds, numpy array and torch tensor. Return the corresponding indices of sampled point cloud.

- Random Sampling
  ```python
  def sample_point_cloud(point_cloud, num_samples):
    if isinstance(point_cloud, np.ndarray):
        # For numpy array
        indices = np.random.choice(point_cloud.shape[0], num_samples, replace=False)
        sampled_points = point_cloud[indices, :]
    elif isinstance(point_cloud, torch.Tensor):
        # For PyTorch tensor
        indices = torch.randperm(point_cloud.size(0))[:num_samples]
        sampled_points = point_cloud[indices, :]
    else:
        raise TypeError("Input must be either a numpy array or a torch tensor")

    return indices
  ```

- Farthest Point Sampling(FPS)

  ```python
    def farthest_point_sample(xyz, npoint, use_cuda=True):
    """
    Modified to support both numpy array and torch tensor.

    Input:
        xyz: pointcloud data, [B, N, 3], can be either numpy array or tensor
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    # Convert numpy array to PyTorch tensor if necessary
    if isinstance(xyz, np.ndarray):
        xyz = torch.from_numpy(xyz)

    # Ensure the tensor is on the correct device
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    xyz = xyz.to(device)

    B, N, C = xyz.shape

    if use_cuda and torch.cuda.is_available():
        print('Use pointnet2_cuda!')
        from pointnet2_ops.pointnet2_utils import furthest_point_sample as fps_cuda
        sampled_points_ids = fps_cuda(xyz, npoint)
    else:
        centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
        distance = torch.ones(B, N).to(device) * 1e10
        farthest = torch.randint(0, N, (B, ), dtype=torch.long).to(device)
        batch_indices = torch.arange(B, dtype=torch.long).to(device)
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid)**2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
        sampled_points_ids = centroids

    return sampled_points_ids
  ```

### RGBD to Point Cloud
Here we provide both step-by-step method and a one-line method to convert RGBD images to point clouds. The step-by-step method is more readable and easier to understand, while the one-line method is more efficient.

- Method using Numpy
    ```python
    import numpy as np

    def rgbd_to_point_cloud(rgb_image, depth_image, camera_intrinsics):
        # Assuming depth_image and rgb_image are numpy arrays of the same size
        # camera_intrinsics is a 3x3 matrix [fx, 0, cx; 0, fy, cy; 0, 0, 1]

        points = []
        colors = []

        height, width = depth_image.shape
        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]

        for v in range(height):
            for u in range(width):
                color = rgb_image[v, u]
                z = depth_image[v, u]

                if z == 0:  # Ignore invalid depth; Add your condition here
                    continue

                x = (u - cx) * z / fx
                y = (v - cy) * z / fy

                points.append([x, y, z])
                colors.append(color)

        return np.array(points), np.array(colors)

    # Usage
    # point_cloud, point_colors = rgbd_to_point_cloud(rgb_image, depth_image, camera_intrinsics)
    ```
- Method using Open3D
    ```python
    import open3d as o3d
    import numpy as np

    def rgbd_to_point_cloud(rgb_image_np, depth_image_np, camera_intrinsics):
        # Load RGB and Depth images
        # For this example, let's assume you have RGB and Depth images as numpy arrays
        # rgb_image_np = <your RGB image as numpy array>
        # depth_image_np = <your Depth image as numpy array>

        # Convert numpy arrays to Open3D images
        rgb_image_o3d = o3d.geometry.Image(rgb_image_np)
        depth_image_o3d = o3d.geometry.Image(depth_image_np)

        # Create an RGBD image from the RGB and Depth images
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image_o3d, depth_image_o3d)

        # Define camera intrinsic parameters
        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width=640, height=480, fx=fx, fy=fy, cx=cx, cy=cy)

        # Convert the RGBD image to a point cloud
        point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

        # Optionally visualize the point cloud
        # o3d.visualization.draw_geometries([point_cloud])

        return np.array(point_cloud.points), np.array(point_cloud.colors)
    ```