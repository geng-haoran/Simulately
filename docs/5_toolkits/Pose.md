---
sidebar_position: 9
---

# 3D Transformation

> **Tutorial on Handling SO(3) and SE(3) Transformations for Simulation**

This tutorial assumes the `numpy` and `transforms3d` Python packages have been installed.

### Inverting Transformation

For an `SO(3)` rotation numpy matrix `R`, you can get the inverse simply as `R.T`. For quaternion `q=(w,x,y,z)`, the inverse
is `(-w,x,y,z)`.

For an `SE(3)` transformation matrix `T`, `np.linalg.inv(T)` can be used to compute the inverse transformation. However,
given its complexity of `O(n^3)`, this isn't efficient. A more efficient method to compute the inverse of the
transformation is:

```python
def inverse_pose(pose: np.ndarray):
  inv_pose = np.eye(4, dtype=pose.dtype)
  inv_pose[:3, :3] = pose[:3, :3].T
  inv_pose[:3, 3] = -pose[:3, :3].T @ pose[:3, 3]
  return inv_pose
```

### SO(3) Rotation Linear Interpolation

```python
def interpolate_rotation(mat1: np.ndarray, mat2: np.ndarray, mat1_weight: float):
  if mat1_weight < 0 or mat1_weight > 1:
    raise ValueError(f"Weight of rotation matrix should be 0-1, but given {mat1_weight}")

  relative_rot = mat1.T @ mat2
  # For numerical stability, first convert to quaternion and then to axis-angle for a not-perfect rotation matrix
  axis, angle = transforms3d.quaternions.quat2axangle(transforms3d.quaternions.mat2quat(relative_rot))

  inter_angle = (1 - mat1_weight) * angle
  inter_rot = transforms3d.axangles.axangle2mat(axis, inter_angle)
```

### SE(3) Transformation Linear Interpolation

```python
def interpolate_transformation(mat1: np.ndarray, mat2: np.ndarray, mat1_weight: float):
  if mat1_weight < 0 or mat1_weight > 1:
    raise ValueError(f"Weight of rotation matrix should be 0-1, but given {mat1_weight}")

  result_pose = np.eye(4)
  rot1 = mat1[:3, :3]
  rot2 = mat2[:3, :3]
  inter_rot = interpolate_rotation(rot1, rot2, mat1_weight)
  inter_pos = mat1[:3, 3] * mat1_weight + mat2[:3, 3] * (1 - mat1_weight)
  result_pose[:3, :3] = inter_rot
  result_pose[:3, 3] = inter_pos
  return result_pose
```

### Creating a Skew Matrix from a Vector

```python
def skew_matrix(vec):
    return np.array([[0, -vec[2], vec[1]],
                     [vec[2], 0, -vec[0]],
                     [-vec[1], vec[0], 0]])
```

### Converting SE(3) Pose to Screw Motion

```python
def pose2se3(pose: np.ndarray):
  axis, theta = transforms3d.axangles.mat2axangle(pose[:3, :3])
  skew = skew_matrix(axis)
  inv_left_jacobian = np.eye(3, dtype=np.float) / theta - 0.5 * skew + (
    1.0 / theta - 0.5 / np.tan(theta / 2)) * skew @ skew
  v = inv_left_jacobian @ pose[:3, 3]
  return np.concatenate([v, axis]), theta
```
