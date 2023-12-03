---
sidebar_position: 9
---

# SE(3) Pose

> **Tutorial for Handling SO(3) and SE(3) Transformation in Simulation**


This tutorial assume that we have already installed the following python pacakge `numpy` and `transforms3d`.

### Inversion

For `SO(3)` rotation numpy matrix `R`, the inversion is simply `R.T`. For quaternion `q`, the inverion is `-q`.

For `SE(3)` transformation matrix `T`, we can use `np.linalg.inv(T)` to compute the inverse transformation. However,
this is a `O(n^3)` complexity, which is not efficient. A better way to compute the inversion of transformation is as
follow:

```python
def inverse_pose(pose: np.ndarray):
  inv_pose = np.eye(4, dtype=pose.dtype)
  inv_pose[:3, :3] = pose[:3, :3].T
  inv_pose[:3, 3] = -pose[:3, :3].T @ pose[:3, 3]
  return inv_pose
```

### Linear Interpolation of SO(3) Rotation


```python

def interpolate_rotation(mat1: np.ndarray, mat2: np.ndarray, mat1_weight: float):
  if mat1_weight < 0 or mat1_weight > 1:
    raise ValueError(f"Weight of rotation matrix should be 0-1, but given {mat1_weight}")

  relative_rot = mat1.T @ mat2
  # For numerical stability, first convert to quaternion and then to axis-angel for not-perfect rotation matrix
  axis, angle = transforms3d.quaternions.quat2axangle(transforms3d.quaternions.mat2quat(relative_rot))

  inter_angle = (1 - mat1_weight) * angle
  inter_rot = transforms3d.axangles.axangle2mat(axis, inter_angle)
```

### Linear Interpolation of SE(3) Transformation

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
