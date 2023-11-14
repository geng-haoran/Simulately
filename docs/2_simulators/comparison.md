---
sidebar_position: 1
---

# Overall Comparison

Here is a table including a brief summary for physics simulators:

|Simulator|Physics Engine|Rendering                                          |Sensor(CHECK) |Dynamics  |Parallelization    |Vectorization|OpenSource|
|:-------:|:-----------:|:--------------------------------------------------:|:------------:|:--------:|:-----------------:|:-----------:|:--------:|
|IsaacSim |PhysX 5      |Rasterization; RayTracing; PathTracing              | RGBD;        |Rigid;Soft;Cloth;Fluid| ✔ |GPU🟡                   | ✘ |
|IsaacGym |PhysX 5, Flex|Rasterization;                                      | RGBD; Force; |Rigid                 | ✔ |CPU;GPU                 | ✘ |
| SAPIEN  |PhysX 4, Warp|Rasterization; RayTracing⭐️;                        | RGBD; Force; |Rigid;Soft;Fluid      | ✔ |CPU;                    | ✔ |
| Pybullet|Bullet       |Rasterization;                                      | RGBD;        |Rigid;Soft;Cloth      | ✘ |CPU;                    | ✔ |
| MuJoCo  |MuJoCo       |Rasterization;                                      | RGBD;        |Rigid;Soft;Cloth      | ✘ |CPU;                    | ✔ |
|CoppeliaSim|MuJoCo; Bullet; ODE; Newton; Vortex|Rasterization; RayTracing🔶;| RGBD;        |Rigid;Soft;Cloth      | ✘ |CPU;                    | ✔ |
|Gazebo   |Bullet; ODE; DART; Simbody|Rasterization;                         | RGBD;        |Rigid;Soft;Cloth      | ✘ |CPU;                    | ✔ |
|Blender  |Bullet       |Rasterization; PathTracing;                         | RGBD;        |Rigid;Soft;Cloth;Fluid| ✘ |CPU;                    | ✔ |

🟡: GPU only, RTX series is required.

⭐️: RayTracing is currently not available in parallel gym-like envs.

🔶: Supported but Limited: CoppeliaSim has built-in functionality for simulating ray-tracing effects to a certain extent. For example, it can perform ray-casting operations, which are useful for sensor simulations and collision detections. However, this is not the same as full-fledged ray tracing for photorealistic rendering.


# Rendering

# Parallelization

# Vectorization