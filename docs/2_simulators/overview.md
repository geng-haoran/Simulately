---
sidebar_position: 1
---

# Overview

Here is a table including a brief summary for physics simulators:

|Simulator|Physics Engine|Rendering|Sensor|Dynamics|Parallelization|Vectorization|OpenSource|
|:-------:|:--------------:|:---------:|:------:|:--------:|:---------------:|:---------:|:----------:|
|IsaacSim|PhysX 5      |Rasterization; RayTracing; PathTracing| RGBD; |Rigid;Soft;Cloth;Fluid| ✔ |CPU;GPU| ✘ |
|IsaacGym|PhysX 5, Flex|Rasterization;| RGBD; |Rigid                 | ✔ |CPU;GPU| ✘ |
| SAPIEN  |PhysX 4, Warp|Rasterization; RayTracing⭐️;| RGBD; |Rigid;Soft;Fluid      | ✔ |CPU;   | ✔ |
| Pybullet|Bullet       |Rasterization;|       |Rigid                 | ✘ |       | ✔ |
| MuJoCo  | MuJoCo      |Rasterization;|       |Rigid;Soft;Cloth      | ✘ |       | ✔ |
|CoppeliaSim|MuJoCo;Bullet;ODE;Newton;Vortex|Rasterization; RayTracing🔶;|       |Rigid;Soft;Cloth      | ✘ |       | ✔ |

<!-- | Blender |             |                  |       |                      | ✘ |       | ✔ |
| AI2-THOR|             |                  |       |                      | ✘ |       | ✔ |
| RLBench |             |                  |       |                      | ✘ |       | ✔ |
| Habitat |             |                  |       |                      | ✘ |       | ✔ | -->

⭐️: RayTracing is currently not available in parallel gym-like envs.

🔶: Supported but Limited: CoppeliaSim has built-in functionality for simulating ray-tracing effects to a certain extent. For example, it can perform ray-casting operations, which are useful for sensor simulations and collision detections. However, this is not the same as full-fledged ray tracing for photorealistic rendering.