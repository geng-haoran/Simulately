---
sidebar_position: 1
---

# Overview

Here is a table including a brief summary for physics simulators:

|Simulator|Physics Engine|Rendering|Sensor|Dynamics|Parallelization|Vectorization|OpenSource|
|:-------:|:--------------:|:---------:|:------:|:--------:|:---------------:|:---------:|:----------:|
|Isaac Sim|PhysX 5      | PhotoRealism     | RGBD; |Rigid;Soft;Cloth;Fluid| ✔ |CPU;GPU| ✘ |
|Isaac Gym|PhysX 5, Flex|   Vulkan         | RGBD; |Rigid                 | ✔ |CPU;GPU| ✘ |
| SAPIEN  |PhysX 4, Warp|Vulkan+RayTracing*| RGBD; |Rigid;Soft;Fluid      | ✔ |CPU;   | ✔ |
| Pybullet|Bullet       |                  |       |Rigid                 | ✘ |       | ✔ |
| MuJoCo  | MuJoCo      |    OpenGL        |       |Rigid;Soft;Cloth      | ✘ |       | ✔ |

<!-- | Blender |             |                  |       |                      | ✘ |       | ✔ |
| AI2-THOR|             |                  |       |                      | ✘ |       | ✔ |
| RLBench |             |                  |       |                      | ✘ |       | ✔ |
| Habitat |             |                  |       |                      | ✘ |       | ✔ | -->

* RayTracing is currently not available in parallel gym-like envs.