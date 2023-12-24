---
sidebar_position: 2
title: Overall Comparison
---

# Overall Comparison

Here is a table including a brief summary for physics simulators:

|  Simulator  |           Physics Engine            |          Rendering           | Sensor(CHECK) |        Dynamics        | Parallelization | Vectorization | OpenSource |
|:-----------:|:-----------------------------------:|:----------------------------:|:-------------:|:----------------------:|:---------------:|:-------------:|:----------:|
|  IsaacSim   |               PhysX 5               |  Rasterization; RayTracing   |     RGBD;     | Rigid;Soft;Cloth;Fluid |        ✔        |     GPU🟡     |     ✘      |
|  IsaacGym   |            PhysX 5, Flex            |        Rasterization;        |     RGBD;     |    Rigid;Soft;Cloth    |        ✔        |    CPU;GPU    |     ✘      |
|   SAPIEN    |            PhysX 5, Warp            | Rasterization; RayTracing⭐️; |     RGBD;     |    Rigid;Soft;Fluid    |        ✔        |     CPU;      |     ✔      |
|  Pybullet   |               Bullet                |        Rasterization;        |     RGBD;     |    Rigid;Soft;Cloth    |        ✘        |     CPU;      |     ✔      |
|   MuJoCo    |               MuJoCo                |        Rasterization;        |     RGBD;     |    Rigid;Soft;Cloth    |        ✘        |     CPU;      |     ✔      |
| CoppeliaSim | MuJoCo; Bullet; ODE; Newton; Vortex | Rasterization; RayTracing🔶; |     RGBD;     |    Rigid;Soft;Cloth    |        ✘        |     CPU;      |     ✔      |
|   Gazebo    |     Bullet; ODE; DART; Simbody      |        Rasterization;        |     RGBD;     |    Rigid;Soft;Cloth    |        ✘        |     CPU;      |     ✔      |
|   Blender   |               Bullet                |  Rasterization; RayTracing   |     RGBD;     | Rigid;Soft;Cloth;Fluid |        ✘        |     CPU;      |     ✔      |

🟡: GPU only, RTX series is required.

⭐️: RayTracing is currently not available in parallel gym-like envs.

🔶: Supported but Limited: CoppeliaSim has built-in functionality for simulating ray-tracing effects to a certain extent.
For example, it can perform ray-casting operations, which are useful for sensor simulations and collision detections.
However, this is not the same as full-fledged ray tracing for photorealistic rendering.

# Rendering

### Comparison of Rendering Speed

We build up the same environment with all the simulators. Here are the rendered images and rendering speed🟡.

<div style={{ display: 'flex', justifyContent: 'space-between' }}>
  <div style={{ textAlign: 'center', marginRight: '10px' }}>
    <img src="/img/comparison/rendering/sapien/color.png" alt="SAPIEN Rendering" style={{ width: 'auto', maxHeight: '200px' }} />
    <p>SAPIEN</p>
  </div>
  <div style={{ textAlign: 'center', marginRight: '10px' }}>
    <img src="/img/comparison/rendering/isaacgym/color.png" alt="IsaacGym Rendering" style={{ width: 'auto', maxHeight: '200px' }} />
    <p>IsaacGym</p>
  </div>
  <div style={{ textAlign: 'center' }}>
    <img src="/img/comparison/rendering/pybullet/color.png" alt="Pybullet Rendering" style={{ width: 'auto', maxHeight: '200px' }} />
    <p>Pybullet</p>
  </div>
</div>

Evaluation 1 🟡:

|     Simulator      | SAPIEN | IsaacGym | IsaacSim | Pybullet                     | MuJoCo | CoppeliaSim | Gazebo |
|:------------------:|:------:|:--------:|:--------:|:----------------------------:|:------:|:-----------:|:------:|
|     RGB @ FPS      | 292.16 |  785.30  |          |22.49(OpenGL) 7.06(TinyRender)|        |             |        |
|    Depth @ FPS     | 260.03 |  788.34  |          |                              |        |             |        |
| Segmentation @ FPS | 279.87 |  800.20  |          |                              |        |             |        |


 found under `code/rendering` folder, see github repo for more details. The number reported here is ran with AMD EPYC 7742 64-Core Processor and A100(80G).


Evaluation 1 🟡:

|     Simulator      | SAPIEN | IsaacGym | IsaacSim | Pybullet | MuJoCo | CoppeliaSim | Gazebo |
|:------------------:|:------:|:--------:|:--------:|:--------:|:------:|:-----------:|:------:|
|     RGB @ FPS      | 742.66 |  1849.71 |          |29.50(OpenGL) 13.68(TinyRender)|        |             |        |
|    Depth @ FPS     | 742.66 |  1849.71 |          |          |        |             |        |
| Segmentation @ FPS | 742.66 |  1849.71 |          |          |        |             |        |


### Comparison of Rendering Performance

pending...

# Parallelization

pending...

# Popularity

|     Simulator            | SAPIEN | IsaacGym | IsaacSim | Pybullet | MuJoCo | CoppeliaSim | Gazebo |
|:------------------------:|:------:|:--------:|:--------:|:--------:|:------:|:-----------:|:------:|
|     Github Star🟡         | 274    |    /     |     /    |   11.4k  |  6.6k  |      88     |  1.1k🔶 |
|Citation  (Google Scholar⭐️) | 302    |  375     |    /     |    /     |  4884  |     1786    |  3949  |

🟡 Last Update: 2023.12.20

⭐️ Last Update: 2023.12.20

🔶 gazebo-classic
