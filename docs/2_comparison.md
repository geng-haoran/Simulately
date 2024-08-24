---
sidebar_position: 2
title: Overall Comparison
---

# Overall Comparison

Here is a table including a brief summary for physics simulators:

|  Simulator                                        |           Physics Engine            |          Rendering           |                    Sensor🤨                       |        Dynamics        | GPU-accelerated Simulation  | Open-Source |
|:-------------------------------------------------:|:-----------------------------------:|:----------------------------:|:-------------------------------------------------:|:----------------------:|:---------------:|:----------:|
|[IsaacSim](https://developer.nvidia.com/isaac-sim) |               PhysX 5               |  Rasterization; RayTracing   |RGBD; Lidar; Force; Effort; IMU; Contact; Proximity| Rigid;Soft;Cloth;Fluid |        ✔        |     ✘      |
|[IsaacGym](https://developer.nvidia.com/isaac-gym) |            PhysX 5, Flex            |        Rasterization;        |RGBD; Force; Contact;                              |    Rigid;Soft;Cloth    |        ✔        |     ✘      |
|[SAPIEN](https://sapien.ucsd.edu/)                 |            PhysX 5, Warp            | Rasterization; RayTracing⭐️; |RGBD; Force; Contact;                              |    Rigid;Soft;Fluid    |        ✔        |     ✔      |
|[Pybullet](https://pybullet.org/wordpress/)        |               Bullet                |        Rasterization;        |RGBD; Force; IMU; Tactile;                         |    Rigid;Soft;Cloth    |        ✘        |     ✔      |
|[MuJoCo](https://mujoco.org/)                      |               MuJoCo                |        Rasterization;        |RGBD; Force; IMU; Tactile;                         |    Rigid;Soft;Cloth    |        ✔💡      |     ✔      |
|[CoppeliaSim](https://www.coppeliarobotics.com/)   | MuJoCo; Bullet; ODE; Newton; Vortex | Rasterization; RayTracing🔶; |RGBD; Force; Contact;                              |    Rigid;Soft;Cloth    |        ✘        |     ✔      |
|[Gazebo](https://gazebosim.org/home)               |     Bullet; ODE; DART; Simbody      |        Rasterization;        |RGBD; Lidar; Force; IMU;                           |    Rigid;Soft;Cloth    |        ✘        |     ✔      |

🤨: Check more information about sensors: [IsaacSim](https://docs.omniverse.nvidia.com/isaacsim/latest/features/sensors_simulation/index.html), IsaacGym(Doc), [SAPIEN](https://sapien.ucsd.edu/docs/latest/index.html), Pybullet, [MuJoCo](https://mujoco.readthedocs.io/en/stable/overview.html?highlight=sensor#sensor), CoppeliaSim, Gazebo

⭐️: RayTracing is currently not available in parallel gym-like envs.

🔶: Supported but Limited: CoppeliaSim has built-in functionality for simulating ray-tracing effects to a certain extent.
For example, it can perform ray-casting operations, which are useful for sensor simulations and collision detections.
However, this is not the same as full-fledged ray tracing for photorealistic rendering.

💡: [MJX](https://mujoco.readthedocs.io/en/stable/mjx.html) allows MuJoCo to run on compute hardware supported 
by the XLA compiler via the JAX framework. MJX runs on a all platforms supported by JAX: Nvidia and 
AMD GPUs, Apple Silicon, and Google Cloud TPUs.

## Rendering

### Comparison of Rendering Speed

We build up the same environment with all the simulators. Here are the rendered images and rendering speed🟡.

Notice:
- The rendering results are evaluated on two machines and we report two evaluation results.
- The rendering code can be found under `code/rendering` folder, see [here](https://github.com/geng-haoran/Simulately/tree/code/rendering) for more details.
- We are only providing as comprehensive a result as possible. The results are not completely authoritative and are not in any way inductive. The test code is all public. If you have any questions about the test, you can first take a look at the corresponding code, and then try to discuss it with us openly in the issue.
- Everyone is welcome to contribute the test code and results to this project.

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

#### Evaluation 1 (RTX4090 ⭐️):

|Simulator|SAPIEN (Rasterization)|IsaacGym (Rasterization)|IsaacSim(Ray Tracing)   |Pybullet (Rasterization)|MuJoCo|
|:------------------:|:------:|:--------:|:--------:|:-----------------------------:|:------:|
|     RGB @ FPS      | 1175.74 | 1917.32🤔 | 182.33 |258.04 (RTX 3090)|2262.59 |
|    Depth @ FPS     | 1467.48 | 1917.32🤔 | 156.31 |- (update soon)|1039.91 |
| Segmentation @ FPS | 1153.03 | 1917.32🤔 | 141.08  |- (update soon)|320.86  |

⭐️: The rendering code can be found under `code/rendering` folder, see github repo for more details. The number reported here is run with 13th Gen Intel Core i9-13900K and RTX 4090.

🤔: In IsaacGym, we cannot decouple depth, segmentation, and RGB rendering. Therefore, we report average FPS across all three rendering modes.

#### Evaluation 2 (A100 🟡):

|Simulator|SAPIEN (Rasterization)|IsaacGym (Rasterization)|IsaacSim(Ray Tracing)  |Pybullet (Rasterization)|MuJoCo|
|:------------------:|:------:|:--------:|:--------:|:----------------------------:|:------:|
|     RGB @ FPS      | 228.23 |  789.25🤔  |102.43    |258.04 (RTX 3090)|88.68  |
|    Depth @ FPS     | 280.56 |  789.25🤔  |92.43    |- (update soon)|288.61  |
| Segmentation @ FPS | 261.06 |  789.25🤔  |97.43    |- (update soon)|119.74  |

🟡: The rendering code can be found under `code/rendering` folder, see github repo for more details. The number reported here is ran with AMD EPYC 7742 64-Core Processor and A100(80G).

🤔: In IsaacGym, we cannot decouple depth, segmentation, and RGB rendering. Therefore, we report average FPS across all three rendering modes.

<!-- ### Comparison of Rendering Performance

pending...

# Parallelization

pending... -->

## Popularity

|     Simulator            | SAPIEN | IsaacGym | IsaacSim | Pybullet | MuJoCo | CoppeliaSim | Gazebo |
|:------------------------:|:------:|:--------:|:--------:|:--------:|:------:|:-----------:|:------:|
|     Github Star🟡         | 274    |    /     |     /    |   11.4k  |  6.6k  |      88     |  1.1k🔶 |
|Citation  (Google Scholar⭐️) | 302    |  375     |    /     |    1942   |  4884  |     1786    |  3949  |

🟡 Last Update: 2023.12.20

⭐️ Last Update: 2023.12.20

🔶 gazebo-classic
