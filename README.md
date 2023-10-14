## TODO

- differentiable simulator

- soft body simulator

- more papers

- simulator details

- Double check: whether need physX version number/correct?

# Simulator Cheatsheet

A universal summarization of current robotics simulators

In this repo, we will summarize the cutting-edge `physics simulators` and related works for `robot learning` systems (and automonous driving if possible). 

What's more, we collect some related resources for robotics simulation development. In the section of `Notice for Simulation Development`, we summarize some common problems and errors for simulation development. We strongly recommended to read them before development and also when meet some errors.

## Simulators

|    Simulator     |# Physics Engine| # Rendering | # Sensor |# Parallelization|# Interface|
|:----------------:|:--------------:|:-----------:|:--------:|:---------------:|:---------:|
|     Isaac Sim    |                |             |          |                 |           |
|     Isaac Gym    | PhysX 5, Flex  |   Vulkan    |          |                 |           |
|      SAPIEN      | PhysX 4, Warp  |             |          |                 |           |
|      Pybullet    |                |             |          |                 |           |
|      MuJoCo      |                |             |          |                 |           |
|      Blender     |                |             |          |                 |           |
|      AI2-THOR    |                |             |          |                 |           |
|      RLBench     |                |             |          |                 |           |
|      Blender     |                |             |          |                 |           |

## Robot Learning Works with Simulators

- [CVPR2023] [GAPartNet: Cross-Category Domain-Generalizable Object Perception and Manipulation via Generalizable and Actionable Parts](https://github.com/PKU-EPIC/GAPartNet): SAPIEN; RGB-D PointCloud

- [CVPR2023] [PartManip: Learning Cross-Category Generalizable Part Manipulation Policy from Point Cloud Observations](https://github.com/PKU-EPIC/PartManip): Isaac Gym; RGB-D PointCloud

- [ICCV2023] [ARNOLD: A Benchmark for Language-Grounded Task Learning With Continuous States in Realistic 3D Scenes](https://arnold-benchmark.github.io/): IsaacSim; RGB-D PointCloud

- [CVPR2023] [UniDexGrasp: Universal Robotic Dexterous Grasping via Learning Diverse Proposal Generation and Goal-Conditioned Policy](https://pku-epic.github.io/UniDexGrasp/): IsaacGym; RGB-D PointCloud
- [ICCV2023] [UniDexGrasp++: Improving Dexterous Grasping Policy Learning via Geometry-aware Curriculum and Iterative Generalist-Specialist Learning](https://pku-epic.github.io/UniDexGrasp++/): IsaacGym; RGB-D PointCloud

- [ICRA2023] [RLAfford: End-to-End Affordance Learning for Robotic Manipulation](https://sites.google.com/view/rlafford/): IsaacGym; RGB-D PointCloud

- [ICRA2023] [GenDexGrasp: Generalizable Dexterous Grasping](https://sites.google.com/view/gendexgrasp/): IsaacGym; RGB-D PointCloud


## Robotic Manipulation Benchmark


## Notice for Simulator Development


## Credit

repo: [awesome-isaac-gym](https://github.com/wangcongrobot/awesome-isaac-gym)

