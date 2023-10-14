# Simulately

A universal summarization of physics simulators for robotic learning research.

## To-Dos for the First Release

- [ ] Website deployment with Docusaurus
  - [ ] Setup automatic compile and deployment pipeline
  - [ ] Use [Dify](https://cloud.dify.ai) to add an AI assistant for Q&A
- [ ] Include more simulators
    - [ ] Differentiable simulation
    - [ ] Soft body simulation
- [ ] Complete the details for simulator
    - [ ] Double check: whether need physX version number/correct?
    - [ ] Double check: whether need physX version number/correct?
- [ ] Include some snippets as initialization
- [ ] Include some FAQs as initialization
- [ ] Include more related work
- [ ] Contribution instructions

> Things below will be moved to sub-pages after deploying Docusaurus.

## About Simulately

**Simulately** is a project where we gather useful information of **physics simulator** for cutting-edge robot learning research. It includes but is not limited to:

- Summary and comparisons of [Simulators](/simulators) for robotic learning research
- [Handy Snippets](/snippets) that you can use by simply copy-and-paste
- [FAQs in Dev](/faq) may help you solve the problems occurred in development
- [Related work](/related-work) presents latest advancements in robotic learning research
- [Contribute](/contribute) provides instructions on contributing to this project

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


## Acknowledgement

Project website powered by:

- [Cloudflare](https://cloudflare.com/)
- [Docusaurus](https://docusaurus.io/)

Repo maintained by:

- [Haoran Geng](https://geng-haoran.github.io/)
- [Yuyang Li](https://yuyangli.com/)

Project inspired by:

- [awesome-isaac-gym](https://github.com/wangcongrobot/awesome-isaac-gym)
- [THU Wiki](https://thu.wiki/)
