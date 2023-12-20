---
sidebar_position: 3
---

<h2 align="center">
  <b>Isaac Gym: High Performance GPU-Based Physics Simulation For Robot Learning</b>

<div align="center">
    <a href="https://developer.nvidia.com/isaac-gym" target="_blank"><img src="https://img.shields.io/badge/Website-IsaacGym-red"></img></a>
    &nbsp;
    <a href="https://arxiv.org/abs/2108.10470" target="_blank"><img src="https://img.shields.io/badge/Paper-arXiv-green" alt="Paper arXiv"></img></a>
    &nbsp;
    <a href="https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/isaac-gym/322" target="_blank"><img src="https://img.shields.io/badge/Forum-IsaacGym-yellow" alt="Project Page"></img></a>
</div>
</h2>

![IsaacGym](imgs/IsaacGym.jpg)
> NVIDIA’s physics simulation environment for reinforcement learning research.

## Official Materials
- [Website](https://developer.nvidia.com/isaac-gym)
- [Paper](https://arxiv.org/abs/2108.10470)
- [Forum](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/isaac-gym/322)
- [OmniIsaacGymEnvs](https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs)

## Related Materials
- [Awesome NVIDIA Isaac Gym](https://github.com/wangcongrobot/awesome-isaac-gym)

## Related Benchmark

### IsaacGymEnvs
This repository contains example RL environments for the NVIDIA Isaac Gym high performance environments described in NVIDIA's NeurIPS 2021 Datasets and Benchmarks paper.

![](imgs/isaacgym/isaacgymenvs.gif)

- [Website](https://developer.nvidia.com/isaac-gym)
- [Code](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs)
- [Paper](https://arxiv.org/abs/2108.10470)

### Bi-DexHands
Bi-DexHands provides a collection of bimanual dexterous manipulations tasks and reinforcement learning algorithms. Reaching human-level sophistication of hand dexterity and bimanual coordination remains an open challenge for modern robotics researchers.

![](imgs/isaacgym/quick_demo3.gif)

- [Website](https://bi-dexhands.ai)
- [Code](https://github.com/PKU-MARL/DexterousHands)
- [Paper](https://arxiv.org/abs/2206.08686)

### DexPBT
DexPBT implement challenging tasks for one- or two-armed robots equipped with multi-fingered hand end-effectors, including regrasping, grasp-and-throw, and object reorientation. And introduce a decentralized Population-Based Training (PBT) algorithm that massively amplify the exploration capabilities of deep reinforcement learning.

![](imgs/isaacgym/dexpbt.gif)

- [Website](https://sites.google.com/view/dexpbt)
- [Code](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs)
- [Paper](https://arxiv.org/abs/2305.12127)

### TimeChamber
TimeChamber is a large scale self-play framework running on parallel simulation. Running self-play algorithms always need lots of hardware resources, especially on 3D physically simulated environments. TimeChamber provide a self-play framework that can achieve fast training and evaluation with ONLY ONE GPU.

![](imgs/isaacgym/humanoid_strike.gif)

- [Website](https://github.com/inspirai/TimeChamber)
- [Code](https://github.com/inspirai/TimeChamber)

## Related Projects
- SIGGRAPH2023: [CALM: Conditional Adversarial Latent Models for Directable Virtual Characters](https://xbpeng.github.io/projects/CALM/index.html): IsaacGym;
- ICCV2023: [UniDexGrasp++: Improving Dexterous Grasping Policy Learning via Geometry-aware Curriculum and Iterative Generalist-Specialist Learning](https://pku-epic.github.io/UniDexGrasp++/): IsaacGym; RGB-D PointCloud
- CoRL2023: [Dynamic Handover: Throw and Catch with Bimanual Hands](https://binghao-huang.github.io/dynamic_handover/): IsaacGym; RGB
- CoRL2023: [Sequential Dexterity: Chaining Dexterous Policies for Long-Horizon Manipulation](https://sequential-dexterity.github.io/): IsaacGym; RGB-D; PointCloud
- CORL2023: [Curiosity-Driven Learning of Joint Locomotion and Manipulation Tasks](https://openreview.net/forum?id=QG_ERxtDAP-&referrer=%5Bthe%20profile%20of%20Clemens%20Schwarke%5D(%2Fprofile%3Fid%3D~Clemens_Schwarke1)): IsaacGym; RL
- CoRL2023: [General In-Hand Object Rotation with Vision and Touch](https://haozhi.io/rotateit/): IsaacGym; RGB-D
- CoRL2023: [Fleet-DAgger: Interactive Robot Fleet Learning with Scalable Human Supervision](https://tinyurl.com/fleet-dagger)
- CVPR2023: [UniDexGrasp: Universal Robotic Dexterous Grasping via Learning Diverse Proposal Generation and Goal-Conditioned Policy](https://pku-epic.github.io/UniDexGrasp/): IsaacGym; RGB-D PointCloud
- CVPR2023: [PartManip: Learning Cross-Category Generalizable Part Manipulation Policy from Point Cloud Observations](https://github.com/PKU-EPIC/PartManip): Isaac Gym; RGB-D PointCloud
- ICRA2023: [RLAfford: Official Implementation of "RLAfford: End-to-end Affordance Learning with Reinforcement Learning](https://github.com/hyperplane-lab/RLAfford): IsaacGym
- ICRA2023: [GenDexGrasp: Generalizable Dexterous Grasping](https://sites.google.com/view/gendexgrasp/): IsaacGym; RGB-D; PointCloud
- ICRA2023: [RLAfford: End-to-End Affordance Learning for Robotic Manipulation](https://sites.google.com/view/rlafford/): IsaacGym; RGB-D; PointCloud
- ICRA2023: [ViNL: Visual Navigation and Locomotion Over Obstacles](http://www.joannetruong.com/projects/vinl.html): IsaacGym;
- RSS2023: [AnyTeleop: A General Vision-Based Dexterous Robot Arm-Hand Teleoperation System](http://anyteleop.com/): IsaacGym
- RSS2023: [DexPBT: Scaling up Dexterous Manipulation for Hand-Arm Systems with Population Based Training](https://sites.google.com/view/dexpbt): IsaacGym
- RSS2023: [Rotating without Seeing: Towards In-hand Dexterity through Touch](https://touchdexterity.github.io/): IsaacGym
- ScienceRobotics2023: [Visual Dexterity: In-Hand Reorientation of Novel and Complex Object Shapes](https://taochenshh.github.io/projects/visual-dexterity): IsaacGym; RGBD; PointCloud
- ICML2023: [On Pre-Training for Visuo-Motor Control: Revisiting a Learning-from-Scratch Baseline](https://arxiv.org/abs/2212.05749): IsaacGym; RGB
- ICML2023: [Parallel Q-Learning: Scaling Off-policy Reinforcement Learning](https://arxiv.org/abs/2307.12983): IsaacGym;
- SIGGRAPHAsia2022: [PADL: Language-Directed Physics-Based Character Control](https://xbpeng.github.io/projects/PADL/index.html): IsaacGym;
- ICRA2023: [Real2Sim2Real: Self-Supervised Learning of Physical Single-Step Dynamic Actions for Planar Robot Casting](https://tinyurl.com/robotcast): Isaac Gym;
- CoRL2022: [In-Hand Object Rotation via Rapid Motor Adaptation](https://haozhi.io/hora/): IsaacGym
- CoRL2022: [Legged Locomotion in Challenging Terrains using Egocentric Vision](https://vision-locomotion.github.io/): IsaacGym
- NIPS2022: [Towards Human-Level Bimanual Dexterous Manipulation with Reinforcement Learning](https://bi-dexhands.ai/): IsaacGym; RGB-D; PointCloud
- ICRA2022: [Data-Driven Operational Space Control for Adaptative and Robust Robot Manipulation](https://github.com/nvlabs/oscar): IsaacGym
- RSS2022: [Rapid Locomotion via Reinforcement Learning](https://agility.csail.mit.edu/): IsaacGym
- RSS2022: [Factory: Fast contact for robotic assembly](https://sites.google.com/nvidia.com/factory): IsaacGym
- SIGGRAPH2022: [ASE: Large-scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters](https://nv-tlabs.github.io/ASE/): IsaacGym
- CoRL2021: [STORM: An Integrated Framework for Fast Joint-Space Model-Predictive Control for Reactive Manipulation](https://github.com/NVlabs/storm): IsaacGym
- ICRA2021: [Causal Reasoning in Simulationfor Structure and Transfer Learning of Robot Manipulation Policies](https://sites.google.com/view/crest-causal-struct-xfer-manip): IsaacGym
- ICRA2021: [In-Hand Object Pose Tracking via Contact Feedback and GPU-Accelerated Robotic Simulation](https://sites.google.com/view/in-hand-object-pose-tracking/): IsaacGym
- IROS2021: [Reactive Long Horizon Task Execution via Visual Skill and Precondition Models](https://arxiv.org/pdf/2011.08694.pdf): IsaacGym
- ICRA2021: [Sim-to-Real for Robotic Tactile Sensing via Physics-Based Simulation and Learned Latent Projections](https://arxiv.org/pdf/2103.16747.pdf): IsaacGym
- RSS2021_VLRR: [A Simple Method for Complex In-Hand Manipulation](https://sites.google.com/view/in-hand-reorientation): IsaacGym
- CoRL2021: [Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning](https://leggedrobotics.github.io/legged_gym/): IsaacGym
- ICRA2021: [Dynamics Randomization Revisited:A Case Study for Quadrupedal Locomotion](https://www.pair.toronto.edu/understanding-dr/): IsaacGym
- NIPS2021: [Isaac Gym: High Performance GPU-Based Physics Simulation For Robot Learning](https://sites.google.com/view/isaacgym-nvidia): IsaacGym
- RAL2021: [Learning a State Representation and Navigation in Cluttered and Dynamic Environments](https://arxiv.org/pdf/2103.04351.pdf): IsaacGym
- CoRL2020: [Learning to Compose Hierarchical Object-Centric Controllers for Robotic Manipulation](https://sites.google.com/view/compositional-object-control/): IsaacGym
- CoRL2020: [Learning a Contact-Adaptive Controller for Robust, Efficient Legged Locomotion](https://sites.google.com/view/learn-contact-controller/home): IsaacGym
- RSS2020: [Learning Active Task-Oriented Exploration Policies for Bridging the Sim-to-Real Gap](https://sites.google.com/view/task-oriented-exploration/): IsaacGym
- ICRA2019: [Closing the Sim-to-Real Loop: Adapting Simulation Randomization with Real World Experience](https://sites.google.com/view/simopt): IsaacGym
- CoRL2018: [GPU-Accelerated Robotics Simulation for Distributed Reinforcement Learning](https://sites.google.com/view/accelerated-gpu-simulation/home): IsaacGym
