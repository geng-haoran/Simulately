---
sidebar_position: 1
---

# Soft Body Simulators

## Overview

- [**SoftGym**](#softgym): a set of benchmark environments for deformable object manipulation including tasks involving fluid, cloth and rope, simulated using FleX.
- [**FluidLab**](#fluidlab): a fully differentiable and multi-material physics simulation platform, supporting **rigid**, **elastic**, **plastic** materials, **inviscid** and **viscous liquid**, and **gaseous phenomena** such as smoke. 
- [**PlasticineLab**](#plasticinelab): a differentiable **soft-body** manipulation simulator and benchmark.
- [**RoboNinja**](#roboninja): a differentiable simulation system for cutting multi-material objects.
- [**DiSECt**](#disect): a differentiable simulation engine for autonomous **robotic cutting**.
- [**ThinShellLab**](#thinshelllab): a differentiable simulator for manipulating **thin-shell materials**, such as cloths and papers.
- [**DaxBench**](#daxbench): a **deformable object manipulation** benchmark with differentiable physics.
- [**SoftMAC**](#softmac): a differentiable soft body simulation with forecast-based contact model, coupling with articulated rigid bodies and clothes.
---

## SoftGym

> **SoftGym** is a set of benchmark environments for deformable object manipulation including tasks involving fluid, cloth and rope, simulated using FleX.

<details open>
<h2 align="center">
  <b>SoftGym: Benchmarking Deep Reinforcement Learning for Deformable Object Manipulation</b>

<div align="center">
    <a href="https://sites.google.com/view/softgym" target="_blank"><img src="https://img.shields.io/badge/Website-SoftGym-red"></img></a>
    &nbsp;
    <a href="http://arxiv.org/abs/2011.07215" target="_blank"><img src="https://img.shields.io/badge/Paper-ArXiv-green"></img></a>
    &nbsp;
    <a href="https://github.com/Xingyu-Lin/softgym" target="_blank"><img src="https://img.shields.io/badge/Source-Code-purple"></img></a>
</div>
</h2>

<div align="center">
<div style={{ textAlign: 'center' }}>
    <img src="https://github.com/Xingyu-Lin/softgym/raw/master/examples/ClothFold.gif"  />
    <img src="https://github.com/Xingyu-Lin/softgym/raw/master/examples/RopeFlatten.gif"  />
    <img src="https://github.com/Xingyu-Lin/softgym/raw/master/examples/PourWater.gif"  />
  </div>
</div>

SoftGym is a set of benchmark environments for deformable object manipulation including tasks involving fluid, cloth and rope. It is built on top of the Nvidia FleX simulator and has standard Gym API for interaction with RL agents.

</details>

## FluidLab

> **FluidLab** and **FluidEngine** is a fully differentiable and multi-material physics simulation platform, supporting **rigid**, **elastic**, **plastic** materials, **inviscid** and **viscous liquid**, and **gaseous phenomena** such as smoke.

<details open> <summary>Details</summary>
<h2 align="center">
  <b>FluidLab: A Differentiable Environment for Benchmarking Complex Fluid Manipulation</b>

<div align="center">
    <a href="https://fluidlab2023.github.io/" target="_blank"><img src="https://img.shields.io/badge/Website-FluidLab-red"></img></a>
    &nbsp;
    <a href="https://arxiv.org/abs/2303.02346" target="_blank"><img src="https://img.shields.io/badge/Paper-ArXiv-green"></img></a>
    &nbsp;
    <a href="https://github.com/zhouxian/FluidLab" target="_blank"><img src="https://img.shields.io/badge/Source-Code-purple"></img></a>
</div>
</h2>

<div align="center">
<video width="800" height="400" controls>
  <source src="https://fluidlab2023.github.io/static/videos/tasks.m4v" type="video/mp4"></source>
</video>
</div>

 FluidLab is a simulation environment with a diverse set of manipulation tasks involving complex fluid dynamics. These tasks address interactions between solid and fluid as well as among multiple fluids. FluidLab is powered by its underlying physics engine, FluidEngine, providing GPU-accelerated simulations and gradient calculations for various material types and their couplings, extending the scope of the existing differentiable simulation engines.

</details>

## PlasticineLab

> **PlasticineLab** is a differentiable **soft-body** manipulation simulator and benchmark.

<details open> <summary>Details</summary>
<h2 align="center">
  <b>PlasticineLab: A Soft-Body Manipulation Benchmark with Differentiable Physics</b>

<div align="center">
    <a href="https://plasticinelab.csail.mit.edu/" target="_blank"><img src="https://img.shields.io/badge/Website-PlasticineLab-red"></img></a>
    &nbsp;
    <a href="https://arxiv.org/abs/2104.03311" target="_blank"><img src="https://img.shields.io/badge/Paper-ArXiv-green"></img></a>
    &nbsp;
    <a href="https://github.com/hzaskywalker/PlasticineLab" target="_blank"><img src="https://img.shields.io/badge/Source-Code-purple"></img></a>
</div>
</h2>

<div align="center">
<div style={{ textAlign: 'center' }}>
    <img src="https://plasticinelab.csail.mit.edu/plb_site_files/tasks.jpg"  />
  </div>
</div>
PlasticineLab is a differentiable physics benchmark including a diverse collection of soft body manipulation tasks. In each task, the agent uses manipulators to deform the plasticine into a desired configuration. The underlying physics engine supports differentiable elastic and plastic deformation using the DiffTaichi system, posing many underexplored challenges to robotic agents. We evaluate several existing RL methods and gradient-based methods on this benchmark. 

</details>

## RoboNinja
> **RoboNinja** is a differentiable simulation system for cutting multi-material objects.

<details open>
<h2 align="center">
  <b>RoboNinja: Learning an Adaptive Cutting Policy for Multi-Material Objects</b>

<div align="center">
    <a href="https://roboninja.cs.columbia.edu/" target="_blank"><img src="https://img.shields.io/badge/Website-RoboNinja-red"></img></a>
    &nbsp;
    <a href="https://arxiv.org/abs/2302.11553" target="_blank"><img src="https://img.shields.io/badge/Paper-ArXiv-green"></img></a>
    &nbsp;
    <a href="https://github.com/columbia-ai-robotics/roboninja" target="_blank"><img src="https://img.shields.io/badge/Source-Code-purple"></img></a>
</div>
</h2>

<div align="center">
<div style={{ textAlign: 'center' }}>
    <img src="https://roboninja.cs.columbia.edu/images/teaser.png"  />
  </div>
</div>
RoboNinja is a learning-based cutting system for multi-material objects (i.e., soft objects with rigid cores such as avocados or mangos). In contrast to prior works using open-loop cutting actions to cut through single-material objects (e.g., slicing a cucumber), RoboNinja aims to remove the soft part of an object while preserving the rigid core, thereby maximizing the yield. Learning such cutting skills directly on a real-world robot is challenging. Yet, existing simulators are limited in simulating multi-material objects or computing the energy consumption during the cutting process. To address this issue, roboninja is a differentiable cutting simulator that supports multi-material coupling and allows for the generation of optimized trajectories as demonstrations for policy learning.
</details>


## DiSECt
> **DiSECt** is a differentiable simulation engine for autonomous **robotic cutting**.

<details open> <summary>Details</summary>
<h2 align="center">
  <b>DiSECt</b>

<div align="center">
    <a href="https://diff-cutting-sim.github.io/" target="_blank"><img src="https://img.shields.io/badge/Website-DiSECt-red"></img></a>
    &nbsp;
    <a href="https://arxiv.org/abs/2105.12244" target="_blank"><img src="https://img.shields.io/badge/Paper-ArXiv-green"></img></a>
    &nbsp;
    <a href="https://github.com/NVlabs/DiSECt" target="_blank"><img src="https://img.shields.io/badge/Source-Code-purple"></img></a>
</div>
</h2>


<div align="center">
<video width="800" height="600" controls>
  <source src="https://diff-cutting-sim.github.io/potato_knife_force_profile.mp4" type="video/mp4"></source>
</video>
</div>

DiSECt is a simulator for the cutting of deformable materials. It uses the Finite Element Method (FEM) to simulate the deformation of the material, and leverages a virtual node algorithm to introduce springs between the two halves of the mesh being cut. These cutting springs are weakened in proportion to the knife forces acting on the material, yielding a continuous model of deformation and crack propagation. By leveraging source code transformation, the back-end of DiSECt automatically generates CUDA-accelerated kernels for the forward simulation and the gradients of the simulation inputs. Such gradient information can be used to optimize the simulation parameters to achieve accurate knife force predictions, optimize cutting actions, and more.

</details>


## ThinShellLab

> **ThinShellLab** is a differentiable simulator for manipulating **thin-shell materials**, such as cloths and papers.

<details open> <summary>Details</summary>
<h2 align="center">
  <b>ThinShellLab: Thin-shell Object Manipulations with Differentiable Physics Simulations</b>

<div align="center">
    <a href="https://thinshelllab.github.io/" target="_blank"><img src="https://img.shields.io/badge/Website-ThinShellLab-red"></img></a>
    &nbsp;
    <a href="https://openreview.net/forum?id=KsUh8MMFKQ&noteId=SZQ8HmKXnI" target="_blank"><img src="https://img.shields.io/badge/Paper-OpenReview-green"></img></a>
    &nbsp;
    <a href="" target="_blank"><img src="https://img.shields.io/badge/Source-Code (coming soon)-purple"></img></a>
</div>
</h2>


<div align="center">
<video width="400" height="300" controls>
  <source src="https://thinshelllab.github.io/static/videos/balance_0.mp4" type="video/mp4"></source>
</video>
<video width="400" height="300" controls>
  <source src="https://thinshelllab.github.io/static/videos/card.mp4" type="video/mp4"></source>
</video>
</div>
ThinShellLab is a fully differentiable simulation platform tailored for robotic interactions with diverse thin-shell materials possessing varying material properties, enabling flexible thin-shell manipulation skill learning and evaluation. This comprehensive endeavor encompasses a triad of experiment task categories, which are as follows: manipulation tasks, inverse design tasks, and real-world experiments.

</details>


## DaxBench

> **DaxBench** is a **deformable object manipulation** benchmark with differentiable physics.

<details open> <summary>Details</summary>
<h2 align="center">
  <b>DaxBench: Benchmarking Deformable Object Manipulation with Differentiable Physics</b>

<div align="center">
    <a href="https://daxbench.github.io/" target="_blank"><img src="https://img.shields.io/badge/Website-DaxBench-red"></img></a>
    &nbsp;
    <a href="https://arxiv.org/abs/2210.13066" target="_blank"><img src="https://img.shields.io/badge/Paper-ArXiv-green"></img></a>
    &nbsp;
    <a href="https://github.com/AdaCompNUS/DaXBench" target="_blank"><img src="https://img.shields.io/badge/Source-Code-purple"></img></a>
</div>
</h2>

<div align="center">
<div style={{ textAlign: 'center' }}>
    <img src="https://github.com/AdaCompNUS/DaXBench/raw/main/docs/images/water_final.gif"  />
  </div>
<div style={{ textAlign: 'center' }}>
    <img src="https://github.com/AdaCompNUS/DaXBench/raw/main/docs/images/cloth_final.gif"  />
  </div>
</div>
DaXBench is a differentiable simulation framework for deformable object manipulation. While existing work often focuses on a specific type of deformable objects, DaXBench supports fluid, rope, cloth, etc ; it provides a general-purpose benchmark to evaluate widely different DOM methods, including planning, imitation learning, and reinforcement learning. DaXBench combines recent advances in deformable object simulation with JAX, a high-performance computational framework. All DOM tasks in DaXBench are wrapped with the OpenAI Gym API for easy integration with DOM algorithms.
</details>


## SoftMac

> **SoftMac** is a differentiable soft body simulation with forecast-based contact model, coupling with articulated rigid bodies and clothes.

<details open> <summary>Details</summary>
<h2 align="center">
  <b>SoftMac</b>

<div align="center">
    <a href="https://sites.google.com/view/softmac" target="_blank"><img src="https://img.shields.io/badge/Website-SoftMac-red"></img></a>
    &nbsp;
    <a href="https://arxiv.org/abs/2312.03297" target="_blank"><img src="https://img.shields.io/badge/Paper-ArXiv-green"></img></a>
    &nbsp;
    <a href="https://github.com/damianliumin/SoftMAC" target="_blank"><img src="https://img.shields.io/badge/Source-Code-purple"></img></a>
</div>
</h2>

<div align="center">
<div style={{ textAlign: 'center' }}>
    <img src="https://lh3.googleusercontent.com/ijg0PhkHk9oR2MYuiY5t9evsDhzKpN-YDhFF3SOlIZ8QY96y70JbLBmi03CR0q5J24VFT72nSa2aBFFtXJxsnoyejjhB5mcjJ82UV2uV1TqgcP_Ic4e-4PFMQhqvRmQRfg=w1280"  />
  </div>
</div>
SoftMAC, a differentiable simulation framework coupling soft bodies with articulated rigid bodies and clothes. SoftMAC simulates soft bodies with the continuum-mechanics-based Material Point Method (MPM).
</details>