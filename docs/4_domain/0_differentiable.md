---
sidebar_position: 0
---

# Differetiable Simulators

## Overview
- [**DiffSim**](#diffsim): a scalable C++ framework for differentiable physics that can support a large number of **rigid** and **deformable** objects and their interactions.
- [**FluidLab**](#fluidlab): a fully differentiable and multi-material physics simulation platform, supporting **rigid**, **elastic**, **plastic** materials, **inviscid** and **viscous liquid**, and **gaseous phenomena** such as smoke. 
- [**PlasticineLab**](#plasticinelab): a differentiable **soft-body** manipulation simulator and benchmark.
- [**DiffArticulated**](#diffarticulated): a differentiable simulation system for **articulated rigid bodies**.
- [**DiSECt**](#disect): a differentiable simulation engine for autonomous **robotic cutting**.
- [**ThinShellLab**](#thinshelllab): a differentiable simulator for manipulating **thin-shell materials**, such as cloths and papers.
- [**DaxBench**](#daxbench): a **deformable object manipulation** benchmark with differentiable physics.
- [**NimblePhysics**](#nimblephysics): a general purpose differentiable physics engine for **articulated rigid bodies**.
- [**MuJoCo XLA(MJX)**](#mjx): a re-implementation of the MuJoCo physics engine in JAX, which is differentiable.

---

## DiffSim

> **DiffSim** is a scalable C++ framework for differentiable physics that can support a large number of rigid and deformable objects and their interactions.

<details open> <summary>Details</summary>
<h2 align="center">
  <b>DiffSim: Scalable Differentiable Physics for Learning and Control</b>
<div align="center">
    <a href="https://gamma.umd.edu/researchdirections/mlphysics/diffsim/" target="_blank"><img src="https://img.shields.io/badge/Website-Diffsim-red"></img></a>
    &nbsp;
    <a href="https://arxiv.org/abs/2007.02168" target="_blank"><img src="https://img.shields.io/badge/Paper-ArXiv-green"></img></a>
    &nbsp;
    <a href="https://github.com/YilingQiao/diffsim" target="_blank"><img src="https://img.shields.io/badge/Source-Code-purple"></img></a>
</div>
</h2>

<div align="center">
<div style={{ textAlign: 'center' }}>
    <img src="https://obj.umiacs.umd.edu/gamma-umd-website-imgs/researchdirections/diffsim/teasericml20.png"  />
  </div>
</div>

DiffSim is a scalable framework for differentiable physics that can support a large number of objects and their interactions. It adopts meshes as its representation and leverage the sparsity of contacts for scalable differentiable collision handling. Collisions are resolved in localized regions to minimize the number of optimization variables even when the number of simulated objects is high. It further accelerates implicit differentiation of optimization with nonlinear constraints. DiffSim requires up to two orders of magnitude less memory and computation in comparison to contemporary particle-based methods.

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

## DiffArticulated

> **DiffArticulated** is a differentiable simulation system for **articulated rigid bodies**.

<details open> <summary>Details</summary>
<h2 align="center">
  <b>DiffArticulated: Efficient Differentiable Simulation of Articulated Bodies</b>

<div align="center">
    <a href="https://github.com/YilingQiao/diffarticulated" target="_blank"><img src="https://img.shields.io/badge/Website-Diffarticulated-red"></img></a>
    &nbsp;
    <a href="https://arxiv.org/abs/2109.07719" target="_blank"><img src="https://img.shields.io/badge/Paper-ArXiv-green"></img></a>
    &nbsp;
    <a href="https://github.com/YilingQiao/diffarticulated" target="_blank"><img src="https://img.shields.io/badge/Source-Code-purple"></img></a>
</div>
</h2>

<div align="center">
<div style={{ textAlign: 'center' }}>
    <img src="https://github.com/YilingQiao/linkfiles/raw/master/icml21/throw.gif"/>
  </div>
</div>

DiffArticulated is a method for efficient differentiable simulation of articulated bodies. This enables integration of articulated body dynamics into deep learning frameworks, and gradient-based optimization of neural networks that operate on articulated bodies. The gradients of the contact solver are derived using spatial algebra and the adjoint method, resulting in simulation speed that's an order of magnitude faster than autodiff tools.
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

## NimblePhysics

> **NimbelPhysics** is a general purpose differentiable physics engine for **articulated rigid bodies**.

<details open> <summary>Details</summary>
<h2 align="center">
  <b>NimblePhysics</b>

<div align="center">
    <a href="https://nimblephysics.org" target="_blank"><img src="https://img.shields.io/badge/Website-NimblePhysics-red"></img></a>
    &nbsp;
    <a href="https://arxiv.org/abs/2103.16021" target="_blank"><img src="https://img.shields.io/badge/Paper-ArXiv-green"></img></a>
    &nbsp;
    <a href="https://github.com/keenon/nimblephysics" target="_blank"><img src="https://img.shields.io/badge/Source-Code-purple"></img></a>
</div>
</h2>

<div align="center">
  <div style={{ textAlign: 'center' }}>
      <img src="https://nimblephysics.org/docs/_images/Falisse_Stair_Up.gif"  />
  </div>
</div>

Nimble is a toolkit for doing AI on human biomechanics (physically simulated realistic human bodies), written in C++ for speed, but with nice Python bindings. It focuses on studying real physical human bodies. Nimble started life as a general purpose differentiable physics engine, as a fork of the (not differentiable) DART physics engine. 
</details>

## MJX

> **MuJoCo XLA(MJX)** is a re-implementation of the MuJoCo physics engine in JAX.

<details open> <summary>Details</summary>
<h2 align="center">
  <b>MuJoCo XLA(MJX)</b>

<div align="center">
    <a href="https://mujoco.readthedocs.io/en/stable/mjx.html" target="_blank"><img src="https://img.shields.io/badge/Website-MJX-red"></img></a>
    &nbsp;
    <a href="https://github.com/google-deepmind/mujoco/tree/main/mjx" target="_blank"><img src="https://img.shields.io/badge/Source-Code-purple"></img></a>
</div>
</h2>

<div align="center">

Starting with version 3.0.0, MuJoCo includes MuJoCo XLA (MJX) under the mjx directory. MJX allows MuJoCo to run on compute hardware supported by the XLA compiler via the JAX framework. MJX runs on a all platforms supported by JAX: Nvidia and AMD GPUs, Apple Silicon, and Google Cloud TPUs. The MJX API is consistent with the main simulation functions in the MuJoCo API.
</div>
</details>