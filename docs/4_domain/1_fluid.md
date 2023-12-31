---
sidebar_position: 2
---

# Fluid Simulators

## Overview
- [**FluidLab**](#fluidlab): a fully differentiable and multi-material physics simulation platform, supporting **rigid**, **elastic**, **plastic** materials, **inviscid** and **viscous liquid**, and **gaseous phenomena** such as smoke. 
- [**FleX**](#flex): a **particle-based simulation** library designed for real-time simulation of particle-based rigid, deformable and fluid bodies using position-based dynamics.
- [**Aquarium**](#aquarium): a differentiable **fluid-structure** interaction system for robotics.
- [**Fish Gym**](#fish-gym): a physics-based simulation framework for physical articulated underwater agent interaction with fluid.

---
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

## FleX

> **FleX** is a **particle-based simulation** library designed for real-time simulation of particle-based rigid, deformable and fluid bodies using position-based dynamics.

<details open> <summary>Details</summary>
<h2 align="center">
  <b>FleX</b>

<div align="center">
    <a href="https://developer.nvidia.com/flex" target="_blank"><img src="https://img.shields.io/badge/Website-Flex-red"></img></a>
    &nbsp;
    <a href="https://github.com/NVIDIAGameWorks/FleX" target="_blank"><img src="https://img.shields.io/badge/Source-Code-purple"></img></a>
</div>
</h2>

<div align="center">
<div style={{ textAlign: 'center' }}>
    <img src="https://d29g4g2dyqv443.cloudfront.net/sites/default/files/akamai/gamedev/images/flex.jpg"/>
  </div>
</div>

FleX is a particle based simulation technique for real-time visual effects. Traditionally, visual effects are made using a combination of elements created using specialized solvers for rigid bodies, fluids, clothing, etc. Because FleX uses a unified particle representation for all object types, it enables new effects where different simulated substances can interact with each other seamlessly. Such unified physics solvers are a staple of the offline computer graphics world, where tools such as Autodesk Maya's nCloth, and Softimage's Lagoa are widely used. The goal for FleX is to use the power of GPUs to bring the capabilities of these offline applications to real-time computer graphics.

</details>

## Aquarium 

> **Aquarium** is a differentiable **fluid-structure** interaction system for robotics.


<details open> <summary>Details</summary>
<h2 align="center">
  <b>Aquarium: Differentiable Fluid-Structure Interaction for Robotics
</b>

<div align="center">
    <a href="https://rexlab.ri.cmu.edu/papers/aquarium.pdf" target="_blank"><img src="https://img.shields.io/badge/Paper-PDF-green"></img></a>
    &nbsp;
    <a href="https://github.com/RoboticExplorationLab/Aquarium.jl" target="_blank"><img src="https://img.shields.io/badge/Source-Code-purple"></img></a>
</div>
</h2>

<div align="center">
<div style={{ textAlign: 'center'}}>
    <img src="https://rexlab.ri.cmu.edu/img/aquarium/aquarium.gif"/>
  </div>
</div>

Aquarium is a differentiable fluid structure interaction solver for robotics that offers stable simulation, accurately coupled fluid-robot physics in two dimensions, and full differentiability with respect to fluid and robot states and parameters. Aquarium achieves stable simulation with accurate flow physics by directly integrating over the incompressible Navier-Stokes equations using a fully implicit Crank-Nicolson scheme with a second-order finite-volume spatial discretization. The fluid and robot physics are coupled using the immersed-boundary method by formulating the no slip condition as an equality constraint applied directly to the Navier-Stokes system. This choice of coupling allows the fluid structure interaction to be posed and solved as a nonlinear optimization problem. This optimization-based formulation is then exploited using the implicit-function theorem to compute derivatives.

</details>


## Fish Gym 

> **Fish Gym** is a physics-based simulation framework for physical articulated underwater agent interaction with fluid.

<details open> <summary>Details</summary>
<h2 align="center">
  <b>FishGym: A High-Performance Physics-based Simulation Framework for Underwater Robot Learning
</b>

<div align="center">
    <a href="https://arxiv.org/abs/2206.01683" target="_blank"><img src="https://img.shields.io/badge/Paper-ArXiv-green"></img></a>
    &nbsp;
    <a href="https://github.com/fish-gym/gym-fish" target="_blank"><img src="https://img.shields.io/badge/Source-Code-purple"></img></a>
</div>
</h2>

<div align="center">
<div style={{ textAlign: 'center'}}>
    <img src="https://gym-fish.readthedocs.io/en/latest/_images/koi_cruising.png"/>
  </div>
</div>

Fish Gym is a physics-based simulation framework for physical articulated underwater agent interaction with fluid. This is the first physics-based environment that support coupled interation between agents and fluid in semi-realtime. Fish Gym is integrated into the OpenAI Gym interface, enabling the use of existing reinforcement learning and control algorithms to control underwater agents to accomplish specific underwater exploration task.


</details>