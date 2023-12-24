---
sidebar_position: 0
---

# Differetiable Simulator

<!-- FluidLab, Diffsim, diffarticulated, DiSect -->

## FluidLab

<details> <summary>Information about FluidLab</summary>
<h2 align="center">
  <b>FluidLab</b>

<div align="center">
    <a href="https://fluidlab2023.github.io/" target="_blank"><img src="https://img.shields.io/badge/Website-FluidLab-red"></img></a>
    &nbsp;
    <a href="https://arxiv.org/abs/2303.02346" target="_blank"><img src="https://img.shields.io/badge/Paper-ArXiv-green"></img></a>
    &nbsp;
    <a href="https://github.com/zhouxian/FluidLab" target="_blank"><img src="https://img.shields.io/badge/Source-Code-purple"></img></a>
</div>
</h2>

<div align="center">
<video width="800" height="600" controls>
  <source src="https://fluidlab2023.github.io/static/videos/tasks.m4v" type="video/mp4"></source>
</video>
</div>

> FluidLab is a differentiable environment with a set of complex fluid manipulation tasks. FluidLab is powered by FluidEngine, a fully differentiable and multi-material physics engine, supporting rigid, elastic, plastic materials, inviscid and viscous liquid, and gaseous phenomena such as smoke.

</details>

## Diffsim

<details> <summary>Information about Diffsim</summary>
<h2 align="center">
  <b>Diffsim</b>

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

> Diffsim: Scalable Differentiable Physics for Learning and Control

</details>

## Diffarticulated

<details> <summary>Information about Diffarticulated</summary>
<h2 align="center">
  <b>Diffarticulated</b>

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

> Diffarticulated: Efficient Differentiable Simulation of Articulated Bodies. This enables integration of articulated body dynamics into deep learning frameworks, and gradient-based optimization of neural networks that operate on articulated bodies.

</details>

## DiSECt

<details> <summary>Information about DiSECt</summary>
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

> DiSECt is a simulator for the cutting of deformable materials. It uses the Finite Element Method (FEM) to simulate the deformation of the material, and leverages a virtual node algorithm to introduce springs between the two halves of the mesh being cut. These cutting springs are weakened in proportion to the knife forces acting on the material, yielding a continuous model of deformation and crack propagation. By leveraging source code transformation, the back-end of DiSECt automatically generates CUDA-accelerated kernels for the forward simulation and the gradients of the simulation inputs. Such gradient information can be used to optimize the simulation parameters to achieve accurate knife force predictions, optimize cutting actions, and more.

</details>