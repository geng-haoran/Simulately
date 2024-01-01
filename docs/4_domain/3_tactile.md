---
sidebar_position: 3
---

# Tactile Simulators

## Overview
- [**Taxim**](#taxim): an example-based simulator for **GelSight tactile** sensors and its variations.
- [**DiffRedMax**](#diffredmax): a differentiable simulator with **penalty-based contact model** and supports implicit time integration.
- [**TACTO**](#tacto): a fast, flexible, and open-source simulator for **vision-based tactile sensors**.
- [**DiffTactile**](#difftactile): a physics-based and fully differentiable tactile simulation system designed to enhance robotic manipulation with **dense and physically-accurate tactile feedback**.

---

## Taxim
> **Taxim** is an example-based simulator for **GelSight tactile** sensors and its variations.

<details open>
<h2 align="center">
  <b>Taxim: An Example-based Simulation Model for GelSight Tactile Sensors
</b>

<div align="center">
    <a href="https://labs.ri.cmu.edu/robotouch/taxim-simulation/" target="_blank"><img src="https://img.shields.io/badge/Website-Taxim-red"></img></a>
    &nbsp;
    <a href="https://arxiv.org/abs/2109.04027" target="_blank"><img src="https://img.shields.io/badge/Paper-ArXiv-green"></img></a>
    &nbsp;
    <a href="https://github.com/Robo-Touch/Taxim" target="_blank"><img src="https://img.shields.io/badge/Source-Code-purple"></img></a>
</div>
</h2>

<div align="center">
<div style={{ textAlign: 'center' }}>
    <img src="https://labs.ri.cmu.edu/robotouch/wp-content/uploads/sites/22/2021/10/cover-2048x822.png"  />
  </div>
</div>

Taxim is a realistic and high-speed simulation model for a vision-based tactile sensor, GelSight. Our simulation framework is the first to incorporate marker motion field simulation together with the optical simulation. We simulate the optical response to the deformation with a polynomial lookup table. This table maps the deformed geometries to pixel intensity sampled by the embedded camera. We apply the linear elastic deformation theory and the superposition principle to simulate the surface markers’ motion that is caused by the surface stretch of the elastomer. The example-based approach requires less than 100 data points from a real sensor to calibrate the simulator and enables the model to easily migrate to other GelSight sensors or their variations.

</details>

## DiffRedMax (previously DiffHand)

> **DiffRedMax** is a differentiable simulator with **penalty-based contact model** and supports implicit time integration.

<details open>
<h2 align="center">
  <b>DiffHand: Efficient Tactile Simulation with Differentiability for Robotic Manipulation</b>

<div align="center">
    <a href="http://tactilesim.csail.mit.edu/" target="_blank"><img src="https://img.shields.io/badge/Website-DiffHand-red"></img></a>
    &nbsp;
    <a href="https://people.csail.mit.edu/jiex/papers/TactileSim/paper.pdf" target="_blank"><img src="https://img.shields.io/badge/Paper-PDF-green"></img></a>
    &nbsp;
    <a href="https://github.com/eanswer/DiffHand" target="_blank"><img src="https://img.shields.io/badge/Source-Code-purple"></img></a>
</div>
</h2>

<div align="center">
<div style={{ textAlign: 'center' }}>
    <img src="https://github.com/eanswer/DiffHand/raw/master/demos/tactile_sim.gif" width="266" />
    <img src="https://github.com/eanswer/DiffHand/raw/master/demos/tactile_depth_map.gif" width="200" />
    <img src="https://github.com/eanswer/DiffHand/raw/master/demos/tactile_force_map.gif" width="200" />
  </div>
</div>

DiffRedMax is a differentiable simulator with penalty-based contact model and supports implicit time integration. It also supports simulating dense tactile force field of both normal and shear directional tactile forces. It also provides the analytical first-order simulation gradients with respect to all the control input and simulation parameters (including kinemactics and dynamics parameters).

</details>


## TACTO

> **TACTO** is a fast, flexible, and open-source simulator for **vision-based tactile sensors**.

<details open>
<h2 align="center">
  <b>TACTO: A Fast, Flexible, and Open-source Simulator for High-Resolution Vision-based Tactile Sensors</b>

<div align="center">
    <a href="https://arxiv.org/abs/2012.08456" target="_blank"><img src="https://img.shields.io/badge/Paper-ArXiv-green"></img></a>
    &nbsp;
    <a href="https://github.com/facebookresearch/tacto" target="_blank"><img src="https://img.shields.io/badge/Source-Code-purple"></img></a>
</div>
</h2>

<div align="center">
<div style={{ textAlign: 'center' }}>
    <img src="https://github.com/facebookresearch/tacto/raw/main/website/static/img/teaser.jpg?raw=true"  />
  </div>
</div>
TACTO is a fast, flexible, and open-source simulator for vision-based tactile sensors. This simulator allows to render realistic high-resolution touch readings at hundreds of frames per second, and can be easily configured to simulate different vision-based tactile sensors, including DIGIT and OmniTact.
</details>

## Tactile-Gym

> **Tactile-Gym** is a suite of simulated environments tailored towards tactile robotics and reinforcement learning.

<details open>
<h2 align="center">
  <b>Tactile Gym 2.0: Sim-to-Real Deep Reinforcement Learning for Comparing Low-Cost High-Resolution Robot Touch</b>

<div align="center">
    <a href="https://ieeexplore.ieee.org/abstract/document/9847020" target="_blank"><img src="https://img.shields.io/badge/Paper-IEEE-green"></img></a>
    &nbsp;
    <a href="https://github.com/ac-93/tactile_gym?tab=readme-ov-file" target="_blank"><img src="https://img.shields.io/badge/Source-Code-purple"></img></a>
</div>
</h2>

<div align="center">
<div style={{ textAlign: 'center' }}>
    <img src="https://github.com/ac-93/tactile_gym/raw/main/docs/readme_images/paper_overview.png"  />
  </div>
</div>
Tactile Gym 2.0 includes three new optical tactile sensors (TacTip, DIGIT and DigiTac) of the two most popular types, Gelsight-style (image-shading based) and TacTip-style (marker based).
</details>


## DiffTactile

> **DiffTactile** is a physics-based and fully differentiable tactile simulation system designed to enhance robotic manipulation with **dense and physically-accurate tactile feedback**.


<details open> <summary>Details</summary>
<h2 align="center">
  <b>DiffTactile: A Physics-based Differentiable Tactile Simulator for Contact-rich Robotic Manipulation</b>

<div align="center">
    <a href="https://difftactile.github.io/" target="_blank"><img src="https://img.shields.io/badge/Website-ThinShellLab-red"></img></a>
    &nbsp;
    <a href="https://difftactile.github.io/static/pdf/paper.pdf" target="_blank"><img src="https://img.shields.io/badge/Paper-PDF-green"></img></a>
    &nbsp;
    <a href="" target="_blank"><img src="https://img.shields.io/badge/Source-Code (coming soon)-purple"></img></a>
</div>
</h2>


<div align="center">
<div style={{ textAlign: 'center' }}>
    <img src="https://difftactile.github.io/static/gifs/surface.gif" width="300" />
    <img src="https://difftactile.github.io/static/gifs/278_1700264468.gif"  width="300"/>
  </div>
</div>

DiffTactile is a physics-based and fully differentiable tactile simulation system designed to enhance robotic manipulation with dense and physically-accurate tactile feedback. In contrast to prior tactile simulators which primarily focus on manipulating rigid bodies and often rely on simplified approximations to model stress and deformations of materials in contact, DiffTactile emphasizes physics-based contact modeling with high fidelity, supporting simulations of diverse contact modes and interactions with objects possessing a wide range of material properties.


</details>
