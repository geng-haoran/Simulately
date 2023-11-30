---
sidebar_position: 0
---

# Getting Started

Welcome to the introductory guide on using MuJoCo. In this tutorial, we will guide you through the steps to initialize
the simulator, load a world XML, simulate the world, and render it to your screen. The code snippets provided here are
adapted from the MuJoCo Documentation to facilitate a quick start.

### Building an XML World Description for a MuJoCo Scene

Unlike many simulators that prefer API functions to construct the world, MuJoCo necessitates the world description to be
saved in an XML file. Once the XML is loaded and the simulation begins, you can't add or remove any objects in the
scene. While this is a limitation of the MuJoCo simulation, it is also a design choice aimed at increasing speed through
pre-allocated data buffers.

We'll start by creating an XML description, which only includes a box in the world. This box is capable of moving freely
in space when a force is applied.

```python
world_xml = r"""
<mujoco>
  <worldbody>
    <body>
      <freejoint/>
      <geom type="box" name="my_box" size="1 1 1"/>
    </body>
  </worldbody>
</mujoco>
"""
```

### Load XML world description and simulate

```python
import mujoco

model = mujoco.MjModel.from_xml_string(world_xml, {})
data = mujoco.MjData(model)
while data.time < 1:
  mujoco.mj_step(model, data)
  print(data.geom_xpos)
```

### Create a renderer and bind the renderer to simulation

Next, we'll need to establish a renderer. The renderer is tasked with illustrating the scene, regardless of whether it's
for headless rendering on a server or for using a viewer on your local machine.

```python
model = mujoco.MjModel.from_xml_string(world_xml, {})
renderer = mujoco.Renderer(model)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
renderer.render()
```

