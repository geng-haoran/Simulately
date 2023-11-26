---
sidebar_position: 0
---

# Getting Started

Welcome to the Getting Started for using MuJoCo.
In this guide, we will walk through the process of initializing the simulator, loading a groud, creating light and
loading actors with SAPIEN API. The provided code snippets are adapted from the SAPIEN Docs to help you get started
quickly.

## Initial the Simulator

### Create a XML world description for MuJoCo scene

Different from many simulators that favor using API function to create world, MuJoCo requires world description to be
stored in a XML file. After loading the xml and begin simulation, you can not add or remove any objects into the scene.
This is one drawback of MuJoCo simulation but is also its design choice to make it faster with pre-allocated data
buffer.

To begin, we need to create a XML description, which contains only a box in the world. This box can move freely in the
space when force applied.

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

Then we need to create a renderer. The renderer is responsible for rendering the scene, no matter whether for headless
rendering on server or using viewer on your own computer.

```python
model = mujoco.MjModel.from_xml_string(world_xml, {})
renderer = mujoco.Renderer(model)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
renderer.render()
```

