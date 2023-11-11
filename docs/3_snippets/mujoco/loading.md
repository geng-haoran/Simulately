---
sidebar_position: 1
---

# Load Environment
> How to load a ground, objects, robots into a SAPIEN env. Some code are borrowed from SAPIEN Doc

### Load a basic object
```python
import mujoco

XML=r"""
<mujoco>
  <asset>
    <mesh file="gizmo.stl"/>
  </asset>
  <worldbody>
    <body>
      <freejoint/>
      <geom type="mesh" name="gizmo" mesh="gizmo"/>
    </body>
  </worldbody>
</mujoco>
"""

ASSETS=dict()
with open('/path/to/gizmo.stl', 'rb') as f:
  ASSETS['gizmo.stl'] = f.read()

model = mujoco.MjModel.from_xml_string(XML, ASSETS)
data = mujoco.MjData(model)
while data.time < 1:
  mujoco.mj_step(model, data)
  print(data.geom_xpos)
```

### Information about env

```python
print('Total number of DoFs in the model:', model.nv)
print('Generalized positions:', data.qpos)
print('Generalized velocities:', data.qvel)
```