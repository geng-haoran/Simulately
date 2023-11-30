---
sidebar_position: 7
---

# [Example]  Minimal Example (box version)
```python
import mujoco

XML = r"""
<mujoco>
  <worldbody>
    <body>
      <freejoint/>
      <geom type="box" name="my_box" size="1 1 1"/>
    </body>
  </worldbody>
</mujoco>
"""

ASSETS = dict()

model = mujoco.MjModel.from_xml_string(XML, ASSETS)
data = mujoco.MjData(model)
while data.time < 1:
    mujoco.mj_step(model, data)
    print(data.geom_xpos)
```