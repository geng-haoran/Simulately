---
sidebar_position: 1
---

# Load Environment
> How to load a ground, objects, robots into a SAPIEN env. Some code are borrowed from SAPIEN Doc

### Set Gravity

```python
    p.setGravity(0,0,-10)
```

### Add Light
```python
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
```

### Add Objects
```python
    planeId = p.loadURDF("plane.urdf")
```

### Add Articulated Objects
```python
    objId = p.loadURDF("path_to_your_urdf.urdf")
    startPos = [0,0,1]
    startOrientation = p.getQuaternionFromEuler([0,0,0])
    objId_with_pose = p.loadURDF("path_to_your_urdf.urdf",startPos, startOrientation)
```

### V-HACD
```python
    import pybullet as p
    import pybullet_data as pd
    import os

    p.connect(p.DIRECT)
    name_in = os.path.join(pd.getDataPath(), "duck.obj")
    name_out = "duck_vhacd2.obj"
    name_log = "log.txt"
    p.vhacd(name_in, name_out, name_log)
```

