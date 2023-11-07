---
sidebar_position: 1
---

# Load Environment
> How to load a ground, objects, robots into a SAPIEN env. Some code are borrowed from SAPIEN Doc

### Add Ground

```python
    scene.add_ground(altitude=0)  # change altitude
```

### Add Light
```python
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
```

### Add Primitive Shapes
```python
    actor_builder = scene.create_actor_builder()
    actor_builder.add_box_collision(half_size=[0.5, 0.5, 0.5])
    actor_builder.add_box_visual(half_size=[0.5, 0.5, 0.5], color=[1., 0., 0.])
    box = actor_builder.build(name='box')  # Add a box
    box.set_pose(sapien.Pose(p=[0, 0, 0.5]))
```

### Add Objects
```python
    builder = scene.create_actor_builder()
    builder.add_collision_from_file(filename='../assets/banana/collision_meshes/collision.obj')
    builder.add_visual_from_file(filename='../assets/banana/visual_meshes/visual.dae')
    mesh = builder.build(name='mesh')
    mesh.set_pose(sapien.Pose(p=[-0.2, 0, 1.0 + 0.05]))
```

### Add Articulated Objects
```python
    loader.fix_root_link = fix_root_link
    art_obj: sapien.Articulation = loader.load("path_to_your_urdf")
    art_obj.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
    art_obj.set_qpos(your_init_qpos)
```

