---
sidebar_position: 0
---

# Getting Started
Welcome to the Getting Started for using SAPIAN.
In this guide, we will walk through the process of initializing the simulator, loading a groud, creating light and loading actors with SAPIEN API. The provided code snippets are adapted from the SAPIEN Docs to help you get started quickly.

## Initial the Simulator
### Create a SAPIEN physical simulation engine

To begin, we need to create a physical simulation engine. The engine is the core of the simulator, which is responsible for the simulation of the physical world. The engine is also responsible for the creation of the scene, which is the container of all the actors in the simulation.
```python
import sapien.core as sapien
engine = sapien.Engine()
```
### Create a renderer and Bind the renderer and the engine
Then we need to create a renderer. The renderer is responsible for rendering the scene. The renderer is also responsible for the creation of the viewer, which is the window that displays the scene.

```python
renderer = sapien.SapienRenderer()
engine.set_renderer(renderer)
```

### Create an instance of simulation world (aka scene)
Now we can create an instance of simulation world (aka scene). The scene is the container of all the actors in the simulation. The scene is also responsible for the simulation of the physical world.

```python
scene = engine.create_scene()
scene.set_timestep(1 / 100.0) # Set the simulation frequency
```

## Create Ground, Actors, Lights and Viewer
### Add a ground
Add a ground to the scene. 
```python
scene.add_ground(altitude=0) # Add a ground
```
### Create an actor builder
Create an actor builder to build actors (rigid bodies). 

```python
actor_builder = scene.create_actor_builder()
```

### Add a box (an example)
Create a box and add it to the scene.
```python
actor_builder.add_box_collision(half_size=[0.5, 0.5, 0.5])
actor_builder.add_box_visual(half_size=[0.5, 0.5, 0.5], color=[1., 0., 0.])
box = actor_builder.build(name='box')  # Add a box
box.set_pose(sapien.Pose(p=[0, 0, 0.5]))
```

### Add some lights
Add some lights so that you can observe the scene.
```python
scene.set_ambient_light([0.5, 0.5, 0.5])
        scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
```

### Create a viewer (window)
Create a viewer (window) and bind the viewer and the scene.
```python
viewer = Viewer(renderer)  # Create a viewer (window)
viewer.set_scene(scene)  # Bind the viewer and the scene

# The coordinate frame in Sapien is: x(forward), y(left), z(upward)
# The principle axis of the camera is the x-axis
viewer.set_camera_xyz(x=-4, y=0, z=2)
# The rotation of the free camera is represented as [roll(x), pitch(-y), yaw(-z)]
# The camera now looks at the origin
viewer.set_camera_rpy(r=0, p=-np.arctan2(2, 4), y=0)
viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)
```

## Run the simulation loop
Now we can run the simulation loop. The simulation loop consists of three steps: (1) simulate the world, (2) update the world to the renderer, (3) render the scene.
```python
while not viewer.closed:  # Press key q to quit
    scene.step()  # Simulate the world
    scene.update_render()  # Update the world to the renderer
    viewer.render()
```

Then you will see:
<p align="center">
  <img src="https://sapien.ucsd.edu/docs/latest/_images/hello_world.png" alt="Move the Box(Official)" width="500"/>
</p>

## Load a robot and control it

Pending...