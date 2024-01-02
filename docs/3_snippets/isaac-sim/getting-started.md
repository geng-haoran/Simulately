---
sidebar_position: 0
---

# Getting Started

Welcome to the Getting Started for using IsaacSim. Here, we will quickly go through:

1. An efficient way of development in Isaac Sim
2. Three workflows (GUI, extensions, and standalone extensions)
3. GUI in Isaac Sim
4. Standalone Python

Some information are borrowed from the [official docs of Isaac Sim](https://docs.omniverse.nvidia.com/isaacsim/latest/index.html), but we provide a more concentrated version to get you ready.

## An efficient way of dev in Isaac Sim

Before we get started, here are several things you need to know:

1. Like other modules like the Create in Omniverse, Isaac Sim is also built on Omniverse Kit. Almost all the functions are implemented as "extensions", *i.e.* every action you do in the GUI can be implemented via coding.
2. You can find the Isaac Sim local folder at `INSTALL_DIRECTORY/pkg/isaac-sim-VERSION/`. For ubuntu, `INSTALL_DIRECTORY` is `~/.local/share/ov/` by default. It would be a rapid way, if you want to add new features to your app, to perform a search in this folder to find possibly relative codes.
3. For docs, always start from [this main page](https://docs.omniverse.nvidia.com/isaacsim/latest/index.html) for the latest version of the documentation, and use the built-in searching (rather than directly searching on Google, which may provide out-of-date links).

## Three workflows in Isaac Sim

There are three workflows in Isaac Sim that you can use to develop:

1. GUI: directly interacting with the Isaac Sim GUI, pushing buttons, connecting action graphs, to implement what you want. This is useful if you want to:
    - constructing a scene or buiding a robot
    - tune the physical parameters (*e.g.*, joint damping and stifness, light temperature, *e.t.c*)
2. Extensions: Start from the menu bar in a launched Isaac Sim, which pops out a panel for your app. You interact with your app inside. This is useful if you want to:
    - interact with your app or provide some arguments (*e.g.* the position you want the robot to go) to the app in play
    - build a plug-in for Isaac Sim, rather than just performing simulations
3. Standalone extensions: Start by running scripts, . This is useful if you want to:
    - run your app in headless mode
    - control over timing of physics and rendering steps

[This page](https://docs.omniverse.nvidia.com/isaacsim/latest/introductory_tutorials/tutorial_intro_workflows.html) provides a more thorough introduction.

## Quick-Start in Extensions

In this section, we will switch to the extensions workflow, which is a more structured way to programming, as you can leverage a pre-defined base classes `BaseSample` and you just to implement some APIs such as scene setup or physics step. Besides, saving the files in the extensions codes triggers hot-reload in Isaac Sim, which boosts your development.

### Create an Extensions with the `BaseSample` class.

Make a directory that is structured as:

```
.
├── config
│   └── extension.toml
└── scripts
    ├── __init__.py
    ├── demo.py
    └── demo_extensions.py
```

#### Edit the `extension.toml`

This file tells the Isaac Sim how to load the extension.

```ini
[core]
reloadable = true
order = 0

[package]
version = "1.0.0"
category = "Simulation"
title = "Isaac Sim Demo"
description = "Isaac Sim Demo for Simulately."
authors = ["Jane Doe"]
repository = ""
keywords = ["simulately", "simulation", "robot"]
writeTarget.kit = true

[dependencies]
"omni.kit.uiapp" = {}
"omni.physx" = {}
"omni.isaac.dynamic_control" = {}
"omni.isaac.motion_planning" = {}
"omni.isaac.synthetic_utils" = {}
"omni.isaac.ui" = {}
"omni.isaac.core" = {}
"omni.isaac.franka" = {}
"omni.isaac.manipulators" = {}
"omni.isaac.dofbot" = {}
"omni.isaac.universal_robots" = {}
"omni.isaac.motion_generation" = {}
"omni.graph.action" = {}
"omni.graph.nodes" = {}
"omni.graph.core" = {}
"omni.isaac.quadruped" = {}
"omni.isaac.wheeled_robots" = {}

[[python.module]]
name = "scripts"

[[test]]
timeout = 960
```

#### Create a Base Extension in `demo.py`

```python
import numpy as np
from omni.isaac.core.objects import FixedCuboid, DynamicCuboid
from omni.isaac.examples.base_sample import BaseSample

class Demo(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        return

    def setup_scene(self):
        # Used to add assets.
        # Only called to load the world starting from an EMPTY stage.
        pass

    async def setup_post_load(self):
        pass

    def physics_step(self, step_size):
        pass
```

- The `setup_scene` allows us to add assets to the scene. It is called to load the world, only if the stage is totally empty.
- The `setup_post_load` is called upon hitting the "LOAD" button, working in an asynchronized way. It comes in handy to get the latest simulation handles.
- The `physics_step` is called in each simulation step when the simulation is running. You can add control logics to it to control the scene.


#### Set Up Extension Loading in `demo_extensions.py`

This file loads essential classes from the surrounding files to load the extension in to the simulation app.

```python
import os
from omni.isaac.examples.base_sample import BaseSampleExtension
from .demo import Demo
from omni.kit.menu.utils import add_menu_items, remove_menu_items
from omni.isaac.ui.menu import make_menu_item_description

EXTENSION_TITLE = "Launch Demo"

class DemoExtension(BaseSampleExtension):
    def on_startup(self, ext_id: str):
        super().on_startup(ext_id)
        super().start_extension(
            menu_name="",
            submenu_name="",
            name="Demo",
            title="Isaac Sim Demo",
            doc_link="",
            overview="Simulately Isaac Sim Demo",
            file_path=os.path.abspath(__file__),
            sample=Demo()
        )
```

#### Import Essentials in `__init__.py`

Just like any other `__init__.py`, this file imports essential libraries to make a package:

```python
from .demo import Demo
from .demo_extension import DemoExtension
```

### Set Up the Scene in `setup_scene`

We will set up the scene using the `setup_scene` function. We first add a ground plane to hold everything:

```python
def setup_scene(self):
    # ...

    world = self.get_world()
    world.scene.add_default_ground_plane()

    # ...
```

It first get the handle to the world to manage it with `self.get_world()`. Then, it adds a default ground plane with `add_default_ground_plane()`, which adds a blue-white plane to the world, which also includes a sphere light and other essential components.

Now, we will add a robot and some objects to the scene. For more information concerning stage management, please refer to [Stage Management in Isaac Sim](./stage_management)

#### Add a Robot

Then, we add a robot to the scene.


```python
def setup_scene(self):
    # ...

    world = self.get_world()

    # ...

    world.scene.add(Franka(prim_path="/World/Franka", name="franka"))
    self._franka = world.scene.get_object(f"franka")
```

#### Add a Table with a Cube

Now, let's add a table (a fixed cuboid) as well a cube on it (a dynamic cuboid), in front of the robot.

```python
def setup_scene(self):
    # ...

    world = self.get_world()

    # ...

    world.scene.add(
        FixedCuboid(
            prim_path=f"/World/table",
            name=f"table",
            scale=np.array([0.5, 1.25, 0.2]),
            color=np.array([1.0, 1.0, 1.0]),
            position=np.array([0.7, 0.0, 0.1])
        )
    )

    world.scene.add(self._franka)
    fancy_cube = world.scene.add(
        DynamicCuboid(
            prim_path="/World/cube",
            name="cube",
            scale=np.array([0.2, 0.2, 0.2]),
            color=np.array([0, 0, 1.0]),
            position=np.array([0.7, 0.0, 0.3])
    ))

    self._table = world.scene.get_object(f"table")
    self._cube = world.scene.get_object("cube")
```

### Set Up Physics Step Callback

In each simulation step, before computing the physics, you can manipulate the simulation with a callback function, which we usually set up using:

```python
async def setup_post_load(self):
    self._world = self.get_world()
    self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
    # ...

def physics_step(self, step_size):
    pass
```

### Physics States

To know about the physics states in the simulation, we can leverage APIs through the `World` instance.

```python
async def setup_post_load(self):
    # ...

    self._cube = self._world.scene.get_object("cube")

    # ...

def physics_step(self, step_size):
    # ...

    pos, rot = self._cube.get_world_pose()
    lin_vel = self._cube.get_linear_velocity()

    # ...
```

### Robot Control

To control the robot, there are several ways. The most direct way is to apply action while providing the desired joint positions:

```python
def physics_step(self, step_size):
    # ...

    self._franka.apply_action(ArticulationAction([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])) # Desired joint positions for arm
    # or
    self._franka.apply_action(ArticulationAction([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.04])) # Desired joint positions for arm and gripper

    # ...
```

This will control the robot to the given joint positions with a controller, and the joint positions in the list is matched following the joint indices. Also, for a robot like Franka Emika Panda, the built-in implementation allows you to interact with the `Robot` and `Articulation` APIs, for instance,

```python
def physics_step(self, step_size):
    # ...

    self._franka.apply_action(ArticulationAction([0.04, 0.04])) # Desired joint positions for gripper
    
    # ...
```

For more advanced ways of controlling the robot, consider referring to [RMPFlow](./rmpflow), [Controlling Robots with ROS Bridge](./control-robots-ros-bridge), *e.t.c*.

## Quick-Start in Standalone Extensions

Coding in Isaac Sim endows powerful simulations. In this section, we will show procedures to migrate the extension code to the standalone extension version. For more information, you can refer to [this page](https://docs.omniverse.nvidia.com/isaacsim/latest/core_api_tutorials/tutorial_core_hello_world.html#converting-the-example-to-a-standalone-application).

A standalone extension is a python script. To run it, go to your installation directory of your isaac sim (`INSTALL_DIRECTORY/pkg/isaac-sim-VERSION`), you will find a `python.sh` here. To start a standalone app, you run:

```shell
./python.sh PATH_TO_YOUR_APP.py
```

### Migrate to / Create a Standalone Extension

Now, let's add something to our script. First, we launch the simulation application with

```python
from omni.isaac.kit import SimulationApp
app = SimulationApp({"headless": False})

from omni.isaac.core import World
# other imports
```

This two lines launches the Kit window, and are recommended to be added as the very first two lines of your script, as importing other dependencies before them may cuase problems.

### Obtaining the `World` instance

To obtain the `World` instance, use:

```python
from omni.isaac.core import World

world = World()
```

### Simulation steps

The substitute for using the `physics_step` function as the callback is to create a loop and step the simulation inside:

```python
world.reset()
N_STEPS = 10000

for i in range(N_STEPS):
    # Do your things

    world.step(render=True)
```

### Closing the simulation app

In the end, close the simulation app:

```python
app.close()
```

