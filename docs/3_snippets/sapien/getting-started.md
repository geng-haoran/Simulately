---
sidebar_position: 0
---

# Getting Started
Welcome to the Getting Started for using SAPIAN.
In this guide, we will walk through the process of initializing the simulator, loading a groud, creating light and loading actors with SAPIEN API. The provided code snippets are adapted from the SAPIEN Docs to help you get started quickly.

Try our demo code at the github `Simulately/code/getting_started`!

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

Try our demo code at the github `Simulately/code/getting_started/sapien_1_helloworld.py`!

<details> <summary>Programming script of hello_world.py</summary>

```python
    """Hello world for Sapien.

    Concepts:
        - Engine and scene
        - Renderer, viewer, lighting
        - Run a simulation loop

    Notes:
        - For one process, you can only create one engine and one renderer.
    """

    import sapien.core as sapien
    from sapien.utils import Viewer
    import numpy as np


    def main():
        engine = sapien.Engine()  # Create a physical simulation engine
        renderer = sapien.SapienRenderer()  # Create a renderer
        engine.set_renderer(renderer)  # Bind the renderer and the engine

        scene = engine.create_scene()  # Create an instance of simulation world (aka scene)
        scene.set_timestep(1 / 100.0)  # Set the simulation frequency

        # NOTE: How to build actors (rigid bodies) is elaborated in create_actors.py
        scene.add_ground(altitude=0)  # Add a ground
        actor_builder = scene.create_actor_builder()
        actor_builder.add_box_collision(half_size=[0.5, 0.5, 0.5])
        actor_builder.add_box_visual(half_size=[0.5, 0.5, 0.5], color=[1., 0., 0.])
        box = actor_builder.build(name='box')  # Add a box
        box.set_pose(sapien.Pose(p=[0, 0, 0.5]))


        # Add some lights so that you can observe the scene
        scene.set_ambient_light([0.5, 0.5, 0.5])
        scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

        viewer = Viewer(renderer)  # Create a viewer (window)
        viewer.set_scene(scene)  # Bind the viewer and the scene

        # The coordinate frame in Sapien is: x(forward), y(left), z(upward)
        # The principle axis of the camera is the x-axis
        viewer.set_camera_xyz(x=-4, y=0, z=2)
        # The rotation of the free camera is represented as [roll(x), pitch(-y), yaw(-z)]
        # The camera now looks at the origin
        viewer.set_camera_rpy(r=0, p=-np.arctan2(2, 4), y=0)
        viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)

        while not viewer.closed:  # Press key q to quit
            scene.step()  # Simulate the world
            scene.update_render()  # Update the world to the renderer
            viewer.render()


    if __name__ == '__main__':
        main()
```

</details>

## Load a robot and control it

After we can setup an environment, the next step is to load a robot and control it. In this section, we will load a robot from a URDF file and control it.

### Load a robot from a URDF file

```python
loader: sapien.URDFLoader = scene.create_urdf_loader()
loader.fix_root_link = True
robot: sapien.Articulation = loader.load("../assets/robot/panda/panda.urdf")
robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
```

### Get basic control information about the loaded robot

From the following code snippet, we can get the basic control information about the loaded robot, including the joint names, the joint limits, the joint types, the joint damping and the joint stiffness. Notice that not all the joints are active. You need to find out the correct active joints to control the robot (`robot.get_active_joints()` can tell you).

```python
# get current qpos and jpints
joints = robot.get_joints() # all joints
active_joints = robot.get_active_joints() # active joints
for joint in joints:
    name = joint.name
    damping = joint.damping
    stiffness = joint.stiffness
    friction = joint.friction
    limits = joint.get_limits()
    child_link_name = joint.get_child_link().name if joint.get_child_link() is not None else None
    parent_link_name = joint.get_parent_link().name if joint.get_parent_link() is not None else None
    print(
        "name: ", name, "damping: ", damping, 
        "stiffness: ", stiffness, "friction: ", friction, 
        "limits: ", limits, "child_link_name: ", child_link_name, 
        "parent_link_name: ", parent_link_name
    )
joint_names = [joint.name for joint in joints]
active_joint_names = [joint.name for joint in active_joints]
print("joint_names: ", joint_names)
print("active_joint_names: ", active_joint_names)
cur_qpos = robot.get_qpos()
print("cur_qpos: ", cur_qpos)
```

### Drive the robot with the PhysX internal PD Controller

SAPIEN provides the builtin PhysX internal PD controller to control either the position or speed of a joint.

```python
active_joints = robot.get_active_joints()
if use_internal_drive:
    for joint_idx, joint in enumerate(active_joints):
        joint.set_drive_property(stiffness=20, damping=5)
        joint.set_drive_target(target_qpos[joint_idx])
    # Or you can directly set joint targets for an articulation
    # robot.set_drive_target(target_qpos)
```

Then apply force during environment step

```python
while not viewer.closed:
    for _ in range(4):  # render every 4 steps
        qf = robot.compute_passive_force(
            gravity=True,
            coriolis_and_centrifugal=True,
        )
        qf += pid_qf
        robot.set_qf(qf)
        scene.step()
    scene.update_render()
    viewer.render()
```

### Drive the robot with the PhysX external PID implementation

We can implement our own PID controller and drive the robot with it. The following code snippet shows how to implement a PID controller and drive the robot with it.

Taking a simple PID controller as an example, we can implement a PID controller as follows:

```python
class SimplePID:
    def __init__(self, kp=0.0, ki=0.0, kd=0.0):
        self.p = kp
        self.i = ki
        self.d = kd

        self._cp = 0
        self._ci = 0
        self._cd = 0

        self._last_error = 0

    def compute(self, current_error, dt):
        self._cp = current_error
        self._ci += current_error * dt
        self._cd = (current_error - self._last_error) / dt
        self._last_error = current_error
        signal = (self.p * self._cp) + \
            (self.i * self._ci) + (self.d * self._cd)
        return signal


def pid_forward(pids: list,
                target_pos: np.ndarray, 
                current_pos: np.ndarray, 
                dt: float) -> np.ndarray:
    errors = target_pos - current_pos
    qf = [pid.compute(error, dt) for pid, error in zip(pids, errors)]
    return np.array(qf)
```

Then we can drive the robot with the PID controller as follows:

```python
pids = []
pid_parameters = [
    (40, 5, 2), (40, 5, 2), (40, 5, 2), (20, 5.0, 2),
    (5, 0.8, 2), (5, 0.8, 2), (5, 0.8, 0.4),
    (0.1, 0, 0.02), (0.1, 0, 0.02), (0.1, 0, 0.02),
    (0.1, 0, 0.02), (0.1, 0, 0.02), (0.1, 0, 0.02),
]
for i, joint in enumerate(active_joints):
    pids.append(SimplePID(*pid_parameters[i]))
```

Then apply force during environment step

```python
while not viewer.closed:
    for _ in range(4):  # render every 4 steps
        qf = robot.compute_passive_force(
            gravity=True,
            coriolis_and_centrifugal=True,
        )
        pid_qf = pid_forward(
            pids,
            target_qpos,
            robot.get_qpos(),
            scene.get_timestep()
        )
        qf += pid_qf
        robot.set_qf(qf)
        scene.step()
    scene.update_render()
    viewer.render()
```

Try our demo code at the github `Simulately/code/getting_started/sapien_1_helloworld.py` with

```bash
python sapien_2_control_robot.py --use-internal-drive # use internal drive
python sapien_2_control_robot.py --use-external-pid   # use external pid
```

<details> <summary>Programming script of sapien_2_control_robot.py</summary>

```python
import sapien.core as sapien
from sapien.utils.viewer import Viewer
import numpy as np


class SimplePID:
    def __init__(self, kp=0.0, ki=0.0, kd=0.0):
        self.p = kp
        self.i = ki
        self.d = kd

        self._cp = 0
        self._ci = 0
        self._cd = 0

        self._last_error = 0

    def compute(self, current_error, dt):
        self._cp = current_error
        self._ci += current_error * dt
        self._cd = (current_error - self._last_error) / dt
        self._last_error = current_error
        signal = (self.p * self._cp) + \
            (self.i * self._ci) + (self.d * self._cd)
        return signal


def pid_forward(pids: list,
                target_pos: np.ndarray, 
                current_pos: np.ndarray, 
                dt: float) -> np.ndarray:
    errors = target_pos - current_pos
    qf = [pid.compute(error, dt) for pid, error in zip(pids, errors)]
    return np.array(qf)


def demo(use_internal_drive, use_external_pid):
    engine = sapien.Engine()
    renderer = sapien.VulkanRenderer()
    engine.set_renderer(renderer)

    scene_config = sapien.SceneConfig()
    scene = engine.create_scene(scene_config)
    # A small timestep for higher control accuracy
    scene.set_timestep(1 / 2000.0)
    scene.add_ground(0)


    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    viewer = Viewer(renderer)
    viewer.set_scene(scene)
    viewer.set_camera_xyz(x=-2, y=0, z=1)
    viewer.set_camera_rpy(r=0, p=-0.3, y=0)

    # Load URDF
    loader: sapien.URDFLoader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot: sapien.Articulation = loader.load("../assets/robot/panda/panda.urdf")
    robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))

    # get current qpos
    
    
    joints = robot.get_joints()
    for joint in joints:
        name = joint.name
        damping = joint.damping
        stiffness = joint.stiffness
        friction = joint.friction
        limits = joint.get_limits()
        child_link_name = joint.get_child_link().name if joint.get_child_link() is not None else None
        parent_link_name = joint.get_parent_link().name if joint.get_parent_link() is not None else None
        print(
            "name: ", name, "damping: ", damping, 
            "stiffness: ", stiffness, "friction: ", friction, 
            "limits: ", limits, "child_link_name: ", child_link_name, 
            "parent_link_name: ", parent_link_name
        )
    joint_names = [joint.name for joint in joints]
    print("joint_names: ", joint_names)
    cur_qpos = robot.get_qpos()
    print("cur_qpos: ", cur_qpos)
    
    # Set joint positions
    init_qpos = cur_qpos
    robot.set_qpos(init_qpos)
    target_qpos = [1, 1, 1, 1, 1, 1, 1, 0.04, 0.04]

    active_joints = robot.get_active_joints()
    # Or other equivalent way to get active joints
    # active_joints = [joint for joint in robot.get_joints() if joint.get_dof() > 0]

    if use_internal_drive:
        for joint_idx, joint in enumerate(active_joints):
            joint.set_drive_property(stiffness=20, damping=5)
            joint.set_drive_target(target_qpos[joint_idx])
        # Or you can directly set joint targets for an articulation
        # robot.set_drive_target(target_qpos)

    if use_external_pid:
        pids = []
        pid_parameters = [
            (40, 5, 2), (40, 5, 2), (40, 5, 2), (40, 5, 2), 
            (40, 5, 2), (40, 5, 2), (40, 5, 2), (0.1, 0, 0.02), (0.1, 0, 0.02),
        ]
        for i, joint in enumerate(active_joints):
            pids.append(SimplePID(*pid_parameters[i]))

    while not viewer.closed:
        for _ in range(4):  # render every 4 steps
            qf = robot.compute_passive_force(
                gravity=True,
                coriolis_and_centrifugal=True,
            )
            if use_external_pid:
                pid_qf = pid_forward(
                    pids,
                    target_qpos,
                    robot.get_qpos(),
                    scene.get_timestep()
                )
                qf += pid_qf
            robot.set_qf(qf)
            scene.step()
        scene.update_render()
        viewer.render()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-internal-drive', action='store_true')
    parser.add_argument('--use-external-pid', action='store_true')
    args = parser.parse_args()

    demo(use_internal_drive=args.use_internal_drive,
         use_external_pid=args.use_external_pid)


if __name__ == '__main__':
    main()
```


</details>
