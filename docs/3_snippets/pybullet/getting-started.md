---
sidebar_position: 0
---

# Getting Started
Welcome to the Getting Started for using PyBullet. In this guide, we we will walk through the process of connecting to the simulator, setting up the environment, loading objects and steping simulation. The provided code snippets are adapted from the [PyBullet Docs](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA) to help you get started quickly.

## Initial the Simulator

### Connect to the Simulator

Like most python script programming, we import the basic modules of PyBullet without much thought. Notably, PyBullet is designed around a client-server driven API, with a client sending commands and a physics server returning the states. So the first thing we need to do is creating the command-client to connect to the physics-server.

```python
import pybullet as p
import pybullet_data

pClient = p.connect(p.GUI) 
## or p.DIRECT for non-graphical processing
## You can create multiple DIRECT clients but only one GUI connection.
```

<details> <summary>Details of connecting DIRECT, GUI from <a href="https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA">Docs</a></summary>

- The DIRECT connection sends the commands directly to the physics engine, without using any transport layer and no graphics visualization window, and directly returns the status after executing the command.

- The GUI connection will create a new graphical user interface (GUI) with 3D OpenGL rendering, within the same process space as PyBullet. On Linux and Windows this GUI runs in a separate thread, while on OSX it runs in the same thread due to operating system limitations. On Mac OSX you may see a spinning wheel in the OpenGL Window, until you run a 'stepSimulation' or other PyBullet command.

</details>

### Set up the Environment

Most simulations require gravity settings and a ground plane. In PyBullet, the environment setup is straightforward:

```python
## set addtional search path for data files
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
## plane.urdf would be searched in the addtional part
planeId = p.loadURDF("plane.urdf")
```

## Construct an Environment

### Load the Robot

Once you want to load something into the environment, [URDF](https://www.mathworks.com/help/sm/ug/urdf-model-import.html), [SDF](http://sdformat.org/) and [MJCF](https://mujoco.readthedocs.io/en/latest/modeling.html) description formats are supported by PyBullet.

```python
## set base pose for the robot
startPos = [0, 0, 2.]
startOrientation = p.getQuaternionFromEuler([0., 0., 0.])
## load the robot from urdf file
robotId = p.loadURDF("r2d2.urdf", startPos, startOrientation, useFixedBase=False)
```

Use `useFixedBase=False` to allow freedom for the base of the robot; setting it to True would make the base static. For more details of loading actors, please refers to [Docs](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.sbnykoneq1me).


### Simulation

After configuring the environment, simulation steps can be performed. A simulation step encompasses collision detection, constraint solving, integration, etc.

```python
while True:
    p.stepSimulation()
    ## by default, the simulation will run at 240Hz
    ## we manually sync the simulation with real time
    time.sleep(1./240.)
    ## we can simply get the base states of the robot
    curPos, curOrn = p.getBasePositionAndOrientation(robotId)
    print(f'Current Position: {curPos}', f'Curren Orientation: {curOrn}')
```

<details> <summary>Programming script of hello.world.py</summary>

```python
import time
import pybullet as p
import pybullet_data

pClient = p.connect(p.GUI) 
## or p.DIRECT for non-graphical processing
## You can create multiple DIRECT clients but only one GUI connection.

## set addtional search path for data files
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
## plane.urdf would be searched in the addtional part
planeId = p.loadURDF("plane.urdf")

## set base pose for the robot
startPos = [0, 0, 2.]
startOrientation = p.getQuaternionFromEuler([0., 0., 0.])
## load the robot from urdf file
robotId = p.loadURDF("r2d2.urdf", startPos, startOrientation, useFixedBase=False)

while True:
    p.stepSimulation()
    ## by default, the simulation will run at 240Hz
    ## we manually sync the simulation with real time
    time.sleep(1./240.)
    ## we can simply get the base states of the robot
    curPos, curOrn = p.getBasePositionAndOrientation(robotId)
    print(f'Current Position: {curPos}', f'Curren Orientation: {curOrn}')

p.disconnect()
```

</details>

## Control the Robot

*Pending...*