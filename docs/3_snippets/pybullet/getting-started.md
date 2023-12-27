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

PyBullet provides `p.setJointMotorControlArray()` API to control the robot, supporting POSITION_CONTROL, VELOCITY_CONTROL, TORQUE_CONTROL and PD_CONTROl. For more details, please refers to [Docs](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.jxof6bt5vhut).

Consider the following two physical equations of each joint:

<p align="center"><img src="https://math.vercel.app/?bgcolor=auto&from=q_%7Bt%2B%5Ctext%7Bd%7Dt%7D%20%3D%20q_t%20%2B%20v_t%20%5Ctext%7Bd%7Dt.svg" /></p>

<p align="center"><img src="https://math.vercel.app/?bgcolor=auto&from=v_%7Bt%2B%5Ctext%7Bd%7Dt%7D%20%3D%20v_t%20%2B%20a_t%20%5Ctext%7Bd%7Dt%20.svg" /></p>

, where *q*, *v*, *a* is the position, velocity and acceleration (a.k.a. force) of the DoF, respectively. *dt* is the time step. The robot can be controlled by 3 ways in pybullet in general.

- **Effort ( target *a* ) Control** involves applying a specified force or torque to the joints of the robot. We can use `p.setJointMotorControlArray()` to control the robot by specifying the `controlMode=p.TORQUE_CONTROL` and `force` parameter. The `force` parameter is the target acceleration of the joint. For example, the following code snippet controls the robot by applying a constant torque of 1.0 to all joints.

```python
numOfJoints = p.getNumJoints(robotId)
jointIds = range(numOfJoints)
p.setJointMotorControlArray(robotId, jointIds, controlMode=p.TORQUE_CONTROL, forces=[1.0]*numOfJoints)
```

- **Velocity ( target *v* ) Control** involves applying a specified velocity to the joints of the robot. We can use `p.setJointMotorControlArray()` to control the robot by specifying the `controlMode=p.VELOCITY_CONTROL` and `targetVelocity` parameter. The `targetVelocity` parameter is the target velocity of the joint. For example, the following code snippet controls the robot by applying a constant velocity of 1.0 to all joints.

```python
numOfJoints = p.getNumJoints(robotId)
jointIds = range(numOfJoints)
p.setJointMotorControlArray(robotId, jointIds, controlMode=p.VELOCITY_CONTROL, targetVelocities=[1.0]*numOfJoints)
```

- **Position ( target *q* ) Control** involves applying a specified position to the joints of the robot. We can use `p.setJointMotorControlArray()` to control the robot by specifying the `controlMode=p.POSITION_CONTROL` and `targetPosition` parameter. The `targetPosition` parameter is the target position of the joint. For example, the following code snippet controls the robot by applying a constant position of 1.0 to all joints.

```python
numOfJoints = p.getNumJoints(robotId)
jointIds = range(numOfJoints)
p.setJointMotorControlArray(robotId, jointIds, controlMode=p.POSITION_CONTROL, targetPositions=[1.0]*numOfJoints)
```

> **_Note:_**  The actual implementation of the joint motor controller is as a constraint for POSITION_CONTROL and VELOCITY_CONTROL, and as an external force for TORQUE_CONTROL. For POSITION_CONTROL and VELOCITY_CONTROL, it acts as a constraint. `POSITION_CONTROL` involves minimizing errors in velocity and position using defined gains. `VELOCITY_CONTROL` enforces pure velocity constraints. For `TORQUE_CONTROL`, it functions as an external force. Starting with VELOCITY_CONTROL or POSITION_CONTROL is recommended, as TORQUE_CONTROL is more challenging and relies on accurate model parameters. Accurate URDF/SDF files and system identification are crucial for simulating correct forces.

Here we offer some straightforward examples of robot control using effort, velocity, and position control methods.

<details> <summary>Effort Control for Franka Robot</summary>

```python
import time
import numpy as np
import pybullet as p
import pybullet_data

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,0)
planeId = p.loadURDF("plane.urdf")
startPos = [0,0,0]
startOrientation = p.getQuaternionFromEuler([0,0,0])
robotId = p.loadURDF("franka_panda/panda.urdf", startPos, startOrientation, useFixedBase=True)
# see the number of joint of the robot
numJoints = p.getNumJoints(robotId)
print(f'Number of joints: {numJoints}')
# find non-fixed joint
jointTypeList = []
for joint in range(numJoints):
    info = p.getJointInfo(robotId, joint)
    jointTypeList.append(info[2])
    print(f'friction and damping of joint {joint}: {info[6:8]}')
jointIds = [j for j in range(numJoints) if jointTypeList[j] != p.JOINT_FIXED]
forces = np.array([0., -1.0, 0., -1.0, 0., 1.0, 1.0, 0.1, 0.1]) * 300
controlPeriod = 240
step = 0
while True:
    p.setJointMotorControlArray(bodyUniqueId=robotId, 
                                jointIndices=jointIds, 
                                controlMode=p.TORQUE_CONTROL, 
                                forces=forces)
    p.stepSimulation()
    if step % controlPeriod == (controlPeriod - 1):
        forces = - forces * 0.1
    time.sleep(1./240.)
    step += 1

p.disconnect()
```
</details>

<details> <summary>Velocity Control for Franka Robot</summary>

```python
import time
import numpy as np
import pybullet as p
import pybullet_data

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,0)
planeId = p.loadURDF("plane.urdf")
startPos = [0,0,0]
startOrientation = p.getQuaternionFromEuler([0,0,0])
robotId = p.loadURDF("franka_panda/panda.urdf", startPos, startOrientation, useFixedBase=True)
# see the number of joint of the robot
numJoints = p.getNumJoints(robotId)
print(f'Number of joints: {numJoints}')
# find non-fixed joint
jointTypeList = []
for joint in range(numJoints):
    info = p.getJointInfo(robotId, joint)
    jointTypeList.append(info[2])
jointIds = [j for j in range(numJoints) if jointTypeList[j] != p.JOINT_FIXED]
targetVelocities = np.array([0., -1.0, 0., -2.0, 0., 1.0, 1.0, 0.1, 0.1])
controlPeriod = 240
step = 0
while True:
    p.setJointMotorControlArray(bodyUniqueId=robotId, 
                                jointIndices=jointIds, 
                                controlMode=p.VELOCITY_CONTROL, 
                                targetVelocities=targetVelocities)
    p.stepSimulation()
    if step % controlPeriod == (controlPeriod - 1):
        targetVelocities = - targetVelocities
    time.sleep(1./240.)
    step += 1

p.disconnect()
```
</details>

<details> <summary>Position Control for Franka Robot</summary>

```python
import time
import numpy as np
import pybullet as p
import pybullet_data

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,0)
planeId = p.loadURDF("plane.urdf")
startPos = [0,0,0]
startOrientation = p.getQuaternionFromEuler([0,0,0])
robotId = p.loadURDF("franka_panda/panda.urdf", startPos, startOrientation, useFixedBase=True)
# see the number of joint of the robot
numJoints = p.getNumJoints(robotId)
print(f'Number of joints: {numJoints}')
# find non-fixed joint
jointTypeList = []
for joint in range(numJoints):
    info = p.getJointInfo(robotId, joint)
    jointTypeList.append(info[2])
jointIds = [j for j in range(numJoints) if jointTypeList[j] != p.JOINT_FIXED]
zeroPositions = np.array([0. for i in range(len(jointIds))])
targetPositions = np.array([0., -0.785, 0., -2.356, 0., 1.571, 0.785, 0.04, 0.04])
controlPeriod = 240
step = 0
while True:
    desiredPositions = (step % controlPeriod) / controlPeriod * targetPositions + (1 - (step % controlPeriod) / controlPeriod) * zeroPositions
    p.setJointMotorControlArray(bodyUniqueId=robotId, 
                                jointIndices=jointIds, 
                                controlMode=p.POSITION_CONTROL, 
                                targetPositions=desiredPositions)
    p.stepSimulation()
    if step % controlPeriod == (controlPeriod - 1):
        temp = zeroPositions
        zeroPositions = targetPositions
        targetPositions = temp
    time.sleep(1./240.)
    step += 1

p.disconnect()
```
</details>