# Controlling Robots with ROS Bridge

This page uses ROS2 Bridge. Similar actions can be also be easily find for ROS Bridge.

## Action Graph for Robot Joint States / Commands

This piece of code creates an action graph at `GRAPH_PATH` for robot at `ROBOT_PRIM_PATH`. It publishes joint states to `/joint_states` and subscribe goal state commands from `/joint_commands` via ROS2 Bridge.

> Note that the action graph should be under the prim of the robot it aims to control.

You need to first obtain the current stage (you can get it by calling `self.get_stage()` in derive classes of `BaseSample`) as `stage`. You also need to import:

```python
import omni.graph.core as og
```

To create the graph:

```python
ROBOT_PRIM_PATH = "/robot"
GRAPH_PATH = "/robot/JointGraph"

(moveit_graphkeys = og.Controller.Keys

 _, _, _) = og.Controller.edit(
    {"graph_path": GRAPH_PATH, "evaluator_name": "execution"},
    {
        og.Controller.Keys.CREATE_NODES: [
            ("OnImpulseEvent", "omni.graph.action.OnImpulseEvent"),
            ("ReadSimTime", "omni.isaac.core_nodes.IsaacReadSimulationTime"),
            ("Context", "omni.isaac.ros2_bridge.ROS2Context"),
            ("PublishClock", "omni.isaac.ros2_bridge.ROS2PublishClock"),
            ("PublishJointState", "omni.isaac.ros2_bridge.ROS2PublishJointState"),
            ("SubscribeJointState", "omni.isaac.ros2_bridge.ROS2SubscribeJointState"),
            ("ArticulationController", "omni.isaac.core_nodes.IsaacArticulationController"),
            ("PublishTF", "omni.isaac.ros2_bridge.ROS2PublishTransformTree"),
        ],
        og.Controller.Keys.CONNECT: [
            ("OnImpulseEvent.outputs:execOut", "PublishJointState.inputs:execIn"),
            ("OnImpulseEvent.outputs:execOut", "SubscribeJointState.inputs:execIn"),
            ("OnImpulseEvent.outputs:execOut", "PublishTF.inputs:execIn"),
            ("OnImpulseEvent.outputs:execOut", "PublishClock.inputs:execIn"),
            ("OnImpulseEvent.outputs:execOut", "ArticulationController.inputs:execIn"),
            ("Context.outputs:context", "PublishJointState.inputs:context"),
            ("Context.outputs:context", "SubscribeJointState.inputs:context"),
            ("Context.outputs:context", "PublishTF.inputs:context"),
            ("Context.outputs:context", "PublishClock.inputs:context"),
            ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
            ("ReadSimTime.outputs:simulationTime", "PublishTF.inputs:timeStamp"),
            ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
            ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
            ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
            ("SubscribeJointState.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
            ("SubscribeJointState.outputs:effortCommand", "ArticulationController.inputs:effortCommand"),
        ],
        og.Controller.Keys.SET_VALUES: [
            ("ArticulationController.inputs:usePath", True),
            ("ArticulationController.inputs:robotPath", ROBOT_PRIM_PATH),
            ("PublishJointState.inputs:topicName", "joint_states"),
            ("SubscribeJointState.inputs:topicName", "joint_commands"),
        ],
    },
)

set_targets(
    prim=stage.GetPrimAtPath(f"{GRAPH_PATH}/PublishJointState"),
    attribute="inputs:targetPrim",
    target_prim_paths=[ROBOT_PRIM_PATH],
)
```

You also need this in your physics step function to set off the impulse:

```python
og.Controller.set(og.Controller.attribute(f"{GRAPH_PATH}/OnImpulseEvent.state:enableImpulse"), True)

```
