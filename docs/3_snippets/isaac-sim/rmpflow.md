# Using RMPFlow to Control Manipulators

[RMPFlow](https://arxiv.org/abs/1811.07049) is a policy synthesis algorithm based on geometrically consistent transformations of Riemannian Motion Policies (RMPs). It can be used to plan motion sequences on high-DoF manipulators in scenes with obstacles.

All Lula algorithms, including RMPFlow, requires a description file for the robot. Isaac Sim supports four manipulators by default, so you can directly use them. To support your robot, use the [Lula robot description editor](https://docs.omniverse.nvidia.com/isaacsim/latest/advanced_tutorials/tutorial_motion_generation_robot_description_editor.html).

Related information in Omniverse Docs:

- [Riemannian Motion Policy](https://docs.omniverse.nvidia.com/isaacsim/latest/reference_glossary.html#riemannian-motion-policy-rmp)
- [RMPFlow](https://docs.omniverse.nvidia.com/isaacsim/latest/concepts/motion_generation/index.html?highlight=RMP%20Flow#rmpflow)
- [Motion Generation](https://docs.omniverse.nvidia.com/isaacsim/latest/advanced_tutorials/tutorials_advanced_motion_generation.html)

## Initialize an RMPFlow Controller on Franka

The following code spawns and controlls a Franka Panda robot with RMPFlow. The code can be used in a `BaseSample` example or a standalone one. [This page](./getting-started) guides you to create an app with this class.

Franka Panda is one of the four robots that is officially supported. Thus, you can directly import the classes without specifying any configuration.

```python
from omni.isaac.franka import Franka
from omni.isaac.franka.controllers import RMPFlowController
```

```python
franka = Franka(prim_path="/Franka", name=f"manipulator")
controller = RMPFlowController(name=f"controller", robot_articulation=franka)
```

To use the controller to generate an action:

```python
target_pos = np.array([0.0, 0.0, 1.0])
target_rot = np.array([1.0, 0.0, 0.0, 0.0]) # wxyz quaternion

actions = controller.forward(
        target_end_effector_position=target_pos,
        target_end_effector_orientation=target_rot,
)
franka.apply_action(actions)
```

## Modify configurations for RMPFlow

By default, RMPFlow controller for franka uses the `right_gripper` frame as the target frame. However, if you want to controll another frame, you need to manually create an `RmpFlow` class instance and specify the target frame as `end_effector_frame_name`. You can check the URDF file the controller use at `isaac_sim-2022.2.1/exts/omni.isaac.motion_generation/motion_policy_configs/franka/lula_franka_gen.urdf`.

For example, to controll the `panda_hand` frame:

```python
from omni.isaac.franka import Franka
from omni.isaac.motion_generation import RmpFlow, ArticulationMotionPolicy
from omni.isaac.franka.controllers import RMPFlowController
from omni.isaac.core.utils.extensions import get_extension_path_from_name

franka = Franka(prim_path="/Franka", name=f"manipulator")

mg_extension_path = get_extension_path_from_name("omni.isaac.motion_generation")
rmp_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")
rmpflow = RmpFlow(
    robot_description_path = rmp_config_dir + "/franka/rmpflow/robot_descriptor.yaml",
    urdf_path = rmp_config_dir + "/franka/lula_franka_gen.urdf",
    rmpflow_config_path = rmp_config_dir + "/franka/rmpflow/franka_rmpflow_common.yaml",
    end_effector_frame_name = "panda_hand",
    maximum_substep_size = 0.00334
)

art_policy = ArticulationMotionPolicy(franka, rmpflow)
rmpflow_controller = RMPFlowController(name=f"controller", robot_articulation=franka)

# Then you can use the rmpflow controller
```
You can also modify these YAML files to customize the controller you want.

## Robot on a moving base

You need to pass the base pose of the manipulator if it is not in the canonical pose:

```python
base_translation, base_orientation = franka.get_world_pose()
rmpflow.set_robot_base_pose(base_translation, base_orientation)
```

## Add obstacles

You can add obstacles that the robot should avoid. Note that Isaac Sim only support several primitives, and it also use several spheres to represent the robot collision, so while it is efficient, it compromises in precision:

```python
obstacle = FixedCuboid("/Obstable", size=0.1,position=np.array([0.4, 0.0, 0.4]),color=np.array([0.,0.,1.]))
rmpflow.add_obstacle(obstacle)
rmpflow.update_world()
```
