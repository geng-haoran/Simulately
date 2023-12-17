# Solving Inverse Kinematics with LULA Kinematic Solver

You can use Lula kinematic solver to solve FK and IK for your robot. Just like the [RMPFlow](./rmpflow), Isaac Sim supports Franka Panda and several other manipulators. You can also add your robots by making configurations for them.

All Lula algorithms, including RMPFlow, requires a description file for the robot. Isaac Sim supports four manipulators by default, so you can directly use them. To support your robot, use the [Lula robot description editor](https://docs.omniverse.nvidia.com/isaacsim/latest/advanced_tutorials/tutorial_motion_generation_robot_description_editor.html).

```python
from omni.isaac.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver, interface_config_loader

kinematics_config = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
_kine_solver = LulaKinematicsSolver(**kinematics_config)
_art_kine_solver = ArticulationKinematicsSolver(_franka, _kine_solver, "right_gripper")
```

You can also check all the frames available in the robot:

```python
all_frame_names = _kine_solver.get_all_frame_names()
```

To solve the inverse kinematics, you need to specify the robot base pose as well as the desired target pose. The solver will generate a motion action to reach the target pose.

```python
# Desired end effector pose
target_pos = np.array([0.0, 0.0, 1.0])
target_rot = np.array([1.0, 0.0, 0.0, 0.0]) # wxyz quaternion

# Solve IK
robot_base_translation, robot_base_orientation = franka.get_world_pose()
kine_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)
action, success = art_kine_solver.compute_inverse_kinematics(target_pos, target_rot)

# Apply action
if success:
    franka.apply_action(action)
else:
    print("IK failed")
```

> Of note, if you use Tasks and creates multiple environments with offsets, add the offset to `target_pos` here.
