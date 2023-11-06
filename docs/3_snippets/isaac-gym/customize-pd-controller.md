# Customize PD controller in IsaacGym

This page provides instructions and snippets of customizing PD controller in IsaacGym.

### Set the actor control mode
If you want to use your hand-write PD controller in IsaacGym, first you need to set the driveMode and the stiffness:

```python
allegro_hand_dof_props = self.gym.get_asset_dof_properties(allegro_hand_asset)

for i in range(self.num_allegro_hand_dofs):
    allegro_hand_dof_props['driveMode'][i] = gymapi.DOF_MODE_EFFORT
    allegro_hand_dof_props['stiffness'][i] = 0
    allegro_hand_dof_props['effort'][i] = 200
    allegro_hand_dof_props['damping'][i] = 80

```
### Set the parameters of the PD controller
Set the parameters of the PD controller when you create your task.

```python
self.p_gain_val = 100.0
self.d_gain_val = 4.0
self.p_gain = torch.ones((self.num_envs, self.num_allegro_hand_dofs * 2), device=self.device, dtype=torch.float) * self.p_gain_val
self.d_gain = torch.ones((self.num_envs, self.num_allegro_hand_dofs * 2), device=self.device, dtype=torch.float) * self.d_gain_val

self.pd_previous_dof_pos = torch.zeros((self.num_envs, self.num_allegro_hand_dofs * 2), device=self.device, dtype=torch.float) * self.p_gain_val
self.pd_dof_pos = torch.zeros((self.num_envs, self.num_allegro_hand_dofs * 2), device=self.device, dtype=torch.float) * self.p_gain_val

self.debug_target = []
self.debug_qpos = []
```

### Update the PD controller
We need to use our custom `update_controller` function instead of IsaacGym's built-in PD controller to give torque to the robot joints. First, let's define the `update_controller` function. 

```python
def update_controller(self):
    self.pd_previous_dof_pos[:, :22] = self.allegro_hand_dof_pos.clone()
    self.pd_previous_dof_pos[:, 22:44] = self.allegro_hand_another_dof_pos.clone()

    self.gym.refresh_dof_state_tensor(self.sim)
    self.gym.refresh_actor_root_state_tensor(self.sim)
    self.gym.refresh_rigid_body_state_tensor(self.sim)
    self.gym.refresh_net_contact_force_tensor(self.sim)

    self.pd_dof_pos[:, :22] = self.allegro_hand_dof_pos.clone()
    self.pd_dof_pos[:, 22:44] = self.allegro_hand_another_dof_pos.clone()
    dof_vel = (self.pd_dof_pos - self.pd_previous_dof_pos) / self.dt
    self.dof_vel_finite_diff = dof_vel.clone()
    torques = self.p_gain * (self.cur_targets - self.pd_dof_pos) - self.d_gain * dof_vel
    self.torques = torques.clone()
    self.torques = torch.clip(self.torques, -2000.0, 2000.0)
    if self.debug_viz:
        self.debug_target.append(self.cur_targets[:, 6:].clone())
        self.debug_qpos.append(self.arm_hand_dof_pos[:, 6:].clone())
    self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))

    return
```

After that, in each step of the tasks, use `update_controller` instead of the `gym.set_dof_position_target_tensor` API.

```python
# self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))
self.update_controller()
```



