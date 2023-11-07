# Rigid Body and Shapes Properties

This page provides instructions and snippets of customizing the rigid sbody and shape properties.

### Customize rigid body and shapes properties

Add the code below to customize the rigid body and shapes properties when you create your environment.

```python
for env in range(num_envs):
    actor_handle = self.gym.create_actor(env, asset, pose, "actor", 0, 1)

    agent_body_props = self.gym.get_actor_rigid_body_properties(env, actor_handle)
    for agent_body_prop in agent_body_props:
        agent_body_prop.mass = agent.rigid_body_mass
    self.gym.set_actor_rigid_body_properties(env, actor_handle, agent_body_props)

    agent_shape_props = self.gym.get_actor_rigid_shape_properties(env, actor_handle)
    for agent_shape_prop in agent_shape_props:
        agent_shape_prop.compliance = agent.rigid_shape_compliance
        agent_shape_prop.contact_offset = agent.rigid_shape_contact_offset
        agent_shape_prop.filter = agent.rigid_shape_filter
        agent_shape_prop.friction = agent.rigid_shape_friction
        agent_shape_prop.rest_offset = agent.rigid_shape_rest_offset
        agent_shape_prop.restitution = agent.rigid_shape_restitution
        agent_shape_prop.rolling_friction = agent.rigid_shape_rolling_friction
        agent_shape_prop.thickness = agent.rigid_shape_thickness
        agent_shape_prop.torsion_friction = agent.rigid_shape_torsion_friction
    self.gym.set_actor_rigid_shape_properties(env, actor_handle, agent_shape_props)  
```
