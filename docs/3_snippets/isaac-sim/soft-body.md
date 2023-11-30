# Deformable Body Simulation

Isaac Sim supports deformable body simulation on GPU. [This page](https://docs.omniverse.nvidia.com/extensions/latest/ext_physics/deformable-bodies.html) in the Omniverse extensions documentation provides detailed (still inadequate yet) information. In this page, we provide some handy snippets.


## Create a deformable mesh for deformable body

As an example, this code creates a cube mesh via the `kit` command:

```python

def create_cube_mesh(stage, target_path):
    _, tmp_path = omni.kit.commands.execute("CreateMeshPrim", prim_type="Cube")
    omni.kit.commands.execute("MovePrim", path_from=tmp_path, path_to=target_path)
    omni.usd.get_context().get_selection().set_selected_prim_paths([], False)
    return UsdGeom.Mesh.Get(stage, target_path)

# Add a cube mesh to `/World/deformable`
stage = omni.usd.get_context().get_stage()
deformable_prim_path = "/World/deformable"
cube_mesh = create_cube_mesh(stage, deformable_prim_path)

# You can specify its pose and scale:
pos = ...
rot = ...
scale = ...
cube_mesh.AddTranslateOp().Set(tuple(Gf.Vec3f(*pos)))
cube_mesh.AddOrientOp().Set(Gf.Quatf(*rot))
cube_mesh.AddScaleOp().Set(tuple(Gf.Vec3f(*scale*)))

# Add PhysX deformable body
deformableUtils.add_physx_deformable_body(
    stage,
    deformable_prim_path,
    simulation_hexahedral_resolution=4,
    collision_simplification=True,
    self_collision=False,
    solver_position_iteration_count=20,
)
```

`simulation_hexahedral_resolution` specifies the resolution of the hexahedron (the blue ones) to be created in order to simulate deformation. Use `collision_simplification` to simplify the collision computation. 20 should be enough for `solver_position_iteration_count`. More details of setting these parameters can be found [here](https://docs.omniverse.nvidia.com/extensions/latest/ext_physics/deformable-bodies.html).

Note that you don't need to add collider and rigid body API to a deformable body mesh, for the deformable body API will handle the collision itself.

## Create a physics material

Normally, adding deformable body material will make a mesh deformable. However, if you want to adjust the parameters, you need to create a physics material, and apply it to the mesh.

```python
def_mat_path = "/DeformableBodyMaterial"
deformable_material_path = omni.usd.get_stage_next_free_path(stage, def_mat_path, True)

deformableUtils.add_deformable_body_material(
    stage,
    self.deformable_material_path,
    youngs_modulus=100000.0,
    poissons_ratio=0.499,
    damping_scale=0.0,
    elasticity_damping=0.0001,
    dynamic_friction=1.0,
    density=300,
)
```

These parameters are vital:

- `youngs_modulus` should be a positive float number. Higher value results in more stiff body.
- `dynamic_friction` should be a positive float number. It indicates the dymaic friction of the contact surface.

## Assign the material to the mesh

```python

physicsUtils.add_physics_material_to_prim(stage, get_prim_at_path(deformable_prim_path), self.deformable_material_path)
physxCollisionAPI = PhysxSchema.PhysxCollisionAPI.Apply(get_prim_at_path(deformable_prim_path))
physxCollisionAPI.CreateRestOffsetAttr().Set(0.0)
physxCollisionAPI.GetContactOffsetAttr().Set(0.0)
```

### Set attachment

Creating attachment will attach some texahedrons (the green ones) to a mesh with either collider API, rigid body API, or deformable body API. A known issue is that the deformable body cannot be attached to a part in an articulation (*e.g.*, on a link of the Franka robot).

```python
# Specify which mesh to attach to. It should have at least one of: collider API, rigid body API, or deformable body API.
attach_prim_path = "/World/base"
attach_prim = stage.GetPrimAtPath(attach_prim_path)

# Set up attachment
attachment_path = cube_mesh.GetPath().AppendElementString("attachment")
attachment = PhysxSchema.PhysxPhysicsAttachment.Define(stage, attachment_path)
attachment.GetActor0Rel().SetTargets([cube_mesh.GetPath()])
attachment.GetActor1Rel().SetTargets([attach_prim.GetPath()])
PhysxSchema.PhysxAutoAttachmentAPI.Apply(attachment.GetPrim())
```
