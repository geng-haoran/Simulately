# Convex Decomposition Mesh in IsaacGym

This page provides instructions and snippets of convex decomposition mesh in IsaacGym.

### Use VHACD method to decompose the 3D mesh

We can add the following code under the `object_asset_option` to use the convex decomposition method that comes with IsaacGym.

```python
object_asset_options = gymapi.AssetOptions()  
object_asset_options.use_mesh_materials = True  
object_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX 
object_asset_options.override_com = True 
object_asset_options.override_inertia = True 
object_asset_options.vhacd_enabled = True 
object_asset_options.vhacd_params = gymapi.VhacdParams() 
object_asset_options.vhacd_params.resolution = 200000 
object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
```

Convex decomposition visualization:

![collision_mesh](imgs/isaacgym/collision_mesh.png)
