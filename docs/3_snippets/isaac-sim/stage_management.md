# Stage Management

### Get stage
```python
stage = omni.usd.get_context().get_stage()
```

### Get Layer
```python
layer = stage.GetRootLayer()
# get layer path: layer.realpath
```

### Get Session Layer
```python
session_layer = stage.GetSessionLayer()
```

### Modify stuffs with session Layer
```python
 with Usd.EditContext(stage, session_layer):
      do_stuff()
```

### Get/Set default prim
```python
self.stage.SetDefaultPrim(self._stage.GetPrimAtPath("/World"))
# .GetDefaultPrim()
```

### Traverse all prims
```python
 self.prim_list = self.stage.TraverseAll()
```


### Create Empty xform
```python
import omni
import pxr
from omni.physx.scripts import physicsUtils

physicsUtils.add_xform(stage, "/xform", pxr.Gf.Vec3f(0.0, 0, 0.0))

# or 
path = omni.usd.get_stage_next_free_path(stage, "/panda", True)
xform_geom = pxr.UsdGeom.Xform.Define(stage, path)
```

### Set/Get stage units, Set Up Axis
```python
from omni.isaac.core.utils.stage import get_stage_units
get_stage_units()

# set up axis to z
pxr.UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
pxr.UsdGeom.SetStageMetersPerUnit(stage, 0.01)
```

### Rename prim
```python
old_prim_name = prim.GetPath().pathString
new_prim_name = prim.GetPath().GetParentPath()
new_prim_name = new_prim_name.AppendChild("Door1")
new_prim_name = omni.usd.get_stage_next_free_path(self.stage, new_prim_name.pathString, False)
print("new_prim_name: ", new_prim_name)

move_dict = {old_prim_name: new_prim_name}
if pxr.Sdf.Path.IsValidPathString(new_prim_name):
    move_dict = {old_prim_name: new_prim_name}
    omni.kit.commands.execute("MovePrims", paths_to_move=move_dict,  on_move_fn=None)
else:
    carb.log_error(f"Cannot rename {old_prim_name} to {new_prim_name} as its not a valid USD path")
```

### Get/Set Prim Attribute
```python
        test_prim = self.stage.GetPrimAtPath("/World/Looks/component_45146_solid_001_wire1/component_45146_solid_001_wire1")
        # shader = pxr.UsdShade.Shader(test_prim)
        # asset = shader.GetSourceAsset("mdl")
        # print("shader", asset)
        attr = test_prim.GetAttribute("inputs:diffuse_texture").Get()
        new_asset_path = str(attr).replace(":","_").replace("@","")
        print("attr", str(attr), new_asset_path)
        test_prim.CreateAttribute("inputs:diffuse_texture", pxr.Sdf.ValueTypeNames.String, False).Set(new_asset_path)

```