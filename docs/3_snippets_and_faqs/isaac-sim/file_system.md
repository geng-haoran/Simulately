### New/Clean Stage

```python

await omni.usd.get_context().new_stage_async()
await omni.kit.app.get_app().next_update_async()
```
Call with corountine:

```python
asyncio.ensure_future(self._create_moveit_sample())
```

### Open stage with USD file
```python
success, error = await omni.usd.get_context().open_stage_async(usd_path)
if not success:
    raise Exception(f"Failed to open usd file: {error}")
```

### Close stage
```python
(result, err) = await omni.usd.get_context().close_stage_async()
# Assert result == True
```

### Reopen stage
```python
(result, err) = await omni.usd.get_context().reopen_stage_async()
```

### Save stage
```python
tmp_file_path = os.path.join(tmpdirname, "tmp.usda")
(result, err, saved_layers) = await omni.usd.get_context().save_as_stage_async(tmp_file_path)

# or
(result, err, saved_layers) = await omni.usd.get_context().save_stage_async()
# result == True
```

### Export stage
```python
(result, err) = await omni.usd.get_context().export_as_stage_async(tmp_file_path)
```

### Get Server
```python
from omni.isaac.core.utils.nucleus import find_nucleus_server
# import carb

result, nucleus_server_path = find_nucleus_server()
# if result is False:
#     carb.log_error("Could not find nucleus server with /Isaac folder")
```

### Import asset `.usd`

### issac-sim
```python
from omni.isaac.core.utils.prims import create_prim

# Add a distant light
create_prim("/DistantLight", "DistantLight", attributes={"intensity":500})

# Add an object from server
create_prim(prim_path="/background", usd_path=self._nucleus_path + "/Isaac/Environments/Simple_Room/simple_room.usd")

# Add an asset from usd_path
prim = create_prim(prim_path = "/Mesh", usd_path=usd_path, scale = np.array([10,10,10]), semantic_label = "mustard")

```
### create
```python
        omni.kit.commands.execute(
            "CreatePrim",
            prim_path="/World/defaultLight",
            prim_type="DistantLight",
            select_new_prim=False,
            attributes={pxr.UsdLux.Tokens.angle: 1.0, pxr.UsdLux.Tokens.intensity: 500},
            create_default_xform=True,
        )
```

### get server and traverse files
```python
        default_server = carb.settings.get_settings().get("/isaac/nucleus/default")
        carb.log_info("default_server: " + str(default_server)) 

        path = "http://localhost:8080/omniverse://127.0.0.1/NVIDIA/Materials/"
        carb.log_info(f"Collecting files for {path}")
        result, entries = omni.client.list(path)
        for e in entries:
            print("result: ", e.relative_path)
```