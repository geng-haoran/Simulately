# Cameras

This page provides instructions and snippets of deploying cameras.

## Creating a Camera

```python
# Create camera
bpy.ops.object.add(type='CAMERA', location=(0, -3.0, 0))
camera = bpy.context.object
camera.data.lens = 35
camera.rotation_euler = Euler((pi/2, 0, 0), 'XYZ')

# Make this the current camera
bpy.context.scene.camera = camera
```

## Get Rendered Output

```python
# Render image
scene = bpy.context.scene
scene.render.resolution_x = 512
scene.render.resolution_y = 512
scene.render.resolution_percentage = 100
scene.render.engine = 'CYCLES' # alternatively 'BLENDER_EEVEE'
scene.render.filepath = 'rendering/output.png'
bpy.ops.render.render(write_still=True)
```
