# Loading and Saving Files
This page provides snippets about how to load and save different file formats with Blender

## Import/Export FBX File
```python
bpy.ops.import_scene.fbx(filepath=file_path)
bpy.ops.export_scene.fbx(filepath=file_path)
```

## Import/export OBJ File
```python
bpy.ops.import_scene.obj(filepath=file_path)
bpy.ops.export_scene.obj(filepath=file_path)
```

## Import/export GLTF File
```python
bpy.ops.import_scene.gltf(filepath=file_path)
bpy.ops.export_scene.gltf(filepath=file_path)
```

## Import/export Extensible 3D (X3D) File
```python
bpy.ops.import_scene.x3d(filepath=file_path)
bpy.ops.export_scene.x3d(filepath=file_path)
```

## Save Current Scene as Blend File
```python
# save blend (untitled.blend if bpy.data.filepath not set)
bpy.ops.wm.save_mainfile()
# save as
bpy.ops.wm.save_as_mainfile(filepath=file_path)
```

## Import/Export a BVH MoCap File
```python
bpy.ops.import_anim.bvh(filepath=file_path)
bpy.ops.export_anim.bvh(filepath=file_path)
```

## Import/Export a PLY Geometry File
```python
bpy.ops.import_mesh.ply(filepath=file_path)
bpy.ops.export_mesh.ply(filepath=file_path)
```


## Import/Export a STL Triangle Mesh File
```python
bpy.ops.import_mesh.stl(filepath=file_path)
bpy.ops.export_mesh.stl(filepath=file_path)
```

