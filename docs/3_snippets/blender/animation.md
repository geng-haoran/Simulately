# Cameras

This page provides code snippets of animation. 
The code here is based on character animation with data in AMASS format, but it can be generalized to any articualted objects (like robots) and any data format.

## Import SMPLX Model

```python
bpy.data.window_managers["WinMan"].smplx_tool.smplx_gender = gender
bpy.ops.scene.smplx_add_gender()
```

## Set joint location and rotation

Assume we have the `data` variable in AMASS format

```python

# set frame properties
total_frame = int(data["mocap_time_length"] * data["mocap_frame_rate"])
bpy.data.scenes["Scene"].frame_start = 0
bpy.data.scenes["Scene"].frame_end = total_frame
bpy.context.view_layer.objects.active = bpy.data.objects['SMPLX-neutral']
bpy.ops.object.mode_set(mode="POSE")
bpy.context.scene.render.fps = round(float(data["mocap_frame_rate"]))
# get the character object
character = bpy.data.objects["NAME_OF_CHARACTER_OBJECT"]
# Clear all keyframes in animation
bpy.context.active_object.animation_data_clear()

# set root joint pos and orn
for frame in range(total_frame):
    root = character.pose.bones["root"]
    root.rotation_mode = "XYZ"
    root.location = data["trans"][frame]
    root.keyframe_insert(data_path="location", frame=frame)
    root.rotation_euler = data["root_orient"][frame]
    root.keyframe_insert(data_path="rotation_euler", frame=frame)

# set other joint orn
# SMPL_X_SKELTON2 is a predefied joint_index to joint_name mapping
for i, joint_name in SMPL_X_SKELTON2.items():
    joint = character.pose.bones[joint_name]
    joint.rotation_mode = "XYZ"
    joint.rotation_euler = data["poses"][frame][i * 3: (i + 1) * 3]
    joint.keyframe_insert(data_path="rotation_euler", frame=frame)
```
