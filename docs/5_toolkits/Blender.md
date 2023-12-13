---
sidebar_position: 6
---

# Blender


> Blender is the free and open source 3D creation suite. It supports the entirety of the 3D pipeline—modeling, rigging, animation, simulation, rendering, compositing and motion tracking, even video editing and game creation. Blender features **Blender’s physics system**, which allows you to simulate a number of different real-world physical phenomena. You can use these systems to create a variety of static and dynamic effects such as water, cloth, smoke, rain, grass, and many more.

> For more information of how to use Blender with python, checkout the code in the snippets section.

### What's the difference between the different renders?

There are 3 renderes in blender, each has its own uniqueness and weaknesses:

- [Workbench](https://docs.blender.org/manual/en/latest/render/workbench/index.html): The default viewport render. The Workbench Engine is a render engine optimized for fast rendering during modeling and animation preview. Although fast, it is not intended to be a render engine that will render final images for a project.
- [Eevee](https://docs.blender.org/manual/en/latest/render/eevee/index.html): Eevee is Blender’s realtime render engine built using OpenGL focused on speed and interactivity while achieving the goal of rendering PBR materials. Eevee can be used interactively in the 3D Viewport but also produce high quality final renders, so it's great for if you want to render out something quick or stylized, but lacks a lot of functionalities and customization Cycles offers.
- [Cycles](https://docs.blender.org/manual/en/latest/render/cycles/index.html): Cycles is Blender’s physically-based path tracer for production rendering. It is designed to provide physically based results out-of-the-box, with artistic control and flexible shading nodes for production needs. It's usually the one people use for more realistic (or realistically-styled) renders, although it's slow and you need good hardware to get it to render at a descent speed.

## Official Materials
- [Website](https://www.blender.org/)
- [Source Code](https://github.com/blender/blender)
- [Forum](https://devtalk.blender.org/)
- [Community](https://www.blender.org/community)
- [Docs](https://docs.blender.org/)
- [Docs on Python API](https://docs.blender.org/api/current/index.html)
- [Docs on Physics in Blender](https://docs.blender.org/manual/en/latest/physics/index.html)

## Related Materials
- [BlenderProc](https://dlr-rm.github.io/BlenderProc/index.html): A procedural Blender pipeline for photorealistic rendering.
- [Phobos](https://github.com/dfki-ric/phobos): A robotics toolkit for Blender
- [Phobos-Motion](https://github.com/YuyangLee/Phobos-Motion): A robotics visualization tool, based on Phobos


## Code Snippets

### 1. Cameras

```python
# =======================================================
# 1. Create camera
bpy.ops.object.add(type='CAMERA', location=(0, -3.0, 0))
camera = bpy.context.object
camera.data.lens = 35
camera.rotation_euler = Euler((pi/2, 0, 0), 'XYZ')
# Make this the current camera
bpy.context.scene.camera = camera

# =======================================================
# 2. Render image
scene = bpy.context.scene
scene.render.resolution_x = 512
scene.render.resolution_y = 512
scene.render.resolution_percentage = 100
scene.render.engine = 'CYCLES' # alternatively 'BLENDER_EEVEE'
scene.render.filepath = 'rendering/output.png'
bpy.ops.render.render(write_still=True)
```

### 2. Animations

```python
# This section provides code snippets of animation. 
# The code here is based on character animation with data in AMASS format, but it can be generalized to any articualted objects (like robots) and any data format.

# =======================================================
# 1. Import SMPLX Model
bpy.data.window_managers["WinMan"].smplx_tool.smplx_gender = gender
bpy.ops.scene.smplx_add_gender()

# =======================================================
# 2. Set joint location and rotation (Assume we have the `data` variable in AMASS format)
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


### 3. Materials

```python
# =======================================================
# 1. Create a new material
def newMaterial(id):
    mat = bpy.data.materials.get(id)
    if mat is None:
        mat = bpy.data.materials.new(name=id)
    mat.use_nodes = True

    if mat.node_tree:
        mat.node_tree.links.clear()
        mat.node_tree.nodes.clear()

    return mat

# =======================================================
# 2. Add a shader to the material
def newShader(id, type, r, g, b):
    mat = newMaterial(id)
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    output = nodes.new(type='ShaderNodeOutputMaterial')

    if type == "diffuse":
        shader = nodes.new(type='ShaderNodeBsdfDiffuse')
        nodes["Diffuse BSDF"].inputs[0].default_value = (r, g, b, 1)
    elif type == "emission":
        shader = nodes.new(type='ShaderNodeEmission')
        nodes["Emission"].inputs[0].default_value = (r, g, b, 1)
        nodes["Emission"].inputs[1].default_value = 1
    elif type == "glossy":
        shader = nodes.new(type='ShaderNodeBsdfGlossy')
        nodes["Glossy BSDF"].inputs[0].default_value = (r, g, b, 1)
        nodes["Glossy BSDF"].inputs[1].default_value = 0

    links.new(shader.outputs[0], output.inputs[0])

    return mat

# =======================================================
# 3. Assign material to object
mat = newShader("Shader1", "diffuse", 1, 1, 1)
bpy.ops.mesh.primitive_cube_add(size=2, align='WORLD', location=(0, 0, 0))
bpy.context.active_object.data.materials.append(mat)
```

### 4. Physics
```python
# This section provides some snippets about how to use the blender game engine (bge) for physics.

# Adding constraints
from bge import logic
from bge import constraints

# get object list
objects = logic.getCurrentScene().objects

# get object named Object1 and Object 2
object_1 = objects["Object1"]
object_2 = objects["Object2"]

# want to use Edge constraint type
constraint_type = 2

# get Object1 and Object2 physics IDs
physics_id_1 = object_1.getPhysicsId()
physics_id_2 = object_2.getPhysicsId()

# use bottom right edge of Object1 for hinge position
edge_position_x = 1.0
edge_position_y = 0.0
edge_position_z = -1.0

# rotate the pivot z axis about 90 degrees
edge_angle_x = 0.0
edge_angle_y = 0.0
edge_angle_z = 90.0

# create an edge constraint
constraints.createConstraint(
    physics_id_1, 
    physics_id_2,
    constraint_type,
    edge_position_x, 
    edge_position_y, 
    edge_position_z,
    edge_angle_x, 
    edge_angle_y, 
    edge_angle_z
)
```
## 5. Loading and Saving Files
```python
# This section provides snippets about how to load and save different file formats with Blender

# Import/Export FBX File
bpy.ops.import_scene.fbx(filepath=file_path)
bpy.ops.export_scene.fbx(filepath=file_path)

# Import/export OBJ File
bpy.ops.import_scene.obj(filepath=file_path)
bpy.ops.export_scene.obj(filepath=file_path)

# Import/export GLTF File
bpy.ops.import_scene.gltf(filepath=file_path)
bpy.ops.export_scene.gltf(filepath=file_path)

# Import/export Extensible 3D (X3D) File
bpy.ops.import_scene.x3d(filepath=file_path)
bpy.ops.export_scene.x3d(filepath=file_path)

# Save Current Scene as Blend File
# save blend (untitled.blend if bpy.data.filepath not set)
bpy.ops.wm.save_mainfile()
# save as
bpy.ops.wm.save_as_mainfile(filepath=file_path)

# Import/Export a BVH MoCap File
bpy.ops.import_anim.bvh(filepath=file_path)
bpy.ops.export_anim.bvh(filepath=file_path)

# Import/Export a PLY Geometry File
bpy.ops.import_mesh.ply(filepath=file_path)
bpy.ops.export_mesh.ply(filepath=file_path)

# Import/Export a STL Triangle Mesh File
bpy.ops.import_mesh.stl(filepath=file_path)
bpy.ops.export_mesh.stl(filepath=file_path)
```


## Related Projects
- SIGGRAPH Asia 2023: [Object Motion Guided Human Motion Synthesis](https://lijiaman.github.io/projects/omomo/): Blender
- CVPR 2023: [CIRCLE: Capture in Rich Contextual Environments](https://stanford-tml.github.io/circle_dataset/): Blender
- CVPR 2023: [A New Benchmark: On the Utility of Synthetic Data with Blender for Bare Supervised Learning and Downstream Domain Adaptation](https://huitangtang.github.io/On_the_Utility_of_Synthetic_Data/): Blender
- ALR 2023: [Simulating dual-arm robot motions to avoid collision by rigid body dynamics for laboratory bench work](https://link.springer.com/article/10.1007/s10015-022-00823-1); Blender
- CVPR 2017: [CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning](https://cs.stanford.edu/people/jcjohns/clevr/): Blender
- Preprint: [Objaverse-XL: A Universe of 10M+ 3D Objects](https://objaverse.allenai.org/): Blender
