# Mesh
> **Toolkits for mesh processing**

### Basic Processing Code with `trimesh`

```python
import trimesh

# Load a mesh from OBJ file
mesh = trimesh.load('path_to_mesh.obj')

# Translate mesh to its centroid
mesh.apply_translation(-mesh.centroid)

# Scale the mesh (1 unit here)
scale_factor = 1.0 / mesh.bounding_box.extents.max()
mesh.apply_scale(scale_factor)

# save the new mesh to OBJ file
mesh.export('output.obj')
```

### Point Cloud to Mesh and Mesh to Point Cloud

```python
# Convert a point cloud array to mesh trimesh object
point_cloud = trimesh.PointCloud(point_cloud_array)

# Save point cloud to a PLY file or OBJ file
point_cloud.export('output.obj')
point_cloud.export('output.ply')
```

### Extracting Features from Mesh

```python
import trimesh

# Load a mesh from OBJ file
mesh = trimesh.load('path_to_mesh.obj')

# Compute vertex normals
vertex_normals = mesh.vertex_normals

# Compute face normals
face_normals = mesh.face_normals

# Compute curvature
curvature = trimesh.curvature.discrete_mean_curvature_measure(mesh, mesh.vertices)
```

### SDF to Mesh and Mesh to SDF

```python

```