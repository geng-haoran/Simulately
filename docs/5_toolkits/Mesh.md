---
sidebar_position: 3
---
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
## Mesh-SDF bi-converter based on trimesh and nvidia kaolin
class MeshSDF:
    ## bound_scale / voxel_resolution = spacing
    def __init__(self, spacing=0.01875, level=0.001, resolution=64):
        self._spacing = spacing
        self._resolution = resolution
        self._upper = spacing * resolution
        self._level = level

    def sdf_to_mesh(self, sdf):
        assert sdf.shape == (self._resolution, self._resolution, self._resolution)
        spacing = (self._spacing, self._spacing, self._spacing)
        vertices, faces, normals, _ = skimage.measure.marching_cubes(sdf, level=self._level, spacing=spacing)
        mesh = trimesh.Trimesh(vertices, faces)
        return mesh

    def mesh_to_sdf(self, mesh):
        def to_tensor(data, device='cuda'):
            if isinstance(data, torch.Tensor):
                return data
            elif isinstance(data, np.ndarray):
                return torch.from_numpy(data, device=device)
            else:
                raise NotImplementedError()

        class KaolinMeshModel():
            def __init__(self, store_meshes=None, device="cuda"):
                """
                Args:
                    `store_meshes` Optional, `list` of `Mesh`.
                """
                self.device = device
                if store_meshes is not None:
                    self.update_meshes(store_meshes)
                
            def update_meshes(self, meshes):
                if meshes is not None:
                    self.object_mesh_list = []
                    self.object_verts_list = []
                    self.object_faces_list = []
                    self.object_face_verts_list = []
                    for mesh in meshes:
                        self.object_mesh_list.append(mesh)
                        self.object_verts_list.append(torch.Tensor(self.object_mesh_list[-1].vertices).to(self.device))
                        self.object_faces_list.append(torch.Tensor(self.object_mesh_list[-1].faces).long().to(self.device))
                        self.object_face_verts_list.append(index_vertices_by_faces(self.object_verts_list[-1].unsqueeze(0), self.object_faces_list[-1]))
                self.num_meshes = len(meshes)

            def mesh_points_sd(self, mesh_idx, points):
                """
                Compute the signed distance of a specified point cloud (`points`) to a mesh (specified by `mesh_idx`).

                Args:
                    `mesh_idx`: Target mesh index in stored.
                    `points`: Either `list`(B) of `ndarrays`(N x 3) or `Tensor` (B x N x 3).

                Returns:
                    `signed_distance`: `Tensor`(B x N)
                """
                points = to_tensor(points)
                verts = self.object_verts_list[mesh_idx].clone().unsqueeze(0).tile((points.shape[0], 1, 1))
                faces = self.object_faces_list[mesh_idx].clone()
                face_verts = self.object_face_verts_list[mesh_idx]
                
                signs = check_sign(verts, faces, points)
                dis, _, _ = point_to_mesh_distance(points, face_verts)      # Note: The calculated distance is the squared euclidean distance.
                dis = torch.sqrt(dis)                  
                return torch.where(signs, -dis, dis)
            
        voxel_resolution = 64
        device = 'cuda'

        xs = np.linspace(0., self._spacing * (self._resolution - 1), self._resolution)
        ys = np.linspace(0., self._spacing * (self._resolution - 1), self._resolution)
        zs = np.linspace(0., self._spacing * (self._resolution - 1), self._resolution)
        xx, yy, zz = np.meshgrid(xs, ys, zs, indexing='ij')

        points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float32).cuda()

        obj_meshes = []
        obj_meshes.append(mesh)
        kl = KaolinMeshModel(store_meshes=obj_meshes, device=device)
        sdf = kl.mesh_points_sd(0, points.unsqueeze(0).contiguous())
        sdf = sdf.reshape((voxel_resolution, voxel_resolution, voxel_resolution)).detach().cpu().numpy()
        
        return sdf
```