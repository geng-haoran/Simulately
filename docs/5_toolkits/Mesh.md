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
# SDF to Mesh (based on trimesh)
def sdf_to_mesh_trimesh(sdf, level=0.02,spacing=(0.01,0.01,0.01)):
    if torch.is_tensor(sdf):
        sdf = sdf.detach().cpu().numpy()
    vertices, faces, normals, _ = skimage.measure.marching_cubes(sdf, level=level, spacing=spacing)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
    return mesh

# Mesh to SDF (based on Kaolin and trimesh)
def mesh_to_sdf_spacing(mesh, spacing):
    def to_tensor(data, device='cuda'):
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        else:
            raise NotImplementedError()

    def compute_unit_cube_spacing_padding(mesh, padding, voxel_resolution):
        """
        returns spacing for marching cube
        add padding, attention!!! padding must be same with mesh_to_voxels_padding
        """
        spacing = (np.max(mesh.bounding_box.extents) + padding) / voxel_resolution
        return spacing

    def scale_to_unit_cube_padding(mesh, padding):
        """
        add padding
        """
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump().sum()

        vertices = mesh.vertices - mesh.bounding_box.centroid
        vertices *= 2 / (np.max(mesh.bounding_box.extents) + padding)

        return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)


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
    upper_bound = spacing * voxel_resolution / 2
    lower_bound = - upper_bound
    device = 'cuda'

    save_spacing_centroid_dic = {}
    ###### calculate spacing before mesh scale to unit cube
    save_spacing_centroid_dic['spacing'] = str(spacing)
    save_spacing_centroid_dic['centroid'] = np.array(mesh.bounding_box.centroid).tolist()

    # voxelize unit cube
    xs = np.linspace(lower_bound, upper_bound, voxel_resolution)
    ys = np.linspace(lower_bound, upper_bound, voxel_resolution)
    zs = np.linspace(lower_bound, upper_bound, voxel_resolution)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing='ij')
    points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float32).cuda()

    obj_meshes = []
    obj_meshes.append(mesh)
    kl = KaolinMeshModel(store_meshes=obj_meshes, device=device)
    sdf = kl.mesh_points_sd(0, points.unsqueeze(0).contiguous())
    sdf = sdf.reshape((voxel_resolution, voxel_resolution, voxel_resolution)).detach().cpu().numpy()
    
    return sdf
```