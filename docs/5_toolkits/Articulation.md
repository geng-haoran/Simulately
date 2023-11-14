---
sidebar_position: 4
---
# Articulation
> Articulated objects and robots are always described using XML files.  URDF(Unified Robot Description Format) is the most popular one used in robotics for describing the structure and properties of a robot. Here we provide code for articulation parsing, calculation and so on.

### Query a Link from URDF file
To query a link in a URDF file, we provide code using urdfpy and xml parser.
- urdfpy
    ```python
    from urdfpy import URDF
    # Load URDF file
    robot = URDF.load('path_to_urdf_file.urdf')

    # Query a link by name
    link_name = 'desired_link_name'
    link = robot.link_map[link_name]

    # Access link properties
    print(f"Link name: {link.name}")
    print(f"Link inertia: {link.inertia}")
    ```
- xml
    ```python
    import xml.etree.ElementTree as ET

    # Load and parse the URDF file
    tree = ET.parse('path_to_urdf_file.urdf')
    root = tree.getroot()

    # Query a link by name
    link_name = 'desired_link_name'
    link = root.find(f".//link[@name='{link_name}']")

    if link is not None:
        print(f"Found link: {link.get('name')}")
    else:
        print("Link not found")
    ```
### Query a Joint from URDF file
To query a joint in a URDF file, we provide code using urdfpy and xml parser.
- urdfpy
  ```python
  from urdfpy import URDF

    # Load URDF file
    robot = URDF.load('path_to_urdf_file.urdf')

    # Query a joint by name
    joint_name = 'desired_joint_name'
    joint = robot.joint_map[joint_name]

    # Access joint properties
    print(f"Joint name: {joint.name}")
    print(f"Joint type: {joint.joint_type}")
  ```
- xml
    ```python
    import xml.etree.ElementTree as ET

    # Load and parse the URDF file
    tree = ET.parse('path_to_urdf_file.urdf')
    root = tree.getroot()

    # Query a joint by name
    joint_name = 'desired_joint_name'
    joint = root.find(f".//joint[@name='{joint_name}']")

    if joint is not None:
        print(f"Found joint: {joint.get('name')}")
    else:
        print("Joint not found")
    ```

### Merging URDF Geometries to OBJ
To merge URDF geometries to an OBJ file, we provide code using xml parser. This code still needs testing.
    ```python
    import xml.etree.ElementTree as ET
    import trimesh
    import numpy as np

    def parse_origin(element):
        """Parse the origin XML element to extract position and rotation."""
        if element is None:
            return np.eye(4)  # Return identity matrix if no origin is specified
        xyz = element.get('xyz', '0 0 0').split()
        rpy = element.get('rpy', '0 0 0').split()
        xyz = [float(val) for val in xyz]
        rpy = [float(val) for val in rpy]

        # Compute the transformation matrix from position and rotation (roll, pitch, yaw)
        translation_matrix = trimesh.transformations.translation_matrix(xyz)
        rotation_matrix = trimesh.transformations.euler_matrix(*rpy)
        return trimesh.transformations.concatenate_matrices(translation_matrix, rotation_matrix)

    # Load and parse the URDF file
    tree = ET.parse('path_to_urdf_file.urdf')
    root = tree.getroot()

    # Initialize an empty scene
    scene = trimesh.Scene()

    # Iterate through each link and process its geometry
    for link in root.findall('link'):
        for visual in link.findall('visual'):
            origin = visual.find('origin')
            transform = parse_origin(origin)
            geometry = visual.find('geometry')
            mesh_element = geometry.find('mesh')
            if mesh_element is not None:
                mesh_filename = mesh_element.get('filename')
                # Load the mesh and apply transformation
                loaded_mesh = trimesh.load(mesh_filename)
                loaded_mesh.apply_transform(transform)
                scene.add_geometry(loaded_mesh)

    # Export the scene to an OBJ file
    scene.export('output.obj')
    ```