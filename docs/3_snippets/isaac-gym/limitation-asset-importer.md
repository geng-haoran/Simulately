# Limitation of Asset Importer

> The asset pipeline in IsaacGym is a work in progress, so there are some limitations. This is borrowed from IsaacGym Doc (Preview 4).

- The URDF importer can only load meshes in OBJ format. Many URDF models come with STL collision meshes and DAE visual meshes, but those need to be manually converted to OBJ for the current importer.

- The MJCF importer supports primitive shapes only, such as boxes, capsules, and spheres. Mesh loading is currently not available in that importer.

- The MJCF importer supports multiple joints between a pair of bodies, which is useful to define independently named and controllable degrees of freedom. This is used in the humanoid_20_5.xml model to define independent motion limits for shoulders, hips, and other compound joints.

- The MJCF importer only supports files where the worldbody has no more than one direct child body. This means that MJCF files that define an entire environment may not be supported. For example, one MJCF file cannot contain both a robot and a ground plane.