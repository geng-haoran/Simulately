---
sidebar_position: 0
---

# Tips for using SAPIEN
> Tips for building up a SAPIEN environment

### Rotation Convention
- SAPIEN use `WXYZ` quaternion convention!
  
### Stiffness and Damping
- Robot Stiffness and damping are important. For motion planing, we need a high stiffness and damping value. 
    ```python
    for joint in self.active_joints:
        joint.set_drive_property(stiffness=1000, damping=200)
    ```
    However, in the Official SAPIEN tutorial, the stiffness and damping in their Gym-like environment are: 
    ```python
    for joint in self.active_joints[:5]:
        joint.set_drive_property(stiffness=0, damping=4.8)
    for joint in self.active_joints[5:7]:
        joint.set_drive_property(stiffness=0, damping=0.72)
    ```