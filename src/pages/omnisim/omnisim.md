<div align="center">
  <img src="/img/omnisim/images/logo_new.png" width="100%"/>
</div>
<div align="center">

  [![Organization](https://img.shields.io/badge/Organization-OmniSim-blue)](https://github.com/RoboOmniSim)
  [![GitHub Repo Stars](https://img.shields.io/github/stars/RoboOmniSim/OmniSim?color=brightgreen&logo=github)](https://github.com/RoboOmniSim/OmniSim/stargazers)
  [![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

**OmniSim is An Infrastructure for Accelerating Robot Learning Research with Cross-Platform Simulation, Evaluation, and Training**


--------------------------------------------------------------------------------
## OmniSim: Effortless Environment Building
Omnisim serves as a powerful simulator API wrapper, including [Isaac Gym](https://developer.nvidia.com/isaac-gym), [Isaac Sim](https://developer.nvidia.com/isaac-sim), [SAPIEN](https://sapien.ucsd.edu/), [MuJoCo](https://mujoco.org/), [PyBullet](https://pybullet.org/wordpress/), [Gensis](https://github.com/Genesis-Embodied-AI/Genesis), streamlining the process of interfacing with complex simulator documentation and APIs.

With Omnisim:

+ **Abstracted Complexity**: Say goodbye to the daunting task of navigating intricate documentation and APIs. Omnisim abstracts away the complexities, providing a simplified interface for interaction.

+ **Effortless Integration**: Seamlessly integrate with various simulators (**config2{all kinds of simulators and useful tools}**) without the need to delve into the intricacies of each individual API.

+ **Streamlined Development**: Focus on developing your simulations rather than wrestling with convoluted documentation. Omnisim empowers you to harness the full potential of simulators with minimal effort.

Examples of the same enviroment in different simulators through config:


Articulated Object Environments           | Bi-Dexterous Hands in Isaac Gym             | Bi-Dexterous Hands in Pybullet
:-------------------------:|:-------------------------:|:-------------------------:
![alt text](/img/omnisim/images/image-2.png)|![alt text](/img/omnisim/images/image-3.png)  |  ![alt text](/img/omnisim/images/image-4.png)

## Omnisim: Bridging Different Simulators for Robot Learning Research
Omnisim acts as a vital bridge connecting various simulators, facilitating seamless communication for robot learning research.

With Omnisim:

+ **Inter-Simulator Communication**: Omnisim enables simulators to exchange data and interact with each other, fostering collaboration and synergy in robot learning research.

+ **Tools-Simulator Communication**: Omnisim seamlessly integrates with popular tools like [Blender](https://www.blender.org/) and [RViz](http://wiki.ros.org/rviz) enabling simulators to communicate and synchronize data effectively.

Examples of Isaac2blender:

Bi-Dexterous Hands HandOver           | Bi-Dexterous Hands Open Box
:-------------------------:|:-------------------------:
![blender1](/img/omnisim/images/blender1_compressed.gif)|![blender2](/img/omnisim/images/blender2_compressed.gif)

## Omnisim: Bridging the Gap Between Simulation and Reality

Omnisim serves as a groundbreaking sim2real and real2sim bridge, pioneering the integration of NeRF technology for world-building and leveraging advanced simulation techniques to optimize the utilization of real-world data in training policies. Our approach involves:

With Omnisim:

+ **NeRF World Building**: We employ cutting-edge NeRF (Neural Radiance Fields) technology to construct highly detailed and realistic virtual environments, ensuring fidelity to real-world scenarios.

+ **Optimized Policy Training with Omnisim**: Utilizing Omnisim's advanced simulation capabilities, we seamlessly integrate real-world data into the training process, allowing for efficient policy optimization in simulated environments.


+ **Deployment and Communication**: Through Omnisim's robust communication infrastructure, trained policies can be deployed back into the simulated environment, enabling iterative refinement and evaluation in a closed-loop system.

Examples of Isaac2RViz:

![alt text](/img/omnisim/images/image-6.png)

![rviz](/img/omnisim/images/rviz_compressed.gif)

Examples of Isaac2Real:

![Train video](/img/omnisim/images/sim_real_align_rl_policy_compressed.gif)


Examples of Real2config through NeRF:

![alt text](/img/omnisim/images/image-5.png)

![Train video](/img/omnisim/images/nerf.gif)

## config details

Please refer to `sim/metasim/config2isaac.py`.

Four interfaces need to be implemented:

- `__init__(config: dict)`
  - `config` is the initial configuration of the environment.
- `get_config(partial_config: dict)`
  - `partial_config` is a dictionary with the same format as the initial config, but with empty leaf nodes.
  - This function needs to assign the current simulator state to all leaf nodes of `partial_config` and return it.
- `set_config(partial_config: dict)`
  - `partial_config` is a dictionary with the same format as the initial config, but with leaf nodes representing the simulator states that need to be updated.
  - This function needs to update the simulator's state according to `partial_config`.
- `step(action, query)`
  - This function will call `set_config(action)` and `get_config(query)` and return the results.
  - It also executes one physical frame of the simulator.

Format of the config:

- `platform`: issacgym/sapien/mujoco/etc.
- `num_envs`: Number of parallel environments.
- `spacing`: Distance between environments (for isaacgym).
- `physical_params`: Default physical parameters for objects in the simulation.
    - `gravity`: [gx, gy, gz].
    - `linear_damping`: The automatic damping for velocity.
    - `angular_damping`: The automatic damping for angular velocity.
    - `static_friction`: Static friction coefficient between objects.
    - `dynamic_friction`: Dynamic friction coefficient between objects.
    - `restitution`: The ratio of velocity after and before collision between objects.
    - `bounce_threshold`: The minimum velocity for a collision to be considered elastic.
- `sim_params`: Parameters for simulators
    - `engine`: Physx/Flex. Physical engine to use.
    - `dt`
- `plane`: Parameters for the ground plane.
    - `altitude`: Altitude of ground plane.
- `viewer`: Parameters for the viewer window.
    - `headless`: Whether not to display the window.
    - `locate_pos`: [x, y, z]
    - `target_pos`: The point the viewer points at.
    - `rot`: [w, x, y, z] or [r, p, y]. Overrides target_pos.
- `cameras`: Parameters for cameras.
    - For each camera: (Cameras are not fully supported now!)
        - `type`: RGB/Depth/Segmentation/Opticalflow.
        - `width`: Width of image.
        - `height`
        - `horizontal_fov`
        - `enable_tensors`
        - `camera_locate_pos`
        - `camera_target_pos`
        - `camera_rot`: [w, x, y, z] or [r, p, y]. Ovverides 
- `agents`: Parameters for agents/objects in the world.
    - For each agent/object:
        - `type`: The urdf file. Or sphere/box/capsule.
        - `asset_root`: The root for urdf if `type` is `urdf`.
        - `asset_file`: The path for the urdf file relative to `asset_root`.
        - `fix_base_link`
        - `disable_gravity`
        - `armature`
        - `flip_visual_attachments`
        - `collapse_fixed_joints`
        - `pos`
        - `rot`: [w, x, y, z] or [r, p, y]
        - `vel`
        - `ang_vel`
        - `dof`
            - `drive_mode`: POS/VEL/TORQ/NONE
            - `pos`: A list, describes the pos of each dof.
            - `vel`: A list, describes the velocity of each dof.
            - `target`: A list，describes the initial target of each dof.
                According to `drive_mode`, this term can have different meanings:
                TORQ: The force applied on each dof.
                POS: The positional target of each dof.
                VEL: The velocity target of each dof.
            - `stiffness`: A list, describes the P term for the PD controller driving each dof.
            - `damping`: A list, describes the D term for the PD controller driving each dof.
        - `rigid_shape_property`: Overrides the default physical parameters.
            - `mass`: Not supported yet.
            - `compliance`
            - `contact_offset`
            - `filter`
            - `friction`
            - `rest_offset`
            - `restitution`
            - `rolling_friction`
            - `thickness`
            - `torsion_friction`

## assets:

+ [mujoco_panda](https://github.com/justagist/mujoco_panda)
+ [YCB objects](https://www.ycbbenchmarks.com/object-models/)
+ [Bi-Dexterous Hands](https://github.com/PKU-MARL/DexterousHands)
+ [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)
+ [robot-assets](https://github.com/ankurhanda/robot-assets)


## License

 This work and the dataset are licensed under [CC BY-NC 4.0][cc-by-nc].

 [![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

 [cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
 [cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png


## Citing OmniSim

If you find OmniSim useful, please cite it in your publications.

```bibtex
@article{OmniSim,
  author = {author1, author2, ...},
  title = {OmniSim},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/RoboOmniSim/OmniSim}},
}
```


## License
OmniSim is released under Apache License 2.0.
