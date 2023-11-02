# Frequently Asked Questions

## I Can't Find the Docs !!!

Isaac Sim is under active development, whose latest documentation could change. As a result, links you find on search engines may only belong to an older version of documentation, which causes a 404 error.

Our recommondendation is to always start from [this main page](https://docs.omniverse.nvidia.com/isaacsim/latest/index.html) for the latest version of the documentation, and use the built-in searching. Here are some portals:

## Camera Intrinsics Error

In Isaac Sim 2022.2.1, the built-in camera API for computing the intrinsics matrix has a bug:

![](img/isaac.sim-2022.2.1-intrinsics.bug.jpeg)

You can fix it in `~/.local/share/ov/pkg/isaac_sim-2022.2.1/exts/omni.isaac.sensor/omni/isaac/sensor/scripts/camera.py` by changing `get_horizontal_aperture` to `get_vertical_aperture`.
