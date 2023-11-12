# Couldn't fine camera with handle -1 in environment 0

This page provides instructions and snippets of Error: couldn't fine camera with handle -1 in environment 0.

### Issue background

It seems like something is wrong when I try to use point cloud observation in the task.
Specifically, using the camera through isaacgym API may have some bugs. I print the 'camera_handle', whose value is -1.

![IsaacGym2](imgs/isaacgym/2.png)

![IsaacGym1](imgs/isaacgym/1.png)

### Solution
Errors that return -1 generally appear when using isaacgym on servers without a monitor. Using --headless may also cause this bug. Currently, isaacgym has many bugs with rendering, maybe you can also go to the isaacgym official [forum](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/isaac-gym/322) to find some answers.

Hope this can help you.


