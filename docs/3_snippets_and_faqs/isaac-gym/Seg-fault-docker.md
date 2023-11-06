# Segmentation fault (core dumped) in Docker

This page provides instructions and snippets of segmentation fault (core dumped) in docker.

### Issue background

- Device NVIDIA A100 40GB PCIe GPU Accelerator

- Method: Docker

- Details:

At the end of the RL training process in Isaac Gym, it shows error in console as following.

```bash
Segmentation fault (core dumped)
```

### Solution
We can use faulthandler to locate your problem. As the docker don't haver a graphic viewer,s o we need to set the parameter "headless" as True.



