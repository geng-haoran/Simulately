---
sidebar_position: 5
---

# Frequently Asked Questions

## Simulation

### QACC Error:

```shell
WARNING: Nan, Inf or huge value in QACC at DOF 1. The simulation is unstable. Time = 0.0600.
```

This is an instability error for your physical simulation in MuJoCo and may originate from many causes.

1. **Large joint force**: Consider adding some damping or armature to your joints to mitigate abrupt force applications.
2. **Large time step**: Try to reduce your time step size, e.g., from 0.005 to 0.002. For debugging purposes, you can
   set the timestep to an extremely small value (e.g., 1e-5) to test the stability of the simulation.
3. **Severe self-collision**: The links of your model, whether it's a robot or an articulated object, might be
   generating very large collision forces. Review the collision mesh you are using to ensure it is accurate and not
   causing excessive force calculations.
4. **More stable integrator**: If your system is particularly complex, consider changing your integrator within
   the `<option>` tag from `Euler` to `RK4` for potentially improved stability.

## Installation

### pip install stuck

```shell
Building wheel for mujoco (setup.py) ... /

(stuck here)
```

If your `pip install` process is stuck, it could be due to several reasons:

- **Binary vs Source**: You might be compiling from source because there is no pre-built binary wheel available for your
  system. This could happen if you're using an older system that does not support
  the [manylinux2014](https://peps.python.org/pep-0599/) platform tag. In such cases, a binary distribution may not be
  suitable for your system.
- **Environment Variables**: Ensure that you have set the environment variables `MUJOCO_PATH` and `MUJOCO_PLUGIN_PATH`
  correctly. These variables help the installer locate the necessary MuJoCo files on your system.

If you continue to experience issues, consider looking for any error messages that occur before the process gets stuck
or checking the system's process manager for any indications of what the installation process is doing at the time it
becomes unresponsive.