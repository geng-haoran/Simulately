---
sidebar_position: 9
---

# Filter and Smoothing

> **Filters are indispensable in the field of robotics, enabling robots to interpret their environment and respond
appropriately.**

One of the most prevalent filters in signal processing is the **Low-Pass Filter (LPF)**. These filters are designed to
permit the passage of signals with frequencies below a specified threshold and to diminish the impact of higher
frequencies, which are typically regarded as undesirable noise.

Low-pass filters hold significant value in the realm of robotics for several reasons:

- Enhancing Sensor Readings: Given that a multitude of sensors are susceptible to signal disruption, LPFs are adept at
  refining the data, resulting in enhanced precision.
- Stabilizing Signal Processing: By mitigating abrupt fluctuations in control signals, LPFs contribute to the seamless
  operation of robotic components like motors and actuators.
- Noise Mitigation: LPFs are effective in curbing high-frequency disruptions originating from multiple sources,
  including electrical disturbances.

## Python Code for Discrete Low-Pass Filtering

Here is a straightforward Python code snippet that demonstrates the implementation of a discrete low-pass filter,
commonly utilized in robotics to enhance the quality of sensor readings.

```python
import numpy as np


class LowPassFilter:
  def __init__(self, control_freq, cutoff_freq):
    dt = 1 / control_freq
    wc = 2 * np.pi * cutoff_freq
    temp = 1 - np.cos(wc * dt)
    self.alpha = -temp + np.sqrt(temp ** 2 + 2 * temp)
    self.y = 0
    self.initialized = False

  def apply(self, x):
    if not self.initialized:
      self.init(x)
    self.y += self.alpha * (x - self.y)
    return self.y

  def init(self, initial_value):
    self.y = initial_value
    self.initialized = True
```

This code presents a simple yet effective approach to applying a first-order discrete low-pass filter, which operates
similarly to an exponential smoothing mechanism due to its straightforward computational structure.

In addition to the custom implementation, the `scipy` library provides facilities to achieve similar results. The
following function demonstrates how to apply a low-pass filter to a series of signals, smoothing the input data
effectively.

```python
import numpy as np
from scipy import signal


def smooth_signal_sequence(signal_sequence: np.ndarray, cutoff_freq=5, sample_rate=25):
  sos = signal.butter(2, cutoff_freq, 'low', fs=sample_rate, output='sos')
  dimensions = signal_sequence.shape
  if len(dimensions) < 2:
    raise ValueError(f"Expected a 2D or 3D array, received shape: {dimensions}")

  smoothed_sequence = np.empty_like(signal_sequence)
  if len(dimensions) == 3:
    for joint_index in range(dimensions[1]):
      for axis_index in range(dimensions[2]):
        smoothed_sequence[:, joint_index, axis_index] = signal.sosfilt(sos, signal_sequence[:, joint_index, axis_index])
  elif len(dimensions) == 2:
    for joint_index in range(dimensions[1]):
      smoothed_sequence[:, joint_index] = signal.sosfilt(sos, signal_sequence[:, joint_index])

  return smoothed_sequence
```


