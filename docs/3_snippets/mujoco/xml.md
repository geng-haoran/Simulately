# MuJoCo XML Reference: Actuator and Position Elements

## Introduction

The MuJoCo (Multi-Joint dynamics with Contact) physics engine utilizes an XML-based language for modeling physical scenes. This document provides a detailed reference to the elements and attributes used within the MJCF (MuJoCo Format) XML schema, focusing on actuators and position elements.

## XML Schema

MJCF files are structured around XML elements and attributes, with no use of text content within elements.

## Actuator Elements

Actuators in MuJoCo are responsible for generating forces on joints or other elements. They are defined within the `<actuator>` tag and can be customized with various attributes to simulate different behaviors.

### General Actuator Attributes

- `name` (optional): A unique identifier for the actuator.
- `class` (optional): A class to apply default settings from.
- `group` (default "0"): An integer group identifier for the actuator.

### Position Actuator (`<position>`)

The `<position>` actuator is a specialized actuator that simulates a position servo with an optional first-order filter. It has the following specific attributes in addition to the general actuator attributes:

- `kp` (default "1"): The position feedback gain.
- `kv` (default "0"): The damping coefficient applied by the actuator.
- `dampratio` (default "0"): The damping ratio, an alternative to `kv` using units of `2kp*sqrt(m)` where `m` is the mass at the reference configuration.

### Actuator Dynamics

- `dyntype`: The type of activation dynamics, such as `none`, `integrator`, `filter`, `filterexact`, `muscle`, or `user`.
- `gaintype`: Determines the output of the force generation mechanism, with options like `fixed`, `affine`, `muscle`, or `user`.
- `biastype`: Specifies how the bias term of the force generation mechanism is calculated.

### Transmission Types

Actuators can be attached to joints, tendons, or other elements through various transmission types, influencing how the actuator's force is applied.

### XML Example

```xml
<mujoco>
  <actuator>
    <position name="positionActuator1" kp="100" kv="10" />
  </actuator>
</mujoco>
```


