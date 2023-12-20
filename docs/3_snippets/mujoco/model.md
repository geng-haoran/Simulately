---
sidebar_position: 3
---

# Robot Models

> **It's crucial to understand that the effectiveness of any simulation hinges on the integrity of the models it's based
on. In MuJoCo, a well-crafted model isn't just beneficial—it's fundamental.**


The journey of a model in MuJoCo commences with its creation in one of two human-readable XML file formats: MJCF (MuJoCo
XML format) or URDF (Unified Robot Description Format). These files serve as the blueprints from which MuJoCo constructs
its dynamic simulations.

For those new to the field, it's captivating to learn that these models are then compiled into an `mjModel`, a low-level
data structure that MuJoCo uses to run simulations efficiently. This step is akin to an architect handing over detailed
plans to a builder who then lays the foundation for what's to come.

## Loading Models

In Python, simplicity is often key, and loading models into MuJoCo is no exception. Whether you're starting with a
snippet of XML or a fully-fledged binary file, the steps remain refreshingly minimal.

Here's how you breathe life into your models:

```python
import mujoco

# A dictionary to hold additional assets
assets = {}

# Loading a model from an XML string
model = mujoco.MjModel.from_xml_string(xml_string, assets)

# Or from an XML file directly
model = mujoco.MjModel.from_xml_path(xml_path, assets)

# And even from a compiled binary MJB file
model = mujoco.MjModel.from_binary_path(binary_path, assets)
```

## Where to Find MuJoCo Models?

### Official Models

![mujoco](https://github.com/google-deepmind/mujoco_menagerie/raw/main/banner.png)

[mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) is a repo a repository curated by the DeepMind MuJoCo team themselves. This
collection boasts an array of high-quality models tailored for the MuJoCo physics engine.Whether you're searching for a
nimble humanoid, a sophisticated robotic arm, or something entirely different, the menagerie is your first port of call.

# Visualize and Convert URDF to MJCF with MuJoCo Viewer

Here's how you can bring a URDF file to MuJoCo. First, open the the MuJoCo Viewer:

```shell
python -m mujoco.viewer
```

Once open, it's a matter of a simple 'drag and drop'—take your URDF file and release it into the viewer. If
the stars align and the URDF model loads successfully, you're halfway there.

With your model now pirouetting in the viewer, you can convert it to native MuJoCo `mjcf` format. This transformation
is as easy as a click on the `Save xml` button. It will be saved as an XML file right in the heart of your current
working directory.

However, keep in mind that MuJoCo and URDF are slightly different in format of the
modeling language. Some components, like `DAE` meshes, might be unsupported in MuJoCo while very common in URDF. Also,
MuJoCo will only take the collision mesh from your URDF and ignore the visual mesh during the loading process. You can
manually integrate these visual elements, ensuring that your robot doesn't just perform well but also looks the part.