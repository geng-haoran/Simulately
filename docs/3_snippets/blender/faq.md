---
sidebar_position: 5
---

# Frequently Asked Questions


## What's the difference between the different renders?

There are 3 renderes in blender, each has its own uniqueness and weaknesses:

- [Workbench](https://docs.blender.org/manual/en/latest/render/workbench/index.html): The default viewport render. The Workbench Engine is a render engine optimized for fast rendering during modeling and animation preview. Although fast, it is not intended to be a render engine that will render final images for a project.
- [Eevee](https://docs.blender.org/manual/en/latest/render/eevee/index.html): Eevee is Blender’s realtime render engine built using OpenGL focused on speed and interactivity while achieving the goal of rendering PBR materials. Eevee can be used interactively in the 3D Viewport but also produce high quality final renders, so it's great for if you want to render out something quick or stylized, but lacks a lot of functionalities and customization Cycles offers.
- [Cycles](https://docs.blender.org/manual/en/latest/render/cycles/index.html): Cycles is Blender’s physically-based path tracer for production rendering. It is designed to provide physically based results out-of-the-box, with artistic control and flexible shading nodes for production needs. It's usually the one people use for more realistic (or realistically-styled) renders, although it's slow and you need good hardware to get it to render at a descent speed.