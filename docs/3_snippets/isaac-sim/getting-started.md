---
sidebar_position: 0
---

# Getting Started

Welcome to the Getting Started for using IsaacSim. Here, we will quickly go through:

1. An efficient way of development in Isaac Sim
2. Three workflows (GUI, extensions, and standalone extensions)
3. GUI in Isaac Sim
4. Standalone Python

Some information are borrowed from the [official docs of Isaac Sim](https://docs.omniverse.nvidia.com/isaacsim/latest/index.html), but we provide a more concentrated version to get you ready.

## An efficient way of dev in Isaac Sim

Before we get started, here are several things you need to know:

1. Like other modules like the Create in Omniverse, Isaac Sim is also built on Omniverse Kit. Almost all the functions are implemented as "extensions", *i.e.* every action you do in the GUI can be implemented via coding.
2. You can find the Isaac Sim local folder at `INSTALL_DIRECTORY/pkg/isaac-sim-VERSION/`. For ubuntu, `INSTALL_DIRECTORY` is `~/.local/share/ov/` by default. It would be a rapid way, if you want to add new features to your app, to perform a search in this folder to find possibly relative codes.
3. For docs, always start from [this main page](https://docs.omniverse.nvidia.com/isaacsim/latest/index.html) for the latest version of the documentation, and use the built-in searching (rather than directly searching on Google, which may provide out-of-date links).

## Three workflows in Isaac Sim

There are three workflows in Isaac Sim that you can use to develop:

1. GUI: directly interacting with the Isaac Sim GUI, pushing buttons, connecting action graphs, to implement what you want. This is useful if you want to:
    - constructing a scene or buiding a robot
    - tune the physical parameters (*e.g.*, joint damping and stifness, light temperature, *e.t.c*)
2. Extensions: Start from the menu bar in a launched Isaac Sim, which pops out a panel for your app. You interact with your app inside. This is useful if you want to:
    - interact with your app or provide some arguments (*e.g.* the position you want the robot to go) to the app in play
    - build a plug-in for Isaac Sim, rather than just performing simulations
3. Standalone extensions: Start by running scripts, . This is useful if you want to:
    - run your app in headless mode
    - control over timing of physics and rendering steps

[This page](https://docs.omniverse.nvidia.com/isaacsim/latest/introductory_tutorials/tutorial_intro_workflows.html) provides a more thorough introduction.

## Quick-Start in Standalone Extensions

Coding in Isaac Sim endows powerful simulations. In this page, we will only focus on the standalone extensions, a way more flexible way. Go to your installation directory of your isaac sim (`INSTALL_DIRECTORY/pkg/isaac-sim-VERSION`), you will find a `python.sh` here. To start a standalone app, you run:

```shell
./python.sh PATH_TO_YOUR_APP.py
```

Now, let's add something to our script.

### Create a world

### Adding a cube and a robot

### Adding a camera to the robot

### Render image
