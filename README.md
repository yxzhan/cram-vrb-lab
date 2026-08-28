# cram-vrb-lab

[![Binder](https://binder.intel4coro.de/badge_logo.svg)](https://binder.dev.intel4coro.de/v2/gh/yxzhan/cram-vrb-lab/main?urlpath=lab%2Ftree%2Fdemos%2Fpanda_aicor_apartment.ipynb)

A virtual research lab for running
[CRAM](https://github.com/cram2/cognitive_robot_abstract_machine)
(Cognitive Robot Abstract Machine) household tasks in NVIDIA Isaac Sim: an
Isaac scene publishes a real-robot-shaped ROS 2 interface, a
[giskardpy](https://github.com/SemRoCo/giskardpy) server does closed-loop
whole-body control against it, and CRAM plans drive the robot through
high-level actions. Three robots (a Hello Robot Stretch, a Franka Panda and
GARMI) across two apartment scenes; the code is organized so further robots and
scenes plug in alongside them.



https://github.com/user-attachments/assets/1d8c8929-a982-46dd-93b9-01c76cb2d7e9



## Quick start

The docker image is pre-built on the GPU-enabled
[AIRCOR Virtual Research Building](https://vrb.ease-crc.org/) and ships this
repository, Isaac Sim, the CRAM venv and the `ros2_ws` overlay already set up.

### Run the pre-built image locally

> Note: Needs Ubuntu 22.04+, an NVIDIA RTX GPU with the drivers installed (`nvidia-smi`), ~16 GB RAM and ~50 GB
of free disk, plus [Docker](https://docs.docker.com/engine/install/) and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

1. Run in Terminal

```bash
curl -fsSL https://raw.githubusercontent.com/yxzhan/cram-vrb-lab/dev/install.sh | bash
```

1. Open WebUI: [http://localhost:8888/files/demos/web_ui/index.html](http://localhost:8888/files/demos/web_ui/index.html)

1. Run Garmi Demo in Terminal

```
/home/jovyan/cram-vrb-lab/binder/cram_python_wrapper.sh /home/jovyan/cram-vrb-lab/demos/garmi_demo.py
```

## Architecture
```
CRAM plan (notebook, CRAM kernel)
   └─► giskard server (CRAM venv)  ◄── joint states / odom / tf ──┐
             └── velocity commands ─────────────────────────────► Isaac Sim scene
                                                                  (Isaac python)
```

| Directory | Content |
|---|---|
| `cram_vrb_lab/` | The Python package (see layout below) |
| `demos/` | Entry scripts + notebooks composing the package into runnable demos |
| `cognitive_robot_abstract_machine/` | CRAM monorepo (git submodule; also provides the python venv) |
| `assets/` | USD scenes/robots; `stretch_urdf/`, `garmi_description/` and `franka_ros/` are the official description submodules |
| `ros2_ws/` | Generated ROS 2 workspace (not tracked — see below) |
| `binder/` | Docker image definition and the jupyter kernel wrappers |

### Package layout

```
cram_vrb_lab/
├── paths.py            # repo-relative paths, single source
├── specs.py            # what a robot / a scene / a robot-in-a-scene is, as data
├── setups.py           # which robot x scene combinations exist -- the registry both entry scripts look up
├── sim/                # generic Isaac Sim infra: app bootstrap, the shared sim loop, ROS msg helpers
├── control/            # generic control infra: robot+scene giskard world, giskard client/server helpers
├── robots/stretch/     # everything Stretch: joint/topic contract, sim node, giskard interface
├── robots/panda/       # everything Panda: URDF patching, semantic model, sim node, giskard interface
├── robots/garmi/       # everything GARMI: URDF patching, omni-drive base, two FR3 arms, CRAM motion overrides
├── scenes/apartment/   # everything apartment: asset paths + USD/URDF alignment, sim loader, giskard world
├── scenes/garmi_apartment/  # the other flat: USD + its MJCF twin, tabletop objects for perception
├── scenes/empty/       # ground and lights, for a robot that needs no scenery
└── scenes/props/       # the manipulable cube, in Isaac physics and in the twin
```

Every demo runs the same two entry scripts, `demos/sim.py` and
`demos/giskard_server.py`, with `--robot`, `--scene` and where the robot starts
(`--spawn-position` / `--spawn-yaw`, default: the map origin — where a robot
belongs in a room is the demo's call, so the notebook passes it and the package
never guesses). A combination is a row in `setups.py`; the robot half and the scene half are combined at run time (the scene
is merged into the robot's giskard world by `control/giskard_world.py`, and the
Isaac side is driven by one loop in `sim/runner.py`), so a new pairing needs no new
entry scripts and usually no new code.

Robot- and scene-specific facts each live in exactly one module: the joint-order
wire contract in `robots/<robot>/joints.py` (imported by both the sim node and
the giskard config) and the apartment placement in
`scenes/apartment/constants.py` (imported by both the Isaac loader and the
giskard world), so the rendered scene and giskard's collision world cannot
drift apart.

Adding a robot means adding one directory under `robots/` plus one line in
`setups.py`. The Panda is the worked example, and the smaller one: a joint/topic contract, a semantic model
for the digital twin (`semantic_digital_twin` ships none for the Panda), a
giskard world and interface config, and an Isaac Sim node.

## Rebuilding ros2_ws

`ros2_ws/` is entirely generated (third-party robot-description and message
repos cloned and colcon-built by the CRAM submodule's setup script) and is not
tracked in git. To (re)build it:

```bash
git lfs install   # the description repos ship meshes via LFS
OVERLAY_WS=$PWD/ros2_ws \
  bash cognitive_robot_abstract_machine/.github/docker/setup_ros_workspace.sh
```

`binder/cram_python_wrapper.sh` and `binder/entrypoint.sh` source
`ros2_ws/install/setup.bash` when it exists.

## Adding a robot

1. Create `cram_vrb_lab/robots/<name>/` with the three pieces the Stretch
   provides:
   - `joints.py` — the controlled-joint order (the wire contract of the
     velocity command topic) and all topic names, plus URDF loading/patching;
   - `isaac_node.py` — spawn/tune the articulation and a ROS 2 `Node`
     bridging joint states, odom, TF and commands (reuse
     `cram_vrb_lab.sim.ros_utils`);
   - `giskard_config.py` — a `RobotInterfaceConfig` wiring those topics, and a
     world config building the robot alone (or reuse a stock one);
   - `spec.py` — a `RobotSpec` tying those four together for the entry scripts.
   The CRAM submodule already ships semantic models and giskard configs for
   many robots (`semantic_digital_twin/.../robots/`,
   `giskardpy/.../scripts/iai_robots/`) — build on them.
2. Add a robot USD under `assets/`.
3. Add a `Setup(robot=..., scene=...)` to `cram_vrb_lab.setups.SETUPS`, and run it
   with `start_isaac_sim(robot=...)` / `start_giskard_server(robot=...)` from your
   notebook. No new entry scripts.

## Adding a scene

1. Create `cram_vrb_lab/scenes/<name>/` with:
   - `constants.py` — asset paths and the USD prim placement (plus, if giskard
     should know the environment, the URDF path and the measured USD↔URDF
     alignment);
   - `isaac_scene.py` — a `load_<name>_scene(world, render)` loader;
   - `giskard_world.py` — an `EnvironmentSpec` (how to parse the environment
     description, and its pose in `map`) for giskard to merge next to any robot;
   - `spec.py` — a `SceneSpec` tying those together for the entry scripts.
2. Put the scene USD under `assets/` (and the URDF wherever it ships from).
3. Pair it with a robot in `cram_vrb_lab.setups.SETUPS` and run it with
   `start_isaac_sim(scene=...)` / `start_giskard_server(scene=...)`.

## Renaming note

This repository was previously named `cram_isaacsim`; GitHub redirects the old
URL.
