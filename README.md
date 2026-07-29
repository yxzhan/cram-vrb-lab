# cram-vrb-lab

A virtual research lab for running
[CRAM](https://github.com/cram2/cognitive_robot_abstract_machine)
(Cognitive Robot Abstract Machine) household tasks in NVIDIA Isaac Sim: an
Isaac scene publishes a real-robot-shaped ROS 2 interface, a
[giskardpy](https://github.com/SemRoCo/giskardpy) server does closed-loop
whole-body control against it, and CRAM plans drive the robot through
high-level actions. Current demo: a Hello Robot Stretch in an apartment scene;
the code is organized so further robots and scenes plug in alongside it.

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
| `assets/` | USD scenes/robots; `stretch_urdf/` is the official URDF submodule |
| `ros2_ws/` | Generated ROS 2 workspace (not tracked — see below) |
| `binder/` | Docker image definition and the jupyter kernel wrappers |

### Package layout

```
cram_vrb_lab/
├── paths.py            # repo-relative paths, single source
├── sim/                # generic Isaac Sim infra: app bootstrap, ROS msg helpers
├── control/            # generic control infra: process launcher, giskard client/server helpers
├── robots/stretch/     # everything Stretch: joint/topic contract, sim node, giskard interface
├── scenes/apartment/   # everything apartment: asset paths + USD/URDF alignment, sim loader, giskard world
└── scenes/props/       # manipulable props: a graspable cube + pedestals, in Isaac physics and in the twin
```

Robot- and scene-specific facts each live in exactly one module: the joint-order
wire contract in `robots/stretch/joints.py` (imported by both the sim node and
the giskard config) and the apartment placement in
`scenes/apartment/constants.py` (imported by both the Isaac loader and the
giskard world), so the rendered scene and giskard's collision world cannot
drift apart.

## Quick start

```bash
git clone --recurse-submodules https://github.com/yxzhan/cram-vrb-lab.git
# environment setup: see binder/Dockerfile (or run the image via binder/docker-compose.yml)
```

Open `demos/stretch_pick_place_cram.ipynb` (kernel: **CRAM**) — it starts the
simulation and the giskard server and runs the simplest full task, picking a
cube off one pedestal and placing it on another. `demos/stretch_apartment_cram.ipynb`
goes further, into the apartment and its drawers; for hand-built giskard motion
goals use `demos/stretch_apartment_giskard.ipynb`.

Details and troubleshooting: `demos/README.md`.

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
   - `giskard_config.py` — a `RobotInterfaceConfig` wiring those topics.
   The CRAM submodule already ships semantic models and giskard configs for
   many robots (`semantic_digital_twin/.../robots/`,
   `giskardpy/.../scripts/iai_robots/`) — build on them.
2. Add a robot USD under `assets/`.
3. Add `demos/<robot>_<scene>_sim.py` and `demos/<robot>_<scene>_giskard_server.py`
   entry scripts composing the modules (copy the Stretch ones), and pass them to
   `cram_vrb_lab.control.launcher.start_isaac_sim(sim_script=...)` /
   `start_giskard_server(server_script=...)` from your notebook.

## Adding a scene

1. Create `cram_vrb_lab/scenes/<name>/` with:
   - `constants.py` — asset paths and the USD prim placement (plus, if giskard
     should know the environment, the URDF path and the measured USD↔URDF
     alignment);
   - `isaac_scene.py` — a `load_<name>_scene(world, render)` loader;
   - `giskard_world.py` — a world config merging the environment URDF next to
     the robot (see the apartment one; ~25 lines).
2. Put the scene USD under `assets/` (and the URDF wherever it ships from).
3. Compose it with a robot in new `demos/` entry scripts.

## Renaming note

This repository was previously named `cram_isaacsim`; GitHub redirects the old
URL.
