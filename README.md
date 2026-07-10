# cram_isaacsim

Running [CRAM](https://github.com/cram2/cognitive_robot_abstract_machine)
(Cognitive Robot Abstract Machine) in NVIDIA Isaac Sim. Current demo: giskardpy
closed-loop whole-body control of a Hello Robot Stretch in an apartment scene
(joint-space / Cartesian end-effector / base / gripper goals).

| Directory | Content |
|---|---|
| `giskard_stretch/` | Sim script, giskard server config, and the `giskard_demo.ipynb` demo |
| `cognitive_robot_abstract_machine/` | CRAM monorepo (git submodule) |
| `assets/` | USD assets; `stretch_urdf/` is the official URDF submodule |
| `ros2_ws/` | json_msgs interface package (giskard's action API) |
| `binder/` | Docker image definition |

## Quick start

```bash
git clone --recurse-submodules <this-repo>   # environment setup: see binder/Dockerfile
```

Open `giskard_stretch/giskard_demo.ipynb` (kernel: **CRAM Python (venv)**) — it
starts the simulation and the giskard server and walks through all motion goals.

Details and troubleshooting: `giskard_stretch/README.md`.
