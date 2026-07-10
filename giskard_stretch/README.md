# giskardpy × Isaac Sim Stretch integration

Controls the Stretch robot simulated by `apartment.py` through
[giskardpy](../cognitive_robot_abstract_machine/giskardpy) (closed-loop
whole-body QP control). Purely additive integration: no changes to giskardpy
core or semantic_digital_twin.

## Architecture

```
Isaac Sim (apartment.py)                              giskard server (this dir)
  /stretch/joint_states       ─────────────────────►  sync_joint_state_topic
  /odom                       ─────────────────────►  sync_odometry_topic (DiffDrive)
  /stretch/cmd_vel            ◄─────────────────────  base Twist (closed loop)
  /stretch/joint_velocity_cmd ◄─────────────────────  joint velocities (Float64MultiArray, ~20 Hz)

  giskard_demo.ipynb ── JsonAction(/giskard/command) ──► giskard
```

- giskard's closed loop outputs **joint velocities**; the sim integrates them
  into position targets every physics step
  (`StretchROS.integrate_joint_velocities`: exact sim dt, anti-windup lead
  clamp, joint-limit clamp, hold after 0.5 s of silence).
- The base Twist goes straight to `/stretch/cmd_vel`; `integrate_base` has a
  1 s cmd_vel watchdog — when a stream (giskard/Nav2) stops, the base stops
  instead of latching the last twist forever. Note this also means one-shot
  manual teleop that relies on latching stops after 1 s (keep publishing for
  continuous motion).
- The URDF comes directly from the official
  [hello-robot/stretch_urdf](https://github.com/hello-robot/stretch_urdf)
  submodule (`assets/stretch_urdf/`, SE3 + DW3 wrist + SG3 gripper variant);
  no URDF file is maintained in this repo. Load-time patches
  (`stretch_joints.load_patched_urdf`): insert the `link_straight_gripper`
  expected by the semantic model, fix the zeroed finger-joint limits in the
  official file, and absolutize the relative mesh paths.
- The joint-order contract lives in `stretch_joints.CONTROLLED_JOINTS` (the
  velocity message carries no joint names; giskard side and sim side must
  match).

## One-time setup (already baked into the Docker image)

```bash
# ROS dependencies
sudo apt-get install -y ros-jazzy-rclpy-message-converter ros-jazzy-py-trees ros-jazzy-py-trees-ros
# CRAM venv (uv workspace) + the two extras giskardpy's ROS2 middleware needs
cd cognitive_robot_abstract_machine
uv sync
.venv/bin/pip install py_trees argcomplete nest_asyncio
cd ..
# CRAM jupyter kernel: a wrapper that sources ROS 2 + the ros2_ws overlay
# before starting the venv python, so notebooks need no pre-sourced jupyter.
cp binder/cram_python_wrapper.sh ~/.local/bin/
mkdir -p ~/.local/share/jupyter/kernels/cram
cp binder/cram-kernel.json ~/.local/share/jupyter/kernels/cram/kernel.json
# giskard's action interface package (workspace at the repo root)
source /opt/ros/jazzy/setup.bash
cd ros2_ws && colcon build --packages-select json_msgs
```

## Running

Everything can be started from `giskard_demo.ipynb` (kernel: **CRAM**) — it
launches the simulation and the giskard server as background
processes and demonstrates joint-space, gripper, Cartesian (arm-only /
whole-body), and base goals.

Manual start (source `/opt/ros/jazzy/setup.bash` and
`ros2_ws/install/setup.bash` in every terminal):

1. **Isaac Sim**: `binder/isaacsim_python_wrapper.sh giskard_stretch/apartment.py`
2. **giskard server** (wait for the `giskard is ready` log line):
   ```bash
   cognitive_robot_abstract_machine/.venv/bin/python giskard_stretch/giskard_stretch_isaac.py
   ```

For custom motion goals follow `giskard_client.py` + the notebook; task types
live in `giskardpy/src/giskardpy/motion_statechart/tasks/`, worked examples in
`giskardpy/doc/examples/`.

## Measured accuracy and known issues

| Path | Measured |
|---|---|
| Joint goals | converge to 0.01-0.02 (demo threshold 0.02) |
| Cartesian EE | reliably reached at 3 cm / 0.03 rad threshold |
| Base | ~5-8 cm (5 cm threshold + settle-phase residual) |

- **Keep the settle phase**: after EndMotion fires, giskard's behavior tree
  keeps publishing the last commanded velocities for ~1 s before its
  terminate-zero message. The cmd_vel watchdog (1 s) bounds the damage, but
  `giskard_client.add_end_conditions` (hold the reached goal for 1 s while the
  QP winds velocities down to zero) stops much more precisely. Use it in your
  own scripts.
- **Self-collision avoidance is not enabled**
  (`WorldWithStretchConfigDiffDrive.setup_collision_config` is empty). Do not
  command EE goals that press into the robot's own body (e.g. straight down
  with a bent wrist). Contact can physically jam the gripper fingers; reset
  through the sim's native interface:
  ```bash
  ros2 topic pub --once /stretch/gripper_command std_msgs/msg/Float64 "{data: 0.05}"
  ```
  and use `/stretch/joint_command` to move back to a neutral pose if needed.
- giskard is a **whole-body controller**: even an "arm-only" goal may slightly
  adjust other controlled joints (including the base).
- Do not teleop manually while giskard is executing (`/stretch/cmd_vel` and
  `/stretch/joint_command` would fight each other).
- giskard and the sim both publish TF (`odom -> base_link` has two sources);
  this only affects RViz display — the control path does not consume TF.
- QP `target_frequency=20` (`giskard_stretch_isaac.py`); below 20 the library
  warns. The sim-side integrator's `VEL_MAX_LEAD` (0.02, apartment.py) clamps
  how far targets may lead the measured position, which also prevents force
  build-up on contact.
