# Demos: Stretch × apartment (giskardpy / CRAM)

Controls the Stretch robot simulated by `stretch_apartment_sim.py` through
[giskardpy](../cognitive_robot_abstract_machine/giskardpy) (closed-loop
whole-body QP control), using the **real-robot interface shape** so the sim
exercises the same giskard code path as physical hardware. Two ways to drive it:

- **`stretch_apartment_giskard.ipynb`** — low-level: hand-build giskard
  `MotionStatechart`s (joint / gripper / Cartesian / base goals). You say *how*
  to move.
- **`stretch_apartment_cram.ipynb`** — high-level: **CRAM** plans (park arm,
  move torso, navigate, …). You say *what* to achieve and CRAM binds it to
  giskard motions at run time.
- **`stretch_pick_place_cram.ipynb`** — the simplest manipulation task, and the
  one to start from: grasp a cube off a pedestal, carry it 1.6 m, put it down.
  The cube is a **real rigid body in Isaac** (`--props`), so the grasp has to
  work in physics, not only in the twin; the sim publishes its true pose so
  every step can be checked against what CRAM believes.

The demo scripts are thin composition layers over the `cram_vrb_lab` package:
robot-specific code lives in `cram_vrb_lab/robots/stretch/`, scene-specific code
in `cram_vrb_lab/scenes/apartment/`, generic infrastructure in
`cram_vrb_lab/sim/` and `cram_vrb_lab/control/`. Purely additive integration: no
changes to giskardpy core or semantic_digital_twin.

## Architecture

```
Isaac Sim (stretch_apartment_sim.py)              giskard server (stretch_apartment_giskard_server.py)
  /stretch/joint_states       ───────────────────►  sync_joint_state_topic
  /odom (odom→base_link, GT)  ───────────────────►  sync_odometry_topic (DiffDrive)
  TF (odom→base_link→links)                         + loads apartment.urdf into its world
  /stretch/cmd_vel            ◄───────────────────  base Twist (closed loop)
  /stretch/joint_velocity_cmd ◄───────────────────  joint velocities (Float64MultiArray, ~15 Hz)
  /stretch/gripper_command    ◄───────────────────  gripper Float64 (native, bypasses giskard)
  /head_camera/image_raw            (rgb8)          head camera, per --camera (perception, not control)
  /head_camera/depth/image_raw      (32FC1, m)

localization stand-in (cram_vrb_lab.control.giskard_server.start_localization_stand_in)
  map→odom (static identity)  ───────────────────►  sync_6dof_joint_with_tf_frame

clients:
  stretch_apartment_giskard.ipynb  MotionStatechart ── JsonAction(/giskard/command) ──►  giskard
  stretch_apartment_cram.ipynb     CRAM plan ──── GiskardWrapper.execute ─────────────►  giskard
```

- **Real-robot interface**
  (`cram_vrb_lab.robots.stretch.giskard_config.StretchRealStyleInterface`):
  giskard consumes `map→odom` (localization) from tf, wheel odometry from
  `/odom`, and joint states from the joint-state topic, and streams base/joint
  velocities back — exactly as against physical hardware. On a real robot only
  the odometry source and the localization node change. Here the sim publishes
  ground-truth `/odom` and a static identity `map→odom` stands in for AMCL/SLAM,
  so `map == odom == the Isaac world frame`.
- **Apartment in the world**
  (`cram_vrb_lab.scenes.apartment.giskard_world.WorldWithStretchAndApartmentDiffDrive`):
  the giskard world giskard plans in also contains the apartment
  (walls/furniture), so motions can avoid it. See the collision-avoidance
  sections in both notebooks. The apartment `.urdf` is aligned to the Isaac
  `.usda` scene by `cram_vrb_lab.scenes.apartment.constants.apartment_pose_in_map`.
- **Head camera** (one RGBD sensor, for perception — not the control loop):
  `stretch_apartment_sim.py --camera {rgb,depth,both,none}` selects the streams
  (rgb -> `/head_camera/image_raw`, depth -> `/head_camera/depth/image_raw` as
  32FC1 metres), plus `camera_info` for both. Images are stamped in
  `camera_color_optical_frame`, which the giskard server already broadcasts as
  part of the Stretch URDF tf tree (REP-103 optical), so they resolve in TF for
  perception/pointclouds — the sim must **not** also publish that frame (two
  parents break tf). From a notebook:
  `start_isaac_sim(camera="both")`. Because the camera drives RTX rendering,
  `--camera none` (or `ISAAC_NO_CAMERA=1`) plus `ISAAC_HEADLESS=1` /
  `ISAAC_RENDER=0` let a machine with no usable GPU display still run the control
  path.
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
  (`cram_vrb_lab.robots.stretch.joints.load_patched_urdf`): insert the
  `link_straight_gripper` expected by the semantic model, fix the zeroed
  finger-joint limits in the official file, and absolutize the relative mesh
  paths.
- The joint-order contract lives in
  `cram_vrb_lab.robots.stretch.joints.CONTROLLED_JOINTS` (the velocity message
  carries no joint names; giskard side and sim side must match — both import
  this single list).
- **Manipulable props** (`--props`, off by default because the pedestals stand
  in floor the apartment demos navigate through): a graspable cube on a
  pedestal plus a second pedestal to carry it to. Isaac gets rigid bodies with
  mass and friction, the twin gets matching boxes, and both are built from the
  one set of numbers in `cram_vrb_lab/scenes/props/constants.py` — the
  apartment's USD/URDF alignment is only approximate, so props that hung off
  apartment furniture would confuse a modelling error with a grasp failure. The
  sim publishes the cube's true pose on `/props/pick_cube_pose` and as the tf
  frame `pick_cube_gt`, which is the only way to tell a real grasp from CRAM
  merely believing it grasped (`AttachNode` moves the twin's cube regardless).
- **Gripper aperture** (`cram_vrb_lab/robots/stretch/gripper.py`): the semantic
  Stretch model's `GripperState.OPEN` is finger angle 0.109 rad, which parts the
  SG3 pads by only 3.6 cm — narrower than most things worth grasping, and the
  grasp then fails for reasons that look like reach or approach-direction
  problems. `open_gripper_to(robot, gap)` redefines that state in metres of pad
  gap, on the robot instance, without touching the vendored model.

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
# Full CRAM ROS 2 workspace at ./ros2_ws: robot descriptions (incl. the
# iai_apartment/iai_kitchen meshes apartment.urdf needs), json_msgs, robokudo
# msgs, ... git-lfs must be initialized first so the mesh LFS objects come down.
git lfs install
OVERLAY_WS=$PWD/ros2_ws \
  bash cognitive_robot_abstract_machine/.github/docker/setup_ros_workspace.sh
```

## Running

Everything can be started from either notebook (kernel: **CRAM**) — each one's
first cell calls `cram_vrb_lab.control.launcher.start_isaac_sim()` /
`start_giskard_server()` to launch the simulation and the giskard server
(background processes, or pass `terminal=True` to open each in a
`gnome-terminal` window):

- `stretch_apartment_giskard.ipynb` — joint-space, gripper, Cartesian
  (arm-only / whole-body), base goals, and external collision avoidance against
  the apartment.
- `stretch_apartment_cram.ipynb` — the same robot driven by high-level CRAM
  plans (park arms, move torso, gripper, navigate).
- `stretch_pick_place_cram.ipynb` — CRAM pick-and-place on the props;
  starts the sim with `start_isaac_sim(props=True)`.

Manual start (source `/opt/ros/jazzy/setup.bash` and
`ros2_ws/install/setup.bash` in every terminal):

1. **Isaac Sim**: `binder/isaacsim_python_wrapper.sh demos/stretch_apartment_sim.py`
2. **giskard server** (wait for the `giskard is ready` log line; it also launches
   the static `map→odom` localization stand-in):
   ```bash
   cognitive_robot_abstract_machine/.venv/bin/python demos/stretch_apartment_giskard_server.py
   ```

For custom motion goals follow `cram_vrb_lab/control/giskard_client.py` +
`stretch_apartment_giskard.ipynb`; task types live in
`giskardpy/src/giskardpy/motion_statechart/tasks/`, worked examples in
`giskardpy/doc/examples/`. For high-level plans follow
`stretch_apartment_cram.ipynb`; CRAM actions live in
`coraplex/src/coraplex/robot_plans/actions/`.

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
- **External collision avoidance** (against the apartment) is opt-in per motion:
  `stretch_apartment_giskard.ipynb`'s `run_goal(..., avoid_collisions=True)` and
  `stretch_apartment_cram.ipynb`'s `real_robot(collision_avoidance=True)` add an
  `ExternalCollisionAvoidance` goal. If the robot refuses to move it may start
  inside a collision body's violated distance — re-check
  `cram_vrb_lab.scenes.apartment.constants.apartment_pose_in_map` or disable
  avoidance.
- **Self-collision avoidance is not enabled**
  (`WorldWithStretchConfigDiffDrive.setup_collision_config` is empty). Do not
  command EE goals that press into the robot's own body (e.g. straight down
  with a bent wrist). Contact can physically jam the gripper fingers; reset the
  gripper through the sim's native interface:
  ```bash
  ros2 topic pub --once /stretch/gripper_command std_msgs/msg/Float64 "{data: 0.05}"
  ```
  To move a twisted arm back, send a giskard joint goal (the raw
  `/stretch/joint_command` position path was removed in the slim-down).
- giskard is a **whole-body controller**: even an "arm-only" goal may slightly
  adjust other controlled joints (including the base).
- Do not teleop the base manually (`/stretch/cmd_vel`) while giskard is
  executing — the two twist streams would fight each other.
- TF has redundant `odom→base_link` publishers (the sim and giskard's own world
  viz); this only affects RViz display — the control path consumes `/odom` (topic)
  and `map→odom` (tf), not `odom→base_link` tf.
- QP `target_frequency=15` (`stretch_apartment_giskard_server.py`); below 20 the
  library warns (harmless here). The sim-side integrator's `VEL_MAX_LEAD`
  (0.02, `cram_vrb_lab/robots/stretch/isaac_node.py`) clamps how far targets may
  lead the measured position, which also prevents force build-up on contact.
