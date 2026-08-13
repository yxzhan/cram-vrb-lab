# Demos: Stretch × apartment (giskardpy / CRAM)

Controls the Stretch robot simulated by `sim.py` through
[giskardpy](../cognitive_robot_abstract_machine/giskardpy) (closed-loop
whole-body QP control), using the **real-robot interface shape** so the sim
exercises the same giskard code path as physical hardware. Two ways to drive it:

- **`stretch_apartment_giskard.ipynb`** — low-level: hand-build giskard
  `MotionStatechart`s (joint / gripper / Cartesian / base goals). You say *how*
  to move.
- **`stretch_apartment_cram.ipynb`** — high-level: **CRAM** plans (park arm,
  move torso, navigate, …). You say *what* to achieve and CRAM binds it to
  giskard motions at run time.
- **`panda_pick_place_cram.ipynb`** — **the one to start from**: the simplest
  manipulation task on the simplest robot. A Franka Panda bolted to a table in
  the apartment grasps a cube off that table and puts it down further along it.
  The cube is a **real rigid body in Isaac**, so the grasp has to work in physics
  and not only in the twin; the sim publishes its true pose, so every step can be
  checked against what CRAM believes.
- **`stretch_pick_place_cram.ipynb`** — the same task on the Stretch, in the
  apartment: grasp the cube, drive 1.6 m holding it, put it down. Everything the
  Panda demo does plus a mobile base, a telescoping arm and a gripper whose pads
  swing on an arc — which is why it is the *second* one to read. **Its cube now
  lands on the apartment floor** rather than on the 0.7 m posts it used to stand
  on, and its standing positions and grasp have not been re-tuned for that.

There are exactly two entry scripts, `sim.py` and `giskard_server.py`, and both
take `--robot`, `--scene` and where the robot starts (`--spawn-position X Y Z`,
`--spawn-yaw RAD`, default: the map origin, unrotated). Which combinations exist,
and everything else that differs between them, is data in
`cram_vrb_lab/setups.py`. The scripts themselves
are thin composition layers over the `cram_vrb_lab` package: robot-specific code
lives in `cram_vrb_lab/robots/<robot>/`, scene-specific code in
`cram_vrb_lab/scenes/<scene>/`, generic infrastructure in `cram_vrb_lab/sim/` and
`cram_vrb_lab/control/`. Purely additive integration: no changes to giskardpy
core or semantic_digital_twin.

## Architecture

```
Isaac Sim (sim.py --robot stretch)                giskard server (giskard_server.py --robot stretch)
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
  (`cram_vrb_lab.scenes.apartment.giskard_world.apartment_environment`, merged
  next to the robot by `cram_vrb_lab.control.giskard_world.build_world_config`):
  the giskard world giskard plans in also contains the apartment
  (walls/furniture), so motions can avoid it. See the collision-avoidance
  sections in both notebooks. The apartment `.urdf` is aligned to the Isaac
  `.usda` scene by `cram_vrb_lab.scenes.apartment.constants.apartment_pose_in_map`.
- **Head camera** (one RGBD sensor, for perception — not the control loop):
  `sim.py --camera {rgb,depth,both,none}` selects the streams
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
- **Watching the sim from another machine**: `ISAAC_LIVESTREAM=1` streams the
  viewport over WebRTC (`omni.services.livestream.nvcf`, port 49100) instead of
  opening a local window, so a browser client can watch and mouse around the
  scene. Set it next to `ISAAC_HEADLESS=1` before `start_isaac_sim()` — the sim
  runs as a subprocess and inherits the environment — and leave rendering on
  (`ISAAC_RENDER=0` would leave nothing to stream). The viewer for that stream,
  NVIDIA's `isaacsim-webrtc-streaming-client`, is installed in the image
  (`binder/Dockerfile`) and `start_streaming_client()` opens it on the
  container's desktop — fire-and-forget like `start_rviz()`, with no ready
  marker to wait for; the connection is made in the client itself.
  `garmi_demo.py` calls it only when `ISAAC_LIVESTREAM=1`, and `stop()` closes
  it with everything else.
- **Grabbing things in the viewport** (shift + left-drag on a rigid body while
  the sim runs, as in the GUI app): that is `omni.physx.ui`, which the GUI's
  experience file loads through `omni.physx.bundle` but the one a `SimulationApp`
  loads (`isaacsim.exp.base.python.kit`) does not. `sim.py` enables it explicitly
  when rendering is on. It moves **rigid bodies** — the props. The robot is not
  one: its base is teleported every step (`integrate_base`) and its joints are
  driven to giskard's targets, so dragging it only fights the controller. Force
  and behaviour are tunable through the usual carb settings
  (`/physics/pickingForce`, `/physics/mousePush`, `/physics/forceGrab`).
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
- **The manipulable cube** (`cram_vrb_lab/scenes/props/`, on the Stretch side
  behind `--props`): Isaac gets a rigid body with mass and friction, the twin
  gets a matching box, and both are built from one `PropLayout`. There are no
  pedestals — the cube is released just above the surface it should land on and
  physics settles it onto whatever the scene provides, so `spawn_props` prints
  the pose it actually settled at. That printout is the only way to learn the
  real height of a surface that belongs to the scene rather than to this code,
  and if that surface is not in the environment description either, giskard
  plans as though nothing were under the cube. The sim publishes the cube's true
  pose on `/props/pick_cube_pose` and as the tf frame `pick_cube_gt`, which is
  the only way to tell a real grasp from CRAM merely believing it grasped
  (`AttachNode` moves the twin's cube regardless). One layout per setup, and for
  a bolted-down arm it is a function of where the arm is bolted:
  `APARTMENT_LAYOUT` is fixed in the room for the Stretch, `panda_layout_at(spawn
  position, yaw)` follows the Panda.
- **The Panda** (`cram_vrb_lab/robots/panda/`) is built entirely from the stock
  `panda_arm_hand.urdf` that ships with Isaac's URDF importer — imported into the
  stage at startup, so no converted USD is checked in, and parsed into the same
  description giskard and the twin plan against. `semantic_model.py` supplies the
  `Panda` robot description that `semantic_digital_twin` has no model for.
- **Gripper aperture.** Both grippers needed their open width corrected, and in
  both cases a too-narrow hand looks exactly like a reach or approach-direction
  failure. For the Stretch (`robots/stretch/gripper.py`) the semantic model's
  `GripperState.OPEN` is finger angle 0.109 rad, which parts the SG3 pads by only
  3.6 cm — narrower than the 5 cm cube; `open_gripper_to(robot, gap)` redefines
  that state in metres of pad gap, on the robot instance, without touching the
  vendored model. The Panda's hand sets its own `GRIPPER_OPEN_TRAVEL`.
- **Closing on an object** (`robots/panda/motions.py`). A gripper-close is a
  joint-position goal, and under closed-loop control it is checked against the
  *measured* finger positions — which a hand closing on something rigid never
  reaches, so the grasp hangs forever holding the object. `PandaMoveGripper` ends
  the close on "position reached **or** pushing for 3 s". The grip then has to
  survive the gap between giskard goals, which is why
  `StreamedVelocityIntegrator` lets nominated joints keep their leading target
  instead of snapping it onto the measured position.

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

- `panda_aicor_apartment.ipynb` — a Franka Panda mounted on a table in the
  apartment, grasping a cube off that table and placing it down further along.
  Same two entry scripts, selected with `start_isaac_sim(robot="panda", ...)` /
  `start_giskard_server(robot="panda", ...)`, on its own topics.
- `stretch_aicor_apartment.ipynb` — the Stretch in the same apartment, driven by
  high-level CRAM plans, with robokudo perception on the head camera.
- `stretch_garmi_apartment.ipynb` — the Stretch in the other flat:
  `start_isaac_sim(scene="garmi_apartment", spawn_position=(0, 6, 0.05), camera="both")`
  and the same spawn pose for the server.
- `panda_garmi_apartment.ipynb` — the Panda in that flat instead, bolted to the
  floor in front of the kitchen run: park, gripper, and four Cartesian goals
  around the cabinet with collision avoidance against the MJCF. No perception and
  no props — and it documents why it stops short of opening a drawer (the
  garmi-apartment USD carries no physics joints, so the MJCF's drawers and doors
  articulate in the twin only).
- `garmi_demo.py` — the same kitchen, worked by **GARMI itself** rather than by a
  bare arm: `start_isaac_sim(robot="garmi", scene="garmi_apartment",
  spawn_position=(..., 0.0259), spawn_yaw=math.pi/2)`. Parks both arms in the
  description's own home pose, **drives** to the cabinet with a `NavigateAction`,
  then opens a drawer with one arm and a cabinet door with the other. The mecanum
  base is an `OmniDrive` on giskard's side and kinematic in the sim — the wheels
  are undriven and `base_link` is teleported, because this description models
  each mecanum wheel as a plain cylinder with no rollers. See
  `cram_vrb_lab/robots/garmi/isaac_node.py:undrive_wheels`.
  The twin is `semantic_digital_twin.robots.garmi.Garmi` (mobile base, lift
  torso, pan/tilt neck, `garmi.srdf` self-collision matrix); the patched URDF
  renames this description's arms and head joints onto the names that model
  expects — see `UPSTREAM_RENAMES` in `cram_vrb_lab/robots/garmi/joints.py`.

Manual start (source `/opt/ros/jazzy/setup.bash` and
`ros2_ws/install/setup.bash` in every terminal):

1. **Isaac Sim**: `binder/isaacsim_python_wrapper.sh demos/sim.py`
   (add `--robot panda`, `--scene garmi_apartment`, `--spawn-position 0 6 0.05`,
   ... — `--help` lists them)
2. **giskard server** (wait for the `giskard is ready` log line; it also launches
   the static `map→odom` localization stand-in), with the *same* `--robot`,
   `--scene` and spawn pose — for a robot bolted to `map` the spawn pose is the
   only thing that tells giskard where it stands, and a value that disagrees with
   the sim's makes it plan for a robot that is not the one being rendered:
   ```bash
   cognitive_robot_abstract_machine/.venv/bin/python demos/giskard_server.py
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
- QP `target_frequency=15` (`giskard_server.py`); below 20 the
  library warns (harmless here). The sim-side integrator's `VEL_MAX_LEAD`
  (0.02, `cram_vrb_lab/robots/stretch/isaac_node.py`) clamps how far targets may
  lead the measured position, which also prevents force build-up on contact.
