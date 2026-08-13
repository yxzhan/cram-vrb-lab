#!/usr/bin/env python
"""Isaac Sim side of every demo: a robot in a scene, controlled over ROS 2.

One entry script for all combinations -- ``--robot`` and ``--scene`` pick one out
of :mod:`cram_vrb_lab.setups`, which is also where a new combination is added.
The scene publishes the interface a real robot would (joint states, odometry, TF,
camera) and subscribes to the commands giskard streams, so the giskard server and
the notebooks work the same way whichever setup is running:

- publishes ``<robot>/joint_states``, and for a mobile base ``/odom`` and TF
  (odom->base_link->links, plus the fixed camera-frame chain as static tf)
- publishes the head camera per ``--camera``: ``/head_camera/image_raw`` (rgb8)
  and/or ``/head_camera/depth/image_raw`` (32FC1, metres) with camera_info,
  stamped in ``camera_color_optical_frame``
- subscribes ``<robot>/joint_velocity_cmd`` (giskard's streamed velocities,
  integrated into position targets each sim step), ``<robot>/gripper_command``
  (Float64), and for a mobile base ``<robot>/cmd_vel`` (Twist, kinematic base
  with a 1 s watchdog)
- with props: spawns the graspable pick-and-place cube and publishes its
  ground-truth pose on ``/props/pick_cube_pose``

Run with the Isaac Sim python (or from the demo notebooks):
    binder/isaacsim_python_wrapper.sh demos/sim.py \
        [--robot NAME] [--scene NAME] [--camera MODE] [--props]
where MODE is rgb (default), depth, both, or none.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cram_vrb_lab.sim.isaac_app import (
    create_simulation_app,
    livestream_enabled,
    parse_scene_args,
    render_enabled,
)

ARGS = parse_scene_args()
simulation_app = create_simulation_app()  # must run before any isaacsim.core import
RENDER = render_enabled()

from isaacsim.core.api import World
from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.ros2.bridge")
# Shift + left-drag in the viewport to grab and push rigid bodies while the sim
# runs. The GUI app (`/isaac-sim/isaac-sim.sh`) has it because its experience file
# pulls in omni.physx.bundle; the experience a SimulationApp loads
# (isaacsim.exp.base.python.kit) does not, so it has to be asked for here. Only
# worth loading when there is a viewport to interact with.
if RENDER:
    enable_extension("omni.physx.ui")

# Watch the viewport from a browser instead of a local window (ISAAC_LIVESTREAM=1).
if livestream_enabled():
    simulation_app.set_setting("/app/window/drawMouse", True)
    enable_extension("omni.services.livestream.nvcf")

my_world = World(stage_units_in_meters=1.0, physics_dt=1 / 200, rendering_dt=8 / 200)
my_world.reset()

from cram_vrb_lab.sim.runner import run

run(simulation_app, my_world, RENDER, ARGS)
