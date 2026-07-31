#!/usr/bin/env python
"""Isaac Sim garmi-apartment scene with a Stretch robot, controlled over ROS 2.

The same simulation side as :mod:`stretch_apartment_sim`, in the other flat: it
swaps ``load_apartment_scene`` for ``load_garmi_apartment_scene`` and spawns the
robot inside the rooms instead of at the map origin (which in this scene lies
outside the flat -- see
:data:`cram_vrb_lab.scenes.garmi_apartment.constants.STRETCH_SPAWN_POSITION`).
Everything else -- the published topics, the subscribed commands, the ROS node --
is unchanged, so the giskard server and the notebooks work the same way:

- publishes /stretch/joint_states, /odom, TF (odom->base_link->links, plus the
  fixed camera-frame chain as static tf), and the head camera per --camera:
  /head_camera/image_raw (rgb8) and/or /head_camera/depth/image_raw (32FC1,
  metres) with camera_info, stamped in camera_color_optical_frame
- subscribes /stretch/cmd_vel (Twist, kinematic base with a 1 s watchdog),
  /stretch/joint_velocity_cmd (giskard's streamed velocities, integrated into
  position targets each sim step), /stretch/gripper_command (Float64)

There is no ``--props`` here: the pick-and-place cube's layout is defined against
the other apartment's geometry (``cram_vrb_lab.scenes.props.constants``), and the
perception notebook this scene exists for does not use it.

Run with the Isaac Sim python (or from the demo notebooks):
    binder/isaacsim_python_wrapper.sh demos/stretch_garmi_apartment_sim.py \
        [--camera MODE]
where MODE is rgb (default), depth, both, or none. The perception notebook needs
``--camera both``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cram_vrb_lab.sim.isaac_app import (
    create_simulation_app,
    parse_scene_args,
    render_enabled,
)

ARGS = parse_scene_args()
simulation_app = create_simulation_app()  # must run before any isaacsim.core import
RENDER = render_enabled()

from isaacsim.core.api import World
from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.ros2.bridge")

my_world = World(stage_units_in_meters=1.0, physics_dt=1 / 200, rendering_dt=8 / 200)
my_world.reset()

from cram_vrb_lab.robots.stretch.isaac_node import (
    StretchROS,
    create_head_camera,
    spawn_stretch,
)
from cram_vrb_lab.scenes.garmi_apartment.constants import STRETCH_SPAWN_POSITION
from cram_vrb_lab.scenes.garmi_apartment.isaac_scene import load_garmi_apartment_scene

load_garmi_apartment_scene(my_world, RENDER)
stretch = spawn_stretch(my_world, RENDER, position=STRETCH_SPAWN_POSITION)
head_cam = (create_head_camera(my_world, RENDER, want_depth=ARGS.want_depth)
            if ARGS.camera != "none" else None)

import rclpy

if not rclpy.ok():
    rclpy.init(args=None)

stretch_node = StretchROS(stretch, head_cam=head_cam,
                          publish_rgb=ARGS.want_rgb, publish_depth=ARGS.want_depth)
# flush=True is load-bearing, not decoration. This is the marker
# launcher.start_isaac_sim polls the log file for, and Isaac's own logging goes to
# fd 1 from C++ while this print lands in Python's block buffer -- which the main
# loop below never writes enough to fill. Without the flush the sim comes up
# fully, publishes every topic, and the notebook's first cell still sits there
# until it times out.
print("StretchROS node ready.", flush=True)

try:
    while simulation_app.is_running():
        dt = my_world.get_rendering_dt()
        stretch_node.integrate_base(dt)
        stretch_node.integrate_joint_velocities(dt)
        my_world.step(render=RENDER)
        # Drain ALL pending callbacks (spin_once handles only ONE message per
        # call; with giskard streaming two topics at ~20 Hz each, a single
        # spin per sim tick falls behind and commands arrive stale).
        for _ in range(16):
            rclpy.spin_once(stretch_node, timeout_sec=0.0)
        stretch_node.publish_joint_states()
        stretch_node.publish_tf()
        stretch_node.publish_odom()
        stretch_node.publish_camera()
except KeyboardInterrupt:
    pass
finally:
    stretch_node.destroy_node()
    rclpy.shutdown()
    simulation_app.close()
