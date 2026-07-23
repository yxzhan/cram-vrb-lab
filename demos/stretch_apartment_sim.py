#!/usr/bin/env python
"""Isaac Sim apartment scene with a Stretch robot, controlled over ROS 2.

Simulation side of the giskard demo (composition of the cram_vrb_lab building
blocks; the giskard server and the demo notebooks live in this same directory):
- publishes /stretch/joint_states, /odom, TF (odom->base_link->links), and the
  head camera per --camera: /head_camera/image_raw (rgb8) and/or
  /head_camera/depth/image_raw (32FC1, metres) with camera_info, stamped in
  camera_color_optical_frame (that frame comes from the giskard server's URDF tf tree)
- subscribes /stretch/cmd_vel (Twist, kinematic base with a 1 s watchdog),
  /stretch/joint_velocity_cmd (giskard's streamed velocities, integrated into
  position targets each sim step), /stretch/gripper_command (Float64)

Run with the Isaac Sim python (or from the demo notebooks):
    binder/isaacsim_python_wrapper.sh demos/stretch_apartment_sim.py [--camera MODE]
where MODE is rgb (default), depth, both, or none.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cram_vrb_lab.sim.isaac_app import (
    create_simulation_app,
    parse_camera_args,
    render_enabled,
)

CAMERA_MODE, WANT_RGB, WANT_DEPTH = parse_camera_args()
simulation_app = create_simulation_app()  # must run before any isaacsim.core import
RENDER = render_enabled()

from isaacsim.core.api import World
from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.ros2.bridge")

my_world = World(stage_units_in_meters=1.0, physics_dt=1 / 200, rendering_dt=8 / 200)
my_world.reset()

from cram_vrb_lab.scenes.apartment.isaac_scene import load_apartment_scene
from cram_vrb_lab.robots.stretch.isaac_node import (
    StretchROS,
    create_head_camera,
    spawn_stretch,
)

load_apartment_scene(my_world, RENDER)
stretch = spawn_stretch(my_world, RENDER)
head_cam = (create_head_camera(my_world, RENDER, want_depth=WANT_DEPTH)
            if CAMERA_MODE != "none" else None)

import rclpy

if not rclpy.ok():
    rclpy.init(args=None)

stretch_node = StretchROS(stretch, head_cam=head_cam,
                          publish_rgb=WANT_RGB, publish_depth=WANT_DEPTH)
print("StretchROS node ready.")

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
