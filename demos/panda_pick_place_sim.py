#!/usr/bin/env python
"""Isaac Sim empty scene with a Franka Panda and the pick-and-place props.

Simulation side of the Panda demo. Deliberately the smallest scene that can
answer "does the grasp work": a bolted-down 7-DoF arm, two stands, and a cube.
No apartment, no mobile base, no cameras -- the Stretch demos cover those, and
each of them is a way for a grasp to fail that has nothing to do with grasping.

- publishes /panda/joint_states and the cube's ground-truth pose on
  /props/pick_cube_pose (plus the pick_cube_gt tf frame)
- subscribes /panda/joint_velocity_cmd (giskard's streamed velocities,
  integrated into position targets each sim step) and /panda/gripper_command
  (Float64, per-finger travel in metres)

Run with the Isaac Sim python (or from the demo notebook):
    binder/isaacsim_python_wrapper.sh demos/panda_pick_place_sim.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cram_vrb_lab.sim.isaac_app import create_simulation_app, render_enabled

simulation_app = create_simulation_app()  # must run before any isaacsim.core import
RENDER = render_enabled()

from isaacsim.core.api import World
from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.ros2.bridge")

my_world = World(stage_units_in_meters=1.0, physics_dt=1 / 200, rendering_dt=8 / 200)
my_world.reset()

from cram_vrb_lab.robots.panda.isaac_node import PandaROS, move_to_park, spawn_panda
from cram_vrb_lab.scenes.empty.isaac_scene import load_empty_scene
from cram_vrb_lab.scenes.props.constants import PANDA_LAYOUT
from cram_vrb_lab.scenes.props.isaac_props import PropsROS, spawn_props

load_empty_scene(my_world, RENDER)
panda = spawn_panda(my_world, RENDER)
cube = spawn_props(my_world, RENDER, layout=PANDA_LAYOUT)
move_to_park(panda, my_world, RENDER)  # last: spawn_props resets the world

import rclpy

if not rclpy.ok():
    rclpy.init(args=None)

panda_node = PandaROS(panda)
props_node = PropsROS(cube)
print("PandaROS node ready.")

try:
    while simulation_app.is_running():
        dt = my_world.get_rendering_dt()
        panda_node.integrate_joint_velocities(dt)
        my_world.step(render=RENDER)
        # Drain ALL pending callbacks: spin_once handles only ONE message per
        # call, and a single spin per sim tick falls behind a 20 Hz command
        # stream, so commands arrive stale.
        for _ in range(16):
            rclpy.spin_once(panda_node, timeout_sec=0.0)
        panda_node.publish_joint_states()
        props_node.publish_props()
except KeyboardInterrupt:
    pass
finally:
    panda_node.destroy_node()
    props_node.destroy_node()
    rclpy.shutdown()
    simulation_app.close()
