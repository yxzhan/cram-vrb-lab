#!/usr/bin/env python
"""Giskard motion-control server for every demo: a robot in a scene (sim.py).

One entry script for all combinations -- ``--robot`` and ``--scene`` pick one out
of :mod:`cram_vrb_lab.setups`, which supplies the world config (the robot with
the scene's walls and furniture merged in, so collision avoidance covers the
room) and the robot interface config (which topics giskard reads and writes).

The interface is the *real-robot* shape wherever the robot has one: giskard reads
joint states and, for a mobile base, wheel odometry over ROS and a ``map -> odom``
localization transform from tf, and streams velocity commands back. The only
sim-specific choice is where ``map -> odom`` comes from: a real robot runs
AMCL/SLAM, here a static identity transform stands in for it (see :func:`main`).
A robot bolted to ``map`` -- the Panda -- has no localization to wait for and no
odometry to sync, and only exchanges joint states and joint velocities.

The props the pick-and-place demos manipulate are not in the world config: the
notebook adds them over ``/world_sync`` once it is connected, which keeps the
scene giskard avoids collisions in identical to the one Isaac renders.

``--spawn-position`` / ``--spawn-yaw`` must repeat what ``sim.py`` was given: for
a robot bolted to ``map`` they are the only thing that tells giskard where it
stands. A robot with a base ignores them and learns its pose from odometry.

Run with the cognitive_robot_abstract_machine venv python, with ROS jazzy and the
ros2_ws overlay sourced (see README.md):
    python demos/giskard_server.py [--robot NAME] [--scene NAME] \
        [--spawn-position X Y Z] [--spawn-yaw RAD]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cram_vrb_lab.control.giskard_server import start_localization_stand_in
from cram_vrb_lab.setups import (
    add_setup_arguments,
    get_setup,
    spawn_pose_from_args,
)

from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.behavior_tree_config import ClosedLoopBTConfig
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.qp.qp_controller_config import QPControllerConfig

from cram_vrb_lab.control.rate import CONTROL_HZ_ENV, control_hz


PREDICTION_HORIZON = 15
"""Steps the QP looks ahead.

Steps, not seconds: the lookahead is ``PREDICTION_HORIZON / control_hz``, so it
stretches when the rate is lowered -- 1.5 s at the default 10 Hz. Left at 15
across rates because the reason to lower the rate is a sim that cannot feed the
controller, and a longer horizon smooths the commands that reach it; lower this
alongside the rate to hold the lookahead in seconds constant instead.
"""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_setup_arguments(parser)
    parser.add_argument(
        "--control-hz",
        type=float,
        default=None,
        help=(
            "rate [Hz] the QP loop runs at. Defaults to ${} if set, else {:g}. "
            "Pick it slightly below the rate the sim sustains -- the [sim] line "
            "in the sim log reports that rate."
        ).format(CONTROL_HZ_ENV, control_hz()),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup = get_setup(args.robot, args.scene)
    spawn_pose = spawn_pose_from_args(args)
    environment = setup.scene.environment() if setup.scene.environment else None
    rate = args.control_hz if args.control_hz is not None else control_hz()
    print(
        f"{setup.name}, robot spawned at {spawn_pose}, QP at {rate:g} Hz "
        f"({PREDICTION_HORIZON} steps = {PREDICTION_HORIZON / rate:.1f} s lookahead)",
        flush=True,
    )

    rospy.init_node("giskard")
    # Needed by an interface config that syncs the map->odom tf frame; started
    # for the others too because RViz and the prop poses are stamped in `odom`
    # and would otherwise have no path to `map`.
    localization = start_localization_stand_in(setup.robot.base_link_height)
    try:
        giskard = Giskard(
            world_config=setup.robot.giskard_world(environment, spawn_pose),
            robot_interface_config=setup.robot.giskard_interface(),
            behavior_tree_config=ClosedLoopBTConfig(),
            # The rate the QP loop can actually sustain alongside the sim on
            # THIS machine -- see cram_vrb_lab.control.rate, which holds the
            # default and the reasoning, and which the sim reads too so its
            # slow-loop warning is about the same number. A nominal rate the loop
            # cannot keep makes the controller's internal dt wrong. (giskardpy
            # warns below 20 Hz -- harmless here.)
            qp_controller_config=QPControllerConfig(
                target_frequency=rate, prediction_horizon=PREDICTION_HORIZON
            ),
        )
        giskard.live()
    finally:
        localization.terminate()


if __name__ == "__main__":
    main()
