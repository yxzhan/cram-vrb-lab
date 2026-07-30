#!/usr/bin/env python
"""Giskard motion-control server for the Panda mounted in the Isaac Sim apartment
(panda_pick_place_sim.py).

Same closed-loop shape as the Stretch server, minus everything the mobile base
brought with it: the robot root is fixed to ``map`` at its mounting pose, so there
is no localization transform to wait for and no odometry to sync. Giskard reads
``/panda/joint_states`` and streams joint velocities back.

``WorldWithPandaConfig`` merges the apartment in beside the arm, so collision
avoidance covers the room. The cube is not part of that world config: the notebook
adds it over ``/world_sync`` once it is connected, which keeps the scene giskard
avoids collisions in identical to the one Isaac renders.

Run with the cognitive_robot_abstract_machine venv python, with ROS jazzy and the
ros2_ws overlay sourced (see README.md).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cram_vrb_lab.control.giskard_server import start_localization_stand_in
from cram_vrb_lab.robots.panda.giskard_config import (
    PandaSimInterface,
    WorldWithPandaConfig,
)
from cram_vrb_lab.robots.panda.joints import load_patched_urdf

from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.behavior_tree_config import ClosedLoopBTConfig
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.qp.qp_controller_config import QPControllerConfig


def main():
    rospy.init_node("giskard")
    # Not needed by the interface config, which syncs no tf frame, but RViz and
    # the prop poses are stamped in `odom` and would otherwise have no path to
    # `map`.
    localization = start_localization_stand_in()
    try:
        giskard = Giskard(
            world_config=WorldWithPandaConfig(urdf=load_patched_urdf()),
            robot_interface_config=PandaSimInterface(),
            behavior_tree_config=ClosedLoopBTConfig(),
            # 15 Hz: the rate the QP loop actually sustains on this machine with
            # the sim running alongside. A nominal rate the loop cannot keep
            # makes the controller's internal dt wrong. (giskardpy warns below
            # 20 Hz -- harmless here.)
            qp_controller_config=QPControllerConfig(
                target_frequency=15, prediction_horizon=15
            ),
        )
        giskard.live()
    finally:
        localization.terminate()


if __name__ == "__main__":
    main()
