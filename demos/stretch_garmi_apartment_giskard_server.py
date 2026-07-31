#!/usr/bin/env python
"""Giskard motion-control server for the Stretch robot in the Isaac Sim
garmi-apartment scene (stretch_garmi_apartment_sim.py).

Identical to :mod:`stretch_apartment_giskard_server` except for the world config:
the environment merged next to the robot is the garmi-apartment MJCF instead of
``apartment.urdf``. The robot interface is unchanged, i.e. the *real-robot*
interface shape -- giskard reads the robot's joint states and wheel odometry over
ROS, reads a ``map -> odom`` localization transform from tf, and streams
base/joint velocity commands back. The only sim-specific choice is where
``map -> odom`` comes from: a real robot runs AMCL/SLAM, here a static identity
transform stands in for it (see :func:`main`).

Run with the cognitive_robot_abstract_machine venv python, with ROS jazzy and
the ros2_ws overlay sourced (see README.md).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cram_vrb_lab.control.giskard_server import start_localization_stand_in
from cram_vrb_lab.robots.stretch.giskard_config import StretchRealStyleInterface
from cram_vrb_lab.robots.stretch.joints import load_patched_urdf
from cram_vrb_lab.scenes.garmi_apartment.giskard_world import (
    WorldWithStretchAndGarmiApartmentDiffDrive,
)

from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.behavior_tree_config import ClosedLoopBTConfig
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.qp.qp_controller_config import QPControllerConfig


def main():
    rospy.init_node("giskard")
    localization = start_localization_stand_in()
    try:
        giskard = Giskard(
            world_config=WorldWithStretchAndGarmiApartmentDiffDrive(
                urdf=load_patched_urdf()
            ),
            robot_interface_config=StretchRealStyleInterface(),
            behavior_tree_config=ClosedLoopBTConfig(),
            # 15 Hz: the highest rate the QP loop actually sustains on this
            # machine while the sim runs alongside; a nominal rate the loop can't
            # keep makes the controller's internal dt wrong. (giskardpy warns
            # below 20 Hz -- harmless here.)
            qp_controller_config=QPControllerConfig(
                target_frequency=15, prediction_horizon=15
            ),
        )
        giskard.live()
    finally:
        localization.terminate()


if __name__ == "__main__":
    main()
