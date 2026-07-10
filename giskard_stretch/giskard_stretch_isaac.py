#!/usr/bin/env python
"""Giskard motion-control server for the Stretch robot in the Isaac Sim
apartment scene (examples/apartment.py).

Run with the cognitive_robot_abstract_machine venv python, with ROS jazzy and
the json_msgs workspace sourced (see README.md).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stretch_joints import (
    CMD_VEL_TOPIC,
    CONTROLLED_JOINTS,
    JOINT_STATES_TOPIC,
    ODOM_TOPIC,
    VELOCITY_CMD_TOPIC,
    load_patched_urdf,
)

from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.behavior_tree_config import ClosedLoopBTConfig
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.robot_interface_config import RobotInterfaceConfig
from giskardpy.middleware.ros2.scripts.iai_robots.stretch.configs import (
    WorldWithStretchConfigDiffDrive,
)
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.world_description.connections import DifferentialDrive


class StretchIsaacInterface(RobotInterfaceConfig):
    """Closed-loop interface against the Isaac Sim topics.

    Unlike StretchVelocityInterface there is no map->odom tf sync (the sim
    runs no SLAM; the localization joint stays identity, so map == odom) and
    joint velocities go to the integrating bridge (joint_velocity_bridge.py)
    instead of a real velocity controller, hence minimum_valid_velocity=0.
    """

    def setup(self):
        diff_drive = self.world.get_connections_by_type(DifferentialDrive)[0]
        self.sync_odometry_topic(ODOM_TOPIC, diff_drive)
        self.add_base_cmd_velocity(cmd_vel_topic=CMD_VEL_TOPIC, joint=diff_drive)
        self.sync_joint_state_topic(JOINT_STATES_TOPIC)
        self.add_joint_velocity_group_controller(
            cmd_topic=VELOCITY_CMD_TOPIC,
            connections=CONTROLLED_JOINTS,
            minimum_valid_velocity=0.0,
        )


def main():
    rospy.init_node("giskard")
    giskard = Giskard(
        world_config=WorldWithStretchConfigDiffDrive(urdf=load_patched_urdf()),
        robot_interface_config=StretchIsaacInterface(),
        behavior_tree_config=ClosedLoopBTConfig(),
        qp_controller_config=QPControllerConfig(
            target_frequency=20, prediction_horizon=15
        ),
    )
    giskard.live()


if __name__ == "__main__":
    main()
