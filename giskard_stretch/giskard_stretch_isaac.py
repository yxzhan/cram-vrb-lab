#!/usr/bin/env python
"""Giskard motion-control server for the Stretch robot in the Isaac Sim
apartment scene (apartment.py).

Uses the *real-robot* interface shape: giskard reads the robot's joint states
and wheel odometry over ROS, reads a ``map -> odom`` localization transform from
tf, and streams base/joint velocity commands back. The only sim-specific choice
is where ``map -> odom`` comes from: a real robot runs AMCL/SLAM, here a static
identity transform stands in for it (see :func:`main`).

Run with the cognitive_robot_abstract_machine venv python, with ROS jazzy and
the ros2_ws overlay sourced (see README.md).
"""

import os
import subprocess
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

from apartment_world_config import WorldWithStretchAndApartmentDiffDrive

from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.behavior_tree_config import ClosedLoopBTConfig
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.robot_interface_config import RobotInterfaceConfig
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    DifferentialDrive,
)


class StretchRealStyleInterface(RobotInterfaceConfig):
    """Real-robot-style closed-loop interface against the Isaac Sim topics.

    Mirrors how giskard talks to a physical Stretch: it consumes
    ``map -> odom`` (localization) from tf, wheel odometry from the ``/odom``
    topic, and joint states from the joint-state topic, and streams base and
    joint velocity commands back. The sim publishes ground-truth odometry and a
    static identity ``map -> odom`` stands in for SLAM (see :func:`main`), so on
    real hardware only those two sources change, not this interface.

    ``minimum_valid_velocity=0`` because the sim integrates streamed joint
    velocities into position targets rather than driving a real velocity
    controller (see ``apartment.py`` ``integrate_joint_velocities``).
    """

    def setup(self):
        self.sync_6dof_joint_with_tf_frame(
            joint=self.world.get_connections_by_type(Connection6DoF)[0],
            tf_parent_frame="map",
            tf_child_frame="odom",
        )
        diff_drive = self.world.get_connections_by_type(DifferentialDrive)[0]
        self.sync_odometry_topic(ODOM_TOPIC, diff_drive)
        self.add_base_cmd_velocity(cmd_vel_topic=CMD_VEL_TOPIC, joint=diff_drive)
        self.sync_joint_state_topic(JOINT_STATES_TOPIC)
        self.add_joint_velocity_group_controller(
            cmd_topic=VELOCITY_CMD_TOPIC,
            connections=CONTROLLED_JOINTS,
            minimum_valid_velocity=0.0,
        )


def start_localization_stand_in() -> subprocess.Popen:
    """Publish a static identity ``map -> odom`` as a localization stand-in.

    On a real robot AMCL/SLAM owns ``map -> odom``; the sim runs none, so the
    robot boots at the map origin with no odometry drift and the transform is
    identity. Giskard's :meth:`sync_6dof_joint_with_tf_frame` blocks until this
    transform is available, so it must run alongside the server.
    """
    return subprocess.Popen(
        [
            "ros2", "run", "tf2_ros", "static_transform_publisher",
            "--x", "0", "--y", "0", "--z", "0",
            "--roll", "0", "--pitch", "0", "--yaw", "0",
            "--frame-id", "map", "--child-frame-id", "odom",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main():
    rospy.init_node("giskard")
    localization = start_localization_stand_in()
    try:
        giskard = Giskard(
            world_config=WorldWithStretchAndApartmentDiffDrive(
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
