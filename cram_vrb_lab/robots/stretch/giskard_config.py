"""Giskard robot-interface config for the Stretch against the Isaac Sim topics."""

from giskardpy.middleware.ros2.robot_interface_config import RobotInterfaceConfig
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    DifferentialDrive,
)

from .joints import (
    CMD_VEL_TOPIC,
    CONTROLLED_JOINTS,
    JOINT_STATES_TOPIC,
    ODOM_TOPIC,
    VELOCITY_CMD_TOPIC,
)


class StretchRealStyleInterface(RobotInterfaceConfig):
    """Real-robot-style closed-loop interface against the Isaac Sim topics.

    Mirrors how giskard talks to a physical Stretch: it consumes
    ``map -> odom`` (localization) from tf, wheel odometry from the ``/odom``
    topic, and joint states from the joint-state topic, and streams base and
    joint velocity commands back. The sim publishes ground-truth odometry and a
    static identity ``map -> odom`` stands in for SLAM (see
    :func:`cram_vrb_lab.control.giskard_server.start_localization_stand_in`),
    so on real hardware only those two sources change, not this interface.

    ``minimum_valid_velocity=0`` because the sim integrates streamed joint
    velocities into position targets rather than driving a real velocity
    controller (see ``StretchROS.integrate_joint_velocities`` in
    :mod:`cram_vrb_lab.robots.stretch.isaac_node`).
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
