"""Giskard world and robot-interface config for GARMI against Isaac Sim.

GARMI drives, so both halves are the mobile-robot shape rather than the Panda's:
the world hangs the robot off ``map -> odom`` through an ``OmniDrive`` instead of
bolting it down, and the interface learns where the robot is from odometry and
localization the way it would on real hardware.

Neither class does much. giskardpy already ships
:class:`~giskardpy.model.world_config.WorldWithOmniDriveRobot`, and
:class:`~giskardpy.middleware.ros2.scripts.iai_robots.hsr.configs.HSRVelocityInterface`
is the same interface against a different robot's topics.

Whatever scene the demo runs in is merged next to the robot by
:func:`cram_vrb_lab.control.giskard_world.build_world_config`.
"""

from dataclasses import dataclass, field

from giskardpy.middleware.ros2.robot_interface_config import RobotInterfaceConfig
from giskardpy.model.world_config import WorldWithOmniDriveRobot
from semantic_digital_twin.robots.garmi import Garmi
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    OmniDrive,
)

from .joints import (
    CMD_VEL_TOPIC,
    CONTROLLED_JOINTS,
    JOINT_STATES_TOPIC,
    ODOM_TOPIC,
    VELOCITY_CMD_TOPIC,
)


@dataclass
class WorldWithGarmiConfig(WorldWithOmniDriveRobot):
    """GARMI on ``map -> odom -> base_link``, the latter an ``OmniDrive``.

    No spawn pose: unlike the Panda's fixed-robot config, nothing here says where
    the robot stands. It hangs off ``odom``, giskard reads ``map -> odom`` from tf
    and the base pose from :data:`~cram_vrb_lab.robots.garmi.joints.ODOM_TOPIC`,
    and the sim publishes odometry from wherever it spawned the robot -- so the
    pose arrives over the wire exactly as a real robot's localization would
    deliver it.
    """

    urdf_view: AbstractRobot = field(kw_only=True, default=Garmi, init=False)


class GarmiSimInterface(RobotInterfaceConfig):
    """Real-robot-style closed-loop interface against the Isaac Sim GARMI topics.

    Four channels: ``map -> odom`` from tf (a static identity stands in for SLAM,
    see :func:`cram_vrb_lab.control.giskard_server.start_localization_stand_in`),
    ground-truth wheel odometry, joint states in, and streamed velocities out --
    a Twist for the base and a Float64MultiArray for everything else.

    ``minimum_valid_velocity=0`` because the sim integrates streamed joint
    velocities into position targets rather than driving a real velocity
    controller, so small commanded velocities are executed faithfully and need
    no deadband.
    """

    def setup(self):
        # [0] picks by position, not by name, and the garmi-apartment MJCF
        # contributes 15 Connection6DoF of its own for its free bodies. It is the
        # right one only because the robot's world is built before the scene is
        # merged onto it -- see cram_vrb_lab.control.giskard_world.
        self.sync_6dof_joint_with_tf_frame(
            joint=self.world.get_connections_by_type(Connection6DoF)[0],
            tf_parent_frame="map",
            tf_child_frame="odom",
        )
        omni_drive = self.world.get_connections_by_type(OmniDrive)[0]
        self.sync_odometry_topic(ODOM_TOPIC, omni_drive)
        self.add_base_cmd_velocity(cmd_vel_topic=CMD_VEL_TOPIC, joint=omni_drive)

        self.sync_joint_state_topic(JOINT_STATES_TOPIC)
        self.add_joint_velocity_group_controller(
            cmd_topic=VELOCITY_CMD_TOPIC,
            connections=CONTROLLED_JOINTS,
            minimum_valid_velocity=0.0,
        )
