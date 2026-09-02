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
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.robots.garmi import Garmi
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    OmniDrive,
)
from semantic_digital_twin.world_description.world_entity import Body

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

    def setup_world(self):
        """Upstream's own body, with the URDF parsed the way ``Garmi`` asks for.

        Copied from :meth:`WorldWithOmniDriveRobot.setup_world` for one word:
        ``use_visual_as_collision_backup``. GARMI's shell -- the two side covers,
        the front and rear covers, the lights, the rockers and the 2D lidars --
        is *drawn* by the description but never described for contact, and
        upstream declares that gap on the robot itself
        (``Garmi.uses_visual_as_collision_backup`` is ``True``) so its collision
        rules may name the covers that bound the base's real width.

        Upstream threads that flag into ``URDFParser`` from
        ``RobotSpecification.spawn`` and from its own test fixtures, but
        giskardpy's world configs still build the parser with its default
        ``False``. Left alone, every collision rule naming a cover raises
        ``BodyHasNoGeometryError`` at ``Executor.compile``, so *every* goal --
        the first ``NavigateAction`` included -- is aborted before it plans.

        Overridden here rather than fixed in giskardpy to keep the submodule
        pristine; delete this the moment ``WorldWithOmniDriveRobot`` reads the
        flag off its own ``urdf_view``.
        """
        map_body = Body(name=self.root_name)
        odom = Body(name=self.odom_body_name)
        self.localization = Connection6DoF.create_with_dofs(
            parent=map_body, child=odom, world=self.world
        )
        self.world.add_connection(self.localization)

        world_with_robot = URDFParser(
            urdf=self.urdf,
            prefix="",
            use_visual_as_collision_backup=(
                self.urdf_view.uses_visual_as_collision_backup
            ),
        ).parse()
        self.robot = self.urdf_view.from_world(world_with_robot)

        drive = OmniDrive.create_with_dofs(
            parent=odom,
            child=world_with_robot.root,
            translation_velocity_limits=0.2,
            rotation_velocity_limits=0.2,
            world=self.world,
        )
        self.world.merge_world(world_with_robot, drive)


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
