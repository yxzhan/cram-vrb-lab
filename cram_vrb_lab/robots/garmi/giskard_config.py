"""Giskard world and robot-interface config for GARMI against Isaac Sim.

The same shape as the Panda's, and for the same reason: with the base frozen
(see :data:`cram_vrb_lab.robots.garmi.joints.FROZEN_JOINTS`) GARMI does not
move, so there is no odometry to sync and no ``map -> odom`` localization. The
robot root is bolted to ``map`` at the demo's spawn pose, and the only channels
are joint states in and streamed joint velocities out.

Whatever scene the demo runs it in is merged next to the robot by
:func:`cram_vrb_lab.control.giskard_world.build_world_config`.
"""

from dataclasses import dataclass, field

from giskardpy.middleware.ros2.robot_interface_config import RobotInterfaceConfig
from giskardpy.model.world_config import WorldWithFixedRobot
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body

from .joints import CONTROLLED_JOINTS, JOINT_STATES_TOPIC, VELOCITY_CMD_TOPIC
from .semantic_model import Garmi


@dataclass
class WorldWithGarmiConfig(WorldWithFixedRobot):
    """GARMI fixed to ``map`` at the pose the sim placed the prim with."""

    urdf_view: AbstractRobot = field(kw_only=True, default=Garmi, init=False)

    robot_pose: HomogeneousTransformationMatrix = field(
        kw_only=True, default_factory=HomogeneousTransformationMatrix
    )
    """Where ``base_link`` sits in ``map``, i.e. the demo's
    :class:`~cram_vrb_lab.specs.SpawnPose`. The stock
    :class:`~giskardpy.model.world_config.WorldWithFixedRobot` bolts the robot to
    the origin with no way to say otherwise, which would leave giskard planning
    for a robot metres away from the rendered one."""

    def setup_world(self) -> None:
        """Build ``map`` and hang the posed robot off it.

        Runs inside the ``modify_world`` context
        :class:`giskardpy.middleware.ros2.giskard.Giskard` already opens around
        ``setup_world``.
        """
        world_root = Body(name=self.root_name)
        # Added explicitly, as the base class does: anything merging onto ``map``
        # after this method -- the scene -- looks the root up inside the same
        # modify_world context.
        self.world.add_body(world_root)

        robot_world = URDFParser(urdf=self.urdf, prefix="").parse()
        self.urdf_view.from_world(robot_world)
        self.robot_root = robot_world.root
        self.world.merge_world(
            robot_world,
            FixedConnection(
                parent=world_root,
                child=self.robot_root,
                parent_T_connection_expression=self.robot_pose,
            ),
        )


class GarmiSimInterface(RobotInterfaceConfig):
    """Closed-loop interface against the Isaac Sim GARMI topics.

    ``minimum_valid_velocity=0`` because the sim integrates streamed joint
    velocities into position targets rather than driving a real velocity
    controller, so small commanded velocities are executed faithfully and need
    no deadband.
    """

    def setup(self):
        self.sync_joint_state_topic(JOINT_STATES_TOPIC)
        self.add_joint_velocity_group_controller(
            cmd_topic=VELOCITY_CMD_TOPIC,
            connections=CONTROLLED_JOINTS,
            minimum_valid_velocity=0.0,
        )
