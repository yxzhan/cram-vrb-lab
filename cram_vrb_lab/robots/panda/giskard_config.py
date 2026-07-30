"""Giskard world and robot-interface config for the Panda against Isaac Sim.

The interface is much simpler than the Stretch's because the Panda does not
move: no odometry to sync, no ``map -> odom`` localization, no base velocity
command. The robot root is bolted to ``map``, and the only channels are joint
states in and streamed joint velocities out.

The world is not simpler, though: the arm is mounted inside the apartment, so
giskard has to know about the apartment to avoid it.
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

from cram_vrb_lab.scenes.apartment.constants import (
    apartment_pose_in_map,
    panda_pose_in_map,
)
from cram_vrb_lab.scenes.apartment.giskard_world import load_apartment_urdf

from .joints import CONTROLLED_JOINTS, JOINT_STATES_TOPIC, VELOCITY_CMD_TOPIC
from .semantic_model import Panda


@dataclass
class WorldWithPandaConfig(WorldWithFixedRobot):
    """The Panda fixed to ``map`` at its mounting pose, with the apartment
    merged in beside it.

    The props the demo manipulates are *not* here: the notebook adds them at
    runtime over ``/world_sync``
    (:func:`cram_vrb_lab.scenes.props.twin_props.add_props_to_twin`), so the
    scene giskard avoids collisions in is built from the same numbers Isaac
    renders from, with no second description of them to keep aligned.
    """

    urdf_view: AbstractRobot = field(kw_only=True, default=Panda, init=False)

    robot_pose: HomogeneousTransformationMatrix = field(
        kw_only=True, default_factory=panda_pose_in_map
    )
    """Where the robot root sits in ``map``. The stock
    :class:`~giskardpy.model.world_config.WorldWithFixedRobot` bolts it to the
    origin, which would put giskard's arm metres away from the rendered one."""

    apartment_urdf: str = field(kw_only=True, default_factory=load_apartment_urdf)
    """URDF string of the environment merged next to the robot."""

    apartment_pose: HomogeneousTransformationMatrix = field(
        kw_only=True, default_factory=apartment_pose_in_map
    )
    """Apartment root pose in ``map``; see :func:`apartment_pose_in_map`."""

    def setup_world(self) -> None:
        """Build ``map``, hang the posed robot off it, then merge the apartment.

        Runs inside the ``modify_world`` context that
        :class:`giskardpy.middleware.ros2.giskard.Giskard` already opens around
        ``setup_world`` (matching the base class, which likewise assumes it).
        """
        world_root = Body(name=self.root_name)

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

        apartment_world = URDFParser(urdf=self.apartment_urdf, prefix="").parse()
        self.world.merge_world(
            apartment_world,
            FixedConnection(
                parent=world_root,
                child=apartment_world.root,
                parent_T_connection_expression=self.apartment_pose,
            ),
        )


class PandaSimInterface(RobotInterfaceConfig):
    """Closed-loop interface against the Isaac Sim Panda topics.

    ``minimum_valid_velocity=0`` because the sim integrates streamed joint
    velocities into position targets rather than driving a real velocity
    controller (see ``PandaROS.integrate_joint_velocities`` in
    :mod:`cram_vrb_lab.robots.panda.isaac_node`), so even very small commanded
    velocities are executed faithfully and do not need a deadband.
    """

    def setup(self):
        self.sync_joint_state_topic(JOINT_STATES_TOPIC)
        self.add_joint_velocity_group_controller(
            cmd_topic=VELOCITY_CMD_TOPIC,
            connections=CONTROLLED_JOINTS,
            minimum_valid_velocity=0.0,
        )
