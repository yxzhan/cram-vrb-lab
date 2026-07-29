"""Giskard world and robot-interface config for the Panda against Isaac Sim.

Much simpler than the Stretch's equivalent because the Panda does not move: no
odometry to sync, no ``map -> odom`` localization, no base velocity command. The
robot root is bolted to ``map``, and the only channels are joint states in and
streamed joint velocities out.
"""

from dataclasses import dataclass, field

from giskardpy.middleware.ros2.robot_interface_config import RobotInterfaceConfig
from giskardpy.model.world_config import WorldWithFixedRobot
from semantic_digital_twin.robots.robot_parts import AbstractRobot

from .joints import CONTROLLED_JOINTS, JOINT_STATES_TOPIC, VELOCITY_CMD_TOPIC
from .semantic_model import Panda


@dataclass
class WorldWithPandaConfig(WorldWithFixedRobot):
    """The Panda alone in an otherwise empty giskard world, its base fixed to
    ``map``.

    Nothing is merged in beside it: the props the demo manipulates are added at
    runtime over ``/world_sync`` by
    :func:`cram_vrb_lab.scenes.props.twin_props.add_props_to_twin`, which keeps
    the scene the robot plans in identical to the one Isaac renders without a
    second environment description to keep aligned.
    """

    urdf_view: AbstractRobot = field(kw_only=True, default=Panda, init=False)


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
