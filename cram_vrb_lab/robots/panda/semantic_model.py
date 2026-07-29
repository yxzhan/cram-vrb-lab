"""Semantic (digital-twin) model of the Franka Emika Panda.

``semantic_digital_twin`` ships models for the robots the CRAM group runs; the
Panda is not one of them, so this module supplies the same shape of description
locally: which bodies form the arm and the hand, what "parked" and "open" mean
in joint values, and how the tool frame is oriented. CRAM's action designators
and giskard's world config both go through this.

The gripper's :attr:`front_facing_orientation` is the piece worth understanding.
CRAM's grasp maths (:class:`coraplex.datastructures.grasp.GraspDescription`)
assumes a tool frame whose **x-axis is the approach axis**; the Panda's hand
frame points its **z-axis** out between the fingers. The quaternion below is the
90 deg rotation about y that reconciles the two, so the standard approach
directions mean what they say:

- ``FRONT`` + ``NoAlignment`` -- come at the object horizontally, along map +x
- ``FRONT`` + ``TOP`` -- come straight down, fingers closing along map y

The second is the natural grasp for something standing on a table, and it is
what :mod:`demos.panda_pick_place_cram` uses.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing_extensions import List, Self

from semantic_digital_twin.collision_checking.collision_rules import (
    AvoidExternalCollisions,
)
from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_part_mixins import HasOneArm, HasTwoFingers
from semantic_digital_twin.robots.robot_parts import (
    AbstractRobot,
    Arm,
    EndEffector,
    Finger,
)
from semantic_digital_twin.spatial_types import Quaternion
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

from .joints import (
    ARM_JOINTS,
    FINGER_JOINTS,
    GRIPPER_OPEN_TRAVEL,
    HAND_LINK,
    LEFT_FINGER_LINK,
    LEFT_FINGERTIP_LINK,
    PARK_CONFIGURATION,
    RIGHT_FINGER_LINK,
    RIGHT_FINGERTIP_LINK,
    ROBOT_ROOT_LINK,
    TOOL_FRAME_LINK,
)


@dataclass(eq=False)
class PandaLeftFinger(Finger):
    """The finger driven by ``panda_finger_joint1``."""

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        return cls(
            root=world.get_body_in_branch_by_name(robot_root, LEFT_FINGER_LINK),
            tip=world.get_body_in_branch_by_name(robot_root, LEFT_FINGERTIP_LINK),
        )


@dataclass(eq=False)
class PandaRightFinger(Finger):
    """The finger driven by ``panda_finger_joint2``."""

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List[JointState]:
        return []

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        return cls(
            root=world.get_body_in_branch_by_name(robot_root, RIGHT_FINGER_LINK),
            tip=world.get_body_in_branch_by_name(robot_root, RIGHT_FINGERTIP_LINK),
        )


@dataclass(eq=False)
class PandaGripper(EndEffector, HasTwoFingers[PandaLeftFinger, PandaRightFinger]):
    """The Franka Hand: two prismatic fingers sliding towards each other."""

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
        finger_connections = [
            self._world.get_connection_by_name(joint) for joint in FINGER_JOINTS
        ]
        gripper_open = JointState.from_mapping(
            name=PrefixedName("gripper_open", prefix=self.name.name),
            mapping=dict(zip(finger_connections, [GRIPPER_OPEN_TRAVEL] * 2)),
            state_type=GripperState.OPEN,
        )
        gripper_close = JointState.from_mapping(
            name=PrefixedName("gripper_close", prefix=self.name.name),
            mapping=dict(zip(finger_connections, [0.0, 0.0])),
            state_type=GripperState.CLOSE,
        )
        return [gripper_open, gripper_close]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        return cls(
            root=world.get_body_in_branch_by_name(robot_root, HAND_LINK),
            tool_frame=world.get_body_in_branch_by_name(robot_root, TOOL_FRAME_LINK),
            # 90 deg about y: turns the hand's approach axis (+z) into the +x
            # axis CRAM's grasp maths expects. See the module docstring.
            front_facing_orientation=Quaternion(
                0.0, math.sqrt(2) / 2, 0.0, math.sqrt(2) / 2
            ),
        )


@dataclass(eq=False)
class PandaArm(Arm[PandaGripper]):
    """The seven revolute joints from the base to the wrist flange."""

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
        arm_connections = [
            self._world.get_connection_by_name(joint) for joint in ARM_JOINTS
        ]
        arm_park = JointState.from_mapping(
            name=PrefixedName("arm_park", prefix=self.name.name),
            mapping=dict(zip(arm_connections, PARK_CONFIGURATION)),
            state_type=StaticJointState.PARK,
        )
        return [arm_park]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        return cls(
            root=world.get_body_in_branch_by_name(robot_root, ROBOT_ROOT_LINK),
            tip=world.get_body_in_branch_by_name(robot_root, HAND_LINK),
        )


@dataclass(eq=False)
class Panda(AbstractRobot, HasOneArm[PandaArm]):
    """The Franka Emika Panda, bolted down: a 7-DoF arm and a two-finger hand.

    ..note::
       No self-collision rules are set up. The stock URDF ships no SRDF, and
       every self-collision pair would otherwise have to be enumerated by hand;
       external collision avoidance (the table, the props) is what this scene
       needs and is configured below.
    """

    @classmethod
    def get_ros_file_path(cls) -> str:
        return "package://franka_description/robots/panda_arm_hand.urdf"

    @classmethod
    def _get_root_body_name(cls) -> str:
        return ROBOT_ROOT_LINK

    def _setup_collision_rules(self):
        self._world.collision_manager.add_default_rule(
            AvoidExternalCollisions(
                buffer_zone_distance=0.05, violated_distance=0.0, robot=self
            )
        )

    def _setup_velocity_limits(self):
        velocity_limits = defaultdict(lambda: 0.4)
        for finger_joint in FINGER_JOINTS:
            # The fingers only ever travel MAX_FINGER_TRAVEL in total; at the
            # arm's rate they would slam shut inside a single control period.
            velocity_limits[self._world.get_connection_by_name(finger_joint)] = 0.05
        self.tighten_dof_velocity_limits_of_1dof_connections(new_limits=velocity_limits)
