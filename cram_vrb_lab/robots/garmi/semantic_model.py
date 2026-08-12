"""Semantic (digital-twin) model of GARMI: two FR3 arms on a standing base.

Same shape of description as :mod:`cram_vrb_lab.robots.panda.semantic_model`,
doubled: the parts are discovered by *type*, so each of the two hands needs its
own pair of finger classes and each arm its own gripper class, even though the
hardware is identical. The bodies they resolve to are the only difference, so
all four fingers, both grippers and both arms share a base that reads the link
names off class attributes.

Left and right are not asserted anywhere: ``HasLeftRightArm`` decides which arm
is which from where its first body sits relative to the robot root, and GARMI's
shoulders are mounted symmetrically about the base's x-axis.

The gripper's :attr:`front_facing_orientation` is the same 90 deg rotation about
y as the Panda's, and for the same reason: CRAM's grasp maths assumes a tool
frame whose x-axis is the approach axis, while a Franka Hand points its z-axis
out between the fingers.
"""

from __future__ import annotations

import math
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
from typing import ClassVar, List, Union

from typing_extensions import Self

from semantic_digital_twin.collision_checking.collision_rules import (
    AvoidExternalCollisions,
)
from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_part_mixins import HasLeftRightArm, HasTwoFingers
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
    FINGER_JOINTS,
    GRIPPER_OPEN_TRAVEL,
    PARK_CONFIGURATION,
    ROBOT_ROOT_LINK,
    arm_joints,
    arm_root_link,
    finger_joints,
    finger_link,
    fingertip_link,
    hand_link,
    tool_frame_link,
)

_FRONT_FACING = Quaternion(0.0, math.sqrt(2) / 2, 0.0, math.sqrt(2) / 2)


@dataclass(eq=False)
class _GarmiFinger(Finger, ABC):
    SIDE: ClassVar[str]
    FINGER: ClassVar[str]

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
            root=world.get_body_in_branch_by_name(
                robot_root, finger_link(cls.SIDE, cls.FINGER)
            ),
            tip=world.get_body_in_branch_by_name(
                robot_root, fingertip_link(cls.SIDE, cls.FINGER)
            ),
        )


@dataclass(eq=False)
class LeftHandLeftFinger(_GarmiFinger):
    SIDE, FINGER = "left", "left"


@dataclass(eq=False)
class LeftHandRightFinger(_GarmiFinger):
    SIDE, FINGER = "left", "right"


@dataclass(eq=False)
class RightHandLeftFinger(_GarmiFinger):
    SIDE, FINGER = "right", "left"


@dataclass(eq=False)
class RightHandRightFinger(_GarmiFinger):
    SIDE, FINGER = "right", "right"


@dataclass(eq=False)
class _GarmiGripper(EndEffector, ABC):
    SIDE: ClassVar[str]

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
        connections = [
            self._world.get_connection_by_name(joint)
            for joint in finger_joints(self.SIDE)
        ]
        return [
            JointState.from_mapping(
                name=PrefixedName("gripper_open", prefix=self.name.name),
                mapping=dict(zip(connections, [GRIPPER_OPEN_TRAVEL] * 2)),
                state_type=GripperState.OPEN,
            ),
            JointState.from_mapping(
                name=PrefixedName("gripper_close", prefix=self.name.name),
                mapping=dict(zip(connections, [0.0, 0.0])),
                state_type=GripperState.CLOSE,
            ),
        ]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        return cls(
            root=world.get_body_in_branch_by_name(robot_root, hand_link(cls.SIDE)),
            tool_frame=world.get_body_in_branch_by_name(
                robot_root, tool_frame_link(cls.SIDE)
            ),
            front_facing_orientation=_FRONT_FACING,
        )


@dataclass(eq=False)
class GarmiLeftGripper(
    _GarmiGripper, HasTwoFingers[LeftHandLeftFinger, LeftHandRightFinger]
):
    SIDE = "left"


@dataclass(eq=False)
class GarmiRightGripper(
    _GarmiGripper, HasTwoFingers[RightHandLeftFinger, RightHandRightFinger]
):
    SIDE = "right"


@dataclass(eq=False)
class _GarmiArm(Arm, ABC):
    SIDE: ClassVar[str]

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
        connections = [
            self._world.get_connection_by_name(joint) for joint in arm_joints(self.SIDE)
        ]
        return [
            JointState.from_mapping(
                name=PrefixedName("arm_park", prefix=self.name.name),
                mapping=dict(zip(connections, PARK_CONFIGURATION)),
                state_type=StaticJointState.PARK,
            )
        ]

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        world = robot_root._world
        return cls(
            root=world.get_body_in_branch_by_name(robot_root, arm_root_link(cls.SIDE)),
            tip=world.get_body_in_branch_by_name(robot_root, hand_link(cls.SIDE)),
        )


@dataclass(eq=False)
class GarmiLeftArm(_GarmiArm, Arm[GarmiLeftGripper]):
    SIDE = "left"


@dataclass(eq=False)
class GarmiRightArm(_GarmiArm, Arm[GarmiRightGripper]):
    SIDE = "right"


@dataclass(eq=False)
class Garmi(AbstractRobot, HasLeftRightArm[GarmiLeftArm, GarmiRightArm]):
    """GARMI with its base frozen: two 7-DoF FR3 arms and two Franka Hands.

    ..note::
       No self-collision rules -- the description ships no SRDF, and every pair
       would have to be enumerated by hand. External collision avoidance (the
       flat, its furniture) is what the demos need and is configured below.
    """

    @classmethod
    def get_ros_file_path(cls) -> str:
        return "package://garmi_description/urdf/garmi.urdf"

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
            # 0.04 m of total stroke; at the arm's rate a finger would slam shut
            # inside a single control period.
            velocity_limits[self._world.get_connection_by_name(finger_joint)] = 0.05
        self.tighten_dof_velocity_limits_of_1dof_connections(new_limits=velocity_limits)
