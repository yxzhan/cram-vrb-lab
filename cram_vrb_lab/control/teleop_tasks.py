"""Giskard tasks whose goal moves while they run, for teleoperation.

A giskard goal is built and parsed once -- ~5 s on the GARMI apartment -- so a
teleoperated arm cannot be one goal per target. These tasks are the other way round:
one long-running goal whose targets live in giskard's float variables, and each
control cycle's ``on_tick`` writes the newest target from a topic into them before the
QP is solved (``Executor.tick`` ticks the statechart, then solves with
``float_variable_data``). ``CartesianPositionTrajectory`` and ``WiggleInsert`` move
their goals the same way.

:class:`FollowFrame` and :class:`TeleopGripper` read their targets off the world -- a
body, a joint -- and :class:`TeleopCartesianPose` off a topic. All start by holding
where the robot is, so a goal sent before any target has moved keeps the arm and the
hand still rather than driving them anywhere.

Imported by the giskard server when it deserializes the goal (by module path), so this
module must stay importable there: plain giskardpy and semantic_digital_twin, nothing
from the Isaac side.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from geometry_msgs.msg import PoseStamped

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.graph_node import NodeArtifacts, Task
from giskardpy.motion_statechart.ros_context import RosContextExtension
from giskardpy.motion_statechart.tasks.cartesian_tasks import HoldPose
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

TELEOP_TOPIC_PREFIX = "/teleop"


def pose_topic(arm: str) -> str:
    """``geometry_msgs/PoseStamped``: where ``arm``'s tool frame should be. Any frame
    the world knows as ``header.frame_id``; ``map`` is the natural one."""
    return f"{TELEOP_TOPIC_PREFIX}/{arm}/pose"


class _Latest:
    """The newest message of one subscription, handed from the ROS thread to the tick."""

    def __init__(self):
        self._lock = threading.Lock()
        self._message = None

    def put(self, message) -> None:
        with self._lock:
            self._message = message

    def take(self):
        """The message received since the last take, or None."""
        with self._lock:
            message, self._message = self._message, None
        return message


@dataclass(eq=False, repr=False)
class TeleopCartesianPose(HoldPose):
    """Keeps ``tip_link`` at the newest pose published on :attr:`topic_name`.

    :class:`HoldPose` with the held pose rewritten from the topic: HoldPose already
    constrains the chain to a pose kept in float variables, and binds it to where the
    tip is on start -- which is the hold this wants until the first target arrives.

    Never converges and never ends; the goal runs until it is cancelled.
    """

    topic_name: str = field(kw_only=True)
    """See :func:`pose_topic`."""

    weight: float = field(
        default=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE, kw_only=True
    )
    """Below collision avoidance, unlike HoldPose's default: a target inside the table
    is followed only as far as the clearance allows."""

    _latest: Optional[_Latest] = field(default=None, init=False, repr=False)
    _subscription: object = field(default=None, init=False, repr=False)

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        artifacts = super().build(context)
        self._latest = _Latest()
        node = context.require_extension(RosContextExtension).ros_node
        self._subscription = node.create_subscription(
            PoseStamped, self.topic_name, self._latest.put, 1
        )
        return artifacts

    def on_tick(self, context: MotionStatechartContext):
        message = self._latest.take()
        if message is None:
            return None
        world = context.world
        try:
            frame = world.get_kinematic_structure_entity_by_name(message.header.frame_id)
        except Exception:
            return None
        position, orientation = message.pose.position, message.pose.orientation
        frame_T_goal = _matrix(
            (position.x, position.y, position.z),
            (orientation.x, orientation.y, orientation.z, orientation.w),
        )
        root_T_goal = world.compute_forward_kinematics_np(self.root_link, frame) @ frame_T_goal
        # The layout ForwardKinematicsBinding.bind writes: the top 3x4, column-major.
        context.float_variable_data.set_value(
            self._pose_to_keep.root_T_tip, root_T_goal[:3, :4].T.flatten()
        )
        return None

    def cleanup(self, context: MotionStatechartContext):
        if self._subscription is not None:
            context.require_extension(RosContextExtension).ros_node.destroy_subscription(
                self._subscription
            )
            self._subscription = None
        super().cleanup(context)


@dataclass(eq=False, repr=False)
class FollowFrame(HoldPose):
    """Keeps ``tip_link`` on :attr:`target_frame`, wherever that frame is moved.

    :class:`HoldPose` with the held pose re-read every control cycle from giskard's own
    world, in which :attr:`target_frame` is moved by whoever moves it in the twin: the
    change reaches giskard over ``/world_sync``, and the control loop applies the
    twin's state updates at the top of every cycle, before this tick reads them.

    The target is a body in the world rather than a topic, so anything that can move a
    body in the twin drives the arm -- a drag in the viewer, a VR controller, a script.
    Never converges and never ends; the goal runs until it is cancelled.
    """

    target_frame: KinematicStructureEntity = field(kw_only=True)
    """The frame :attr:`tip_link` is kept on. Must exist before the goal is sent: a body
    added while a goal runs is a model change, which aborts it."""

    weight: float = field(
        default=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE, kw_only=True
    )
    """Below collision avoidance, unlike HoldPose's default: a target inside the table
    is followed only as far as the clearance allows."""

    translation_deadband: float = field(default=0.01, kw_only=True)
    """[m] the target has to move from where the hand was last sent before the hand
    follows. A hand on a controller or a mouse never holds perfectly still, and without
    this every tremor of the marker is a motion of the arm."""

    rotation_deadband: float = field(default=0.05, kw_only=True)
    """[rad] the same, for turning -- about 3 degrees."""

    _sent: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    """The target last written into the goal, root_T_target."""

    def on_start(self, context: MotionStatechartContext):
        self._sent = None
        self._follow(context)

    def on_tick(self, context: MotionStatechartContext):
        self._follow(context)
        return None

    def _follow(self, context: MotionStatechartContext) -> None:
        root_T_target = context.world.compute_forward_kinematics_np(
            self.root_link, self.target_frame
        )
        if self._sent is not None and not self._moved(self._sent, root_T_target):
            return
        self._sent = np.array(root_T_target)
        # The layout ForwardKinematicsBinding.bind writes: the top 3x4, column-major.
        context.float_variable_data.set_value(
            self._pose_to_keep.root_T_tip, root_T_target[:3, :4].T.flatten()
        )

    def _moved(self, before: np.ndarray, after: np.ndarray) -> bool:
        """Whether ``after`` is out of the deadband around ``before``."""
        if np.linalg.norm(after[:3, 3] - before[:3, 3]) > self.translation_deadband:
            return True
        cosine = (np.trace(before[:3, :3].T @ after[:3, :3]) - 1.0) / 2.0
        return float(np.arccos(np.clip(cosine, -1.0, 1.0))) > self.rotation_deadband


@dataclass(eq=False, repr=False)
class TeleopGripper(Task):
    """Drives a hand's finger joints to the opening of :attr:`command_joint`.

    The command is a joint in the world rather than a topic, for the reason
    :class:`FollowFrame`'s target is a body: whatever moves that joint in the twin --
    the fingers of a teleop marker, squeezed shut from a VR controller -- opens and
    closes the hand, and the change reaches giskard over ``/world_sync``.

    Read into a float variable each tick rather than used as the joint's own position
    symbol: :attr:`command_joint` is an active degree of freedom in giskard's world, and
    an expression over it would let the QP satisfy the task by moving the command
    instead of the fingers.

    One opening in ``[0, 1]`` for the whole hand -- where :attr:`command_joint` stands
    between its limits -- interpolated per finger joint between :attr:`closed` and
    :attr:`opened`. Closing on an object does not converge -- the fingers stop at the
    object -- and is meant not to: the error that remains is what keeps the sim's
    integrator leading the fingers into it, which is the grip
    (``velocity_integrator.MAX_LEAD``).
    """

    connections: List[str] = field(kw_only=True)
    """The finger joints, by connection name."""

    closed: List[float] = field(kw_only=True)
    """Each joint's position when the hand is shut."""

    opened: List[float] = field(kw_only=True)
    """Each joint's position when the hand is wide open."""

    command_joint: str = field(kw_only=True)
    """The connection whose position commands the opening: at its lower limit the hand
    is shut, at its upper one wide open."""

    max_velocity: float = field(default=0.1, kw_only=True)
    """Reference velocity of the fingers [m/s]."""

    weight: float = field(
        default=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE, kw_only=True
    )

    _opening: sm.FloatVariable = field(default=None, init=False, repr=False)

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()
        self._opening = sm.FloatVariable(f"{self.name}_opening")
        context.float_variable_data.register_expression(self._opening)
        context.float_variable_data.set_value(self._opening, self._commanded(context))
        for name, shut, wide in zip(self.connections, self.closed, self.opened):
            connection = context.world.get_connection_by_name(name)
            current = connection.dof.variables.position
            target = shut + self._opening * (wide - shut)
            artifacts.constraints.add_equality_constraint(
                name=name,
                reference_velocity=self.max_velocity,
                equality_bound=target - current,
                quadratic_weight=self.weight,
                task_expression=current,
            )
        artifacts.observation = sm.Scalar.const_true()
        return artifacts

    def _commanded(self, context: MotionStatechartContext) -> float:
        """The opening :attr:`command_joint` stands at, in ``[0, 1]``."""
        connection = context.world.get_connection_by_name(self.command_joint)
        limits = connection.dof.limits
        lower, upper = limits.lower.position, limits.upper.position
        if lower is None or upper is None or upper == lower:
            return 0.0
        return float(np.clip((connection.position - lower) / (upper - lower), 0.0, 1.0))

    def on_start(self, context: MotionStatechartContext):
        context.float_variable_data.set_value(self._opening, self._commanded(context))

    def on_tick(self, context: MotionStatechartContext):
        context.float_variable_data.set_value(self._opening, self._commanded(context))
        return None


@dataclass(eq=False, repr=False)
class HoldJoints(Task):
    """Keeps :attr:`connections` at the positions they had when this task started.

    The joint-space :class:`HoldPose`, for joints no teleop target should move: a torso
    lift both arms hang off. Held above collision avoidance, because that is what moves
    them otherwise -- a task on one arm, rooted at its mount, never reaches the lift,
    but keeping a link clear of the table may, and a lift that moves carries both arm
    mounts with it, so each hand's target shifts under the other's motion.
    """

    connections: List[str] = field(kw_only=True)
    """The joints held, by connection name."""

    reference_velocity: float = field(default=0.1, kw_only=True)
    """Normalization of the hold [m/s or rad/s]."""

    weight: float = field(
        default=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE, kw_only=True
    )

    _held: List[sm.FloatVariable] = field(default_factory=list, init=False, repr=False)

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()
        self._held = []
        for name in self.connections:
            connection = context.world.get_connection_by_name(name)
            held = sm.FloatVariable(f"{self.name}_{name}_held")
            context.float_variable_data.register_expression(held)
            context.float_variable_data.set_value(held, float(connection.position))
            self._held.append(held)
            current = connection.dof.variables.position
            artifacts.constraints.add_equality_constraint(
                name=name,
                reference_velocity=self.reference_velocity,
                equality_bound=held - current,
                quadratic_weight=self.weight,
                task_expression=current,
            )
        artifacts.observation = sm.Scalar.const_true()
        return artifacts

    def on_start(self, context: MotionStatechartContext):
        for name, held in zip(self.connections, self._held):
            context.float_variable_data.set_value(
                held, float(context.world.get_connection_by_name(name).position)
            )


def _matrix(position, quaternion_xyzw) -> np.ndarray:
    """4x4 from a position and an ``(x, y, z, w)`` quaternion."""
    x, y, z, w = quaternion_xyzw
    norm = np.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    matrix = np.eye(4)
    matrix[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    matrix[:3, 3] = position
    return matrix
