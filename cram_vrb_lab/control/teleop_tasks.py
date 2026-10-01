"""Giskard tasks whose goal moves while they run, for teleoperation.

A giskard goal is built and parsed once -- ~5 s on the GARMI apartment -- so a
teleoperated arm cannot be one goal per target. These tasks are the other way round:
one long-running goal whose targets live in giskard's float variables, and each
control cycle's ``on_tick`` writes the newest target from a topic into them before the
QP is solved (``Executor.tick`` ticks the statechart, then solves with
``float_variable_data``). ``CartesianPositionTrajectory`` and ``WiggleInsert`` move
their goals the same way.

:class:`FollowFrame` reads its target off a body in the world, :class:`TeleopCartesianPose`
off a topic. Both start by holding where the robot is, so a goal sent before any target
has moved keeps the arm still rather than driving it anywhere. The grippers are not
here: a hand is opened and shut straight through the sim's finger drives
(``robots.garmi.joints.gripper_topic``), with nothing for a QP to solve.

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
class BodyMarker:
    """A marker in front of the robot's chest that moves the whole robot: the base goes
    under it and turns with it, the lift raises the torso to its height.

    At rest the marker sits :attr:`forward` ahead of the base and :attr:`height` above
    it plus the lift's travel, turned with the base; moving it away from there is what
    asks the robot to follow. Only its position and its heading count -- how it is
    tipped means nothing to a base that drives on the floor.
    """

    marker: KinematicStructureEntity
    """The marker body."""

    base: KinematicStructureEntity
    """The robot's root, which the base drives."""

    lift_connections: List[str]
    """The lift's joints, by connection name; their travel adds up to the lift's."""

    forward: float
    """[m] how far ahead of the base the marker sits at rest."""

    height: float
    """[m] how far above the base the marker sits at rest with the lift all the way
    down."""

    translation_tolerance: float = 0.02
    """[m] how close the base has to be to where the marker puts it to have arrived."""

    rotation_tolerance: float = 0.05
    """[rad] the same, for its heading -- about 3 degrees."""

    lift_tolerance: float = 0.01
    """[m] the same, for the lift."""

    def base_target(self, world) -> np.ndarray:
        """``map_T_base`` where the marker puts the base: level, at the base's own
        height, turned as the marker is."""
        map_T_marker = world.compute_forward_kinematics_np(world.root, self.marker)
        map_T_base = world.compute_forward_kinematics_np(world.root, self.base)
        yaw = float(np.arctan2(map_T_marker[1, 0], map_T_marker[0, 0]))
        target = np.eye(4)
        target[:3, :3] = _yaw_matrix(yaw)
        target[:2, 3] = map_T_marker[:2, 3] - target[:2, :2] @ np.array([self.forward, 0.0])
        target[2, 3] = map_T_base[2, 3]
        return target

    def lift_targets(self, world) -> List[float]:
        """Where each lift joint goes for the torso to reach the marker's height: the
        travel shared out evenly, held inside every joint's limits."""
        map_T_marker = world.compute_forward_kinematics_np(world.root, self.marker)
        map_T_base = world.compute_forward_kinematics_np(world.root, self.base)
        remaining = map_T_marker[2, 3] - map_T_base[2, 3] - self.height
        connections = [world.get_connection_by_name(name) for name in self.lift_connections]
        targets = []
        for index, connection in enumerate(connections):
            limits = connection.dof.limits
            share = remaining / (len(connections) - index)
            value = float(np.clip(share, limits.lower.position, limits.upper.position))
            targets.append(value)
            remaining -= value
        return targets

    def settled(self, world) -> bool:
        """Whether the robot stands where the marker puts it."""
        map_T_base = world.compute_forward_kinematics_np(world.root, self.base)
        if not _within(map_T_base, self.base_target(world),
                       self.translation_tolerance, self.rotation_tolerance):
            return False
        return all(
            abs(float(world.get_connection_by_name(name).position) - target)
            <= self.lift_tolerance
            for name, target in zip(self.lift_connections, self.lift_targets(world))
        )

    def rest_pose(self, world) -> np.ndarray:
        """``map_T_marker`` at rest on the robot as it stands."""
        map_T_base = world.compute_forward_kinematics_np(world.root, self.base)
        yaw = float(np.arctan2(map_T_base[1, 0], map_T_base[0, 0]))
        lift = sum(float(world.get_connection_by_name(name).position)
                   for name in self.lift_connections)
        pose = np.eye(4)
        pose[:3, :3] = _yaw_matrix(yaw)
        pose[:3, 3] = map_T_base[:3, 3] + pose[:3, :3] @ np.array(
            [self.forward, 0.0, self.height + lift]
        )
        return pose


@dataclass(eq=False, repr=False)
class FollowBodyBase(HoldPose):
    """Drives the base to where :attr:`body` puts it, every control cycle.

    With the marker at rest that is where the base already is, so the base holds still
    until the marker is moved. No deadband, unlike :class:`FollowFrame`: a marker let
    go of does not tremble, and the arms wait for the base to arrive exactly.
    """

    body: BodyMarker = field(kw_only=True)

    weight: float = field(
        default=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE, kw_only=True
    )
    """Below collision avoidance: the base is braked short of the furniture."""

    def on_tick(self, context: MotionStatechartContext):
        # The layout ForwardKinematicsBinding.bind writes: the top 3x4, column-major.
        context.float_variable_data.set_value(
            self._pose_to_keep.root_T_tip,
            self.body.base_target(context.world)[:3, :4].T.flatten(),
        )
        return None


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

    body: Optional[BodyMarker] = field(default=None, kw_only=True)
    """The marker the whole robot follows, if there is one. While the robot has not
    reached it the arm is locked where it is on its mount -- the hand stops at once and
    is carried along -- and :attr:`target_frame` is not followed until the robot has
    arrived and the target has been put back on the hand."""

    reset_tolerance: float = field(default=0.01, kw_only=True)
    """[m] how close :attr:`target_frame` has to be to the hand for a locked arm to take
    it up again."""

    _sent: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    """The target last written into the goal, root_T_target."""

    _locked: bool = field(default=False, init=False, repr=False)
    """Whether the arm is held on its mount while the robot follows :attr:`body`."""

    def on_start(self, context: MotionStatechartContext):
        self._sent = None
        self._locked = False
        self._follow(context)

    def on_tick(self, context: MotionStatechartContext):
        self._follow(context)
        return None

    def _follow(self, context: MotionStatechartContext) -> None:
        world = context.world
        if self.body is not None:
            if self._locked:
                if not (self.body.settled(world) and self._target_on_hand(world)):
                    return
                self._locked = False
                self._sent = None
            elif not self.body.settled(world):
                self._locked = True
                self._write(context, world.compute_forward_kinematics_np(
                    self.root_link, self.tip_link
                ))
                return
        root_T_target = world.compute_forward_kinematics_np(
            self.root_link, self.target_frame
        )
        if self._sent is not None and not self._moved(self._sent, root_T_target):
            return
        self._sent = np.array(root_T_target)
        self._write(context, root_T_target)

    def _write(self, context: MotionStatechartContext, root_T_target: np.ndarray) -> None:
        # The layout ForwardKinematicsBinding.bind writes: the top 3x4, column-major.
        context.float_variable_data.set_value(
            self._pose_to_keep.root_T_tip, root_T_target[:3, :4].T.flatten()
        )

    def _target_on_hand(self, world) -> bool:
        tip_T_target = world.compute_forward_kinematics_np(self.tip_link, self.target_frame)
        return float(np.linalg.norm(tip_T_target[:3, 3])) <= self.reset_tolerance

    def _moved(self, before: np.ndarray, after: np.ndarray) -> bool:
        """Whether ``after`` is out of the deadband around ``before``."""
        if np.linalg.norm(after[:3, 3] - before[:3, 3]) > self.translation_deadband:
            return True
        cosine = (np.trace(before[:3, :3].T @ after[:3, :3]) - 1.0) / 2.0
        return float(np.arccos(np.clip(cosine, -1.0, 1.0))) > self.rotation_deadband


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


@dataclass(eq=False, repr=False)
class FollowBodyLift(HoldJoints):
    """Raises and lowers the lift to the height of :attr:`body`, every control cycle.
    :attr:`connections` must be the body's lift joints, in the same order."""

    body: BodyMarker = field(kw_only=True)

    def on_tick(self, context: MotionStatechartContext):
        for held, target in zip(self._held, self.body.lift_targets(context.world)):
            context.float_variable_data.set_value(held, target)
        return None


def _yaw_matrix(yaw: float) -> np.ndarray:
    """3x3 rotation about z."""
    cosine, sine = np.cos(yaw), np.sin(yaw)
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def _within(before: np.ndarray, after: np.ndarray, translation: float, rotation: float) -> bool:
    """Whether two 4x4 poses are within ``translation`` [m] and ``rotation`` [rad]."""
    if np.linalg.norm(after[:3, 3] - before[:3, 3]) > translation:
        return False
    cosine = (np.trace(before[:3, :3].T @ after[:3, :3]) - 1.0) / 2.0
    return float(np.arccos(np.clip(cosine, -1.0, 1.0))) <= rotation


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
