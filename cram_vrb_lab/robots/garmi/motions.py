"""GARMI-specific motion mappings for CRAM.

Two repairs live here.

**1. Working a container ignores ``full_body_controlled``.** That flag is read in
exactly one place, ``coraplex.robot_plans.motions.gripper`` -- the TCP motions --
where it decides whether a Cartesian goal is posed from ``map`` or from
``base_link``. Giskard's ``Open`` / ``Close`` never consults it, and could not
usefully: their "hold the handle" task is ``CartesianPose(root_link=handle,
tip_link=tool_frame)``, and the chain from the handle to the tool frame runs
``handle -> ... -> map -> odom -> base_link -> ... -> tool_frame``, so the drive's
degrees of freedom sit in the error expression whatever the flag says and the QP
uses them. The result is a demo that approaches a drawer with the arm alone and
then pulls it open with the whole robot. :class:`_GarmiContainerMotion` pins the
base for the pull when the flag is off, so one switch means one thing.

**2. Giskard has no timeout of its own.** There is no
``max_trajectory_length`` or equivalent in its config, and the client side waits
on the action result with no deadline either
(``ros2_interface.ActionClient.get_result`` awaits ``wait_until_not_none``), so a
goal whose monitor never turns TRUE blocks the calling notebook cell for good.
The only mechanism giskard offers is a monitor node --
:class:`~giskardpy.motion_statechart.monitors.payload_monitors.CountSeconds` --
put in ``Parallel`` with the real goal under ``minimum_success=1``, so whichever
finishes first ends the motion.

Each of the motions below has a goal that can legitimately never converge:

- the gripper, because a hand closing on a rigid object stops short of the
  commanded finger positions and never reaches them;
- opening and closing a container, because the goal is satisfied only when the
  container's own joint reaches its limit -- and that joint is moved by physics
  in the sim, not by a controller, so it can stall anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.goals.templates import Parallel
from giskardpy.motion_statechart.graph_node import (
    Goal,
    MotionStatechartNode,
    NodeArtifacts,
)
from giskardpy.motion_statechart.monitors.payload_monitors import CountSeconds
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from coraplex.datastructures.enums import ExecutionType
from coraplex.robot_plans import MoveGripperMotion
from coraplex.robot_plans.motions.base import AlternativeMotion
from coraplex.robot_plans.motions.container import ClosingMotion, OpeningMotion
from coraplex.view_manager import ViewManager
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.spatial_types.spatial_types import Pose

from semantic_digital_twin.robots.garmi import Garmi

CLOSE_TIMEOUT = 3.0
"""Seconds to keep pushing the fingers shut before calling the hand closed.

Comfortably longer than the 0.8 s an unobstructed close takes, so it only ever
decides the *obstructed* case.
"""

CONTAINER_TIMEOUT = 20.0
"""Seconds to keep working a drawer or door before calling the motion done.

**Coupled to how fast the furniture is allowed to move.** A drawer's full travel
is 0.466 m and it is pulled at
:data:`cram_vrb_lab.scenes.garmi_apartment.giskard_world.PRISMATIC_VELOCITY_LIMIT`,
so the motion needs at least ``0.466 / that`` seconds: about 5 s at 0.1 m/s, but
47 s at 0.01 m/s. Not imported from there on purpose -- a robot has no business
knowing which flat it is standing in -- so **if you lower that limit, raise this
too**, or the timeout fires with the drawer half open and it looks like the grasp
slipped.

20 s is roughly four times what the 0.1 m/s setting needs.
"""


@dataclass(eq=False, repr=False)
class WhileHolding(Goal):
    """Runs ``goal`` with ``held`` constraining what the goal leaves free.

    A plain ``Parallel`` cannot express this. Its observation is a vote over its
    children -- all of them, or ``minimum_success`` of them -- while a hold task
    is satisfied from the first tick, so the motion would either end immediately
    (``minimum_success=1``) or be held hostage by a millimetre of slack in
    something nobody asked to move. Here the observation is the wrapped goal's,
    full stop; the hold tasks only ever contribute constraints.

    They apply for exactly as long as the goal does. A node's ``end_condition``
    defaults to false, so a hold task never retires itself: it ends when this
    wrapper ends, and this wrapper ends when the caller's sequencing says the goal
    is finished.
    """

    goal: MotionStatechartNode = field(kw_only=True)
    """The motion that was actually asked for."""

    held: List[MotionStatechartNode] = field(default_factory=list, kw_only=True)
    """Tasks constraining what the goal does not. See :func:`hold_base`."""

    def expand(self, context: MotionStatechartContext) -> None:
        self.add_nodes([self.goal, *self.held])

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts(observation=self.goal.observation_variable)


def hold_base(robot, world) -> CartesianPose:
    """A task that keeps the base where it is, in ``map``.

    Built the same way giskard's own ``Open`` holds the handle: a goal pose whose
    reference frame *is* the tip link, so it reads "stay at the pose you were at
    when this started" -- bound once on start, then held.

    Weighted like the tasks it runs against (``Open`` gives both of its
    ``WEIGHT_ABOVE_COLLISION_AVOIDANCE``), so it neither overrides the container
    goal nor gets quietly optimised away. If the two ever really conflict -- a
    handle the arm cannot follow from a standing base -- the drawer stalls and
    :data:`CONTAINER_TIMEOUT` ends the motion, rather than the base silently
    winning.
    """
    return CartesianPose(
        root_link=world.root,
        tip_link=robot.root,
        goal_pose=Pose(reference_frame=robot.root),
        weight=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE,
        name="HoldBase",
    )


@dataclass
class GarmiMoveGripper(MoveGripperMotion, AlternativeMotion[Garmi]):
    """Closes the hand on "reached the position *or* pushed long enough", so
    closing on an object terminates. Opening keeps the plain position goal:
    nothing obstructs it, and a sloppy one would leave the hand too narrow for
    the next grasp."""

    execution_type = ExecutionType.SIMULATED, ExecutionType.REAL

    def perform(self):
        return

    @property
    def _motion_chart(self):
        end_effector = ViewManager().get_end_effector_view(self.gripper, self.robot)
        goal_state = end_effector.get_joint_state_by_type(self.motion)

        if self.motion != GripperState.CLOSE:
            return JointPositionList(goal_state=goal_state, name="OpenGripper")

        return Parallel(
            [
                JointPositionList(goal_state=goal_state, name="CloseGripper"),
                CountSeconds(seconds=CLOSE_TIMEOUT, name="CloseGripperTimeout"),
            ],
            minimum_success=1,
        )


@dataclass
class _GarmiContainerMotion(AlternativeMotion[Garmi]):
    """Runs the stock container goal against a :data:`CONTAINER_TIMEOUT` clock.

    The goal itself is untouched -- ``super()._motion_chart`` is giskard's own
    ``Open`` / ``Close``, which drives the container's joint to its limit while
    holding the tool on the handle. What changes is that it is no longer the only
    way for the motion to end.

    That matters more here than for the gripper, because the container's joint is
    **not measured**: nothing in the sim publishes the apartment's articulation
    back (see the topic list -- only robot joints, odometry and tf), so giskard
    moves the drawer in its own model and has no way to notice that PhysX left it
    somewhere else. If the handle slips, the goal simply never converges.

    Also pins the base unless the robot is whole-body controlled -- point 1 of the
    module docstring. Only the *pull* needs this; the approach is already
    arm-only, because that one goes through ``MoveToolCenterPointMotion``, which
    reads the flag itself.

    Measured from the station ``drive_to`` parks at, with the base pinned: drawer
    1 comes out to its full 0.466 m in 246 control cycles (241 with the base
    free), and cabinet door 1 -- the harder case, its handle sweeping an arc of
    its own 0.41 m radius rather than pulling straight -- still reaches its full
    1.571 rad in 871 cycles (808 free). The arm can follow both from a standing
    base; what the base was doing was moving 0.41 m and 1.02 m respectively while
    the idle arm hung off it.

    .. note::
       If some container does stall at :data:`CONTAINER_TIMEOUT` with the arm
       stretched out, give the base back for that step:
       ``robot.mobile_base.full_body_controlled = True``.
    """

    execution_type = ExecutionType.SIMULATED, ExecutionType.REAL

    def perform(self):
        return

    @property
    def _motion_chart(self):
        goal = super()._motion_chart
        if not self.robot.mobile_base.full_body_controlled:
            goal = WhileHolding(
                goal=goal,
                held=[hold_base(self.robot, self.world)],
                name=f"{type(self).__name__}WithBaseHeld",
            )
        return Parallel(
            [
                goal,
                CountSeconds(
                    seconds=CONTAINER_TIMEOUT,
                    name=f"{type(self).__name__}Timeout",
                ),
            ],
            minimum_success=1,
        )


@dataclass
class GarmiOpeningMotion(_GarmiContainerMotion, OpeningMotion):
    """Opening a container, with a deadline. See :class:`_GarmiContainerMotion`."""


@dataclass
class GarmiClosingMotion(_GarmiContainerMotion, ClosingMotion):
    """Closing a container, with a deadline. See :class:`_GarmiContainerMotion`."""


GARMI_MOTION_MAPPINGS = [GarmiMoveGripper, GarmiOpeningMotion, GarmiClosingMotion]
"""Pass as ``Context(alternative_motion_mappings=...)``.

``AlternativeMotion.check_for_alternative`` picks an entry by
``issubclass(alternative, motion)`` plus a robot-class match, so each of these
substitutes for exactly the stock motion it derives from.
"""
