"""GARMI-specific motion mappings for CRAM.

All three are the same repair: **giskard has no timeout of its own**. There is no
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

from dataclasses import dataclass

from giskardpy.motion_statechart.goals.templates import Parallel
from giskardpy.motion_statechart.monitors.payload_monitors import CountSeconds
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from coraplex.datastructures.enums import ExecutionType
from coraplex.robot_plans import MoveGripperMotion
from coraplex.robot_plans.motions.base import AlternativeMotion
from coraplex.robot_plans.motions.container import ClosingMotion, OpeningMotion
from coraplex.view_manager import ViewManager
from semantic_digital_twin.datastructures.definitions import GripperState

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
    """

    execution_type = ExecutionType.SIMULATED, ExecutionType.REAL

    def perform(self):
        return

    @property
    def _motion_chart(self):
        return Parallel(
            [
                super()._motion_chart,
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
