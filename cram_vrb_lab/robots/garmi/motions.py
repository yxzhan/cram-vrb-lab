"""GARMI-specific motion mappings for CRAM.

The Panda's gripper mapping, for the same hardware and the same reason: the
generic close motion waits for the *measured* finger positions to reach the
commanded ones, which a hand closing on a rigid object never does, so the grasp
hangs with the object between the fingers. What marks a successful close is that
the fingers stopped early, hence the timeout branch.
"""

from __future__ import annotations

from dataclasses import dataclass

from giskardpy.motion_statechart.goals.templates import Parallel
from giskardpy.motion_statechart.monitors.payload_monitors import CountSeconds
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from coraplex.datastructures.enums import ExecutionType
from coraplex.robot_plans import MoveGripperMotion
from coraplex.robot_plans.motions.base import AlternativeMotion
from coraplex.view_manager import ViewManager
from semantic_digital_twin.datastructures.definitions import GripperState

from .semantic_model import Garmi

CLOSE_TIMEOUT = 3.0
"""Seconds to keep pushing the fingers shut before calling the hand closed.

Comfortably longer than the 0.8 s an unobstructed close takes, so it only ever
decides the *obstructed* case.
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


GARMI_MOTION_MAPPINGS = [GarmiMoveGripper]
"""Pass as ``Context(alternative_motion_mappings=...)``."""
