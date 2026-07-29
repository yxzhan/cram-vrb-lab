"""Panda-specific motion mappings for CRAM.

CRAM lets a robot substitute its own motion for a generic one
(:class:`coraplex.alternative_motion_mapping.AlternativeMotion`); this module
holds the Panda's, and the notebook passes them to its ``Context``.

Only one is needed, and it fixes a hang rather than improving a motion.
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

from .semantic_model import Panda

CLOSE_TIMEOUT = 3.0
"""Seconds to keep pushing the fingers shut before calling the hand closed.

Comfortably longer than the 0.8 s an unobstructed close takes (0.04 m of stroke
at the finger velocity limit), so it only ever decides the *obstructed* case.
"""


@dataclass
class PandaMoveGripper(MoveGripperMotion, AlternativeMotion[Panda]):
    """Closes the hand on a "reached the position *or* pushed long enough" rule,
    so closing on an object terminates.

    The generic motion turns a gripper state into a
    :class:`~giskardpy.motion_statechart.tasks.joint_tasks.JointPositionList`
    and waits for the *measured* finger positions to arrive. A hand closing on a
    rigid object never gets there -- the object stops the fingers half its own
    width short of closed -- so the motion waits forever and the whole grasp
    hangs with the cube between the fingers, which is a particularly confusing
    failure because everything up to that point looks right.

    Widening the position tolerance instead does not work: it would end the
    motion *before* the fingers touch anything narrower than the tolerance,
    which is a grasp of thin air. What actually marks a successful close is that
    the fingers stopped early, so the second branch is a timeout:
    :data:`CLOSE_TIMEOUT` of pushing counts as closed.

    Opening keeps the plain position goal. Nothing obstructs it, so reaching the
    commanded width is a fair requirement, and a sloppy one would leave the hand
    too narrow for the next grasp.

    ..note::
       Ending the motion is not the same as letting go. The fingers keep the
       target they were driving towards, which by then lies inside the object,
       so the drives go on squeezing through the lift and the carry -- see
       :class:`cram_vrb_lab.sim.velocity_integrator.StreamedVelocityIntegrator`.
    """

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


PANDA_MOTION_MAPPINGS = [PandaMoveGripper]
"""Pass as ``Context(alternative_motion_mappings=...)``."""
