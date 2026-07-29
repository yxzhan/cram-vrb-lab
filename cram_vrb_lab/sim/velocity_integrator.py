"""Turns giskard's streamed joint velocities into Isaac Sim position targets.

Giskard's closed-loop controller does not send trajectories; it re-solves a QP
every control period and streams the resulting joint *velocities*. Isaac's
articulation drives take positions, so something has to integrate. Doing that
naively produces two failure modes that this class exists to avoid, both learned
the hard way on the Stretch:

- a target that runs away from the measured position while a joint is blocked,
  so the drive builds up an arbitrarily large error and lunges when it frees;
- a target that keeps following the measured position while the commanded
  velocity is zero, which leaves the drive with no error to work against, so a
  gravity-loaded joint sags a little every step and the following target
  ratchets it all the way down.

Shared by the Stretch and Panda sim nodes, which differ only in which joints
they hand over.
"""

import time

import numpy as np

MAX_LEAD = 0.02
"""How far [rad or m] a target may lead the measured position.

An anti-windup clamp: it bounds the drive's position error, and with it the
force the drive can develop. It also sets the grip force of a closing gripper,
which is ``finger stiffness * MAX_LEAD``.
"""

COMMAND_TIMEOUT = 0.5
"""Seconds without a command before the integrator lets go and holds position."""

MAX_STEP = 0.2
"""Longest wall-clock step [s] to integrate over in one go.

A hitch in the sim (asset loading, a slow render frame) must not be integrated
as one huge jump.
"""


class StreamedVelocityIntegrator:
    """Integrates streamed velocities for one fixed set of articulation DOFs.

    :param robot: the Isaac ``Articulation`` to command.
    :param joint_names: joint names in the order the velocity command carries
        them -- the contract with giskard's joint-group velocity controller,
        whose Float64MultiArray has no names.
    :param holding_joints: joints that keep their leading target when the
        commanded velocity drops to zero, instead of snapping it to the measured
        position. Gripper fingers: a finger that has stopped because an object is
        in the way must go on pressing into it, and snapping the target onto the
        measured position would zero the drive's error and drop the object. For
        every other joint the snap is what prevents gravity-driven ratcheting.
    """

    def __init__(self, robot, joint_names, holding_joints=()):
        self.robot = robot
        self.joint_names = list(joint_names)
        self.dof_indices = np.array(
            [robot.get_dof_index(name) for name in self.joint_names]
        )
        self._holds_target = np.array(
            [name in set(holding_joints) for name in self.joint_names]
        )
        dof_limits = robot.get_dof_limits()[0]
        self._lower = dof_limits[self.dof_indices, 0]
        self._upper = dof_limits[self.dof_indices, 1]
        self._command = None
        self._command_time = None
        self._targets = None
        self._was_zero = None
        self._last_tick = None
        self._stale = False

    def accept(self, velocities) -> bool:
        """Latch a velocity command; return whether it had the expected length."""
        if len(velocities) != len(self.dof_indices):
            return False
        self._command = np.asarray(velocities, dtype=float)
        self._command_time = time.time()
        return True

    def step(self, nominal_dt: float):
        """Advance the position targets. Call once per sim step.

        Integration uses measured **wall-clock** time, not the nominal step:
        giskard's QP plans in wall time, while a loaded sim steps slower than its
        nominal rate. Integrating the nominal step would execute commands at a
        load-dependent fraction of the commanded speed, which the controller
        perceives as lag and answers with overshoot.
        """
        now = time.time()
        dt, self._last_tick = (
            (min(now - self._last_tick, MAX_STEP), now)
            if self._last_tick is not None
            else (nominal_dt, now)
        )
        if self._command is None:
            return
        if now - self._command_time > COMMAND_TIMEOUT:
            # The stream stopped: hold whatever the drives were last given and
            # wait. The targets are kept rather than cleared, because a gap
            # between two giskard goals is the normal case -- and a gripper that
            # dropped its target here would relax mid-grasp and put the object
            # down.
            self._stale = True
            return

        measured = self.robot.get_joint_positions()[0][self.dof_indices]
        if self._targets is None:
            self._targets = measured.copy()
            self._was_zero = np.ones(len(self.dof_indices), dtype=bool)
        elif self._stale:
            # Re-seed after a gap so no stale lead is carried into the new goal,
            # except on holding joints, whose lead *is* the grip.
            self._targets = np.where(self._holds_target, self._targets, measured)
            self._was_zero = np.ones(len(self.dof_indices), dtype=bool)
        self._stale = False

        zero = self._command == 0.0
        newly_zero = zero & ~self._was_zero & ~self._holds_target
        moving = np.clip(
            self._targets + self._command * dt,
            measured - MAX_LEAD,
            measured + MAX_LEAD,
        )
        # Zero-velocity joints keep the target they were holding, snapped to the
        # measured position once on the transition into zero -- this is what
        # kills the end-of-goal overshoot from a leading target. Holding joints
        # skip the snap and keep pressing.
        target = np.where(
            zero, np.where(newly_zero, measured, self._targets), moving
        )
        self._targets = np.clip(target, self._lower, self._upper)
        self._was_zero = zero
        self.robot.set_joint_position_targets(
            self._targets.reshape(1, -1), joint_indices=self.dof_indices
        )
