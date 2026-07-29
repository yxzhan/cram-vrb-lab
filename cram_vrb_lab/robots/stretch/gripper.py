"""How wide the SG3 gripper actually opens, and how to ask for a given width.

The semantic Stretch model ships a single hard-coded pair of finger angles for
``GripperState.OPEN`` / ``GripperState.CLOSE``
(``semantic_digital_twin.robots.stretch.StretchGripper.setup_joint_states``,
``[0.109, 0.109]`` and ``[0.0, 0.0]``). Measured against the SG3 fingertip
collision meshes of ``SIM_URDF_PATH``, those angles mean:

===============  ==========================
finger angle     gap between the pads
===============  ==========================
0.000 rad        0.000 m  (pads touching)
0.109 rad        0.036 m
0.250 rad        0.082 m
0.600 rad        0.191 m  (the patched limit)
===============  ==========================

So the default "open" leaves only 3.6 cm between the pads: the gripper cannot be
brought around anything wider than that, and a grasp of a 5 cm object fails
before it starts -- not because the reach or the approach direction is wrong,
but because the hand is still nearly shut. Widen the state to suit the object
with :func:`open_gripper_to`.

``0.0`` for CLOSE is right: the pads meet there, so closing on an object always
commands a genuine squeeze.
"""

from __future__ import annotations

from coraplex.datastructures.enums import Arms
from coraplex.view_manager import ViewManager
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world import World

PAD_GAP_PER_RADIAN = 0.3255
"""Slope [m/rad] of pad gap against finger angle.

The fingers swing on an arc, so the relation is only near-linear; this fit is
accurate to well under a millimetre up to 0.35 rad and drifts to ~4 mm at the
0.6 rad limit.
"""

MAX_FINGER_ANGLE = 0.6
"""Upper finger limit [rad] that :func:`.joints.load_patched_urdf` writes into the
URDF (the official one ships the finger joints with zeroed limits)."""


FINGER_JOINT_NAME = "joint_gripper_finger_left"
"""The finger whose measured angle stands for the whole gripper: the two fingers
are commanded to the same value and mirror each other."""


def finger_angle_for_gap(gap: float) -> float:
    """Finger angle [rad] that opens the pads to ``gap`` metres."""
    angle = gap / PAD_GAP_PER_RADIAN
    if angle > MAX_FINGER_ANGLE:
        raise ValueError(
            f"a gap of {gap:.3f} m needs {angle:.3f} rad, beyond the "
            f"{MAX_FINGER_ANGLE} rad finger limit (max gap ~0.19 m)"
        )
    return angle


def pad_gap_for_finger_angle(angle: float) -> float:
    """Pad gap [m] at a finger angle -- the inverse of :func:`finger_angle_for_gap`."""
    return angle * PAD_GAP_PER_RADIAN


def measured_pad_gap(world: World) -> float:
    """Pad gap [m] the fingers are currently at, from the synchronized world state.

    Worth printing after an open: the URDF limits are patched to ±0.6 rad but the
    sim's articulation carries its own, so a commanded width is not proof of an
    achieved one.
    """
    return pad_gap_for_finger_angle(
        world.get_connection_by_name(FINGER_JOINT_NAME).position
    )


def open_gripper_to(robot: AbstractRobot, gap: float, arm: Arms = Arms.LEFT) -> float:
    """Redefine what ``GripperState.OPEN`` means for ``robot`` to a pad gap of
    ``gap`` metres, and return the finger angle it now commands.

    Affects every later ``SetGripperAction(..., GripperState.OPEN)`` and the
    opening phase of ``PickUpAction``, because both resolve the target through
    the gripper view's joint states.
    """
    angle = finger_angle_for_gap(gap)
    joint_state = ViewManager().get_end_effector_view(arm, robot).get_joint_state_by_type(
        GripperState.OPEN
    )
    joint_state.target_values = [angle] * len(joint_state.target_values)
    return angle
