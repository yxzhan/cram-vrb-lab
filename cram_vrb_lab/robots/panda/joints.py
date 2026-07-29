"""Shared constants for the giskard <-> Isaac Sim Franka Emika Panda integration.

The Panda plays the same role here as the Stretch does in the apartment demos,
minus the mobile base: a 7-DoF arm bolted to the floor with a two-finger hand.
That makes it a much easier robot to get a grasp out of -- no base to drive, no
telescoping arm, and a gripper whose fingers slide straight at each other.

The joint order of :data:`CONTROLLED_JOINTS` is the contract between giskard's
joint-group velocity controller and the sim's velocity-command integrator
(:class:`cram_vrb_lab.robots.panda.isaac_node.PandaROS`): the Float64MultiArray
velocity command carries no joint names, only values in this order.

The description is the stock ``panda_arm_hand.urdf`` that ships with Isaac Sim's
URDF importer sample data, so the sim and the twin are built from the same file
and no asset has to be checked in. :func:`load_patched_urdf` adds what the
semantic model needs on top of it.
"""

import os
import xml.etree.ElementTree as ElementTree

PANDA_DESCRIPTION_DIR = (
    "/isaac-sim/exts/isaacsim.asset.importer.urdf/data/urdf/robots/franka_description"
)
"""Franka description shipped with Isaac Sim's URDF importer sample data."""

PANDA_URDF_PATH = os.path.join(
    PANDA_DESCRIPTION_DIR, "robots", "panda_arm_hand.urdf"
)

ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
FINGER_JOINTS = ["panda_finger_joint1", "panda_finger_joint2"]
CONTROLLED_JOINTS = ARM_JOINTS + FINGER_JOINTS

VELOCITY_CMD_TOPIC = "/panda/joint_velocity_cmd"
JOINT_STATES_TOPIC = "/panda/joint_states"
GRIPPER_CMD_TOPIC = "/panda/gripper_command"

ROBOT_ROOT_LINK = "panda_link0"
HAND_LINK = "panda_hand"
TOOL_FRAME_LINK = "panda_tool_frame"
LEFT_FINGER_LINK = "panda_leftfinger"
RIGHT_FINGER_LINK = "panda_rightfinger"
LEFT_FINGERTIP_LINK = "panda_leftfinger_tip"
RIGHT_FINGERTIP_LINK = "panda_rightfinger_tip"

TOOL_FRAME_OFFSET = 0.1034
"""Distance [m] along ``panda_hand`` +z to the point between the fingertips.

Franka's own ``panda_EE`` frame, i.e. where the pads meet when the hand closes
on something thin. The URDF ships no such link, so :func:`load_patched_urdf`
adds it; grasp poses are commanded for this frame.
"""

FINGER_TIP_OFFSET = TOOL_FRAME_OFFSET - 0.0584
"""Distance [m] from a finger's own origin to its pad tip.

0.0584 m is where ``panda_finger_joint1`` mounts the finger on the hand, so this
is the remaining length out to :data:`TOOL_FRAME_OFFSET`.
"""

MAX_FINGER_TRAVEL = 0.04
"""Per-finger stroke [m] from the URDF. Both fingers move, so the widest the
pads ever stand apart is twice this -- 0.08 m."""

GRIPPER_OPEN_TRAVEL = 0.038
"""Per-finger travel [m] that :data:`~semantic_digital_twin.datastructures.
definitions.GripperState.OPEN` means for this hand, i.e. a 0.076 m opening.

Just short of the :data:`MAX_FINGER_TRAVEL` hard stop, so the drive settles on
the commanded value instead of pushing into the limit, and as wide as that
allows: the hand has to come down *around* an object, and every millimetre of
clearance is a millimetre of approach error the grasp tolerates before a finger
knocks the object off its stand.
"""

PARK_CONFIGURATION = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
"""Franka's own "ready" pose, in :data:`ARM_JOINTS` order.

Elbow up and the hand held above the table in front of the base, which is both
well clear of the joint limits and a sane place to start a reach from.
"""

# These constants and the two functions below live here rather than with the
# semantic model because the Isaac Sim python has no semantic_digital_twin
# installed, and the sim node needs them too.


def gripper_pad_gap(finger_travel: float) -> float:
    """Distance [m] between the finger pads at a given per-finger travel.

    Both fingers move symmetrically, so the opening is simply twice the travel
    -- unlike the Stretch's SG3, whose pads swing on an arc.
    """
    return 2.0 * finger_travel


def finger_travel_for_pad_gap(gap: float) -> float:
    """Per-finger travel [m] that opens the pads to ``gap`` metres."""
    travel = gap / 2.0
    if travel > MAX_FINGER_TRAVEL:
        raise ValueError(
            f"a gap of {gap:.3f} m needs {travel:.3f} m of travel per finger, "
            f"beyond the {MAX_FINGER_TRAVEL} m stroke (max gap "
            f"{gripper_pad_gap(MAX_FINGER_TRAVEL):.3f} m)"
        )
    return travel

_MESH_PREFIX = "package://franka_description/"

_FINGER_MIMIC = '<mimic joint="panda_finger_joint1"/>'
"""The stock URDF couples the right finger to the left one with a mimic tag.

That is faithful to the real hand -- one motor drives both -- but it makes the
two joints share a single degree of freedom in the digital twin, while giskard's
velocity group controller is handed both joint *names*. The QP then receives two
velocity constraints on one variable and goes infeasible the moment they
disagree. Dropping the mimic gives each finger its own DOF on both sides; the
gripper joint states command them to the same value, so the hand still behaves
as one motor.
"""

_TOOL_LINKS = f"""  <link name="{TOOL_FRAME_LINK}"/>
  <joint name="{TOOL_FRAME_LINK}_joint" type="fixed">
    <origin rpy="0 0 0" xyz="0 0 {TOOL_FRAME_OFFSET}"/>
    <parent link="{HAND_LINK}"/>
    <child link="{TOOL_FRAME_LINK}"/>
  </joint>
  <link name="{LEFT_FINGERTIP_LINK}"/>
  <joint name="{LEFT_FINGERTIP_LINK}_joint" type="fixed">
    <origin rpy="0 0 0" xyz="0 0 {FINGER_TIP_OFFSET}"/>
    <parent link="{LEFT_FINGER_LINK}"/>
    <child link="{LEFT_FINGERTIP_LINK}"/>
  </joint>
  <link name="{RIGHT_FINGERTIP_LINK}"/>
  <joint name="{RIGHT_FINGERTIP_LINK}_joint" type="fixed">
    <origin rpy="0 0 0" xyz="0 0 {FINGER_TIP_OFFSET}"/>
    <parent link="{RIGHT_FINGER_LINK}"/>
    <child link="{RIGHT_FINGERTIP_LINK}"/>
  </joint>
</robot>"""


def load_patched_urdf() -> str:
    """Read the Panda URDF and make it usable by the twin, giskard and the sim.

    Three changes: absolute mesh paths, because nothing resolves the
    ``package://franka_description`` prefix here; the tool and fingertip frames
    the URDF has no links for (:data:`TOOL_FRAME_LINK`,
    :data:`LEFT_FINGERTIP_LINK`, :data:`RIGHT_FINGERTIP_LINK`); and the removal
    of the finger mimic (see :data:`_FINGER_MIMIC`).
    """
    with open(PANDA_URDF_PATH) as urdf_file:
        urdf = urdf_file.read()

    urdf = urdf.replace(_MESH_PREFIX, f"{PANDA_DESCRIPTION_DIR}/")

    assert urdf.count(_FINGER_MIMIC) == 1, "expected exactly one finger mimic tag"
    urdf = urdf.replace(_FINGER_MIMIC, "")

    assert urdf.count("</robot>") == 1, "expected exactly one closing robot tag"
    return urdf.replace("</robot>", _TOOL_LINKS)


def joint_limits() -> dict[str, tuple[float, float]]:
    """Position limits [rad or m] of every controlled joint, read from the URDF.

    The sim clamps integrated velocity targets to these, and the park pose is
    checked against them.
    """
    root = ElementTree.fromstring(load_patched_urdf())
    limits = {}
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        if joint.get("name") in CONTROLLED_JOINTS and limit is not None:
            limits[joint.get("name")] = (
                float(limit.get("lower")),
                float(limit.get("upper")),
            )
    return limits
