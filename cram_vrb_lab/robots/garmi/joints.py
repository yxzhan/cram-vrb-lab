"""Shared constants for the giskard <-> Isaac Sim GARMI integration.

GARMI is a Clearpath-Ridgeback-style mobile base carrying a lift, a head and two
Franka FR3 arms with Franka Hands. The base is not driven here: the drawer demo
works from one standing position, so :func:`load_patched_urdf` freezes the
wheels, the lift and the head and the robot is bolted to ``map`` like the Panda.
What is left controlled is both arms and their fingers.

The joint order of :data:`CONTROLLED_JOINTS` is the contract between giskard's
joint-group velocity controller and the sim's velocity-command integrator: the
Float64MultiArray velocity command carries no joint names, only values in this
order.
"""

import os
import re
import tempfile
import xml.etree.ElementTree as ElementTree
from pathlib import Path

from cram_vrb_lab.paths import ASSETS_DIR

GARMI_DESCRIPTION_DIR = str(ASSETS_DIR / "garmi_description" / "garmi_description")
"""The geriatronics/garmi_description submodule."""

GARMI_URDF_PATH = os.path.join(GARMI_DESCRIPTION_DIR, "urdf", "garmi.urdf")

ROBOT_NAME = "garmi"
"""What the patched URDF calls the robot, i.e. the Isaac prim path.

The description's own name is ``r100-0603``, which the importer would sanitize
into a prim path nothing can predict.
"""

SIDES = ("left", "right")


def arm_joints(side: str) -> list[str]:
    return [f"{side}_fr3_joint{i}" for i in range(1, 8)]


def finger_joints(side: str) -> list[str]:
    return [f"{side}_fr3_finger_joint{i}" for i in (1, 2)]


ARM_JOINTS = [joint for side in SIDES for joint in arm_joints(side)]
FINGER_JOINTS = [joint for side in SIDES for joint in finger_joints(side)]
CONTROLLED_JOINTS = ARM_JOINTS + FINGER_JOINTS

FROZEN_JOINTS = [
    "front_left_wheel_joint",
    "front_right_wheel_joint",
    "rear_left_wheel_joint",
    "rear_right_wheel_joint",
    "lift_0_lower_joint",
    "lift_0_upper_joint",
    "o1_motor_1",
    "o1_motor_2",
]
"""Joints made ``fixed`` by :func:`load_patched_urdf`.

Everything nothing in this repo commands. Left movable they would be free DOFs
in both worlds: giskard would happily solve a goal by driving the head or the
lift, and in the sim they would sag under their own weight with no drive to
hold them.
"""

VELOCITY_CMD_TOPIC = "/garmi/joint_velocity_cmd"
JOINT_STATES_TOPIC = "/garmi/joint_states"
GRIPPER_CMD_TOPIC = "/garmi/gripper_command"

ROBOT_ROOT_LINK = "base_link"


def arm_root_link(side: str) -> str:
    return f"{side}_fr3_link0"


def hand_link(side: str) -> str:
    return f"{side}_fr3_hand"


def tool_frame_link(side: str) -> str:
    """The description's own TCP link, 0.1034 m out along the hand's +z."""
    return f"{side}_fr3_hand_tcp"


def finger_link(side: str, finger: str) -> str:
    return f"{side}_fr3_{finger}finger"


def fingertip_link(side: str, finger: str) -> str:
    """Added by :func:`load_patched_urdf`; the description ships no such link."""
    return f"{side}_fr3_{finger}finger_tip"


TOOL_FRAME_OFFSET = 0.1034
"""Distance [m] along the hand's +z to the point between the fingertips."""

FINGER_TIP_OFFSET = TOOL_FRAME_OFFSET - 0.0584
"""Distance [m] from a finger's own origin to its pad tip: 0.0584 m is where the
finger joint mounts the finger on the hand."""

MAX_FINGER_TRAVEL = 0.04
"""Per-finger stroke [m] from the URDF; the pads stand at most 0.08 m apart."""

GRIPPER_OPEN_TRAVEL = 0.038
"""Per-finger travel [m] that ``GripperState.OPEN`` means, just short of the hard
stop so the drive settles on the commanded value instead of pushing into it."""

FINGER_MASS = 0.0291
"""Mass [kg] of one FR3 finger link, from the description.

The importer authors *acceleration* drives, so the force-drive gains in
:mod:`cram_vrb_lab.robots.garmi.isaac_node` have to be divided by this to mean
what they say -- see the Panda's ``FINGER_MASS`` for the full account.
"""

PARK_CONFIGURATION = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
"""GARMI's own home pose, in :func:`arm_joints` order. The same for both arms.

Not chosen here: the description states it twice, and both agree. The URDF's
``ros2_control`` block gives every FR3 joint an ``initial_value`` (this, to three
decimals), and ``mujoco/garmi.xml`` carries it again as the ``home`` keyframe --
whose ``qpos`` also opens with ``0 0 0.0259``, which is where
:data:`~cram_vrb_lab.robots.garmi.isaac_node.BASE_LINK_HEIGHT` comes from.

It is Franka's own "ready" pose, applied to both arms unmirrored. On GARMI's
tilted shoulders (see ``arm_mount_*_joint``) that holds the hands 0.79 m in front
of ``base_link`` at 0.84 m -- arms out in front, chest height. Worth knowing when
picking a spawn pose: a robot parked less than ~0.8 m from a worktop parks its
hands inside it.
"""

_MESH_PREFIX = "package://garmi_description/"

_STRIPPED_JOINT_TAGS = ("axis", "limit", "mimic", "dynamics", "safety_controller")

_MESH_ALIAS_DIR = Path(tempfile.gettempdir()) / "garmi_mesh_aliases"
"""Where :func:`_usd_safe_mesh` puts its symlinks.

Outside the repo on purpose: it is generated, and a checkout should stay clean.
Regenerated on demand, so deleting it costs nothing.
"""


def _usd_safe_mesh(path: str) -> str:
    """``path``, or a symlink to it whose *file name* is a legal USD prim name.

    Isaac's URDF importer names the prim it creates for a mesh after that mesh's
    file, so ``body-collision.stl`` asks it for the prim path
    ``/colliders/chassis_link/body-collision``. Hyphens are not legal in an
    SdfPath: the path fails to parse, the importer carries on against a null prim
    and eventually raises ``Used null prim``, which says nothing about meshes at
    all. Three of this description's meshes are hyphenated
    (``body-collision``, ``end-cover``, ``side-cover``); everything else is passed
    through untouched.

    Renaming rather than editing the description keeps the submodule pristine, and
    all three are plain STLs -- no ``.mtl`` sidecar is looked up by name, so a
    renamed alias resolves to exactly the same geometry.
    """
    name = os.path.basename(path)
    safe = re.sub(r"[^0-9A-Za-z_.]", "_", name)
    if safe == name:
        return path

    _MESH_ALIAS_DIR.mkdir(parents=True, exist_ok=True)
    alias = _MESH_ALIAS_DIR / safe
    if not alias.is_symlink() or os.readlink(alias) != path:
        alias.unlink(missing_ok=True)
        alias.symlink_to(path)
    return str(alias)


def _tool_links() -> str:
    """The fingertip frames the semantic model looks bodies up by.

    The tool frame itself is not among them: unlike the stock Panda URDF, this
    description already carries ``*_fr3_hand_tcp``.
    """
    links = []
    for side in SIDES:
        for finger in ("left", "right"):
            tip, parent = fingertip_link(side, finger), finger_link(side, finger)
            links.append(
                f'  <link name="{tip}"/>\n'
                f'  <joint name="{tip}_joint" type="fixed">\n'
                f'    <origin rpy="0 0 0" xyz="0 0 {FINGER_TIP_OFFSET}"/>\n'
                f'    <parent link="{parent}"/>\n'
                f'    <child link="{tip}"/>\n'
                f"  </joint>\n"
            )
    return "".join(links) + "</robot>"


def load_patched_urdf() -> str:
    """Read the GARMI URDF and make it usable by the twin, giskard and the sim.

    Six changes: the robot is renamed (:data:`ROBOT_NAME`); the Gazebo and
    ros2_control blocks are dropped, as neither parser here has any use for them;
    :data:`FROZEN_JOINTS` become fixed joints; every ``mimic`` is removed, which
    on the fingers is what stops giskard's velocity group from putting two
    constraints on one DOF and going infeasible; mesh paths are made absolute
    (nothing resolves ``package://garmi_description`` here) and, where the file
    name would not survive Isaac's importer, aliased -- see
    :func:`_usd_safe_mesh`; and the fingertip frames are appended.
    """
    root = ElementTree.parse(GARMI_URDF_PATH).getroot()
    root.set("name", ROBOT_NAME)

    for tag in ("gazebo", "ros2_control"):
        for element in root.findall(tag):
            root.remove(element)

    frozen = set(FROZEN_JOINTS)
    for joint in root.findall("joint"):
        if joint.get("name") in frozen:
            joint.set("type", "fixed")
            for child in [c for c in joint if c.tag in _STRIPPED_JOINT_TAGS]:
                joint.remove(child)
        for mimic in joint.findall("mimic"):
            joint.remove(mimic)

    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if filename.startswith(_MESH_PREFIX):
            filename = filename.replace(_MESH_PREFIX, f"{GARMI_DESCRIPTION_DIR}/", 1)
        mesh.set("filename", _usd_safe_mesh(filename))

    urdf = ElementTree.tostring(root, encoding="unicode")

    assert urdf.count("</robot>") == 1, "expected exactly one closing robot tag"
    return urdf.replace("</robot>", _tool_links())


def joint_limits() -> dict[str, tuple[float, float]]:
    """Position limits [rad or m] of every controlled joint, read from the URDF."""
    root = ElementTree.fromstring(load_patched_urdf())
    controlled = set(CONTROLLED_JOINTS)
    limits = {}
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        if joint.get("name") in controlled and limit is not None:
            limits[joint.get("name")] = (
                float(limit.get("lower")),
                float(limit.get("upper")),
            )
    return limits
