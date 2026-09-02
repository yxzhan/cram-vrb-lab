"""Shared constants for the giskard <-> Isaac Sim GARMI integration.

GARMI is a Clearpath-Ridgeback-style mecanum base carrying a lift, a pan/tilt
head and two Franka FR3 arms with Franka Hands. All of it is live here: the base
drives, the lift raises the torso and the head turns.

The twin side does not model any of that locally --
:mod:`semantic_digital_twin.robots.garmi` already ships a full GARMI model
(``GarmiMobileBase(MobileBase[OmniDrive])``, ``GarmiTorso``, ``GarmiNeck``,
``GarmiCamera``) plus a ``garmi.srdf`` self-collision matrix. That model is
written against *this* description's own names -- ``left_fr3_*`` /
``right_fr3_*`` for the arms and grippers, ``o1_motor_*`` for the head -- so
:func:`load_patched_urdf` passes them through unchanged and everything
downstream uses the upstream classes.

The joint order of :data:`CONTROLLED_JOINTS` is the contract between giskard's
joint-group velocity controller and the sim's velocity-command integrator: the
Float64MultiArray velocity command carries no joint names, only values in this
order.

.. note::
   This module is imported by the Isaac python too, which has no
   ``semantic_digital_twin``. Everything here is therefore plain python, and the
   constants that duplicate an upstream value (:data:`PARK_CONFIGURATION`) say so.
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

ARM_PREFIX = {"left": "left", "right": "right"}
"""How :mod:`semantic_digital_twin.robots.garmi` names the two arms.

The same word the description uses: ``GarmiLeftArm`` resolves its tip as
``left_fr3_link8`` off ``arm_mount_left_link``. Kept as a mapping rather than
inlined because the demos index it by :class:`~coraplex.datastructures.enums.Arms`.
"""


def arm_joints(side: str) -> list[str]:
    return [f"{ARM_PREFIX[side]}_fr3_joint{i}" for i in range(1, 8)]


def finger_joints(side: str) -> list[str]:
    return [f"{ARM_PREFIX[side]}_fr3_finger_joint{i}" for i in (1, 2)]


ARM_JOINTS = [joint for side in SIDES for joint in arm_joints(side)]
FINGER_JOINTS = [joint for side in SIDES for joint in finger_joints(side)]

LIFT_JOINTS = ["lift_0_lower_joint", "lift_0_upper_joint"]
"""The two prismatic segments of the lift column.

The description makes the upper one *mimic* the lower; the mimic is dropped (see
:func:`load_patched_urdf`) and ``GarmiTorso`` drives both to the same value
instead, which is what its low/mid/high states do.
"""

HEAD_JOINTS = ["o1_motor_1", "o1_motor_2"]
"""The description's own names for the neck motors -- pan then tilt, in the
``neck_1 -> neck_2 -> head`` chain. ``GarmiNeck`` looks them up by these names."""

WHEEL_JOINTS = [
    "front_left_wheel_joint",
    "front_right_wheel_joint",
    "rear_left_wheel_joint",
    "rear_right_wheel_joint",
]
"""The four mecanum wheels.

Deliberately **not** in :data:`CONTROLLED_JOINTS`. The base is commanded as a
whole through giskard's ``OmniDrive`` connection and driven kinematically in the
sim (``GarmiROS.integrate_base``), which owns these joints and spins them
cosmetically; nothing in either world reads their angle back.
"""

CONTROLLED_JOINTS = ARM_JOINTS + FINGER_JOINTS + LIFT_JOINTS + HEAD_JOINTS
"""Every joint giskard streams velocities for -- the base excepted, which goes
over :data:`CMD_VEL_TOPIC` as a Twist instead."""

VELOCITY_CMD_TOPIC = "/garmi/joint_velocity_cmd"
JOINT_STATES_TOPIC = "/garmi/joint_states"
GRIPPER_CMD_TOPIC = "/garmi/gripper_command"
CMD_VEL_TOPIC = "/garmi/cmd_vel"

ODOM_TOPIC = "/odom"

# --- Head camera ----------------------------------------------------------
#
# The same topic names and the same frame id as the Stretch's head camera, and
# deliberately so rather than by copy-paste: demos/rviz/garmi.rviz already ships a
# DepthCloud pointed at these two topics, and cram_vrb_lab.perception.pipeline
# imports CAMERA_FRAME_ID and the topics straight out of the Stretch's joints
# module. Reusing the names is what lets both work against GARMI unchanged, and
# only one sim runs at a time so there is nothing to collide with.

RGB_IMAGE_TOPIC = "/head_camera/image_raw"
RGB_INFO_TOPIC = "/head_camera/camera_info"
DEPTH_IMAGE_TOPIC = "/head_camera/depth/image_raw"
DEPTH_INFO_TOPIC = "/head_camera/depth/camera_info"

CAMERA_FRAME_ID = "camera_color_optical_frame"
"""Frame the head-camera images are stamped in.

Unlike the Stretch, GARMI's URDF has **no camera link at all** -- 63 joints and not
one of them a camera; the description carries two IMUs and two 2D lidars and nothing
else. So this frame is not a link the twin or giskard knows about: it is invented
here and published by the sim as static tf off :data:`CAMERA_PARENT_LINK`, which the
per-step tf already emits.
"""

CAMERA_PARENT_LINK = "head"
"""Link the camera rides on -- the tip of the ``neck_1 -> neck_2 -> head`` chain, so
it pans and tilts with :data:`HEAD_JOINTS`."""

CAMERA_IN_HEAD = (0.13, 0.034, 0.02)
"""Where the camera sits in the ``head`` link frame [m].

Measured off ``head.obj``, which lands in that frame at x in [-0.087, 0.127],
y in [-0.101, 0.169], z in [-0.112, 0.090] once the visual origin's rpy is applied.
x = 0.13 puts the lens just proud of the face, y = 0.034 is the head's own centre
line (the tilt joint is mounted 34 mm off it), and z = 0.02 is a little above centre,
where eyes would be.
"""

CAMERA_OPTICAL_IN_HEAD_QUAT = (-0.5, 0.5, -0.5, 0.5)
"""``head -> camera_color_optical_frame`` rotation as (x, y, z, w).

The head looks along its own **+x** (the same axis
``semantic_digital_twin.robots.garmi.GarmiCamera`` declares as its
``forward_facing_axis``), with +y left and +z up. A REP-103 optical frame wants +z
forward, +x right and +y down, and this quaternion is exactly that relabelling:
optical +z -> head +x, optical +x -> head -y, optical +y -> head -z.
"""
"""Unnamespaced, as on the Stretch: the localization stand-in
(:func:`cram_vrb_lab.control.giskard_server.start_localization_stand_in`)
publishes a static ``map -> odom`` and only one robot runs at a time."""

BASE_LINK_HEIGHT = 0.0259
"""Height [m] of ``base_link`` above the floor when the wheels are on it.

The wheel centres sit 0.05 m above ``base_link`` and the wheels have a 0.0759 m
radius; the description's own MuJoCo ``home`` keyframe opens with the same number.
A demo that spawns GARMI at z = 0 sinks it into the floor by this much.

It is also where ``odom`` sits. Odometry on a wheeled robot is planar, so the
``OmniDrive`` the twin hangs the robot off has no z degree of freedom at all
("we can't measure its z-axis position, so z=0" -- ``OmniDrive``'s own
docstring). The only place that height can live is therefore ``map -> odom``:
:func:`cram_vrb_lab.control.giskard_server.start_localization_stand_in` publishes
it there, and :meth:`~cram_vrb_lab.robots.garmi.isaac_node.GarmiROS.publish_tf`
subtracts it again so ``odom -> base_link`` stays planar. Leave the two out of
step and every frame above the base is this much lower in the twin than in the
sim -- which reaches a modelled handle 2.6 cm high and sinks a perceived object
2.6 cm into the worktop.
"""

ROBOT_ROOT_LINK = "base_link"
BASE_LINK = "chassis_link"
"""What ``GarmiMobileBase`` uses as its root."""


def hand_link(side: str) -> str:
    return f"{ARM_PREFIX[side]}_fr3_hand"


def tool_frame_link(side: str) -> str:
    return f"{ARM_PREFIX[side]}_fr3_hand_tcp"

TOOL_FRAME_REACH = 0.1034 - 0.01
"""Distance [m] along the hand's +z to the point between the fingertips."""

TOOL_FRAME_LATERAL_OFFSET = 0.0
"""Shift [m] of the TCP along the hand's closing axis.

CRAM aims a grasp at the handle body's *origin*, which is not where the rod it
has to close around actually is -- see ``report_grasp_geometry`` in the demo.
This takes up that gap.

Signed **in the hand frame**, so it has to be mirrored between the two arms
(:func:`tool_frame_offset`): the correction is fixed in the world, while the two
hands meet a grasp rolled 180 degrees apart.
"""


def tool_frame_offset(side: str) -> str:
    """Where to anchor ``*_hand_tcp``, in its hand's frame."""
    lateral = TOOL_FRAME_LATERAL_OFFSET if side == "left" else -TOOL_FRAME_LATERAL_OFFSET
    return f"0 {lateral} {TOOL_FRAME_REACH}"

MAX_FINGER_TRAVEL = 0.04
"""Per-finger stroke [m] from the URDF; the pads stand at most 0.08 m apart."""

GRIPPER_OPEN_TRAVEL = 0.038
"""Per-finger travel [m] the sim opens the hand to when it parks.

Just short of the :data:`MAX_FINGER_TRAVEL` hard stop, so the drive settles on
the commanded value instead of pushing into the limit. Upstream's
``GripperState.OPEN`` is the full 0.04; the difference only matters to the sim's
park, since giskard commands the upstream value.
"""

FINGER_MASS = 0.0291
"""Mass [kg] of one FR3 finger link, from the description.

The importer authors *acceleration* drives, so the force-drive gains in
:mod:`cram_vrb_lab.robots.garmi.isaac_node` have to be divided by this to mean
what they say -- see the Panda's ``FINGER_MASS`` for the full account.
"""

# PARK_CONFIGURATION = [
#     0.0,
#     -1.6,
#     -1.0,
#     -2.356194490192345,
#     0.0,
#     1.5707963267948966,
#     0.7853981633974483,
# ]

PARK_CONFIGURATION = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]

"""GARMI's park pose, in :func:`arm_joints` order. The same numbers on both arms.

Duplicated from ``ARM_PARK_CONFIGURATION`` in ``GarmiLeftArm.setup_joint_states``
(:mod:`semantic_digital_twin.robots.garmi`), and it has to be: the sim's park runs
under the Isaac python, which has no ``semantic_digital_twin`` to read it from --
and upstream keeps it in a local variable rather than a class attribute, so there
is nothing importable to read even where the import would work.
**Keep the two in sync** -- if they drift, the sim parks somewhere giskard does
not think it is.

The description does *not* state these numbers. Its ``ros2_control``
``initial_value``s and ``mujoco/garmi.xml``'s ``home`` keyframe both still carry
Franka's own "ready" pose, ``0 -0.785 0 -2.356 0 1.571 0.785`` (the keyframe's
``qpos`` also opens with ``0 0 0.0259``, which is where
:data:`~cram_vrb_lab.robots.garmi.isaac_node.BASE_LINK_HEIGHT` comes from); this
pose pulls the shoulders down and rolls the upper arms in, off that home pose.

Applied unmirrored to two mirrored shoulders, so unlike the old "ready" pose --
which sat symmetrically at ``x=0.85, y=+-0.23, z=0.90`` in ``base_link`` -- it
does not hold the two hands symmetrically: the left lands at ``(0.54, 0.22,
1.21)`` and the right at ``(0.69, -0.54, 0.64)``.
"""


_MESH_PREFIX = "package://garmi_description/"

_MESH_ALIAS_DIR = Path(tempfile.gettempdir()) / "garmi_mesh_aliases"
"""Where :func:`_usd_safe_mesh` puts its symlinks.

Outside the repo on purpose: it is generated, and a checkout should stay clean.
Regenerated on demand, so deleting it costs nothing.
"""


UPSTREAM_RENAMES: list[tuple[str, str]] = []
"""Link/joint renames taking this description to the names
:mod:`semantic_digital_twin.robots.garmi` looks bodies up by -- none, now.

The upstream model used to be written against another export of the same robot,
which prefixed the arms ``arm_0_`` / ``arm_1_``, put the grippers in their own
``_gripper_`` namespace and called the neck motors ``head_pan_joint`` /
``head_tilt_joint``; :func:`load_patched_urdf` rewrote all of it on the way past.
It now looks bodies up by this description's own names (``left_fr3_link8``,
``left_fr3_hand``, ``o1_motor_1``, ...), so there is nothing left to rename.

Kept as an empty list rather than deleted: it is the seam where the two
namespaces meet, and the next export that disagrees goes here.
"""

_TOOL_FRAME_RPY = {
    "left": "0 0 0",
    "right": "0 0 3.141592653589793",
}
"""Rotation put on each ``*_hand_tcp_joint``.

The left value is the description's own, i.e. no rotation at all; the right one is a
half turn about the approach axis. Both are chosen so that **the tool frame's +z
points out between the fingers**, because that is the convention the pose CRAM
actually commands is built in.

Which way is "front" is genuinely ambiguous upstream -- two pieces of coraplex read
``EndEffector.front_facing_orientation`` (``Quaternion.from_rpy(0, pi/2, 0)`` for both
of GARMI's grippers) in opposite senses:

- :class:`~semantic_digital_twin.robots.robot_parts.EndEffector` takes the **x column**
  of its rotation matrix, ``R @ x``, which for that quaternion is ``(0, 0, -1)`` -- the
  tool's **-z**. This is ``front_facing_axis``.
- ``GraspDescription.calculate_end_effector_axis`` takes ``R.inverse() @ x``, which is
  ``(0, 0, +1)`` -- the tool's **+z**. ``grasp_orientation`` multiplies
  ``front_facing_orientation`` on the right for the same reason, so the tool pose it
  commands puts **+z** on the object.

The two differ by exactly the half turn this constant used to carry. Only the second
one moves the robot: ``GraspDescription.grasp_pose_sequence`` is what
``GraspingAction`` reaches to. Measured against the real drawer -- the commanded
pre-grasp frame for ``drawer_2_handle``, tool at ``(-0.09, 6.997, 0.6)`` and the handle
0.126 m away at ``(-0.09, 7.123, 0.6)`` -- the direction the fingers physically point,
dotted with the direction from the tool to the handle, comes out as:

======================================  ======  ==============================
``*_hand_tcp_joint`` rpy                dot     what you see
======================================  ======  ==============================
``0 0 0`` / ``0 0 pi``  (these)         ``+1``  fingers reach the handle
``pi 0 0`` / ``0 pi 0``  (until now)    ``-1``  **wrist** reaches the handle
``0 -pi/2 0``  (before that)             ``0``  approach square across it
======================================  ======  ==============================

The commanded tool pose does not depend on this constant at all -- ``grasp_orientation``
only ever sees ``front_facing_orientation`` -- so that table covers the whole family and
is not specific to this handle or this arm.

The cost of choosing this convention is that ``front_facing_axis`` now reads backwards,
and one place uses it: ``ReachMotion._calculate_pose_sequence`` backs its pre-pose off by
``-0.05`` along it (``coraplex/robot_plans/motions/gripper.py``), so that pre-pose lands
5 cm past the target instead of 5 cm short of it. That is the lesser of the two errors by
a wide margin, and ``GraspingAction`` does not go through ``ReachMotion`` anyway -- it
issues ``MoveToolCenterPointMotion`` directly. Upstream fixing its own disagreement is
what would remove the trade-off.

The right side carries the same half turn about the *approach* axis that it always did,
now expressed about z rather than about y because z is the approach axis again. It is
what makes the two arms usable with one grasp description. The hands are attached to
their ``link8`` identically, but ``arm_mount_right_joint`` mirrors
``arm_mount_left_joint`` (rpy ``1.0389 0.1675 0.6876`` against ``-1.0389 0.1675
-0.6876``), so the right hand hangs rolled half a turn from the left.

``GraspDescription.grasp_orientation`` knows nothing about which arm it is planning for
-- for a ``FRONT``/``NoAlignment`` grasp it is the identity, and both grippers pass the
same ``front_facing_orientation`` -- so both arms get commanded the *same* tool roll. The
left arm is already there; the right one would have to twist its wrist half a turn out of
its natural posture to reach it, which is the contorted reach at the drawers. Rolling its
tool frame instead lets it hold the goal the way its shoulder wants to.

A parallel gripper is symmetric about its approach axis, so this changes nothing
physical: the fingers still close on the same line, just labelled the other way round.
What it does change is the sign of :data:`TOOL_FRAME_LATERAL_OFFSET`, which is why
:func:`tool_frame_offset` mirrors it.
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


def load_patched_urdf() -> str:
    """Read the GARMI URDF and make it usable by the twin, giskard and the sim.

    Five changes:

    - the robot is renamed (:data:`ROBOT_NAME`), and the Gazebo and ros2_control
      blocks are dropped -- neither parser here has any use for them;
    - every ``mimic`` is removed. On the fingers that is what stops giskard's
      velocity group from putting two constraints on one DOF and going
      infeasible; on the lift it lets ``GarmiTorso`` drive both segments;
    - mesh paths are made absolute (nothing resolves
      ``package://garmi_description`` here) and, where the file name would not
      survive Isaac's importer, aliased -- see :func:`_usd_safe_mesh`;
    - the TCP frames are given :data:`_TOOL_FRAME_RPY` and re-anchored at
      :func:`tool_frame_offset`, both of which differ between the two arms
      because their shoulders are mirrored;
    - the arms, grippers and head joints are renamed to what the upstream
      semantic model expects -- see :func:`_upstream_renames`.

    Nothing is frozen: the wheels, the lift and the head are all real joints, and
    the upstream model claims all three.
    """
    root = ElementTree.parse(GARMI_URDF_PATH).getroot()
    root.set("name", ROBOT_NAME)

    for tag in ("gazebo", "ros2_control"):
        for element in root.findall(tag):
            root.remove(element)

    for joint in root.findall("joint"):
        for mimic in joint.findall("mimic"):
            joint.remove(mimic)
        name = joint.get("name", "")
        if name.endswith("_hand_tcp_joint"):
            side = next(s for s in SIDES if name.startswith(f"{s}_"))
            origin = joint.find("origin")
            origin.set("rpy", _TOOL_FRAME_RPY[side])
            origin.set("xyz", tool_frame_offset(side))

    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if filename.startswith(_MESH_PREFIX):
            filename = filename.replace(_MESH_PREFIX, f"{GARMI_DESCRIPTION_DIR}/", 1)
        mesh.set("filename", _usd_safe_mesh(filename))

    urdf = ElementTree.tostring(root, encoding="unicode")
    for old, new in UPSTREAM_RENAMES:
        urdf = urdf.replace(old, new)
    return urdf


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
