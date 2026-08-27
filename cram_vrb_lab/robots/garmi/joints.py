"""Shared constants for the giskard <-> Isaac Sim GARMI integration.

GARMI is a Clearpath-Ridgeback-style mecanum base carrying a lift, a pan/tilt
head and two Franka FR3 arms with Franka Hands. All of it is live here: the base
drives, the lift raises the torso and the head turns.

The twin side does not model any of that locally --
:mod:`semantic_digital_twin.robots.garmi` already ships a full GARMI model
(``GarmiMobileBase(MobileBase[OmniDrive])``, ``GarmiTorso``, ``GarmiNeck``,
``GarmiCamera``) plus a ``garmi.srdf`` self-collision matrix. That model was
written against a differently-named export of this same robot, so
:func:`load_patched_urdf` renames the arms, grippers and head joints to match it
(:data:`UPSTREAM_RENAMES`) and everything downstream uses the upstream classes.

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

ARM_PREFIX = {"left": "arm_0", "right": "arm_1"}
"""How :mod:`semantic_digital_twin.robots.garmi` names the two arms.

``GarmiLeftArm`` resolves its tip as ``arm_0_fr3_link8`` off
``arm_mount_left_link``, so arm 0 is the left one.
"""


def arm_joints(side: str) -> list[str]:
    return [f"{ARM_PREFIX[side]}_fr3_joint{i}" for i in range(1, 8)]


def finger_joints(side: str) -> list[str]:
    return [f"{ARM_PREFIX[side]}_gripper_fr3_finger_joint{i}" for i in (1, 2)]


ARM_JOINTS = [joint for side in SIDES for joint in arm_joints(side)]
FINGER_JOINTS = [joint for side in SIDES for joint in finger_joints(side)]

LIFT_JOINTS = ["lift_0_lower_joint", "lift_0_upper_joint"]
"""The two prismatic segments of the lift column.

The description makes the upper one *mimic* the lower; the mimic is dropped (see
:func:`load_patched_urdf`) and ``GarmiTorso`` drives both to the same value
instead, which is what its low/mid/high states do.
"""

HEAD_JOINTS = ["head_pan_joint", "head_tilt_joint"]
"""``o1_motor_1`` / ``o1_motor_2`` after the rename -- pan then tilt, in the
``neck_1 -> neck_2 -> head`` chain."""

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
    return f"{ARM_PREFIX[side]}_gripper_fr3_hand"


def tool_frame_link(side: str) -> str:
    return f"{ARM_PREFIX[side]}_gripper_fr3_hand_tcp"

TOOL_FRAME_REACH = 0.1034 - 0.01
"""Distance [m] along the hand's +z to the point between the fingertips."""

TOOL_FRAME_LATERAL_OFFSET = -0.004
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

PARK_CONFIGURATION = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
"""GARMI's home pose, in :func:`arm_joints` order. The same for both arms.

Duplicated from ``GarmiLeftArm.ARM_PARK_CONFIGURATION`` in
:mod:`semantic_digital_twin.robots.garmi`, and it has to be: the sim's park runs
under the Isaac python, which has no ``semantic_digital_twin`` to read it from.
**Keep the two in sync** -- if they drift, the sim parks somewhere giskard does
not think it is. The description states the same numbers twice more, in the
URDF's ``ros2_control`` ``initial_value``s and in ``mujoco/garmi.xml``'s ``home``
keyframe (whose ``qpos`` also opens with ``0 0 0.0259``, which is where
:data:`~cram_vrb_lab.robots.garmi.isaac_node.BASE_LINK_HEIGHT` comes from).

It is Franka's own "ready" pose, applied to both arms unmirrored. On GARMI's
tilted shoulders (``arm_mount_*_joint``) it holds the hands 0.79 m in front of
``base_link`` at 0.84 m -- arms out, chest height.
"""

_MESH_PREFIX = "package://garmi_description/"

_MESH_ALIAS_DIR = Path(tempfile.gettempdir()) / "garmi_mesh_aliases"
"""Where :func:`_usd_safe_mesh` puts its symlinks.

Outside the repo on purpose: it is generated, and a checkout should stay clean.
Regenerated on demand, so deleting it costs nothing.
"""


def _upstream_renames() -> list[tuple[str, str]]:
    """Link/joint renames taking this description to the names
    :mod:`semantic_digital_twin.robots.garmi` looks bodies up by.

    The upstream model was written against another export of the same robot: its
    ``garmi.srdf`` names every non-arm link exactly as this URDF does
    (``chassis_link``, ``axle_link``, ``lift_0_*``, ``front_rocker_link``,
    ``lidar2d_*``, ``head``, ``neck_1``), and only the arm/gripper prefixes and
    the two head joints differ.

    **Order matters.** Each substitution is a plain prefix replacement, and the
    specific ones have to run before the catch-all ``left_fr3_`` -> ``arm_0_fr3_``
    or the fingers and hand would be swept into the arm's namespace. Replacing
    ``left_fr3_hand`` covers ``_hand_tcp``, ``_hand_joint`` and
    ``_hand_tcp_joint`` in one go, for the same reason.
    """
    renames = []
    for side in SIDES:
        arm = ARM_PREFIX[side]
        renames += [
            (f"{side}_fr3_hand", f"{arm}_gripper_fr3_hand"),
            (f"{side}_fr3_leftfinger", f"{arm}_gripper_fr3_leftfinger"),
            (f"{side}_fr3_rightfinger", f"{arm}_gripper_fr3_rightfinger"),
            (f"{side}_fr3_finger_joint", f"{arm}_gripper_fr3_finger_joint"),
            (f"{side}_fr3_", f"{arm}_fr3_"),
        ]
    return renames + [("o1_motor_1", "head_pan_joint"), ("o1_motor_2", "head_tilt_joint")]


UPSTREAM_RENAMES = _upstream_renames()

_TOOL_FRAME_RPY = {
    "left": "0 -1.5707963267948966 0",
    "right": "3.141592653589793 -1.5707963267948966 0",
}
"""Rotation put on each ``*_hand_tcp_joint``, replacing the description's ``0 0 0``.

Reconciles two conventions that would otherwise disagree silently.
:class:`~semantic_digital_twin.robots.robot_parts.EndEffector` derives its
approach direction as the **x** column of ``front_facing_orientation``, and
upstream's ``GarmiLeftGripper`` passes the identity quaternion -- so it expects a
tool frame whose x-axis already points out between the fingers. A Franka Hand
points its **z** out, and this description's TCP joint carries no rotation of its
own, so without this the approach axis would come out sideways and every grasp
would reach across the object instead of at it.

A quarter turn about y maps z onto x and leaves y -- the closing axis -- alone,
which is exactly the convention
:mod:`cram_vrb_lab.robots.panda.semantic_model` states for the Panda.

The sign is **negative**, and it is worth being sure about rather than reasoning
about: a rotation of ``theta`` about y has ``(cos(theta), 0, -sin(theta))`` as
its first column, so only ``-pi/2`` puts the tool's x on the hand's ``+z``.
``+pi/2`` lands on ``-z``, which preserves the closing axis and looks entirely
plausible while making every grasp approach the object from behind. Checked by
comparing the tool frame's x against the hand's z in the parked pose.

The right side carries an extra half turn about that same x, which is what makes
the two arms usable with one grasp description. The hands are attached to their
``link8`` identically, but ``arm_mount_right_joint`` mirrors
``arm_mount_left_joint`` (rpy ``1.0389 0.1675 0.6876`` against ``-1.0389 0.1675
-0.6876``), so at :data:`PARK_CONFIGURATION` -- the same numbers on both arms --
the right hand hangs rolled half a turn from the left: in ``base_link`` the
closing axis reads ``(-0.211, -0.483, 0.850)`` on the left and ``(0.211, -0.483,
-0.850)`` on the right, up against down.

``GraspDescription.grasp_orientation`` knows nothing about which arm it is
planning for -- for a ``FRONT``/``NoAlignment`` grasp it is the identity, and
both grippers pass the identity ``front_facing_orientation`` -- so both arms get
commanded the *same* tool roll. The left arm is already there; the right one has
to twist its wrist half a turn out of its natural posture to reach it, which is
the contorted reach at the drawers. Rolling its tool frame instead lets it hold
the goal the way its shoulder wants to.

A parallel gripper is symmetric about its approach axis, so this changes nothing
physical: the fingers still close on the same line, just labelled the other way
round. What it does change is the sign of
:data:`TOOL_FRAME_LATERAL_OFFSET`, which is why
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
            # still the description's own names here; the renames run further down
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
