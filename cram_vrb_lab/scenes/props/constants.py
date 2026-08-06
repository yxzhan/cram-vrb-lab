"""Geometry of the pick-and-place prop: one graspable cube.

The smallest scene that answers "can this robot actually pick something up and
put it somewhere else": a cube resting on a surface, and a second place on that
surface to carry it to.

The props are deliberately *not* part of any environment asset.
``apartmentICRA.usda`` (what Isaac renders) and ``apartment.urdf`` (what the twin
plans in) are only approximately aligned -- see
:mod:`cram_vrb_lab.scenes.apartment.constants` -- so a cube placed on apartment
furniture would sit at a slightly different height in the render than in the
twin, and a failed grasp could not be told apart from a mis-modelled surface.
Both sides build the props from the numbers below instead, so the rendered
physics bodies and their twin counterparts agree by construction and a failed
grasp is a real grasp failure.

The cube itself is the same everywhere; where the props *stand* is not, because
a mobile robot and a bolted-down arm need completely different scenes. That part
lives in a :class:`PropLayout`, of which there is one per demo scene.

All poses are in the giskard ``map`` frame, which coincides with the Isaac world
frame (see
:data:`cram_vrb_lab.scenes.apartment.constants.USD_PRIM_POSITION_IN_MAP`).
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

CUBE_SIZE = 0.05
"""Edge length [m] of the graspable cube.

Small enough for both grippers in this repo to close past it -- the Panda hand
opens to 0.08 m, the Stretch's SG3 pads meet at finger angle 0 -- so a grasp ends
in a real squeeze rather than bottoming out.
"""

CUBE_GRIPPER_GAP = 0.09
"""Pad gap [m] to open the Stretch's gripper to before going for the cube.

Two centimetres of clearance either side of :data:`CUBE_SIZE`, enough that small
errors in the approach do not knock the cube off its surface. The Stretch's
default ``GripperState.OPEN`` is far too narrow for that; see
:mod:`cram_vrb_lab.robots.stretch.gripper`. The Panda hand cannot open this wide
and sets its own width in
:data:`cram_vrb_lab.robots.panda.semantic_model.GRIPPER_OPEN_TRAVEL`.
"""

CUBE_MASS = 0.05
"""[kg]. Light enough that the friction of a two-finger pinch carries it."""

CUBE_COLOR = (0.85, 0.15, 0.15)
"""RGB, so the cube is obvious against the surface it sits on."""

CUBE_FRICTION = 1.2
"""Static friction of the cube's physics material.

Isaac averages the two materials in a contact, so this is high on purpose: it
lifts the finger/cube pair to a coefficient where a light pinch does not let the
cube slide out during the carry.
"""

CUBE_BODY_NAME = "pick_cube"
"""Name of the cube's body in the digital twin."""

CUBE_POSE_TOPIC = "/props/pick_cube_pose"
"""Ground-truth cube pose published by the sim (a stand-in for perception).

The twin only knows where it *believes* the cube is; after a grasp attempt the
two can disagree, which is exactly the thing under test. See
:func:`cram_vrb_lab.scenes.props.twin_props.sync_cube_from_sim`.
"""

CUBE_GROUND_TRUTH_FRAME = "pick_cube_gt"
"""TF frame the sim publishes the true cube pose on. Deliberately *not* the twin
body's name, so RViz shows the believed and the actual cube as two frames."""

PROPS_FRAME_ID = "odom"
"""Frame the sim stamps prop poses in.

For the Stretch the sim publishes ground-truth ``odom -> base_link`` and the
localization stand-in makes ``map == odom``, so this is the Isaac world frame
under the name the tf tree already carries. The Panda scene has no odometry and
publishes the same static ``map -> odom`` identity, so the name still holds.
"""


@dataclass(frozen=True)
class PropLayout:
    """Where the cube starts and where it is meant to end up, in one scene.

    Kept apart from the cube's own geometry because each demo works at a
    different place and height. The cube is always released just above the
    surface it should land on and settled by physics, so what it actually rests
    on is whatever the scene provides -- a table, a counter, the floor.
    """

    pick_position: Tuple[float, float]
    """(x, y) the cube starts at."""

    place_position: Tuple[float, float]
    """(x, y) the cube is carried to."""

    surface_z: float = 0.0
    """Height [m] of the surface the cube rests on."""

    drop_height: float = 0.03
    """How far [m] above the resting height the cube is spawned.

    The exact height of a surface that comes from the scene's own geometry is not
    known here, so the cube is released just above where it is expected to land
    and physics settles it. :func:`~cram_vrb_lab.scenes.props.isaac_props.spawn_props`
    prints where it ended up, which is the only way to learn that surface's true
    height.
    """

    @property
    def cube_rest_z(self) -> float:
        """Height [m] of the cube's centre once it is resting."""
        return self.surface_z + CUBE_SIZE / 2

    @property
    def cube_start_position(self) -> Tuple[float, float, float]:
        """Where the cube is spawned -- above the surface, so it falls onto it."""
        return (*self.pick_position, self.cube_rest_z + self.drop_height)

    @property
    def cube_target_position(self) -> Tuple[float, float, float]:
        """Cube centre once placed, i.e. resting on the surface."""
        return (*self.place_position, self.cube_rest_z)


APARTMENT_LAYOUT = PropLayout(
    # North of the Stretch's spawn at (-1.5, 0) rather than south of it: with the
    # props on the south side the robot ended up boxed in between them and had no
    # clear lane to its standing position.
    pick_position=(-1.0, 0.6),
    # 1.6 m away, so the base has to drive while holding the cube. That drive is
    # what makes the Stretch demo a transport test rather than a grasp test.
    place_position=(-2.6, 0.6),
)
"""Cube positions for the Stretch in the apartment.

..warning::
   ``surface_z`` is the apartment floor, so the cube lands on the floor rather
   than at gripper height. This demo used to stand it on 0.7 m posts; those went
   away with the pedestal support, and the Stretch's standing positions and
   grasp have not been re-tuned for a cube on the floor."""

PANDA_REACH_OFFSETS = ((0.45, -0.2), (0.45, 0.2))
"""Pick and place positions in the Panda's **own base frame**, as (x, y).

Both 0.49 m from the base, well inside the 0.85 m reach and far enough out that
the arm is not folded back on itself; left and right of centre, so the transfer
is a real motion rather than a nudge. Kept in the robot frame so the layout
follows the arm wherever the scene puts it.
"""


def panda_layout_at(base_position, base_yaw: float) -> PropLayout:
    """The Panda's prop layout for a base standing at ``base_position`` [m] with
    ``base_yaw`` [rad], both in ``map``.

    The arm is bolted to a table and works on that table, so ``surface_z`` is
    simply the height the base is mounted at. The table itself comes from the
    apartment USD and is not modelled here.

    Rotating :data:`PANDA_REACH_OFFSETS` into ``map`` rather than writing map
    coordinates down keeps the cube in front of the arm by construction, so
    moving the robot cannot silently move it out of reach.
    """
    cos_yaw, sin_yaw = math.cos(base_yaw), math.sin(base_yaw)

    def to_map(offset):
        x, y = offset
        return (
            base_position[0] + cos_yaw * x - sin_yaw * y,
            base_position[1] + sin_yaw * x + cos_yaw * y,
        )

    pick_offset, place_offset = PANDA_REACH_OFFSETS
    return PropLayout(
        pick_position=to_map(pick_offset),
        place_position=to_map(place_offset),
        surface_z=base_position[2],
    )

# --- The Stretch's standing positions -------------------------------------
#
# Chosen from the Stretch's kinematics rather than by eye. The tool frame
# (link_grasp_center) sits at base_link y = -0.415 with the arm retracted and
# y = -0.935 fully extended, at z = 0.11 + joint_lift -- so the arm reaches out
# of the base's RIGHT side and the base must be parked with the cube abeam, not
# in front of it. Standing 0.75 m to the cube's +y side puts the arm at roughly
# 2/3 extension, comfortably inside both limits, and leaves the cube clear of the
# base footprint.
#
# The same kinematics fix which way the gripper may come at the cube. At the
# natural posture (wrist yaw 0) the tool frame's axes in base_link are
#
#     tool x (the approach axis) = base -y      tool z = base +z
#
# so with the base at yaw 0 the gripper approaches along map -y, straight down
# the arm. That is CRAM's ``ApproachDirection.LEFT``. ``FRONT`` would ask for the
# gripper to point along map +x instead, which costs a 90 deg wrist yaw and
# swings the grasp centre round to base (0.248, -0.145) -- inside the base's own
# footprint. The whole-body QP answers that by driving the base away and then
# goes infeasible. Grasp along the arm, not across it.

BASE_STANDOFF = 0.75
"""Lateral distance [m] from the base to the cube it is manipulating."""

PICK_BASE_POSITION = (
    APARTMENT_LAYOUT.pick_position[0],
    APARTMENT_LAYOUT.pick_position[1] + BASE_STANDOFF,
)
"""Where to park the base to reach the cube at the pick position."""

PLACE_BASE_POSITION = (
    APARTMENT_LAYOUT.place_position[0],
    APARTMENT_LAYOUT.place_position[1] + BASE_STANDOFF,
)
"""Where to park the base to reach the place position. Same y as
:data:`PICK_BASE_POSITION`, so the drive between them runs straight along a lane
that clears both."""

APPROACH_WAYPOINT = (-2.0, PICK_BASE_POSITION[1])
"""Waypoint to drive through on the way from the spawn to the pick position.

There is no path planning here -- ``DifferentialDriveBaseGoal`` drives roughly
straight at its target -- and the straight line from the Stretch's spawn at
``(-1.5, 0)`` to :data:`PICK_BASE_POSITION` cuts the corner at the pick
position. Going west first and then east along the standing lane clears both by
a wide margin.
"""
