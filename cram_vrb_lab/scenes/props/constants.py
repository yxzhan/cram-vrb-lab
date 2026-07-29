"""Geometry of the pick-and-place props: one graspable cube and two pedestals.

The smallest scene that answers "can the Stretch gripper actually pick something
up and carry it somewhere else": a cube sitting on a pedestal, and a second
pedestal to carry it to.

The props are deliberately *not* part of the apartment. ``apartmentICRA.usda``
(what Isaac renders) and ``apartment.urdf`` (what the twin plans in) are only
approximately aligned -- see
:mod:`cram_vrb_lab.scenes.apartment.constants` -- so a cube placed on apartment
furniture would sit at a slightly different height in the render than in the
twin, and a failed grasp could not be told apart from a mis-modelled surface.
Both sides build the props from the constants below instead, so the rendered
physics bodies and their twin counterparts agree by construction and a failed
grasp is a real grasp failure.

All poses are in the giskard ``map`` frame, which coincides with the Isaac world
frame (see
:data:`cram_vrb_lab.scenes.apartment.constants.USD_PRIM_POSITION_IN_MAP`). The
props stand in the open floor west of the kitchen counter -- the apartment
furniture all sits at x > 0.2 -- next to where the Stretch spawns, at
``(-1.5, 0)``.
"""

CUBE_SIZE = 0.05
"""Edge length [m] of the graspable cube.

Small enough that the SG3 pads close well past it -- they meet at finger angle
0 -- so the grasp ends in a real squeeze rather than bottoming out. It is *not*
small enough for the semantic model's default open width, which parts the pads
by only 0.036 m; see :data:`CUBE_GRIPPER_GAP`.
"""

CUBE_GRIPPER_GAP = 0.09
"""Pad gap [m] to open the gripper to before going for the cube.

Two centimetres of clearance either side of :data:`CUBE_SIZE`, enough that small
errors in the approach do not knock the cube off its pedestal. The default
``GripperState.OPEN`` is far too narrow for that; see
:mod:`cram_vrb_lab.robots.stretch.gripper`.
"""

CUBE_MASS = 0.05
"""[kg]. Light enough that the friction of a two-finger pinch carries it."""

CUBE_COLOR = (0.85, 0.15, 0.15)
"""RGB, so the cube is obvious against the grey pedestals in the viewport."""

CUBE_FRICTION = 1.2
"""Static friction of the cube's physics material.

Isaac averages the two materials in a contact, so this is high on purpose: it
lifts the finger/cube pair to a coefficient where a light pinch does not let the
cube slide out during the carry.
"""

PEDESTAL_SIZE = (0.05, 0.05, 0.7)
"""(x, y, z) extents [m] of each pedestal. The top face is at z = extent z."""

PEDESTAL_COLOR = (0.4, 0.4, 0.45)

PICK_PEDESTAL_POSITION = (-1.0, 0.6)
"""(x, y) of the pedestal the cube starts on.

North of the Stretch's spawn at ``(-1.5, 0)`` rather than south of it: with the
pedestals on the south side the robot ended up boxed in between them and had no
clear lane to its standing position.
"""

PLACE_PEDESTAL_POSITION = (-2.6, 0.6)
"""(x, y) of the pedestal the cube is carried to -- 1.6 m away, so the base has
to drive while holding the cube. That drive is what makes this a transport test
rather than a grasp test."""

PEDESTAL_TOP_Z = PEDESTAL_SIZE[2]
"""Height [m] of the pedestal top faces, i.e. the surface the cube rests on."""

CUBE_START_POSITION = (*PICK_PEDESTAL_POSITION, PEDESTAL_TOP_Z + CUBE_SIZE / 2 + 0.1)
"""Cube centre when it is resting on the pick pedestal."""

CUBE_TARGET_POSITION = (*PLACE_PEDESTAL_POSITION, PEDESTAL_TOP_Z + CUBE_SIZE / 2)
"""Cube centre when it has been placed on the place pedestal."""

# Base poses, chosen from the Stretch's kinematics rather than by eye. The tool
# frame (link_grasp_center) sits at base_link y = -0.415 with the arm retracted
# and y = -0.935 fully extended, at z = 0.11 + joint_lift -- so the arm reaches
# out of the base's RIGHT side and the base must be parked with the cube abeam,
# not in front of it. Standing 0.75 m to the cube's +y side puts the arm at
# roughly 2/3 extension, comfortably inside both limits, and leaves the pedestal
# (0.4 m deep) clear of the base footprint.
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
    PICK_PEDESTAL_POSITION[0],
    PICK_PEDESTAL_POSITION[1] + BASE_STANDOFF,
)
"""Where to park the base to reach the cube on the pick pedestal."""

PLACE_BASE_POSITION = (
    PLACE_PEDESTAL_POSITION[0],
    PLACE_PEDESTAL_POSITION[1] + BASE_STANDOFF,
)
"""Where to park the base to reach the place pedestal. Same y as
:data:`PICK_BASE_POSITION`, so the drive between them runs straight along a lane
that clears both pedestals."""

APPROACH_WAYPOINT = (-2.0, PICK_BASE_POSITION[1])
"""Waypoint to drive through on the way from the spawn to the pick position.

There is no path planning here -- ``DifferentialDriveBaseGoal`` drives roughly
straight at its target -- and the straight line from the Stretch's spawn at
``(-1.5, 0)`` to :data:`PICK_BASE_POSITION` shaves the pick pedestal's near
corner. Going west first and then east along the standing lane clears both
pedestals by a wide margin.
"""

CUBE_BODY_NAME = "pick_cube"
"""Name of the cube's body in the digital twin."""

PICK_PEDESTAL_BODY_NAME = "pick_pedestal"
PLACE_PEDESTAL_BODY_NAME = "place_pedestal"

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
"""Frame the sim stamps prop poses in. The sim publishes ground-truth
``odom -> base_link`` and the localization stand-in makes ``map == odom``, so
this is the Isaac world frame under the name the tf tree already carries."""
