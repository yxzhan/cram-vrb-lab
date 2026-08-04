"""Apartment scene constants: asset locations and the measured USD/URDF alignment.

Single source for where the apartment lives -- consumed by both the Isaac Sim
side (:mod:`cram_vrb_lab.scenes.apartment.isaac_scene`) and the giskard side
(:mod:`cram_vrb_lab.scenes.apartment.giskard_world`).
"""

import math

from cram_vrb_lab.paths import ASSETS_DIR, ROS2_WS_DIR

GRID_USD_PATH = str(ASSETS_DIR / "Grid" / "default_environment.usd")
"""Ground/grid environment referenced under the apartment."""

APARTMENT_USD_PATH = str(ASSETS_DIR / "apartment" / "apartmentICRA.usda")
"""The apartment scene rendered by Isaac Sim."""

APARTMENT_URDF_PATH = str(
    ROS2_WS_DIR / "src" / "iai_maps" / "iai_apartment" / "urdf" / "apartment.urdf"
)
"""Location of the apartment description: the iai_apartment source checkout in
``ros2_ws`` (same xacro lineage and root frame as the copy coraplex ships, so
the measured USD alignment below applies unchanged)."""

USD_PRIM_POSITION_IN_MAP = (-6.0, 5.0, 0.0701)
"""Where Isaac places the apartment USD prim, in the giskard ``map`` frame.

Consumed by BOTH the Isaac side (``isaac_scene.load_apartment_scene``) and the
giskard side (:func:`apartment_pose_in_map`), so the rendered apartment and
giskard's collision world stay aligned by construction. Giskard's ``map`` frame
coincides with the Isaac world frame because the localization joint is identity
(``map == odom``) and ``/odom`` reports the ground-truth pose.
"""

# APARTMENT_URDF_OFFSET_IN_USD = (9.55, -2.594, -0.0701)
APARTMENT_URDF_OFFSET_IN_USD = (9.516, -2.594, -0.08)

"""Translation, in the (world-aligned) USD-origin frame, applied to
``apartment.urdf`` after the yaw rotation to line its root up with the USD origin.

x/y refined by matching ``handle_cab10_m`` (URDF forward kinematics) against the
``SM_Kitchen_10_Drawer_02/SM_Kitchen_Handle19`` prim's world-space bbox center in
``apartmentICRA.usda``; the drawer body confirms the same x/y residual. z is left
untouched: the two probes disagree there (link-origin vs bbox-center conventions),
so vertical alignment should be judged in RViz, not from this measurement."""

APARTMENT_URDF_YAW_IN_USD = math.pi
"""Rotation about Z applied to ``apartment.urdf`` to line it up with the USD origin;
the URDF is authored 180 deg flipped relative to ``apartmentICRA.usda``."""


def apartment_pose_in_map():
    """Pose of ``apartment.urdf``'s root in the giskard ``map`` frame, as a
    ``HomogeneousTransformationMatrix``.

    Composes the Isaac USD prim placement (``map_T_usd``) with the measured
    URDF-to-USD alignment (``usd_T_urdf`` = yaw 180 deg then the offset). Because
    the USD prim is placed with identity rotation this reduces to a translation of
    ``(3.5, 7.5, 0.0701)`` and a 180 deg yaw, but the composition keeps the two
    documented facts (prim placement, URDF/USD alignment) editable in isolation.
    """
    # Imported lazily: semantic_digital_twin exists only in the CRAM venv, and
    # the Isaac Sim side imports this module for the plain-tuple constants.
    from semantic_digital_twin.spatial_types.spatial_types import (
        HomogeneousTransformationMatrix,
    )

    map_T_usd = HomogeneousTransformationMatrix.from_xyz_rpy(*USD_PRIM_POSITION_IN_MAP)
    usd_T_urdf = HomogeneousTransformationMatrix.from_xyz_rpy(
        *APARTMENT_URDF_OFFSET_IN_USD, yaw=APARTMENT_URDF_YAW_IN_USD
    )
    return map_T_usd @ usd_T_urdf


PANDA_BASE_POSITION_IN_USD = (7.14, -5.30, 0.933)
# PANDA_BASE_POSITION_IN_USD = (7.9, -5.30, 0.0)

"""Where the Panda stands, in ``apartmentICRA.usda``'s own frame.

Given in the USD's frame rather than in ``map`` because that is where it was
measured: with the apartment spawned at the origin this is the arm's correct
place in the scene. Our scene offsets the apartment prim
(:data:`USD_PRIM_POSITION_IN_MAP`), so the robot has to move with it -- see
:data:`PANDA_BASE_POSITION_IN_MAP`.
"""

PANDA_BASE_ORIENTATION_WXYZ = (0.0, 0.0, 0.0, -1.0)
"""The Panda's orientation, quaternion in USD/Isaac ``(w, x, y, z)`` order.

``w = 0, z = -1`` is a half turn about z, so the arm faces along ``map`` -x. The
apartment prim is placed with identity rotation, so this orientation carries
into ``map`` unchanged.
"""

PANDA_BASE_POSITION_IN_MAP = (
    PANDA_BASE_POSITION_IN_USD[0] + USD_PRIM_POSITION_IN_MAP[0],
    PANDA_BASE_POSITION_IN_USD[1] + USD_PRIM_POSITION_IN_MAP[1],
    PANDA_BASE_POSITION_IN_USD[2],
)
"""The Panda's base in the giskard ``map`` frame: (1.14, -0.30, 0.933).

x and y are a plain sum with the prim placement -- plain because the apartment
prim carries no rotation; if that ever changes this becomes a full composition.

z is taken **verbatim** from :data:`PANDA_BASE_POSITION_IN_USD`, without the
prim's 0.0701 lift. That lift belongs to the apartment geometry, not to the
surface the arm is bolted to, and adding it stood the robot 7 cm above its table.
"""

PANDA_BASE_YAW_IN_MAP = 2.0 * math.atan2(
    PANDA_BASE_ORIENTATION_WXYZ[3], PANDA_BASE_ORIENTATION_WXYZ[0]
)
"""The Panda's heading [rad] in ``map``, from :data:`PANDA_BASE_ORIENTATION_WXYZ`.

The orientation is a pure z rotation, so the quaternion reduces to a yaw. Used to
place the props in front of the arm without pulling in the twin's spatial types.
"""


def panda_pose_in_map():
    """Pose of the Panda's base in the giskard ``map`` frame, as a
    ``HomogeneousTransformationMatrix``.

    The giskard world config fixes the robot root to ``map`` with this, so the
    arm giskard plans for stands exactly where Isaac renders it.
    """
    from semantic_digital_twin.spatial_types.spatial_types import (
        HomogeneousTransformationMatrix,
    )

    w, x, y, z = PANDA_BASE_ORIENTATION_WXYZ
    return HomogeneousTransformationMatrix.from_xyz_quaternion(
        *PANDA_BASE_POSITION_IN_MAP, x, y, z, w
    )
