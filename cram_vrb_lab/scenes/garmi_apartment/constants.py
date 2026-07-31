"""garmi-apartment scene constants: asset locations and the USD/MJCF alignment.

Single source for where this apartment lives -- consumed by both the Isaac Sim
side (:mod:`cram_vrb_lab.scenes.garmi_apartment.isaac_scene`) and the giskard side
(:mod:`cram_vrb_lab.scenes.garmi_apartment.giskard_world`), so the rendered scene
and giskard's collision world stay aligned by construction.

.. note::
   The contrast with :mod:`cram_vrb_lab.scenes.apartment.constants` is the whole
   reason this module is short. There, ``apartment.urdf`` and
   ``apartmentICRA.usda`` are *independent* descriptions of the same flat, so
   lining them up needed a measured 180 deg yaw plus a probed x/y/z offset. Here
   ``scene-bodies.xml`` was **converted from** ``world.usda``, so the two agree
   exactly and the alignment is the identity -- see
   :data:`MJCF_OFFSET_IN_USD`. Verified by spot-checking the shared frames: the
   USD ``Root/Meshes/Base/Wall`` prim and the MJCF ``wall_0`` body both sit at
   ``(-1.4967, 4.8020, 1.5)``.
"""

from cram_vrb_lab.paths import ASSETS_DIR, ROS2_WS_DIR

GRID_USD_PATH = str(ASSETS_DIR / "Grid" / "default_environment.usd")
"""Ground/grid environment referenced under the apartment.

The apartment USD carries its own ``Base/Floor`` mesh, so this is not what the
robot drives on; it is here for the same reason as in the other scenes -- an
infinite physics ground plane, so a robot that leaves the flat does not fall
forever.
"""

GARMI_APARTMENT_USD_PATH = str(ASSETS_DIR / "garmi-apartment" / "world.usda")
"""The apartment scene rendered by Isaac Sim."""

GARMI_APARTMENT_MJCF_PATH = str(
    ROS2_WS_DIR
    / "src"
    / "iai_maps"
    / "iai_garmi_apartment"
    / "mjcf"
    / "scene-bodies.xml"
)
"""The digital twin: an MJCF converted from :data:`GARMI_APARTMENT_USD_PATH`.

79 bodies and 116 degrees of freedom -- the furniture plus articulated drawers,
cabinet doors and room doors. Meshes and textures are referenced with the MJCF
``meshdir``/``texturedir`` attributes and resolve relative to this file, so
(unlike ``apartment.urdf``) nothing has to be dropped to make it parse.
"""

USD_PRIM_POSITION_IN_MAP = (0.0, 0.0, 0.0)
"""Where Isaac places the apartment USD prim, in the giskard ``map`` frame.

Left at the origin **on purpose**. The USD's own coordinates are already the ones
the MJCF twin uses, so spawning the prim anywhere else would mean carrying the
same offset on the giskard side (:func:`garmi_apartment_pose_in_map`) and would
buy nothing. Giskard's ``map`` frame coincides with the Isaac world frame because
the localization joint is identity (``map == odom``) and ``/odom`` reports the
ground-truth pose.
"""

MJCF_OFFSET_IN_USD = (0.0, 0.0, 0.0)
"""Translation lining the MJCF root up with the USD origin: none needed.

Kept as a named constant rather than inlined so that a future re-export that
*does* shift the origin is a one-line change here, with both sides picking it up.
"""

MJCF_YAW_IN_USD = 0.0
"""Rotation about Z lining the MJCF up with the USD: none needed.

``apartment.urdf`` needs 180 deg here; a converted asset does not.
"""


def garmi_apartment_pose_in_map():
    """Pose of the MJCF root in the giskard ``map`` frame, as a
    ``HomogeneousTransformationMatrix``.

    Composes the Isaac prim placement with the MJCF-to-USD alignment, exactly as
    :func:`cram_vrb_lab.scenes.apartment.constants.apartment_pose_in_map` does.
    Both factors are currently identity, so this returns identity -- the
    composition is kept anyway so the two documented facts stay editable in
    isolation and the two scenes read the same way.
    """
    # Imported lazily: semantic_digital_twin exists only in the CRAM venv, and the
    # Isaac Sim side imports this module for the plain-tuple constants.
    from semantic_digital_twin.spatial_types.spatial_types import (
        HomogeneousTransformationMatrix,
    )

    map_T_usd = HomogeneousTransformationMatrix.from_xyz_rpy(*USD_PRIM_POSITION_IN_MAP)
    usd_T_mjcf = HomogeneousTransformationMatrix.from_xyz_rpy(
        *MJCF_OFFSET_IN_USD, yaw=MJCF_YAW_IN_USD
    )
    return map_T_usd @ usd_T_mjcf


# --- Where the robot stands and what it looks at --------------------------------
#
# The flat occupies x in [-5.3, 2.3], y in [1.2, 8.7], so the map origin is
# OUTSIDE it -- the Stretch's usual spawn at (-1.5, 0) would put it in the void
# beyond the south wall. Everything below is therefore given explicitly.

STRETCH_SPAWN_POSITION = (0.0, 6.0, 0.05)
"""Where the Stretch is spawned, in ``map``. Spawned facing ``map`` +x.

Standing in the living room with a clear view down +x: the coffee table is 0.9 m
ahead, the sofa and the floor lamp a little beyond it to either side. Clear of
the bookshelf behind it (x = -1.9), of ``door_1`` (y = 1.3) and of the coffee
table itself. z is the same 0.05 m lift ``spawn_stretch`` uses elsewhere, so the
wheels start just above the floor and settle onto it.
"""

LIVING_ROOM_FLOOR = (1.0, 2.7, 0.1)
"""Point in ``map`` the head camera is aimed at: the floor 1.3 m ahead of the robot.

Aimed at the **floor**, which is what makes the detection cell work in this
scene, and is the one substantive difference from
``stretch_perception_cram.ipynb``. That demo aims at a kitchen counter carrying
four small objects; this apartment has no such group -- every surface in it
(dining table, kitchen worktop, nightstand, bay-window platform) is bare, and its
only clustered small objects are books shelved *inside* a bookshelf, where the
shelf above occludes the view from any standing height and the dominant plane in
frame is the shelf's back panel rather than a support surface.

Aiming at the floor gives the pipeline the geometry it actually wants: one large
dominant plane with several well-separated objects standing on it -- the coffee
table, the sofa and the floor lamp. See ``FLOOR_CROP``.
"""

FURNITURE_IN_VIEW = ("coffee_table_0", "sofa_0", "floor_lamp_0")
"""MJCF bodies expected to stand on the plane at :data:`LIVING_ROOM_FLOOR`.

Used **only** to score the detections after the fact, never to produce them.
Because the twin was converted from the very USD the camera renders, these
bodies are exact ground truth -- something the original apartment scene could not
offer, and what the notebook's comparison step is built on.
"""

FLOOR_CROP = {
    # Camera optical frame: +x right, +y down, +z forward. Bounds the depth so the
    # far walls and the rooms beyond the living room never reach the plane fit,
    # and stays wide in x/y because the camera is tilted well down: with the
    # optical axis pointing at the floor 1.3 m ahead, the floor sweeps across most
    # of the frame and a tight lateral crop would clip the very objects standing
    # on it.
    "min_z": 0.4,
    "max_z": 4.0,
    "min_x": -2.5,
    "max_x": 2.5,
    "min_y": -2.0,
    "max_y": 2.0,
}
"""Crop passed to ``build_pipeline`` for the living-room floor view.

Replaces :data:`cram_vrb_lab.perception.pipeline.CAMERA_CROP`, which is bounded for
a counter about 2.4 m away at roughly camera height.

.. note::
   ``max_z`` deliberately leaves the window wall at x = 2.34 in frame, so the run
   returns a fourth, thin (~1.18 x 0.05 x 0.80 m) cluster alongside the three
   pieces of furniture. That is a correct answer to the question the pipeline is
   asked -- a wall does stand on the floor plane -- and cropping it out would mean
   pulling ``max_z`` in past the floor lamp at 2.1 m depth and losing the lamp with
   it. The notebook's comparison step prints every detection's extents so the slab
   is easy to recognise.
"""

FLOOR_CLUSTER_TUNING = {
    # The opposite problem to the counter demo. There, objects 2.4 m away covered
    # ~12x38 px and robokudo's stock min_cluster_count of 1000 rejected all of
    # them. Here a sofa 1.5 m away is thousands of points, so the floor is instead
    # a huge plane whose stray points cluster easily -- the counts go back up to
    # keep the leftovers of the floor itself from being reported as objects.
    "min_cluster_count": 300,
    "dbscan_min_cluster_count": 40,
    "min_on_plane_point_count": 200,
}
"""Cluster tuning for the floor view; overrides
:data:`cram_vrb_lab.perception.pipeline.CLUSTER_TUNING`."""
