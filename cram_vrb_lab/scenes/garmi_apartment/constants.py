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

from dataclasses import dataclass
from typing import Tuple

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


# --- Tabletop objects -----------------------------------------------------------
#
# This apartment ships without a single graspable object: every surface in it --
# worktop, dining table, nightstand, bay-window platform -- is bare, and the only
# small objects are books shelved *inside* a bookshelf, occluded from any standing
# height (which is why LIVING_ROOM_FLOOR aims at the floor instead). The objects
# below are added by the Isaac side at load time; neither ``world.usda`` nor
# ``scene-bodies.xml`` is touched.

YCB_ASSET_DIR = "/Isaac/Props/YCB/Axis_Aligned"
"""Directory of the YCB props, relative to the Isaac Sim assets root.

Resolved against ``isaacsim.storage.native.get_assets_root_path()`` rather than
:data:`ASSETS_DIR`, so these are the stock Isaac assets (currently served from
NVIDIA's cloud bucket, and cached locally by omniclient after the first load)
rather than another copy vendored into this repo.
"""

YCB_UPRIGHT_ROLL = -1.5707963267948966  # -pi/2
"""Roll [rad] about X that stands a YCB ``Axis_Aligned`` asset up in this Z-up world.

These assets are authored with their vertical axis along **Y**: the mustard bottle's
axis-aligned bounding box is 0.096 x 0.191 x 0.058 m, i.e. its 19.1 cm height lies
along the asset's Y. The sign is the part worth writing down -- the axis points
*down*, so it is the asset's **-Y** that has to become world +Z, which is a roll of
**-90 deg**, not +90. Both signs give an upright bounding box, so this cannot be
checked by measuring extents; +90 stands every object on its head (rendered and
looked at: the mustard label upside down, the soup can resting on its lid).

Leaves the bottle and the soup can standing, the tuna can on its base and the banana
lying flat -- how each of them would actually be found on a table.
"""


@dataclass(frozen=True)
class YCBProp:
    """One YCB object standing on a surface, in the giskard ``map`` frame."""

    name: str
    """Prim name under :data:`~cram_vrb_lab.scenes.garmi_apartment.isaac_scene.YCB_PROPS_ROOT`,
    and the name the spawn report prints."""

    asset: str
    """File name inside :data:`YCB_ASSET_DIR`."""

    position: Tuple[float, float, float]
    """(x, y, surface_z): where on the surface, and the height of the surface itself.

    Deliberately the **surface** height rather than the object's centre: how far a
    centre sits above the surface depends on the mesh, so
    :func:`~cram_vrb_lab.scenes.garmi_apartment.isaac_scene.spawn_ycb_props` measures
    each asset's bounding box after rotating it upright and releases it with its
    underside :data:`YCB_DROP_HEIGHT` above ``surface_z``.
    """

    mass: float
    """[kg]. The object's real mass, from the YCB object-and-model set.

    Given rather than left to PhysX's density default so a grasp has to hold the
    weight the real object has.
    """

    yaw: float = 0.0
    """Rotation [rad] about world Z, applied after :data:`YCB_UPRIGHT_ROLL`."""


YCB_DROP_HEIGHT = 0.005
"""How far [m] above its surface an object is released.

Small on purpose: the surfaces are measured (see :data:`KITCHEN_WORKTOP`) and each
asset is grounded on its own bounding box, so the drop only has to cover the error in
those two numbers. Releasing from a height instead of spawning exactly at rest is what
makes the settled pose evidence that the object is really standing on the surface
rather than hovering a centimetre above it or sunk into it.
"""


KITCHEN_WORKTOP = (0.5, 7.42, 0.945)
"""Centre of the kitchen worktop, in ``map``, at the height an object rests at.

Every z here is a **raycast** against the assembled scene -- a vertical ray dropped on
a grid over the surface, which is the only measurement that answers the question that
actually matters ("what would an object released here land on?") rather than a
question about geometry. Bounding boxes and the USD's own numbers both mislead:

- the ``cabinet`` prim's ``xformOp:translate`` in ``world.usda`` says 1.0, but that is
  the asset's pivot, near the middle of a run reaching from the floor to the wall
  units at z = 2.0
- the highest *vertices* of the cabinet's static mesh in this column sit at 0.900,
  which is the worktop's front edge profile, not its face

The rays come back 0.945 across the whole free run.

.. warning::
   The worktop is **not** clear at this centre. The sink is cut into it from x = 0.70
   to x = 1.00 (rays fall through to the basin at 0.76, with a rim at 0.956) and the
   tap stands behind it at (1.00, 7.60). Solid worktop runs from x = 0.10 to x = 0.65
   at every depth y in [7.14, 7.70], and again from x = 1.10 to x = 1.50. This centre
   is kept as the anchor because it is the middle of the worktop; :data:`YCB_PROPS`
   offsets its two objects into the free stretch to its left. A soup can released at
   x = 0.62 -- 0.08 m short of the cut-out -- slid off the edge and ended up in the
   sink, which is how the cut-out was found.
"""

DINING_TABLE_TOP = (1.85, 4.78, 0.771)
"""Centre of the dining table, in ``map``, at the height an object rests at.

Raycast like :data:`KITCHEN_WORKTOP`, and 0.771 everywhere on the top: the whole
surface is usable, 0.85 m across x by 1.35 m along y (rendered bounding box
x in [1.433, 2.283], y in [4.101, 5.452]).

The 11 mm between this and the top of the *visual* mesh at 0.760 is the table's own
collider: it ships from ``world.usda`` as a ``convexDecomposition`` approximation,
which does not follow the mesh exactly. Nothing to correct -- 0.771 is where an object
comes to rest, and the render shows it resting on the table -- but it is the reason a
number measured off the geometry is the wrong one to place from.
"""

YCB_PROPS = (
    # Two on the worktop, spread along the run (the cabinets face -y, so the free
    # direction is x) and pushed left of centre to clear the sink cut-out that starts
    # at x = 0.70 -- see the warning on KITCHEN_WORKTOP. 0.28 m apart centre-to-centre
    # leaves ~0.19 m of bare worktop between them, which is what keeps them two
    # clusters rather than one for the Euclidean clustering in
    # cram_vrb_lab.perception.pipeline.
    YCBProp("mustard_bottle", "006_mustard_bottle.usd",
            (KITCHEN_WORKTOP[0] - 0.20, KITCHEN_WORKTOP[1], KITCHEN_WORKTOP[2]),
            mass=0.603, yaw=0.35),
    YCBProp("tomato_soup_can", "005_tomato_soup_can.usd",
            (KITCHEN_WORKTOP[0] + 0.08, KITCHEN_WORKTOP[1] + 0.03, KITCHEN_WORKTOP[2]),
            mass=0.349),
    # Two on the dining table, spread along its long axis (y) for the same reason.
    # The banana is 0.197 m long and lies along x, well inside the table's 0.85 m.
    YCBProp("banana", "011_banana.usd",
            (DINING_TABLE_TOP[0], DINING_TABLE_TOP[1] - 0.13, DINING_TABLE_TOP[2]),
            mass=0.066, yaw=-0.5),
    YCBProp("tuna_fish_can", "007_tuna_fish_can.usd",
            (DINING_TABLE_TOP[0] + 0.03, DINING_TABLE_TOP[1] + 0.13, DINING_TABLE_TOP[2]),
            mass=0.171),
)
"""The four YCB objects the Isaac side adds to this apartment.

Both surfaces get a tall object and a low one, so a detection can be scored on
height as well as position. The positions here are where each object is *released*;
what it settles at is decided by physics and printed by
:func:`~cram_vrb_lab.scenes.garmi_apartment.isaac_scene.spawn_ycb_props`.

.. note::
   These are real rigid bodies, but neither the assets nor this apartment come that
   way, so both halves of the contact are built at load time (see
   :func:`~cram_vrb_lab.scenes.garmi_apartment.isaac_scene.spawn_ycb_props`): the YCB
   ``Axis_Aligned`` assets are bare meshes with no rigid body and no collider, and of
   the apartment's own prims only the 15 free bodies (dining table, chairs, floor
   lamp, books) carry a collider -- the kitchen cabinet, walls and floor carry none,
   which is why the robot drives on the invisible ground plane instead of the
   apartment's floor mesh. The dining table therefore supports an object out of the
   box; the worktop needs a collider adding first, or anything released above it
   falls straight through to the floor.
"""
