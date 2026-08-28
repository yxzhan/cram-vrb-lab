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

import os
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
# OUTSIDE it: a robot spawned at the default origin ends up in the void beyond the
# south wall. A demo in this scene has to pass a spawn position -- (0.0, 6.0, 0.05)
# puts the Stretch in the living room facing +x, with the coffee table 0.9 m ahead,
# the sofa and floor lamp a little beyond it, and clear of the bookshelf behind it
# (x = -1.9), of door_1 (y = 1.3) and of the coffee table itself. The 0.05 m is the
# usual lift that lets the wheels settle onto the floor rather than start in it.

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
   is kept as the anchor because it is the middle of the worktop; :data:`KITCHEN_PROPS`
   offsets its four objects into the free stretch to its left. A soup can released at
   x = 0.62 -- 0.08 m short of the cut-out -- slid off the edge and ended up in the
   sink, which is how the cut-out was found, and why nothing here is placed past
   x = 0.56.
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
    # All four on the dining table, spread along its long axis (y) so the gaps stay
    # wide enough for the Euclidean clustering in cram_vrb_lab.perception.pipeline to
    # call them four objects rather than one. The banana is 0.197 m long and lies
    # along x, well inside the table's 0.85 m; the tightest pair here clears by
    # 0.153 m. Bounds checked against the rendered table top,
    # x in [1.433, 2.283], y in [4.101, 5.452].
    #
    # The worktop is left to KITCHEN_PROPS -- the mustard bottle and the soup can
    # used to stand on it, and moved here when those four took the free run over.
    YCBProp("banana", "011_banana.usd",
            (DINING_TABLE_TOP[0], DINING_TABLE_TOP[1] - 0.13, DINING_TABLE_TOP[2]),
            mass=0.066, yaw=-0.5),
    YCBProp("tuna_fish_can", "007_tuna_fish_can.usd",
            (DINING_TABLE_TOP[0] + 0.03, DINING_TABLE_TOP[1] + 0.13, DINING_TABLE_TOP[2]),
            mass=0.171),
    YCBProp("mustard_bottle", "006_mustard_bottle.usd",
            (DINING_TABLE_TOP[0], DINING_TABLE_TOP[1] - 0.44, DINING_TABLE_TOP[2]),
            mass=0.603, yaw=0.35),
    YCBProp("tomato_soup_can", "005_tomato_soup_can.usd",
            (DINING_TABLE_TOP[0], DINING_TABLE_TOP[1] + 0.40, DINING_TABLE_TOP[2]),
            mass=0.349),
)
"""The four YCB objects the Isaac side adds to this apartment.

Tall objects and low ones together, so a detection can be scored on height as well as
position. The positions here are where each object is *released*; what it settles at
is decided by physics and printed by
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


# --- Kitchen objects on the cabinet worktop ---------------------------------
#
# The four props extracted out of ``apartmentICRA.usda`` into
# ``assets/kitchen-objects`` (see the README there). Unlike the YCB objects above
# these are *not* bare meshes: each asset already carries its own rigid body,
# mass, inertia tensor and collider, so the Isaac side only has to place it.

KITCHEN_OBJECTS_DIR = ASSETS_DIR / "kitchen-objects"
"""Directory of the standalone kitchen assets, vendored into this repo.

Local, unlike :data:`YCB_ASSET_DIR`, which resolves against the Isaac assets root:
these four were cut out of the ICRA apartment rather than downloaded, so nothing
here needs the Isaac asset bucket to be reachable.
"""

KITCHEN_PROPS_ENV = "ISAAC_KITCHEN_PROPS"
"""Environment variable that opts these objects in; see :func:`kitchen_props_enabled`."""


def kitchen_props_enabled() -> bool:
    """Whether to put the kitchen objects on the worktop. Off unless asked.

    An environment variable rather than a ``--`` flag because that is how the
    notebooks already steer the Isaac process (``ISAAC_RENDER``, ``ISAAC_HEADLESS``,
    ``ISAAC_WINDOW``): the demo sets it at the top of the file and
    ``launcher.start_isaac_sim`` inherits it into the sim subprocess. A scene flag
    would have to be threaded through the launcher, the entry script and the runner
    to reach the one place that reads it.

    Off by default so the other garmi demos keep the worktop they were tuned with.
    """
    return os.environ.get(KITCHEN_PROPS_ENV, "0") == "1"


@dataclass(frozen=True)
class KitchenProp:
    """One kitchen object standing on a surface, in the giskard ``map`` frame."""

    name: str
    """Asset directory name under :data:`KITCHEN_OBJECTS_DIR`, which is also the
    prim name and the name the spawn report prints."""

    position: Tuple[float, float, float]
    """(x, y, surface_z) -- the **surface** height, like :class:`YCBProp`, not the
    object's centre. The spawn measures each asset's bounding box and releases it
    :data:`YCB_DROP_HEIGHT` above the surface."""

    yaw: float = 0.0
    """Rotation [rad] about world Z. No roll: unlike the YCB ``Axis_Aligned``
    assets these are authored Z-up and stand the right way up unrotated (their
    bounding boxes measure 0.070 x 0.200 x **0.300** for the cereal box,
    0.133 x 0.133 x **0.067** for the bowl)."""

    @property
    def usd_path(self) -> str:
        """The asset's entry-point layer, whose ``defaultPrim`` is the object."""
        return str(KITCHEN_OBJECTS_DIR / self.name / f"{self.name}.usda")


KITCHEN_PROPS = (
    # All four go on the stretch of worktop LEFT of the sink, x in [0.10, 0.65] --
    # see the warning on KITCHEN_WORKTOP; the 0.70..1.00 cut-out to their right is
    # the sink itself. This is the run the mustard bottle and the soup can used to
    # stand on, before YCB_PROPS moved them to the dining table. It is the better of
    # the two runs to work: 0.55 m against 0.40 m, and no tap standing over it.
    #
    # Two rows, because even 0.55 m of run does not hold four objects side by side
    # with clustering gaps between them (their x footprints sum to 0.384 m). Tall at
    # the back, low at the front, so nothing hides behind anything else from a camera
    # looking in from -y:
    #
    #     back  (y = 7.55)   cereal box (0.30 m)   milk box (0.20 m)
    #     front (y = 7.30)   bowl (0.067 m)        cup (0.087 m)
    #
    # Every footprint below was checked against the free run and the sink cut-out
    # with the yaws applied; the tightest pair clears by 0.062 m, and the rightmost
    # edge (the cup's, at x = 0.556) keeps 0.094 m from where the worktop ends.
    KitchenProp("SM_CerealBox", (-0.1, 7.25, KITCHEN_WORKTOP[2]), yaw=1.5707963267948966),
    KitchenProp("SM_MilkBox", (0.15, 7.35, KITCHEN_WORKTOP[2]), yaw=-1.5707963267948966),
    KitchenProp("SM_SmallBowl", (0.55, 7.25, KITCHEN_WORKTOP[2])),
    # -pi/2 turns the handle, which the mesh puts on +x, to face -y -- i.e. towards
    # the robot, which works this run from the open side of the room.
    KitchenProp("SM_Cup", (0.3, 7.30, KITCHEN_WORKTOP[2]), yaw=-1.5707963267948966),
)
"""The four kitchen objects the Isaac side puts on the worktop when
:func:`kitchen_props_enabled`.

Cereal box, milk box, bowl and cup: the makings of one breakfast task, which is why
these four were the ones cut out of the ICRA apartment.

.. note::
   These carry their own physics, so the spawn must **not** re-apply a rigid body or
   a collider the way :func:`~cram_vrb_lab.scenes.garmi_apartment.isaac_scene.spawn_ycb_props`
   does for the YCB meshes -- that would overwrite the authored masses and flatten the
   cup's and bowl's convex decompositions back to a single hull, filling in the
   handle and the bowl's cavity. The worktop still needs its collider adding, exactly
   as for the YCB props.
"""
