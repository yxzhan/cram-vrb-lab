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
from typing import Optional, Tuple

from cram_vrb_lab.paths import ASSETS_DIR, CRAM_SUBMODULE_DIR, ROS2_WS_DIR

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


# --- Surfaces objects are placed on ---------------------------------------------
#
# This apartment ships without a single graspable object: every surface in it --
# worktop, dining table, nightstand, bay-window platform -- is bare, and the only
# small objects are books shelved *inside* a bookshelf, occluded from any standing
# height (which is why LIVING_ROOM_FLOOR aims at the floor instead). Whatever
# stands on these surfaces is put there by the Isaac side at load time; neither
# ``world.usda`` nor ``scene-bodies.xml`` is touched.

PROP_DROP_HEIGHT = 0.005
"""How far [m] above its surface an object is released.

Small on purpose: the surfaces are measured (see :data:`KITCHEN_WORKTOP`) and each
asset is grounded on its own bounding box (see
:func:`~cram_vrb_lab.scenes.garmi_apartment.isaac_scene._release_above_surface`), so
the drop only has to cover the error in those two numbers. Releasing from a height instead of spawning exactly at rest is what
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

# --- Kitchen objects on the cabinet worktop ---------------------------------
#
# The props extracted out of ``apartmentICRA.usda`` into ``assets/kitchen-objects``
# (see the README there). Each asset already carries its own rigid body, mass,
# inertia tensor and collider, so the Isaac side only has to place it -- these are
# the only graspable objects this scene has.

KITCHEN_OBJECTS_DIR = ASSETS_DIR / "kitchen-objects"
"""Directory of the standalone kitchen assets, vendored into this repo.

Local rather than resolved against the Isaac assets root: these were cut out of the
ICRA apartment rather than downloaded, so nothing here needs the Isaac asset bucket
to be reachable.
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
    """Prim name under :data:`~cram_vrb_lab.scenes.garmi_apartment.isaac_scene.KITCHEN_PROPS_ROOT`,
    and the name the spawn report prints.

    **Unique across** :data:`KITCHEN_PROPS`, which is the whole reason it is a field
    of its own rather than doubling as the asset name the way it used to. A prim path
    identifies exactly one prim, so two objects called ``SM_Cup`` are not two cups --
    the second ``create_prim`` lands on the path the first already took. Put a second
    copy of an asset on the worktop by giving it its own name and naming the asset
    explicitly, as :data:`KITCHEN_PROPS` does::

        KitchenProp("cup_left", (...), asset="SM_Cup")
    """

    position: Tuple[float, float, float]
    """(x, y, surface_z) -- the **surface** height, not the object's centre. The
    spawn measures each asset's bounding box and releases it
    :data:`PROP_DROP_HEIGHT` above the surface."""

    yaw: float = 0.0
    """Rotation [rad] about world Z. No roll: these assets are authored Z-up and
    stand the right way up unrotated (their bounding boxes measure
    0.070 x 0.200 x **0.300** for the cereal box, 0.133 x 0.133 x **0.067** for the
    bowl)."""

    asset: Optional[str] = None
    """Directory name under :data:`KITCHEN_OBJECTS_DIR`; defaults to :attr:`name`.

    Only worth giving when one asset is placed more than once, or when the prim
    should read as its role in the scene rather than as the file it came from.
    """

    @property
    def asset_name(self) -> str:
        """The directory this object's USD lives in -- :attr:`asset`, or :attr:`name`."""
        return self.asset or self.name

    @property
    def usd_path(self) -> str:
        """The asset's entry-point layer, whose ``defaultPrim`` is the object."""
        return str(
            KITCHEN_OBJECTS_DIR / self.asset_name / f"{self.asset_name}.usda"
        )


KITCHEN_PROPS = (
    # All six go on the stretch of worktop LEFT of the sink -- see the warning on
    # KITCHEN_WORKTOP; the 0.70..1.00 cut-out to their right is the sink itself. The
    # run this uses is x in [-0.10, 0.56], y in [7.14, 7.70].
    #
    # Two rows, tall at the back, low at the front, so nothing hides behind anything
    # else from a camera looking in from -y:
    #
    #     back  (y = 7.55)   cereal box (0.30 m)   milk box (0.20 m)
    #     front (y = 7.28)   bowl  cup  bowl  cup
    #
    # Footprints, with each yaw applied (half-extents, m):
    #
    #     cereal box, yaw +pi/2   x 0.100   y 0.035
    #     milk box,   yaw -pi/2   x 0.0475  y 0.030
    #     bowl                    x 0.0665  y 0.0665
    #     cup,        yaw -pi/2   x 0.0461  y -0.064 .. +0.0571  (handle on -y)
    #
    # which puts the front row's edges at [-0.097, 0.037], [0.104, 0.196],
    # [0.254, 0.387] and [0.454, 0.546]: the tightest pair clears by 0.057 m -- enough
    # for the Euclidean clustering in cram_vrb_lab.perception.pipeline to call them
    # separate objects -- and the rightmost edge keeps 0.154 m from the sink cut-out.
    # The rows clear each other by 0.169 m.
    #
    # The names are roles, not asset names, because two of these assets are placed
    # twice; see KitchenProp.name.
    # KitchenProp("cereal_box", (0.10, 7.55, KITCHEN_WORKTOP[2]),
    #             yaw=1.5707963267948966, asset="SM_CerealBox"),
    # KitchenProp("milk_box", (0.38, 7.55, KITCHEN_WORKTOP[2]),
    #             yaw=-1.5707963267948966, asset="SM_MilkBox"),
    KitchenProp("bowl_left", (-0.03, 7.23, KITCHEN_WORKTOP[2]),
                asset="SM_SmallBowl"),
    # -pi/2 turns the handle, which the mesh puts on +x, to face -y -- i.e. towards
    # the robot, which works this run from the open side of the room.
    KitchenProp("cup_left", (0.15, 7.20, KITCHEN_WORKTOP[2]),
                yaw=-1.5707963267948966, asset="SM_Cup"),
    KitchenProp("bowl_right", (0.32, 7.24, KITCHEN_WORKTOP[2]),
                asset="SM_SmallBowl"),
    KitchenProp("cup_right", (0.50, 7.18, KITCHEN_WORKTOP[2]),
                yaw=0, asset="SM_Cup"),
)
"""The six kitchen objects the Isaac side puts on the worktop when
:func:`kitchen_props_enabled`.

Cereal box, milk box and two bowl/cup place settings: the makings of one breakfast
task, which is why these assets were the ones cut out of the ICRA apartment. Two
cups and two bowls rather than one of each so a plan has to pick *which* cup, and
so a perception run has to separate two instances of the same object.

.. note::
   These carry their own physics, so the spawn must **not** re-apply a rigid body or
   a collider -- that would overwrite the authored masses and flatten the cup's and
   bowl's convex decompositions back to a single hull, filling in the handle and the
   bowl's cavity. The worktop is the half of the contact that *is* missing: it ships
   without a collider, and gets one at load time (see
   :func:`~cram_vrb_lab.scenes.garmi_apartment.isaac_scene.load_garmi_apartment_scene`).
"""


# --- Upstream's transport objects -------------------------------------------
#
# The bowl and the spoon that coraplex's own GARMI demonstration
# (``coraplex/demos/coraplex_garmi_demo/demo.py``) carries across this apartment.
# That demonstration spawns them into the *twin* only, as meshes at fixed poses, so
# a run against physics closes the hand on nothing: Isaac has no bowl and no spoon
# standing where the plan reaches. These constants put the same two meshes into the
# render, and both sides read them from here -- the pattern
# :mod:`cram_vrb_lab.scenes.props.constants` states outright, so the rendered rigid
# body and its twin counterpart agree by construction rather than by two people
# typing the same numbers.

TRANSPORT_OBJECTS_DIR = (
    CRAM_SUBMODULE_DIR / "coraplex" / "resources" / "objects"
)
"""Where upstream keeps the meshes, read in place rather than copied.

Deliberately *not* vendored into :data:`ASSETS_DIR` the way
:data:`KITCHEN_OBJECTS_DIR` is. Those four were cut out of an apartment this repo
owns; these two belong to coraplex, which is a submodule pinned at a commit. One
file on disk, read by the twin (``BodySpecification.mesh``) and by the Isaac side
(:func:`~cram_vrb_lab.scenes.garmi_apartment.isaac_scene.spawn_transport_props`), is
the only arrangement in which the two cannot drift apart.

The cost is that a USD conversion happens at every scene load instead of once. It is
a few thousand triangles; if that ever shows up in a startup profile, convert them
once into ``assets/`` and accept the copy.
"""

TRANSPORT_PROPS_ENV = "ISAAC_TRANSPORT_PROPS"
"""Environment variable that opts these objects in; see
:func:`transport_props_enabled`."""


def transport_props_enabled() -> bool:
    """Whether to put upstream's bowl and spoon into the scene. Off unless asked.

    An environment variable for the same reason :func:`kitchen_props_enabled` is one:
    it is how the notebooks already steer the Isaac process, and
    ``launcher.start_isaac_sim`` inherits it into the sim subprocess.

    Separate from :data:`KITCHEN_PROPS_ENV` because the two sets are for different
    demos and, on the worktop, for different runs -- see the warning on
    :data:`TRANSPORT_PROPS`.
    """
    return os.environ.get(TRANSPORT_PROPS_ENV, "0") == "1"


SDF_RESOLUTION = 256
"""Voxels along the longest axis of an ``sdf`` collider's signed distance field.

PhysX's own default is 256. It decides how faithfully a thin feature survives -- the
spoon's bowl is under 2 mm of shell in places, and at a coarse resolution the field
closes it into a solid lump. Cooking cost and memory both scale with the cube of
this, so raise it only against a measured contact problem.
"""

MAX_CONVEX_HULLS = 32
"""Hull budget for a ``convexDecomposition`` collider.

PhysX's default, and the same number ``assets/kitchen-objects`` settled on for the
cup and the bowl there; see that README for why shape *count* rather than triangle
count is what costs.
"""


@dataclass(frozen=True)
class TransportProp:
    """One of upstream's mesh objects, placed in this scene."""

    name: str
    """Prim name under
    :data:`~cram_vrb_lab.scenes.garmi_apartment.isaac_scene.TRANSPORT_PROPS_ROOT`, and
    the name the spawn report prints. Matches upstream's ``BOWL_NAME`` / ``SPOON_NAME``
    so the twin body and the rendered prim are called the same thing."""

    asset: str
    """File name inside :data:`TRANSPORT_OBJECTS_DIR`."""

    mass: float
    """[kg]. Authored rather than left to PhysX, and it has to be.

    Neither mesh is watertight (``trimesh`` reports ``is_watertight=False`` for both,
    and the spoon's ``volume`` comes out at exactly 0), so there is no enclosed volume
    for a density to act on and nothing to derive a mass from.
    """

    approximation: str
    """``physics:approximation`` for the collider. See :data:`TRANSPORT_PROPS`."""

    position: Tuple[float, float, float]
    """``(x, y, surface_z)`` in ``map``, exactly as in :class:`KitchenProp`.

    Always ``map``, and always the height of the *surface* the object rests on rather
    than of the object itself -- both spawns ground the mesh on its own measured
    bounding box. That holds for something in a drawer too; the surface is then the
    drawer's base plate.

    Stating it in ``map`` is not a style choice. It was drawer-relative at first,
    reusing upstream's own ``(-0.09, 0, -0.069)`` on the theory that both sides could
    resolve it against the drawer. They cannot: ``drawer_1``'s frame is rotated -90
    degrees about z relative to ``map`` (its ``map_T_drawer`` rotation is
    ``[[0,-1,0],[1,0,0],[0,0,1]]``), and the Isaac side has no equivalent frame to
    resolve against -- its drawer Xform sits near the world origin while the geometry
    is under the worktop, so the only thing it can measure is a world-*axis*-aligned
    bounding box, which silently applies the offset along the wrong axis. Measured on
    a real run: the spoon landed 208 mm from where the twin had it. ``map`` has no
    such ambiguity.
    """

    yaw: float = 0.0
    """Rotation [rad] about world Z. Both meshes are authored Z-up, so no roll."""

    inside: Optional[str] = None
    """Name of the body this object is *in*, e.g. ``"drawer_1"``.

    Only the twin acts on it, and only to choose a parent: a body hanging off the
    drawer travels with it when the plan pulls it open, which is what a kinematic
    world needs to be told. Isaac needs no equivalent -- the object is a rigid body
    resting in a real cavity, and contact carries it.

    Not used for placement on either side; see :attr:`position`.
    """

    @property
    def stl_path(self) -> str:
        """The mesh, as an absolute path."""
        return str(TRANSPORT_OBJECTS_DIR / self.asset)


TRANSPORT_PROPS = (
    TransportProp(
        "bowl",
        "bowl.stl",
        mass=0.058,
        approximation="sdf",
        position=(0.0, 7.2, KITCHEN_WORKTOP[2]),
    ),
    TransportProp(
        "spoon",
        "spoon.stl",
        mass=0.05,
        approximation="sdf",
        position=(-0.09, 7.313, 0.705),
        inside="drawer_1",
    ),
)
"""Upstream's two transported objects, placed against *this* scene's measurements.

Upstream states ``BOWL_POSE = (0, 7.2, 1.0)`` and hangs the spoon off ``drawer_1`` at
``(-0.09, 0, -0.069)``. Neither number is reused:

- ``1.0`` is not this worktop. It raycasts at :data:`KITCHEN_WORKTOP`'s 0.945, and the
  bowl mesh carries its origin at its bounding-box centre, so spawning at 1.0 leaves
  it hovering ~2 cm and dropping. The z here is the *surface*, and the spawn grounds
  the mesh on its own measured box -- see :func:`KitchenProp.position`.
- The spoon's ``(-0.09, 7.313, 0.705)`` is upstream's ``(-0.09, 0, -0.069)``
  *resolved*, not replaced. x and y are that offset put through the drawer's actual
  frame -- ``map_T_drawer`` is a -90 degree turn about z at ``(-0.09, 7.403, 0.8)``,
  so upstream's -0.09 along the drawer's x comes out as -0.09 along ``map``'s y. The z
  is the drawer's base plate measured off its own mesh (its vertices cluster at
  0.705 with the outer shell at 0.701), rather than upstream's -0.069, which sits
  16 mm above the plate -- harmless in a kinematic world, but in a physical one the
  spoon would drop those 16 mm and the two sides would disagree about where it is.

.. warning::
   The bowl stands on the same worktop as :data:`KITCHEN_PROPS`. It is placed in the
   free band between that set's two rows (front row at y in [7.18, 7.24], back row at
   y = 7.55), and 0.18 m clear of the sink cut-out that starts at x = 0.70 -- so the
   two sets *can* be enabled together. They are still two answers to the same
   question, and a plan that reaches for "the bowl" with both on has three bowls to
   choose from.

.. note::
   The spoon lands in a real cavity. ``drawer_1``'s collider is a
   ``convexDecomposition`` with ``shrinkWrap``, and the description says why in its
   own comment: "this drawer is an open box, and a single hull fills the cavity so
   nothing can be put inside it". So the spoon rests on the drawer's floor and is
   carried along by contact when the drawer opens -- it does not need parenting to
   the drawer the way the twin's does.

Both colliders are ``sdf``
==========================

The spoon has no real alternative. It measures 0.2167 x 0.0463 x **0.0189** m --
thin, concave and not watertight, which is the worst case for everything else:
``convexHull`` swallows it into a wedge, ``convexDecomposition`` on an open shell
yields near-degenerate hulls that jitter and tunnel, and the exact triangle mesh
(``"none"``) is not allowed on a dynamic rigid body at all. A signed distance field
is what PhysX 5 offers for exactly this shape.

The bowl is the interesting one, and it is here on purpose rather than by default.
``convexDecomposition`` would also work -- it is what ``assets/kitchen-objects``
settled on for ``SM_SmallBowl``, and both hold a cavity open where a single hull
would fill it in. What an SDF adds is that the *rim* stays a rim: a decomposition
approximates the 6 mm wall with hulls that bulge slightly inward and outward, and
the rim is exactly where the plan grasps this bowl (see
``demo_ori.py``'s ``GRASP_ORIGIN_IN_BOWL`` -- the fingers have to find 6 mm of wall,
because a Franka Hand opens to 0.08 m and the bowl is 0.1397 m across).

What it costs: SDF colliders cook at load and hold a voxel grid per shape
(:data:`SDF_RESOLUTION` cubed), and contact generation against one is dearer than
against a handful of hulls. This scene has two of them, which is nothing; a worktop
full would not be.

To go back, set ``approximation="convexDecomposition"`` on the bowl -- the spawn
reads it straight off this table and
:data:`MAX_CONVEX_HULLS` still applies.
"""
