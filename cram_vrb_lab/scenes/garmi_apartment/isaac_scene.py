"""garmi-apartment scene loading on the Isaac Sim side.

.. warning::
   Import only after :func:`cram_vrb_lab.sim.isaac_app.create_simulation_app`
   has run -- this module imports ``isaacsim.core`` at module scope.
"""

import numpy as np
from isaacsim.core.utils import viewports
from isaacsim.core.utils.prims import create_prim, define_prim
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from .constants import (
    GARMI_APARTMENT_USD_PATH,
    GRID_USD_PATH,
    USD_PRIM_POSITION_IN_MAP,
    YCB_ASSET_DIR,
    YCB_DROP_HEIGHT,
    YCB_PROPS,
    YCB_UPRIGHT_ROLL,
)

APARTMENT_PRIM = "/World/GarmiApartment"

WORKTOP_MESH_PRIM = f"{APARTMENT_PRIM}/Meshes/Assets/cabinet/Actor_0000/Static/geom"
"""The kitchen run's static mesh -- carcase, plinth and worktop in one mesh.

The surface the objects on :data:`~cram_vrb_lab.scenes.garmi_apartment.constants.KITCHEN_WORKTOP`
rest on. It ships without a collider, so :func:`spawn_ycb_props` adds one; see there.
The drawer fronts and cabinet doors are separate prims under the same asset and are
left alone -- nothing is standing on those.
"""

YCB_PROPS_ROOT = "/World/YCBProps"
"""Prim the tabletop objects are spawned under, i.e. outside /World/GarmiApartment.

Kept separate from the apartment's own prim on purpose: the apartment is a reference
to ``world.usda`` and its twin ``scene-bodies.xml`` is a conversion of that same file,
so anything added inside it would be geometry the digital twin does not know about.
These objects are meant to be *found by perception*, not read out of the twin.
"""

def _world_aabb(stage, prim):
    """Axis-aligned bounding box of ``prim`` in world coordinates, as (min, max).

    Computed over the ``default`` and ``render`` purposes, i.e. what the camera sees.
    A fresh :class:`pxr.UsdGeom.BBoxCache` per call, because the callers below move
    prims between calls and a cache would answer from the pose before the move.
    """
    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    aabb = box.ComputeWorldBound(prim).ComputeAlignedRange()
    return aabb.GetMin(), aabb.GetMax()


def _release_above_surface(stage, prim, surface_z, drop_height):
    """Lift ``prim`` so the bottom of its bounding box is ``drop_height`` above
    ``surface_z``, and return that bounding box's size.

    Measured rather than tabulated: a YCB asset's origin is at the centre of its
    bounding box, so how far its centre sits above a table depends on the mesh and on
    the rotation it was spawned with. Measuring the rotated box means the placement
    constants stay surface heights (see
    :class:`~cram_vrb_lab.scenes.garmi_apartment.constants.YCBProp`) and swapping in
    another asset needs no new numbers.
    """
    minimum, maximum = _world_aabb(stage, prim)
    translate = prim.GetAttribute("xformOp:translate")
    position = translate.Get()
    translate.Set(
        Gf.Vec3d(
            position[0],
            position[1],
            position[2] + surface_z + drop_height - minimum[2],
        )
    )
    return tuple(round(float(hi - lo), 4) for lo, hi in zip(minimum, maximum))


def _add_static_collider(prim):
    """Give ``prim``'s mesh a static collider: the full triangle mesh, no rigid body.

    ``approximation = "none"`` means PhysX uses the triangles as they are, which only
    static geometry may do -- and is what a worktop with a sink cut out of it needs,
    since any convex approximation would fill the sink in and round the edges off.
    """
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")


def _make_rigid_body(prim, mass):
    """Turn a referenced YCB asset into a dynamic rigid body of ``mass`` [kg].

    The assets are geometry only, so both halves have to be added: the rigid body (and
    with it gravity) on the reference's root, and a collider on each mesh underneath
    it. ``convexHull`` rather than the exact triangles because a dynamic body must be
    convex in PhysX; every one of these four objects is close enough to convex that a
    hull is a fair stand-in, the banana's inner curve being the worst case.
    """
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
    for mesh in (p for p in Usd.PrimRange(prim) if p.IsA(UsdGeom.Mesh)):
        UsdPhysics.CollisionAPI.Apply(mesh)
        UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr("convexHull")


def spawn_ycb_props(world, render, props=YCB_PROPS):
    """Drop the YCB objects in :data:`YCB_PROPS` onto their surfaces and settle them.

    Referenced straight from the Isaac Sim assets root, made into rigid bodies, rolled
    upright, yawed, released :data:`~cram_vrb_lab.scenes.garmi_apartment.constants.YCB_DROP_HEIGHT`
    above their surface and then stepped until they stop moving. The worktop gets a
    collider first, because it has none of its own -- without it the two kitchen
    objects fall through the cabinet and land on the floor.

    :return: ``{name: (settled_centre, size)}`` for every object, in ``map``. Printed
        as well: the release poses are known from the constants, but where an object
        actually came to rest is only knowable from the simulation, and those centres
        are the ground truth a detection gets scored against.
    """
    assets_root = get_assets_root_path()
    if assets_root is None:
        raise RuntimeError(
            "Isaac Sim assets root not reachable, so the YCB props cannot be "
            f"referenced from {YCB_ASSET_DIR}. Everything else in this scene is local "
            "to the repo; these four objects are the stock Isaac assets."
        )

    stage = world.stage
    _add_static_collider(stage.GetPrimAtPath(WORKTOP_MESH_PRIM))

    define_prim(YCB_PROPS_ROOT, "Xform")
    sizes = {}
    for prop in props:
        prim = create_prim(
            prim_path=f"{YCB_PROPS_ROOT}/{prop.name}",
            usd_path=f"{assets_root}{YCB_ASSET_DIR}/{prop.asset}",
            position=np.array(prop.position),  # z corrected below
            orientation=euler_angles_to_quat([YCB_UPRIGHT_ROLL, 0.0, prop.yaw]),
        )
        _make_rigid_body(prim, prop.mass)
        sizes[prop.name] = _release_above_surface(
            stage, prim, prop.position[2], YCB_DROP_HEIGHT
        )

    # reset() registers the new rigid bodies with the physics scene; the steps then let
    # them fall the few millimetres onto their surface and stop bouncing.
    world.reset()
    for _ in range(120):
        world.step(render=render)

    placed = {}
    for prop in props:
        minimum, maximum = _world_aabb(stage, stage.GetPrimAtPath(
            f"{YCB_PROPS_ROOT}/{prop.name}"))
        centre = tuple(round(float((lo + hi) / 2), 4)
                       for lo, hi in zip(minimum, maximum))
        placed[prop.name] = (centre, sizes[prop.name])
        print(f"YCB prop {prop.name}: released at {prop.position} + "
              f"{YCB_DROP_HEIGHT} m, settled centre {centre}, "
              f"size {sizes[prop.name]}", flush=True)
    return placed


def load_garmi_apartment_scene(world, render, camera_eye=None, camera_target=None):
    """Ground grid, the apartment USD, the tabletop objects, and the camera view.

    :param camera_eye: viewport camera position; defaults to the view saved in
        ``world.usda``'s own ``customLayerData``, which frames the living room.
    :param camera_target: what the viewport looks at.
    """
    # Ground
    define_prim("/World/Ground", "Xform").GetReferences().AddReference(GRID_USD_PATH)
    UsdGeom.Imageable(
        world.stage.GetPrimAtPath("/World/Ground")
    ).MakeInvisible()

    # Apartment. Placed at the origin so the render and the MJCF twin share one
    # coordinate frame; see constants.USD_PRIM_POSITION_IN_MAP.
    create_prim(
        usd_path=GARMI_APARTMENT_USD_PATH,
        prim_path="/World/GarmiApartment",
        position=np.array(USD_PRIM_POSITION_IN_MAP),
    )

    # The apartment ships no graspable objects at all, so perception gets nothing to
    # find; these four put objects on the worktop and the dining table without
    # touching either the USD or its MJCF twin.
    spawn_ycb_props(world, render)

    # Defaults lifted from world.usda's saved Perspective camera, i.e. the view
    # the scene was authored from: over the robot's shoulder into the living room.
    viewports.set_camera_view(
        eye=np.array(
            camera_eye if camera_eye is not None else [-1.0, 2.0, 1.5]
        ),
        target=np.array(
            camera_target if camera_target is not None else [1, 6.0, 0.8]
        ),
    )

    for _ in range(30):
        world.step(render=render)
