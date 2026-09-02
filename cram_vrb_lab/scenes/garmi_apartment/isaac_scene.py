"""garmi-apartment scene loading on the Isaac Sim side.

.. warning::
   Import only after :func:`cram_vrb_lab.sim.isaac_app.create_simulation_app`
   has run -- this module imports ``isaacsim.core`` at module scope.
"""

import numpy as np
from isaacsim.core.utils import viewports
from isaacsim.core.utils.prims import create_prim, define_prim
from isaacsim.core.utils.rotations import euler_angles_to_quat
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from .constants import (
    GARMI_APARTMENT_USD_PATH,
    GRID_USD_PATH,
    KITCHEN_PROPS,
    PROP_DROP_HEIGHT,
    USD_PRIM_POSITION_IN_MAP,
    kitchen_props_enabled,
)

APARTMENT_PRIM = "/World/GarmiApartment"

WORKTOP_MESH_PRIM = f"{APARTMENT_PRIM}/Meshes/Assets/cabinet/Actor_0000/Static/geom"
"""The kitchen run's static mesh -- carcase, plinth and worktop in one mesh.

The surface the objects on :data:`~cram_vrb_lab.scenes.garmi_apartment.constants.KITCHEN_WORKTOP`
rest on. It ships without a collider, so :func:`load_garmi_apartment_scene` adds one;
without it anything released above the worktop falls straight through the cabinet
onto the floor. The drawer fronts and cabinet doors are separate prims under the same
asset and are left alone -- nothing is standing on those.
"""

KITCHEN_PROPS_ROOT = "/World/KitchenProps"
"""Prim the kitchen objects are spawned under, i.e. outside /World/GarmiApartment.

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

    Measured rather than tabulated: how far an asset's origin sits above a table
    depends on the mesh and on the rotation it was spawned with. Measuring the rotated
    box means the placement constants stay surface heights (see
    :class:`~cram_vrb_lab.scenes.garmi_apartment.constants.KitchenProp`) and swapping
    in another asset needs no new numbers.
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


def spawn_kitchen_props(world, render, props=KITCHEN_PROPS):
    """Put the kitchen objects in :data:`~cram_vrb_lab.scenes.garmi_apartment.constants.KITCHEN_PROPS`
    on the worktop and settle them.

    Each kitchen asset is referenced as a complete physics body --
    ``PhysicsRigidBodyAPI``, an authored mass and inertia tensor, and its own collider
    (``convexDecomposition`` for the cup and the bowl, so the handle opening and the
    bowl's cavity survive; see ``assets/kitchen-objects/README.md``). So nothing here
    applies a rigid body or a collider: doing so would overwrite those masses and
    collapse the decompositions to a single hull.

    What is still needed is a collider on the worktop, which the apartment ships
    without, and grounding each object on its own measured bounding box so the
    constants can stay surface heights.

    :return: ``{name: (settled_centre, size)}`` in ``map``, and prints the same --
    where an object came to rest is only knowable from the simulation.
    :raises ValueError: if two props share a name, and so would share a prim path.
    """
    # Checked before anything is created, because the failure it catches is not
    # obvious from what Isaac reports: two props sharing a name share a prim path,
    # and the second create_prim lands on the prim the first one made rather than
    # adding an object. Give the copy its own name and an explicit ``asset``.
    duplicates = sorted({p.name for p in props
                         if [q.name for q in props].count(p.name) > 1})
    if duplicates:
        raise ValueError(
            f"KitchenProp names must be unique -- {', '.join(duplicates)} used more "
            f"than once, and would collide under {KITCHEN_PROPS_ROOT}. Name each "
            "copy separately and pass asset= to say which USD it loads."
        )

    stage = world.stage
    # Idempotent: load_garmi_apartment_scene has already applied this, but this
    # function has to stand on its own when called directly -- without the collider
    # every object falls through to the floor.
    _add_static_collider(stage.GetPrimAtPath(WORKTOP_MESH_PRIM))

    define_prim(KITCHEN_PROPS_ROOT, "Xform")
    sizes = {}
    for prop in props:
        prim = create_prim(
            prim_path=f"{KITCHEN_PROPS_ROOT}/{prop.name}",
            usd_path=prop.usd_path,
            position=np.array(prop.position),  # z corrected below
            orientation=euler_angles_to_quat([0.0, 0.0, prop.yaw]),
        )
        sizes[prop.name] = _release_above_surface(
            stage, prim, prop.position[2], PROP_DROP_HEIGHT
        )

    world.reset()
    for _ in range(120):
        world.step(render=render)

    placed = {}
    for prop in props:
        minimum, maximum = _world_aabb(stage, stage.GetPrimAtPath(
            f"{KITCHEN_PROPS_ROOT}/{prop.name}"))
        centre = tuple(round(float((lo + hi) / 2), 4)
                       for lo, hi in zip(minimum, maximum))
        placed[prop.name] = (centre, sizes[prop.name])
        print(f"Kitchen prop {prop.name}: released at {prop.position} + "
              f"{PROP_DROP_HEIGHT} m, settled centre {centre}, "
              f"size {sizes[prop.name]}", flush=True)
    return placed


def load_garmi_apartment_scene(world, render, camera_eye=None, camera_target=None):
    """Ground grid, the apartment USD, the worktop objects, and the camera view.

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

    # The worktop ships without a collider, so anything released above it falls
    # through the cabinet onto the floor. Applied unconditionally rather than left to
    # the prop spawn: it is a property of the scene, and every caller that puts
    # something on the worktop needs it. spawn_kitchen_props asks for it again, which
    # is free -- _add_static_collider is idempotent.
    _add_static_collider(world.stage.GetPrimAtPath(WORKTOP_MESH_PRIM))

    # The apartment ships no graspable objects at all, so perception gets nothing to
    # find; these put cup, bowl, cereal and milk on the worktop left of the sink
    # without touching either the USD or its MJCF twin. Opt-in, because the other
    # garmi demos were tuned against a bare worktop -- demos/garmi_demo.py sets
    # ISAAC_KITCHEN_PROPS=1.
    if kitchen_props_enabled():
        spawn_kitchen_props(world, render)

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
