"""garmi-apartment scene loading on the Isaac Sim side.

.. warning::
   Import only after :func:`cram_vrb_lab.sim.isaac_app.create_simulation_app`
   has run -- this module imports ``isaacsim.core`` at module scope.
"""

import numpy as np
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils import viewports
from isaacsim.core.utils.prims import create_prim, define_prim
from isaacsim.core.utils.rotations import euler_angles_to_quat
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from .constants import (
    FIXED_CAMERA_CLIPPING,
    FIXED_CAMERA_FOCAL_LENGTH,
    FIXED_CAMERA_RESOLUTION,
    FIXED_CAMERA_TARGET,
    FIXED_CAMERAS,
    GARMI_APARTMENT_USD_PATH,
    GRID_USD_PATH,
    KITCHEN_PROPS,
    PROP_DROP_HEIGHT,
    USD_PRIM_POSITION_IN_MAP,
    fixed_camera_field_of_view,
    fixed_cameras_enabled,
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

CABINET_PRIM = f"{APARTMENT_PRIM}/Meshes/Assets/cabinet/Actor_0000"
"""The kitchen run's articulated asset: the static carcase plus every door and drawer.

:data:`WORKTOP_MESH_PRIM` is its ``Static`` part; the doors and drawers are siblings
of it, one rigid body each with the geometry on a ``geom`` child.
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


FIXED_CAMERAS_ROOT = "/World/FixedCameras"
"""Prim the fixed camera rig is built under.

Outside ``/World/GarmiApartment`` for the same reason as :data:`KITCHEN_PROPS_ROOT`:
the apartment prim is a reference to ``world.usda``, whose conversion is the twin
giskard plans against, and nothing should be added inside it that the twin does not
know about. A camera would be invisible geometry to collide with.
"""

FIXED_CAMERA_SENSORS = {}
"""``{name: Camera}`` for the rig this module last built, or empty.

Module state because the only caller is :attr:`~cram_vrb_lab.specs.SceneSpec.load`,
whose return value :func:`cram_vrb_lab.sim.runner.build` discards -- the scene loader
is typed as building a stage, not as handing sensors back. :func:`spawn_fixed_cameras`
returns the same dict, so a caller that has one does not need this; it is here so that
a recorder added later (piece B of ``docs/vla-data-collection-assessment.md``) can
find the cameras without the scene interface growing a camera-shaped field that only
one scene would ever fill.
"""


def _look_at_orientation(eye, target, up=(0.0, 0.0, 1.0)):
    """Orientation ``(w, x, y, z)`` that puts a USD camera at ``eye`` looking at
    ``target``, with ``up`` up.

    A USD camera looks down its own -z with +y up, which is exactly the convention
    ``Gf.Matrix4d.SetLookAt`` is written for -- it builds the *view* matrix (world to
    eye), so the camera's own transform is its inverse and the rotation falls out of
    that. Done this way rather than by composing basis vectors by hand because the
    hand-rolled version has to get the handedness right in code that nothing type
    checks, and gets it silently wrong when it does not.

    :raises ValueError: if the view direction is parallel to ``up``, where the
        construction is degenerate -- ``SetLookAt`` answers that with a matrix rather
        than an error, and the camera ends up pointing somewhere arbitrary.
    """
    direction = np.asarray(target, dtype=float) - np.asarray(eye, dtype=float)
    norm = np.linalg.norm(direction)
    if norm == 0.0 or abs(np.dot(direction / norm, np.asarray(up, dtype=float))) > 0.999:
        raise ValueError(
            f"camera at {tuple(eye)} cannot look at {tuple(target)}: the view "
            f"direction is parallel to the up vector {tuple(up)}. Offset it "
            "sideways, or pass a different up."
        )
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(*up))
    quaternion = view.GetInverse().ExtractRotationQuat()
    return (quaternion.GetReal(), *quaternion.GetImaginary())


def spawn_fixed_cameras(world, render, cameras=FIXED_CAMERAS,
                        target=FIXED_CAMERA_TARGET):
    """Build the fixed camera rig: one camera per entry of ``cameras``, every one of
    them aimed at ``target``.

    Simpler than :func:`~cram_vrb_lab.robots.garmi.isaac_node.create_head_camera`,
    which is the template, in the one way that matters: these hang off ``/World``
    rather than off a link the articulation moves, so their pose is set once in world
    coordinates and nothing has to be expressed in a parent link's frame.

    The orientation is set on the prim rather than passed to ``Camera(orientation=)``
    for the reason ``create_head_camera`` documents: that argument goes through
    Isaac's own ROS<->USD camera axis conversion, and these quaternions are already
    in USD's.

    Each camera is an RTX render product that the sim renders **every cycle, whether
    or not anything reads a frame**, which is why the caller below only builds the rig
    when :func:`~cram_vrb_lab.scenes.garmi_apartment.constants.fixed_cameras_enabled`.

    :return: ``{name: Camera}``, also stored in :data:`FIXED_CAMERA_SENSORS`. Read a
        frame with ``camera.get_rgba()[..., :3]``.
    :raises ValueError: if two entries share a name, and so would share a prim path.
    """
    # Same check, and the same failure, as spawn_kitchen_props: two cameras sharing a
    # name share a prim path, so the second Camera would attach to the first's prim
    # and the rig would silently be one camera short.
    duplicates = sorted({c.name for c in cameras
                         if [d.name for d in cameras].count(c.name) > 1})
    if duplicates:
        raise ValueError(
            f"FixedCamera names must be unique -- {', '.join(duplicates)} used more "
            f"than once, and would collide under {FIXED_CAMERAS_ROOT}."
        )

    # Imported here rather than at module scope to keep the cost off every other demo
    # in this scene: isaacsim.sensors.camera pulls in the replicator, which is a
    # second or so of extension loading for a rig that is off by default.
    from isaacsim.sensors.camera import Camera

    stage = world.stage
    define_prim(FIXED_CAMERAS_ROOT, "Xform")
    width, height = FIXED_CAMERA_RESOLUTION
    sensors = {}
    for spec in cameras:
        prim_path = f"{FIXED_CAMERAS_ROOT}/{spec.name}"
        usd_camera = UsdGeom.Camera.Define(stage, prim_path)
        # Isaac only supports square pixels and re-derives the vertical aperture from
        # the render product's aspect ratio at initialize() -- with a carb warning per
        # camera, because the USD default (20.955 x 15.2908) is 4:3 and this rig is
        # 1:1. Setting it here makes the two agree before it ever looks.
        usd_camera.GetVerticalApertureAttr().Set(
            usd_camera.GetHorizontalApertureAttr().Get() * height / width
        )
        camera = Camera(
            prim_path=prim_path,
            frequency=30,
            resolution=FIXED_CAMERA_RESOLUTION,
        )
        position = spec.position_in(target)
        XFormPrim(prim_path).set_world_poses(
            positions=np.array([position]),
            orientations=np.array([_look_at_orientation(position, target)]),
        )
        camera.initialize()
        camera.set_focal_length(FIXED_CAMERA_FOCAL_LENGTH)
        camera.set_clipping_range(*FIXED_CAMERA_CLIPPING)
        sensors[spec.name] = camera
        # The aperture is read back rather than assumed: it is what turns the focal
        # length into an angle, and initialize() is free to have rewritten it to keep
        # pixels square.
        print(f"Fixed camera {spec.name}: at {position} looking at {tuple(target)}, "
              f"{width}x{height}, focal length {FIXED_CAMERA_FOCAL_LENGTH:g} "
              f"({fixed_camera_field_of_view(camera.get_horizontal_aperture()):.1f} "
              "deg fov)", flush=True)

    # The render product needs a few frames before get_rgba() returns anything but an
    # empty array -- the same wait create_head_camera does, and for the same reason.
    for _ in range(20):
        world.step(render=render)

    FIXED_CAMERA_SENSORS.clear()
    FIXED_CAMERA_SENSORS.update(sensors)
    return sensors


def load_garmi_apartment_scene(world, render, camera_eye=None, camera_target=None):
    """Ground grid, the apartment USD, the worktop objects, the fixed camera rig and
    the viewport's own view.

    The last two are different things and only one of them renders an image a program
    can read: ``camera_eye`` / ``camera_target`` frame the *viewport*, i.e. what a
    person watching the livestream sees, while :func:`spawn_fixed_cameras` builds
    sensors with render products behind them.

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

    # Three room-fixed cameras framing the worktop workspace, for recording
    # demonstrations. Opt-in: they are rendered every cycle from here on, and the
    # sim's cycle is what feeds giskard. See fixed_cameras_enabled.
    if fixed_cameras_enabled():
        if render:
            spawn_fixed_cameras(world, render)
        else:
            # An RTX camera has nothing to read out of a sim that is not rendering,
            # and would sit there costing a render product for empty frames. Said out
            # loud because the alternative is a recording that is silently all black.
            print("Fixed cameras: skipped -- ISAAC_RENDER=0, nothing is rendered",
                  flush=True)

    # Defaults lifted from world.usda's saved Perspective camera, i.e. the view
    # the scene was authored from: over the robot's shoulder into the living room.
    viewports.set_camera_view(
        eye=np.array(
            camera_eye if camera_eye is not None else [-2.8, 6.8, 1.6]
        ),
        target=np.array(
            camera_target if camera_target is not None else [0, 6.4, 1.0]
        ),
    )

    for _ in range(30):
        world.step(render=render)
