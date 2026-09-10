"""The Isaac half of :mod:`cram_vrb_lab.sim.scene_sync`: applies sync requests.

.. warning::
   Import only after :func:`cram_vrb_lab.sim.isaac_app.create_simulation_app` has
   run -- this module imports ``isaacsim.core`` at module scope.
"""

from __future__ import annotations

import json
import threading
from typing import Dict, List, Optional

import numpy as np
import trimesh
from isaacsim.core.prims import RigidPrim
from isaacsim.core.utils.prims import (
    create_prim,
    define_prim,
    delete_prim,
    is_prim_path_valid,
)
from pxr import Gf, UsdGeom, UsdPhysics
from std_msgs.msg import String

from cram_vrb_lab.sim.ros_utils import SimBridge, as_np
from cram_vrb_lab.sim.scene_sync import (
    SCENE_SYNC_ACK_TOPIC,
    SCENE_SYNC_POSE_TOPIC,
    SCENE_SYNC_TOPIC,
    SYNC_ROOT,
)

SYNCED_COLOR = (0.9, 0.4, 0.1)
"""RGB of a body this bridge created. Unlike anything the apartment or the prop
sets use, so it is obvious in the viewport which geometry came from the twin
rather than from the scene."""


class SceneSyncROS(SimBridge):
    """Forces prims to the poses the twin asks for, between physics steps.

    Requests arrive on :data:`~cram_vrb_lab.sim.scene_sync.SCENE_SYNC_TOPIC` and are
    **queued**, not applied, by the subscription callback: rclpy delivers on its own
    thread, and USD authoring and PhysX writes from a thread other than the one
    stepping the sim corrupt the stage. :meth:`apply_commands` is the sim loop's own
    hook, called between steps, which is the only safe place to touch either.

    That queue is also what makes a request atomic. Everything in one request lands
    in the same gap between two steps, so a scene the twin changed in several places
    never renders half-updated and no object is ever seen mid-flight.
    """

    receives_commands = True  # the whole point is the subscription

    def __init__(self, world, retune=None, robot=None):
        """
        :param retune: called after physics is restarted, to put the drive tuning
            back. **Deleting a prim mid-simulation requires stopping and replaying
            physics** (see :meth:`_delete`), and replaying makes PhysX re-read the
            drive parameters authored on the prims -- discarding every gain and force
            budget set through the tensor API. Without this the lift silently stops
            working the first time anything is deleted; see
            :func:`~cram_vrb_lab.robots.garmi.isaac_node.tune_drives`.
        """
        super().__init__("scene_sync_ros")
        self.world = world
        self.retune = retune
        self.robot = robot
        self._requests: List[Dict] = []
        self._lock = threading.Lock()
        self.create_subscription(String, SCENE_SYNC_TOPIC, self._on_request, 10)
        self._ack = self.create_publisher(String, SCENE_SYNC_ACK_TOPIC, 10)
        self._poses = self.create_publisher(String, SCENE_SYNC_POSE_TOPIC, 10)
        self._tracked: List[str] = []
        self._spawned: Dict[str, Dict] = {}
        self._view: Optional[RigidPrim] = None
        self._viewed: List[str] = []
        self._attached: Dict[str, Dict] = {}
        self._carry_view: Optional[RigidPrim] = None
        self._carry_viewed: List[str] = []
        define_prim(SYNC_ROOT, "Xform")

    def _release_view(self) -> None:
        """Drop the physics view before anything it covers is deleted.

        A ``RigidPrim`` is a PhysX *tensor view*, and deleting a prim while a view
        covers it does not just orphan that one entry -- it invalidates the whole
        ``simulationView``, and Isaac ends the process:

            prim '/World/SceneSync/tracked_box' was deleted while being used by a
            shape in a tensor view class. The physics.tensors simulationView was
            invalidated.

        Measured: a scene reset deleting a tracked object took the simulator down
        with it. Releasing first is what makes deletion safe, and it has to be
        every reference, which is why the view is cached rather than rebuilt each
        step -- a per-step temporary is still alive in PhysX's registration when
        the delete lands.
        """
        self._view = None
        self._viewed = []
        self._carry_view = None
        self._carry_viewed = []

    def _on_request(self, message: String) -> None:
        """Queue a request. Deliberately does no stage work; see the class docstring."""
        with self._lock:
            self._requests.append(json.loads(message.data))

    def apply_commands(self, dt: float) -> None:
        """Apply every queued request and acknowledge each one.

        Called by :func:`cram_vrb_lab.sim.runner.run` before ``world.step``, so the
        step that follows renders the scene the twin asked for -- and, because the
        poses are written rather than settled, renders it exactly.
        """
        with self._lock:
            requests, self._requests = self._requests, []
        for request in requests:
            # A request is arbitrary input from another process, and this runs on the
            # sim loop thread: an exception here does not fail the request, it takes
            # down the simulator. The runner catches only KeyboardInterrupt, so
            # anything else unwinds into its `finally`, and `simulation_app.close()`
            # ends the process before Python can print the traceback -- which is why
            # a malformed request looked like a clean, silent shutdown.
            try:
                report = self._apply(request)
            except Exception as failure:  # noqa: BLE001 - a bad request must not be fatal
                report = {
                    "id": request.get("id"),
                    "moved": [],
                    "created": [],
                    "removed": [],
                    "attached": [],
                    "detached": [],
                    "missing": [],
                    "error": f"{type(failure).__name__}: {failure}",
                }
                self.get_logger().error(f"scene sync failed: {report['error']}")
            self._ack.publish(String(data=json.dumps(report)))

        if self._attached:
            try:
                self._carry_attached()
            except Exception as failure:  # noqa: BLE001 - same reason as above
                self.get_logger().error(f"carry failed: {failure}")
                self._attached.clear()

    def _apply(self, request: Dict) -> Dict:
        report = {
            "id": request.get("id"),
            "moved": [],
            "created": [],
            "removed": [],
            "attached": [],
            "detached": [],
            "missing": [],
        }
        for entry in request.get("objects", []):
            name = entry["name"]
            path = f"{SYNC_ROOT}/{name}"
            existing = self.world.stage.GetPrimAtPath(path)
            if existing and not existing.IsActive():
                # Placed again after a remove: revive rather than rebuild, so the
                # name keeps meaning one object.
                existing.SetActive(True)
            if not is_prim_path_valid(path):
                if "size" not in entry and "mesh" not in entry:
                    # Asked to place something that is not here and that carries no
                    # shape. Inventing geometry would put a body in the render the
                    # twin never described, so say so instead.
                    report["missing"].append(name)
                    continue
                self._create(path, entry)
                report["created"].append(name)
            # Remembered so a reset can put it back where it was asked for; see
            # restore_spawned.
            self._spawned[name] = {
                "position": list(entry["position"]),
                "orientation": list(entry["orientation"]),
            }
            self._place(path, entry)
            report["moved"].append(name)
            # Opt-in per object and per request, so "who owns this pose" is answered
            # where the object is placed rather than by a mode set once elsewhere.
            if entry.get("track") and name not in self._tracked:
                self._tracked.append(name)
            elif not entry.get("track") and name in self._tracked:
                self._tracked.remove(name)

        for entry in request.get("attach", []):
            name = entry["name"]
            if self._attach(name, entry["link"]):
                report["attached"].append(name)
            else:
                report["missing"].append(name)

        for name in request.get("detach", []):
            if self._detach(name):
                report["detached"].append(name)

        for name in request.get("remove", []):
            path = f"{SYNC_ROOT}/{name}"
            # Scoped to SYNC_ROOT by construction: a request cannot name its way out
            # to the apartment or the robot, whatever string it sends.
            if is_prim_path_valid(path):
                self._delete(path)
                report["removed"].append(name)
            self._spawned.pop(name, None)
            if name in self._tracked:
                self._tracked.remove(name)
            self._detach(name)
        return report

    @staticmethod
    def _matrix(position, orientation) -> np.ndarray:
        """4x4 from a position and a ``(w, x, y, z)`` quaternion."""
        w, x, y, z = (float(value) for value in orientation)
        return np.array(
            Gf.Matrix4d(
                Gf.Rotation(Gf.Quatd(w, Gf.Vec3d(x, y, z))),
                Gf.Vec3d(*(float(value) for value in position)),
            ),
            dtype=float,
        ).T

    @staticmethod
    def _quaternion(matrix: np.ndarray) -> List[float]:
        """The rotation of a 4x4, as ROS ``(x, y, z, w)``."""
        rotation = Gf.Matrix4d(*matrix.T.ravel().tolist()).GetOrthonormalized()
        quaternion = rotation.ExtractRotationQuat()
        imaginary = quaternion.GetImaginary()
        return [float(v) for v in imaginary] + [float(quaternion.GetReal())]

    def _link_poses(self, paths: List[str]) -> Dict[str, np.ndarray]:
        """Link poses off the robot's own articulation view.

        Never a ``RigidPrim``: it decides whether it holds articulation links from
        ``_prim_paths[0]`` alone, and when it decides wrong its constructor rewrites
        every prim's xform ops -- which deletes the ``xformOp:transform`` the URDF
        importer authored and takes the link out of the render.
        """
        if self.robot is None:
            return {}
        transforms = as_np(self.robot._physics_view.get_link_transforms()).reshape(-1, 7)
        index_of = {n.split("/")[-1]: i for i, n in enumerate(self.robot.body_names)}
        poses = {}
        for path in paths:
            index = index_of.get(path.split("/")[-1])
            if index is not None:
                t = transforms[index]
                # this view is (x, y, z, w); _matrix takes (w, x, y, z)
                poses[path] = self._matrix(t[:3], (t[6], t[3], t[4], t[5]))
        return poses

    def _object_poses(self, paths: List[str]) -> Dict[str, np.ndarray]:
        """Synced-object poses, from a cached view over prims this bridge created."""
        if self._carry_viewed != paths:
            self._carry_view = None
            self._carry_viewed = []
            view = RigidPrim(paths, name="scene_sync_carry", reset_xform_properties=False)
            # else get_world_poses silently falls back to the stale stage
            view.initialize()
            self._carry_view, self._carry_viewed = view, paths
        positions, orientations = self._carry_view.get_world_poses()
        return {
            path: self._matrix(positions[i], orientations[i])
            for i, path in enumerate(paths)
        }

    @staticmethod
    def _set_kinematic(prim, kinematic: bool) -> None:
        UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr().Set(bool(kinematic))

    def _robot_link_paths(self, link_path: str) -> List[str]:
        """Every link prim of the articulation ``link_path`` belongs to.

        Read off the articulation rather than assumed from the carry link alone: the
        object hangs from one link, but the ones it is actually in contact with are
        that link's siblings -- the two fingers -- and while the base drives it can
        reach the torso as well.

        Falls back to the carry link if the robot is unknown, which filters the pair
        that matters most rather than nothing at all.
        """
        root = link_path.rsplit("/", 1)[0]
        names = getattr(self.robot, "body_names", None) or []
        paths = [f"{root}/{str(name).split('/')[-1]}" for name in names]
        return [path for path in paths if is_prim_path_valid(path)] or [link_path]

    def _filter_collisions(self, path: str, against: List[str]) -> None:
        """Stop ``path`` colliding with ``against`` for as long as it is carried.

        Without this the weld and the gripper fight each other, and the arm loses.
        A carried object is kinematic, so it has infinite mass and every contact it
        has with the hand is resolved entirely on the *robot's* side. Meanwhile the
        fingers are told to keep pressing -- they are
        :class:`~cram_vrb_lab.sim.velocity_integrator.StreamedVelocityIntegrator`
        holding joints, whose leading target *is* the grip force -- and
        :meth:`_carry_attached` writes the object to the link pose of the step
        *before* the one about to run, so it trails the hand by one step of its
        motion. The solver answers the overlap by shoving the articulation, harder
        the faster the hand moves, which is what the jitter is.

        A weld has no such contact to resolve, and ``physics:filteredPairs`` is how
        USD says so: the pair never reaches narrow phase, so the hand can hold the
        mesh exactly where it was grasped without the two pushing at each other. The
        object goes on colliding with everything else on the stage.
        """
        prim = self.world.stage.GetPrimAtPath(path)
        UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel().SetTargets(
            list(against)
        )

    def _unfilter_collisions(self, path: str) -> None:
        """Give ``path`` its collisions with the robot back, on release.

        Cleared wholesale rather than target by target: this bridge authored the prim
        and is the only thing that filters pairs on it.
        """
        prim = self.world.stage.GetPrimAtPath(path)
        if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
            UsdPhysics.FilteredPairsAPI(prim).CreateFilteredPairsRel().SetTargets([])

    def _attach(self, name: str, link_path: str) -> bool:
        """Weld ``name`` onto ``link_path``. The offset is measured on the next step.

        Three writes make the weld: the pair is filtered so the robot and the object
        cannot push each other, the object turns kinematic so nothing else can move it
        either, and :meth:`_carry_attached` then carries it on the link from the sim's
        own measurements. The twin is not consulted again until it says the carry
        ended -- deliberately, since a pose crossing the boundary every step is a
        round trip the render cannot afford and a source of disagreement the weld does
        not need.
        """
        path = f"{SYNC_ROOT}/{name}"
        if not is_prim_path_valid(path) or not self._link_poses([link_path]):
            return False
        self._attached[name] = {"link": link_path, "link_T_object": None}
        self._filter_collisions(path, self._robot_link_paths(link_path))
        self._set_kinematic(self.world.stage.GetPrimAtPath(path), True)
        return True

    def _detach(self, name: str) -> bool:
        if self._attached.pop(name, None) is None:
            return False
        path = f"{SYNC_ROOT}/{name}"
        if is_prim_path_valid(path):
            prim = self.world.stage.GetPrimAtPath(path)
            self._unfilter_collisions(path)
            self._set_kinematic(prim, False)
            for attribute in ("physics:velocity", "physics:angularVelocity"):
                if prim.HasAttribute(attribute):
                    prim.GetAttribute(attribute).Set(Gf.Vec3f(0.0))
        return True

    def _carry_attached(self) -> None:
        """Put every attached object back on its link. Kinematic, so this is the only
        thing that moves it."""
        alive = {}
        for name, carry in list(self._attached.items()):
            path = f"{SYNC_ROOT}/{name}"
            if is_prim_path_valid(path):
                alive[name] = path
            else:
                self._attached.pop(name, None)
        if not alive:
            self._carry_view, self._carry_viewed = None, []
            return

        object_poses = self._object_poses(list(alive.values()))
        link_poses = self._link_poses(
            list({self._attached[n]["link"] for n in alive})
        )
        for name, path in alive.items():
            carry = self._attached[name]
            world_T_link = link_poses.get(carry["link"])
            if world_T_link is None:
                self._detach(name)
            elif carry["link_T_object"] is None:
                carry["link_T_object"] = np.linalg.inv(world_T_link) @ object_poses[path]
            else:
                world_T_object = world_T_link @ carry["link_T_object"]
                self._place(
                    path,
                    {
                        "position": world_T_object[:3, 3].tolist(),
                        "orientation": self._quaternion(world_T_object),
                    },
                )

    def _delete(self, path: str) -> None:
        """Remove a prim, around the stop/play that makes it survivable.

        Deleting a rigid body while physics runs invalidates the **global**
        ``physics.tensors`` simulationView -- not merely a view covering that prim.
        Every wrapper built on it loses its handle, the robot's ``Articulation``
        included, and the next read of the robot dies with ``AttributeError:
        'Articulation' object has no attribute '_physics_view'``. Both
        ``stage.RemovePrim`` and ``delete_prim`` do it; the Kit command routes
        through PhysX, but routing through PhysX *is* what invalidates the view.

        ``play()`` "does one step internally to propagate all physics handles
        properly", which is what rebuilds it -- and then :attr:`retune` puts back
        what replaying discarded.
        """
        self._release_view()
        was_playing = self.world.is_playing()
        if was_playing:
            self.world.stop()
        delete_prim(path)
        if was_playing:
            self.world.play()
            if self.retune is not None:
                self.retune()

    def delete_spawned(self) -> List[str]:
        """Delete every object this bridge created; returns their names.

        What a scene reset wants: objects a plan spawned postdate the reset
        snapshot, so restoring that snapshot says nothing about them and they would
        survive as leftovers of a run that is over.

        One stop/play for the whole set rather than one per object, because each is
        a physics restart and each costs the drive tuning a round trip.
        """
        names = [
            prim.GetName()
            for prim in self.world.stage.GetPrimAtPath(SYNC_ROOT).GetChildren()
        ]
        if not names:
            return []
        self._release_view()
        was_playing = self.world.is_playing()
        if was_playing:
            self.world.stop()
        for name in names:
            delete_prim(f"{SYNC_ROOT}/{name}")
        if was_playing:
            self.world.play()
            if self.retune is not None:
                self.retune()
        self._tracked.clear()
        self._spawned.clear()
        self._attached.clear()
        return names

    def restore_spawned(self) -> List[str]:
        """Put every object this bridge created back where it was asked for.

        What a scene reset needs from here. Objects a *plan* spawned postdate the
        reset snapshot, so restoring that snapshot says nothing about them and they
        would otherwise keep whatever pose physics left them in.

        **Restored, not deleted, and that is not a preference.** Deleting a
        rigid-body prim while the simulation is running invalidates the *global*
        ``physics.tensors`` simulationView -- not just a view covering that prim.
        Every wrapper built on it loses its handle, the robot's ``Articulation``
        included, and the next read of the robot dies with::

            prim '/World/SceneSync/bowl' was deleted while being used by a tensor
            view class. The physics.tensors simulationView was invalidated.
            ...
            AttributeError: 'Articulation' object has no attribute '_physics_view'

        (``Articulation`` *deletes* that attribute on teardown -- see
        ``articulation.py``'s ``del self._physics_view`` -- which is why it reads as
        a missing attribute rather than a ``None``. It comes back only on the next
        physics-ready event.) Releasing this bridge's own view first is not enough,
        because the view that matters is the global one.

        Repeating a task therefore does not need the objects gone: a spawn names the
        state it wants, so spawning the same name again moves the object that is
        already there rather than adding a second. Nothing accumulates.
        """
        for name in list(self._attached):
            self._detach(name)
        restored = []
        for name, pose in self._spawned.items():
            path = f"{SYNC_ROOT}/{name}"
            if not is_prim_path_valid(path):
                continue
            self._place(path, pose)
            restored.append(name)
        if restored:
            self._release_view()
            bodies = RigidPrim([f"{SYNC_ROOT}/{n}" for n in restored])
            bodies.set_velocities(np.zeros((len(restored), 6)))
        return restored

    def _create(self, path: str, entry: Dict) -> None:
        """Build a box of the twin's extents as a dynamic rigid body.

        A box because that is what the twin has to offer: the bodies this interface
        exists for are ``add_boxes``' perceived detections, which *are* boxes -- the
        perception pipeline fits an oriented bounding box and claims nothing more.
        Rendering a box is therefore not an approximation of the twin, it is the
        twin.

        ``convexHull`` rather than a decomposition for the same reason: a box is
        already convex, so the cheap approximation is also the exact one.
        """
        # A unit cube scaled to the extents, so one prim type covers every size --
        # and the scale is handed to create_prim rather than added afterwards.
        # create_prim already authors translate, orient *and* scale, so a later
        # AddScaleOp is a duplicate and USD raises on it: "The xformOp
        # 'xformOp:scale' already exists in xformOpOrder". That threw on the sim
        # loop thread, which used to end the simulator outright.
        if "mesh" in entry:
            prim = self._create_mesh(path, entry)
        else:
            create_prim(
                prim_path=path,
                prim_type="Cube",
                attributes={"size": 1.0},
                scale=[float(value) for value in entry["size"]],
            )
            prim = self.world.stage.GetPrimAtPath(path)
        UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*SYNCED_COLOR)])

        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(
            entry.get(
                "collider", "convexHull" if "mesh" not in entry else "convexDecomposition"
            )
        )
        UsdPhysics.RigidBodyAPI.Apply(prim)
        if "mass" in entry:
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(entry["mass"]))

    def _create_mesh(self, path: str, entry: Dict):
        """Author a ``UsdGeom.Mesh`` from the mesh file the twin loaded.

        Read with trimesh and written out as points and face indices rather than
        converted into a USD file kept alongside, so **one** copy of the geometry
        exists and the render cannot drift from the plan. This is the same approach
        as :func:`~cram_vrb_lab.scenes.garmi_apartment.isaac_scene._stl_mesh_prim`,
        which is what a hardcoded prop table uses; going through here instead is
        what lets a plan spawn its own objects rather than pick from that table.

        ``force="mesh"`` because a file with several solids loads as a ``Scene``;
        concatenating is what a caller means either way. The stage is authored in
        metres, as are these files, so the vertices go in unscaled.
        """
        mesh = trimesh.load(entry["mesh"], force="mesh")
        geometry = UsdGeom.Mesh.Define(self.world.stage, path)
        geometry.CreatePointsAttr(
            [Gf.Vec3f(*(float(value) for value in point)) for point in mesh.vertices]
        )
        geometry.CreateFaceVertexIndicesAttr(
            [int(index) for index in np.asarray(mesh.faces).ravel()]
        )
        geometry.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
        geometry.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        prim = geometry.GetPrim()
        # _place writes through whichever ops exist, so give it the pair it expects.
        UsdGeom.Xformable(prim).AddTranslateOp()
        UsdGeom.Xformable(prim).AddOrientOp()
        return prim

    def _place(self, path: str, entry: Dict) -> None:
        """Write the pose onto the prim and stop it carrying any motion.

        The velocity reset is the half that is easy to forget and impossible to see
        until something drifts: a rigid body keeps the linear and angular velocity it
        had before it was moved, so a teleport alone hands PhysX an object that is
        somewhere new *and* still travelling. Zeroing both is what makes this a
        placement rather than a shove.
        """
        prim = self.world.stage.GetPrimAtPath(path)
        xformable = UsdGeom.Xformable(prim)

        position = [float(value) for value in entry["position"]]
        x, y, z, w = (float(value) for value in entry["orientation"])

        operations = {op.GetOpName(): op for op in xformable.GetOrderedXformOps()}
        translate = operations.get("xformOp:translate") or xformable.AddTranslateOp()
        orient = operations.get("xformOp:orient") or xformable.AddOrientOp()

        # Written at whatever precision the op was *authored* at, not at one this
        # module picks. ``create_prim`` authors these as double, and USD refuses a
        # value of the other width outright -- "Type mismatch for xformOp:orient:
        # expected 'GfQuatd', got 'GfQuatf'" -- so assuming either way breaks half
        # the prims this can be pointed at.
        if translate.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
            translate.Set(Gf.Vec3d(*position))
        else:
            translate.Set(Gf.Vec3f(*position))

        # Isaac and USD take (w, x, y, z); the wire carries ROS (x, y, z, w).
        if orient.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
            orient.Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
        else:
            orient.Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))

        for attribute, zero in (
            ("physics:velocity", Gf.Vec3f(0.0)),
            ("physics:angularVelocity", Gf.Vec3f(0.0)),
        ):
            if prim.HasAttribute(attribute):
                prim.GetAttribute(attribute).Set(zero)

    def publish(self) -> None:
        """Report where physics has taken the tracked objects.

        Silent unless something was placed with ``track=True``, which is what keeps
        the Isaac -> twin direction opt-in: a scene nobody asked to track publishes
        nothing at all, and the two sides stay in the one-way relationship the rest
        of this repo assumes.
        """
        if not self._tracked:
            return
        try:
            self._publish_tracked()
        except Exception as failure:  # noqa: BLE001 - same reason as apply_commands
            self.get_logger().error(f"pose report failed: {failure}")
            self._tracked.clear()

    def _publish_tracked(self) -> None:
        alive = [
            (name, path)
            for name, path in ((n, f"{SYNC_ROOT}/{n}") for n in self._tracked)
            if is_prim_path_valid(path)
        ]
        if not alive:
            self._release_view()
            return

        paths = [path for _, path in alive]
        # Rebuilt only when the set changes: see _release_view for why a fresh view
        # every step is not merely wasteful but unsafe.
        if paths != self._viewed:
            self._release_view()
            self._view = RigidPrim(paths)
            self._viewed = paths
        positions, orientations = self._view.get_world_poses()
        report = {}
        for index, (name, _) in enumerate(alive):
            x, y, z = (float(value) for value in positions[index])
            # Isaac hands back (w, x, y, z); the wire carries ROS (x, y, z, w), the
            # order every twin boundary in this repo takes.
            w, qx, qy, qz = (float(value) for value in orientations[index])
            report[name] = {
                "position": [x, y, z],
                "orientation": [qx, qy, qz, w],
            }
        self._poses.publish(String(data=json.dumps(report)))
