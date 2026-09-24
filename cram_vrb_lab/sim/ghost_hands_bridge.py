"""The Isaac half of :mod:`cram_vrb_lab.sim.ghost_hands`: show the hands, drag with them.

.. warning::
   Import only after :func:`cram_vrb_lab.sim.isaac_app.create_simulation_app` has
   run -- this module imports ``isaacsim.core`` at module scope.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from isaacsim.core.prims import RigidPrim
from isaacsim.core.utils.prims import define_prim, is_prim_path_valid
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from std_msgs.msg import String

from cram_vrb_lab.sim.ghost_hands import GHOST_HANDS_TOPIC
from cram_vrb_lab.sim.numpy_bridge import numpy_view
from cram_vrb_lab.sim.ros_utils import SimBridge

GHOST_ROOT = "/World/GhostHands"
"""Where the hands are drawn, as the blocks cramera draws a viewer's hands in. Visual
prims only: no rigid body, no collider, so nothing in the scene ever touches them."""

HAND_SCALE = (0.12, 0.035, 0.045)
"""[m] of the block a hand is drawn as, +X forward -- cramera's avatar hand, used when a
report does not say (see :data:`~cram_vrb_lab.sim.ghost_hands.GHOST_HANDS_TOPIC`)."""

HAND_COLOR = "#4ac2ff"
"""The first viewer's colour in cramera, used when a report does not say."""

HAND_OPACITY = 0.5
"""Half see-through: a ghost, and never hiding what it is reaching for."""

GRAB_RADIUS = 0.05
"""[m] from a body's bounds a hand may be and still take it: all but touching, so a
hand reaching past one thing for another takes the one it is at."""

HAND_TIMEOUT = 3.0
"""[s] without a report before a hand is taken out -- a headset taken off, a page
closed. Viewers report at least once a second while they are there, but the demo
process relaying them pauses for up to two seconds at a time, and a hand taken out
lets go of what it holds."""

LINEAR_GAIN = 12.0
"""[1/s] how hard a held body is pulled to where the hand holds it: its velocity is this
times the distance still to go, i.e. it closes 1/e of the gap every 1/12 s. A spring
critically damped by construction, since what is set is a velocity, not a force."""

ANGULAR_GAIN = 10.0
"""[1/s] the same for turning it."""

MAX_LINEAR_SPEED = 3.0
"""[m/s] a held body is never pulled faster than -- a hand that jumps (a controller
losing tracking for a moment) must not fling what it holds across the room."""

MAX_ANGULAR_SPEED = 12.0
"""[rad/s] the same for turning."""


def _matrix(position, quaternion_wxyz) -> np.ndarray:
    w, x, y, z = (float(v) for v in quaternion_wxyz)
    return np.array(
        Gf.Matrix4d(Gf.Rotation(Gf.Quatd(w, Gf.Vec3d(x, y, z))), Gf.Vec3d(*map(float, position))),
        dtype=float,
    ).T


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """The axis-angle vector of a 3x3 rotation."""
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cosine))
    if angle < 1e-6:
        return np.zeros(3)
    axis = np.array([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ])
    norm = np.linalg.norm(axis)
    if norm < 1e-9:  # half a turn: any axis perpendicular to the plane will do
        axis = np.sqrt(np.clip((np.diag(rotation) + 1.0) / 2.0, 0.0, None))
        norm = np.linalg.norm(axis) or 1.0
    return axis / norm * angle


def _rgb(hex_color: str):
    """``#rrggbb`` as three floats in 0..1."""
    value = hex_color.lstrip("#")
    return tuple(int(value[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _clamp(vector: np.ndarray, limit: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector if norm <= limit else vector * (limit / norm)


@dataclass
class _Grab:
    path: str
    hand_T_body: np.ndarray
    view: Optional[RigidPrim] = None


@dataclass
class _Hand:
    prim: str
    grab: Optional[_Grab] = None
    was_grabbing: bool = False
    look: Optional[tuple] = None     # (scale, color) last drawn


class GhostHandsROS(SimBridge):
    """Draws the viewers' hands and drags what they hold, between physics steps.

    Reports are queued by the subscription and applied in :meth:`apply_commands`, the
    sim loop's own hook, for the reason :class:`SceneSyncROS` gives: USD and PhysX are
    written only from the thread that steps them.

    A held body is driven by its velocity, set every step towards where the hand holds
    it, rather than by a joint to the hand: nothing is added to or removed from the
    stage while physics runs, and removing prims mid-run is what takes this simulator
    down (see :meth:`SceneSyncROS._delete`).
    """

    receives_commands = True

    def __init__(self, world, robot=None, sync_bridge=None):
        """
        :param robot: the robot, whose links are never taken -- they belong to its
            drives.
        :param sync_bridge: the scene-sync bridge, if any. It deletes prims (a reset,
            a removed object), and a physics view still covering a deleted prim ends
            the process, so every grab is let go of first.
        """
        super().__init__("ghost_hands_ros")
        self.world = world
        self._robot_prefix = None
        if robot is not None and len(robot.prim_paths) > 0:
            self._robot_prefix = "/" + str(robot.prim_paths[0]).strip("/").split("/")[0]
        self._lock = threading.Lock()
        self._reports: Dict[str, tuple] = {}         # key -> (report, received at)
        self._gone: List[str] = []
        self._hands: Dict[str, _Hand] = {}
        self._bounds: Dict[str, Gf.Range3d] = {}     # body path -> local bounds
        self.create_subscription(String, GHOST_HANDS_TOPIC, self._on_message, 10)
        if sync_bridge is not None:
            sync_bridge.before_delete.append(self.release_all)
        if not is_prim_path_valid(GHOST_ROOT):
            define_prim(GHOST_ROOT, "Xform")

    # %% input
    def _on_message(self, message: String) -> None:
        try:
            data = json.loads(message.data)
        except ValueError:
            return
        now = time.monotonic()
        with self._lock:
            for key, report in (data.get("hands") or {}).items():
                self._reports[str(key)] = (report, now)
            for key in data.get("gone") or []:
                self._reports.pop(str(key), None)
                self._gone.append(str(key))

    # %% the step
    def apply_commands(self, dt: float) -> None:
        with self._lock:
            reports = dict(self._reports)
            gone, self._gone = self._gone, []
        now = time.monotonic()
        for key in gone:
            self._drop(key)
        for key, (report, received) in reports.items():
            if now - received > HAND_TIMEOUT:
                with self._lock:
                    self._reports.pop(key, None)
                self._drop(key)
                continue
            try:
                self._step(key, report)
            except Exception as failure:  # noqa: BLE001 - a bad report must not end the sim
                self.get_logger().error(f"ghost hand {key}: {failure}")
                self._release(self._hands.get(key))

    def _step(self, key: str, report: Dict) -> None:
        position = report.get("position")
        orientation = report.get("orientation")
        if position is None or orientation is None:
            return
        x, y, z, w = orientation
        world_T_hand = _matrix(position, (w, x, y, z))
        hand = self._hands.get(key) or self._make(key)
        self._draw(hand, world_T_hand, report.get("scale"), report.get("color"))

        grabbing = bool(report.get("grab"))
        if grabbing and not hand.was_grabbing and hand.grab is None:
            hand.grab = self._take(world_T_hand)
        elif not grabbing:
            self._release(hand)
        hand.was_grabbing = grabbing
        if hand.grab is not None:
            self._pull(hand, world_T_hand)

    # %% drawing
    def _make(self, key: str) -> _Hand:
        """The hand for ``key``, drawn -- reusing its block if it was drawn before."""
        name = re.sub(r"[^A-Za-z0-9_]", "_", key)
        path = f"{GHOST_ROOT}/{name if not name[:1].isdigit() else '_' + name}"
        prim = self.world.stage.GetPrimAtPath(path)
        if prim.IsValid():
            UsdGeom.Imageable(prim).MakeVisible()
        else:
            # a unit cube, sized by its scale op: the translate-orient-scale order
            # every pose below is written in
            cube = UsdGeom.Cube.Define(self.world.stage, path)
            cube.GetSizeAttr().Set(1.0)
            cube.GetDisplayOpacityAttr().Set([HAND_OPACITY])
            cube.AddTranslateOp()
            cube.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
            cube.AddScaleOp()
        hand = _Hand(prim=path)
        self._hands[key] = hand
        return hand

    def _draw(self, hand: _Hand, world_T_hand: np.ndarray, scale=None, color=None) -> None:
        """Put the hand's block on the hand, in the viewer's size and colour."""
        prim = self.world.stage.GetPrimAtPath(hand.prim)
        if not prim.IsValid():
            return
        translate, orient, scale_op = UsdGeom.Xformable(prim).GetOrderedXformOps()
        translate.Set(Gf.Vec3d(*world_T_hand[:3, 3]))
        # Gf is row-vector: the transpose of the column-vector numpy rotation
        rotation = (
            Gf.Matrix3d(*world_T_hand[:3, :3].T.ravel().tolist()).ExtractRotation().GetQuat()
        )
        orient.Set(Gf.Quatd(rotation.GetReal(), Gf.Vec3d(rotation.GetImaginary())))
        look = (tuple(scale or HAND_SCALE), color or HAND_COLOR)
        if look != hand.look:
            scale_op.Set(Gf.Vec3d(*look[0]))
            UsdGeom.Gprim(prim).GetDisplayColorAttr().Set([Gf.Vec3f(*_rgb(look[1]))])
            hand.look = look

    def _drop(self, key: str) -> None:
        """Take a hand out: let go of what it holds and hide it.

        Hidden, never deleted, though the sphere has no physics at all: *any* prim
        deletion reaches every physics view in the process -- Isaac's deletion callback
        is not filtered by path -- and the robot's ``Articulation`` answers it by
        dropping its physics view outright, whatever was deleted. The next joint-state
        read then ends the sim with ``'Articulation' object has no attribute
        '_physics_view'``. A viewer entering VR, or a hand whose reports paused past
        :data:`HAND_TIMEOUT`, did exactly that. The sphere is reused if the hand
        comes back.
        """
        hand = self._hands.pop(key, None)
        if hand is None:
            return
        self._release(hand)
        prim = self.world.stage.GetPrimAtPath(hand.prim)
        if prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()

    # %% grabbing
    def _candidates(self) -> List[str]:
        """Every body a hand may take: dynamic, enabled, and not the robot's."""
        paths = []
        for prim in Usd.PrimRange(self.world.stage.GetPseudoRoot()):
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            path = str(prim.GetPath())
            if self._robot_prefix and path.startswith(self._robot_prefix):
                continue
            body = UsdPhysics.RigidBodyAPI(prim)
            enabled = body.GetRigidBodyEnabledAttr().Get() if body.GetRigidBodyEnabledAttr() else None
            kinematic = body.GetKinematicEnabledAttr().Get() if body.GetKinematicEnabledAttr() else None
            if enabled is False or kinematic:
                continue
            paths.append(path)
        return paths

    def _local_bounds(self, path: str) -> Gf.Range3d:
        """A body's bounds in its own frame. Cached: a body's shape does not change."""
        if path not in self._bounds:
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
            self._bounds[path] = cache.ComputeUntransformedBound(
                self.world.stage.GetPrimAtPath(path)
            ).ComputeAlignedRange()
        return self._bounds[path]

    def _take(self, world_T_hand: np.ndarray) -> Optional[_Grab]:
        """The smallest body within :data:`GRAB_RADIUS` of the hand, held where it is.

        Poses from physics rather than from the stage, which lags behind it; the view
        used to read them is dropped right after (see :meth:`release_all`).
        """
        paths = self._candidates()
        if not paths:
            return None
        view = numpy_view(RigidPrim(paths, name="ghost_candidates", reset_xform_properties=False))
        view.initialize()
        positions, orientations = view.get_world_poses()
        del view
        # Of everything within reach, the smallest. Reach is measured to a body's
        # bounding box, and a container's box holds everything in it: a hand in a
        # drawer is inside the drawer's box, at distance 0 from it, however close it is
        # to the spoon lying there. The smaller of the two is what it went in for -- a
        # drawer is taken by its handle, outside its box's contents.
        point = world_T_hand[:3, 3]
        best, best_key, best_pose = None, None, None
        for index, path in enumerate(paths):
            world_T_body = _matrix(positions[index], orientations[index])
            local = world_T_body[:3, :3].T @ (point - world_T_body[:3, 3])
            bounds = self._local_bounds(path)
            if bounds.IsEmpty():
                continue
            low, high = np.array(bounds.GetMin()), np.array(bounds.GetMax())
            distance = float(np.linalg.norm(np.maximum(0.0, np.maximum(low - local, local - high))))
            if distance > GRAB_RADIUS:
                continue
            key = (float(np.prod(high - low)), distance)
            if best_key is None or key < best_key:
                best, best_key, best_pose = path, key, world_T_body
        if best is None:
            return None
        self.get_logger().info(f"ghost hand takes {best}")
        return _Grab(path=best, hand_T_body=np.linalg.inv(world_T_hand) @ best_pose)

    def _pull(self, hand: _Hand, world_T_hand: np.ndarray) -> None:
        grab = hand.grab
        if not is_prim_path_valid(grab.path):
            self._release(hand)
            return
        valid = getattr(grab.view, "is_physics_handle_valid", None)
        if grab.view is None or (callable(valid) and not valid()):
            grab.view = numpy_view(
                RigidPrim([grab.path], name="ghost_grab", reset_xform_properties=False)
            )
            grab.view.initialize()
        positions, orientations = grab.view.get_world_poses()
        world_T_body = _matrix(positions[0], orientations[0])
        target = world_T_hand @ grab.hand_T_body
        linear = _clamp((target[:3, 3] - world_T_body[:3, 3]) * LINEAR_GAIN, MAX_LINEAR_SPEED)
        turn = _rotation_vector(target[:3, :3] @ world_T_body[:3, :3].T)
        angular = _clamp(turn * ANGULAR_GAIN, MAX_ANGULAR_SPEED)
        grab.view.set_velocities(np.array([[*linear, *angular]]))

    def _release(self, hand: Optional[_Hand]) -> None:
        """Let go: the body keeps the velocity it was last given, so it can be thrown."""
        if hand is not None and hand.grab is not None:
            hand.grab.view = None
            hand.grab = None

    def release_all(self) -> None:
        """Let go of everything and drop every physics view, before prims are deleted."""
        for hand in self._hands.values():
            self._release(hand)
            # a grab dropped mid-press must not be taken up again until the trigger is
            # pressed afresh
            hand.was_grabbing = True
