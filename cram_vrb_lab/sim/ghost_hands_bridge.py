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
from cram_vrb_lab.sim.ros_utils import SimBridge, as_np, qrot
from cram_vrb_lab.sim.scene_joints_bridge import AXES

GHOST_ROOT = "/World/GhostHands"
"""Where the hands are drawn, as cramera draws a viewer's hands: a ball on the hand, a
bar back from it. Visual prims only: no rigid body, no collider, so nothing in the
scene ever touches them."""

HAND_SCALE = (0.03, 0.03, 0.03)
"""[m] the ball a hand is drawn as, its diameters -- cramera's avatar hand, used when a
report does not say (see :data:`~cram_vrb_lab.sim.ghost_hands.GHOST_HANDS_TOPIC`)."""

HAND_BAR = (0.1, 0.015, 0.015)
"""[m] the bar behind the ball, (length, width, height), running back along -X."""

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
class _JointAxis:
    """The one motion a joint leaves a held body, in that body's frame."""

    revolute: bool
    axis: np.ndarray
    """Unit vector: the hinge's axis, or the direction a slide runs in."""
    anchor: np.ndarray
    """A point on the hinge's axis (unused for a slide)."""


@dataclass
class _Grab:
    path: str
    hand_T_body: np.ndarray
    view: Optional[RigidPrim] = None
    joint: Optional[_JointAxis] = None
    """The joint the body hangs on, if any: it is then pulled by one point, not by
    its pose."""
    point: Optional[np.ndarray] = None
    """Where on a jointed body the hand took it, in the body's frame."""
    center_of_mass: Optional[np.ndarray] = None
    """The body's centre of mass in its frame, where physics takes its velocity at."""


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
        self._draw(
            hand, world_T_hand, report.get("scale"), report.get("bar"), report.get("color")
        )

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
            # the hand's frame, posed each step; the ball on its origin and the bar
            # behind it, both sized by scale ops (a unit sphere, a unit cube)
            hand = UsdGeom.Xform.Define(self.world.stage, path)
            hand.AddTranslateOp()
            hand.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
            ball = UsdGeom.Sphere.Define(self.world.stage, f"{path}/ball")
            ball.GetRadiusAttr().Set(0.5)
            ball.AddScaleOp()
            bar = UsdGeom.Cube.Define(self.world.stage, f"{path}/bar")
            bar.GetSizeAttr().Set(1.0)
            bar.AddTranslateOp()
            bar.AddScaleOp()
            for part in (ball, bar):
                part.GetDisplayOpacityAttr().Set([HAND_OPACITY])
        hand = _Hand(prim=path)
        self._hands[key] = hand
        return hand

    def _draw(self, hand: _Hand, world_T_hand: np.ndarray, scale=None, bar=None,
              color=None) -> None:
        """Put the hand's ball and bar on the hand, in the viewer's size and colour."""
        prim = self.world.stage.GetPrimAtPath(hand.prim)
        if not prim.IsValid():
            return
        translate, orient = UsdGeom.Xformable(prim).GetOrderedXformOps()
        translate.Set(Gf.Vec3d(*world_T_hand[:3, 3]))
        # Gf is row-vector: the transpose of the column-vector numpy rotation
        rotation = (
            Gf.Matrix3d(*world_T_hand[:3, :3].T.ravel().tolist()).ExtractRotation().GetQuat()
        )
        orient.Set(Gf.Quatd(rotation.GetReal(), Gf.Vec3d(rotation.GetImaginary())))
        look = (tuple(scale or HAND_SCALE), tuple(bar or HAND_BAR), color or HAND_COLOR)
        if look != hand.look:
            ball_size, bar_size, hex_color = look
            ball = self.world.stage.GetPrimAtPath(f"{hand.prim}/ball")
            bar_prim = self.world.stage.GetPrimAtPath(f"{hand.prim}/bar")
            UsdGeom.Xformable(ball).GetOrderedXformOps()[0].Set(Gf.Vec3d(*ball_size))
            bar_translate, bar_scale = UsdGeom.Xformable(bar_prim).GetOrderedXformOps()
            # back from the ball along -X, starting at its surface
            bar_translate.Set(Gf.Vec3d(-(ball_size[0] / 2 + bar_size[0] / 2), 0.0, 0.0))
            bar_scale.Set(Gf.Vec3d(*bar_size))
            for part in (ball, bar_prim):
                UsdGeom.Gprim(part).GetDisplayColorAttr().Set([Gf.Vec3f(*_rgb(hex_color))])
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
        grab = _Grab(path=best, hand_T_body=np.linalg.inv(world_T_hand) @ best_pose)
        grab.joint = self._joint_axis(best)
        if grab.joint is not None:
            # the point of the body nearest the hand: on it, or the hand itself when
            # the hand is inside the body's bounds
            bounds = self._local_bounds(best)
            local = best_pose[:3, :3].T @ (point - best_pose[:3, 3])
            grab.point = np.clip(local, np.array(bounds.GetMin()), np.array(bounds.GetMax()))
            grab.center_of_mass = np.array(bounds.GetMidpoint(), dtype=float)
        return grab

    def _joint_axis(self, path: str) -> Optional[_JointAxis]:
        """The revolute or prismatic joint ``path`` hangs on, in its frame; None for a
        free body.

        The joint frame's position is authored in the body's scaled space and the
        pose physics reports is unscaled, so the scale is folded in, as
        :class:`~cram_vrb_lab.sim.scene_joints_bridge._Joint` does.
        """
        stage = self.world.stage
        for prim in Usd.PrimRange(stage.GetPseudoRoot()):
            revolute = prim.IsA(UsdPhysics.RevoluteJoint)
            if not (revolute or prim.IsA(UsdPhysics.PrismaticJoint)):
                continue
            joint = UsdPhysics.Joint(prim)
            sides = (
                (joint.GetBody0Rel().GetTargets(), joint.GetLocalPos0Attr(), joint.GetLocalRot0Attr()),
                (joint.GetBody1Rel().GetTargets(), joint.GetLocalPos1Attr(), joint.GetLocalRot1Attr()),
            )
            for targets, position, rotation in sides:
                if not targets or str(targets[0]) != path:
                    continue
                typed = (UsdPhysics.RevoluteJoint if revolute else UsdPhysics.PrismaticJoint)(prim)
                world = np.array(
                    UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(
                        Usd.TimeCode.Default()
                    )
                ).T
                scale = np.linalg.norm(world[:3, :3], axis=0)
                quaternion = rotation.Get()
                axis = qrot(
                    np.array([*quaternion.GetImaginary(), quaternion.GetReal()], dtype=float),
                    np.array(AXES[typed.GetAxisAttr().Get()]),
                )
                return _JointAxis(
                    revolute=revolute,
                    axis=axis / np.linalg.norm(axis),
                    anchor=scale * np.array(position.Get(), dtype=float),
                )
        return None

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
            if grab.joint is not None:
                try:
                    grab.center_of_mass = as_np(grab.view.get_coms()[0]).reshape(-1, 3)[0]
                except Exception:  # noqa: BLE001 - the bounds' centre stands in
                    pass
        positions, orientations = grab.view.get_world_poses()
        world_T_body = _matrix(positions[0], orientations[0])
        if grab.joint is not None:
            grab.view.set_velocities(
                np.array([self._pull_point(grab, world_T_body, world_T_hand[:3, 3])])
            )
            return
        target = world_T_hand @ grab.hand_T_body
        linear = _clamp((target[:3, 3] - world_T_body[:3, 3]) * LINEAR_GAIN, MAX_LINEAR_SPEED)
        turn = _rotation_vector(target[:3, :3] @ world_T_body[:3, :3].T)
        angular = _clamp(turn * ANGULAR_GAIN, MAX_ANGULAR_SPEED)
        grab.view.set_velocities(np.array([[*linear, *angular]]))

    @staticmethod
    def _pull_point(grab: _Grab, world_T_body: np.ndarray, hand: np.ndarray) -> np.ndarray:
        """The velocity ``[linear, angular]`` of a jointed body pulled by one point.

        An elastic band from the hand to where the body was taken: that point is
        wanted moving towards the hand at :data:`LINEAR_GAIN` times the distance, and
        the body is given the part of that its joint allows -- the slide's component,
        or the turn about the hinge that moves the point that way. How the hand is
        turned does not enter into it.
        """
        rotation, origin = world_T_body[:3, :3], world_T_body[:3, 3]
        point = rotation @ grab.point + origin
        wanted = _clamp((hand - point) * LINEAR_GAIN, MAX_LINEAR_SPEED)
        axis = rotation @ grab.joint.axis
        if not grab.joint.revolute:
            return np.concatenate([axis * float(axis @ wanted), np.zeros(3)])
        anchor = rotation @ grab.joint.anchor + origin
        arm = point - anchor
        arm = arm - axis * float(axis @ arm)
        reach = float(arm @ arm)
        if reach < 1e-6:  # taken on the hinge line: no pull there turns it
            return np.zeros(6)
        rate = float(np.clip(np.cross(axis, arm) @ wanted / reach,
                             -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED))
        angular = axis * rate
        # physics takes a body's linear velocity at its centre of mass
        center = rotation @ grab.center_of_mass + origin
        return np.concatenate([np.cross(angular, center - anchor), angular])

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
