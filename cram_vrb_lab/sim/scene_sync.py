"""Force the Isaac scene to match the twin, on demand and without physics.

The twin is authoritative for a plan: CRAM adds perceived boxes, moves a carried
object, resets a scene. Isaac knows none of that -- the two sides share *numbers*,
never a library, and there is no path from a ``semantic_digital_twin`` world into
the render at all. So a plan can believe it put a bowl on the table while the bowl
Isaac simulates is still on the worktop, and nothing reports the disagreement.

This module is that missing path, driven by hand rather than continuously:

    from cram_vrb_lab.sim.scene_sync import SceneSyncClient

    sync = SceneSyncClient(node)
    sync.place("perceived_0", position, orientation)          # teleport, or create
    report = sync.apply()                                     # blocks until applied

**Why not just publish over** ``/semantic_digital_twin/world_sync``: because the
Isaac process cannot read it. Isaac runs its own python (3.11) with no
``semantic_digital_twin`` in it, and it never will -- the render side deliberately
depends on nothing the twin side does. What crosses is JSON on a topic, which is
exactly how the twin's own ``WorldSynchronizer`` talks to giskard.

**"Without physics" is the point.** The obvious way to put an object somewhere is
to drop it and let it settle, which is what
:func:`~cram_vrb_lab.scenes.garmi_apartment.isaac_scene.spawn_kitchen_props` does at
load time and why it prints where things *came to rest*. That is the wrong tool for
synchronising: it takes hundreds of steps, it lands somewhere the twin did not ask
for, and a body already in a gripper would be dragged out of it. Here the pose is
written straight onto the prim and the body's velocities are zeroed, so it is
exactly where the twin says with no momentum left over.

Applied between physics steps rather than in the ROS callback, because USD and
PhysX are not thread-safe: see
:meth:`~cram_vrb_lab.sim.scene_sync_bridge.SceneSyncROS.apply_commands`.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Dict, List, Optional, Sequence

from std_msgs.msg import String

SCENE_SYNC_TOPIC = "/cram_vrb_lab/scene_sync"
"""Requests, as JSON. See :func:`encode_request`."""

SCENE_SYNC_POSE_TOPIC = "/cram_vrb_lab/scene_sync_poses"
"""Where the sim reports what physics did to synced objects, as JSON.

**Only published when a request asked for it**, and the asking is per object: see
:meth:`SceneSyncClient.place`'s ``track``. Off by default because it inverts who
owns a pose, and owning it in both places at once is how a sync loop starts.

The rule that keeps it a stream rather than a loop: **an inbound pose may never
trigger an outbound one.** :meth:`SceneSyncClient.pull` writes the twin and stops;
nothing in this module publishes in response to a pose.
"""

SCENE_SYNC_ACK_TOPIC = "/cram_vrb_lab/scene_sync_ack"
"""What the sim did with each request, as JSON, keyed by its ``id``.

A separate topic rather than a service because a service would need an interface
package built into ``ros2_ws``, and because this is the shape the twin already uses
to cross the same gap. The client still *behaves* like a service call:
:meth:`SceneSyncClient.apply` blocks until the ack for its own id arrives.
"""

SYNC_ROOT = "/World/SceneSync"
"""Prim the sim creates synced objects under.

Its own root, like ``KitchenProps``: these are objects the *twin* asked for, and
keeping them out of the apartment and out of the load-time prop sets is what makes
a stage tree readable when something is wrong.
"""

def prim_name(name: str) -> str:
    """The prim an object of this name is spawned under, below :data:`SYNC_ROOT`.

    An object name doubles as a USD prim name, and prim names are identifiers: a dot
    starts a property path, so ``bowl.stl`` names no prim at all. The twin's body names
    carry the mesh suffix on purpose -- the live viewer reads it to tell an object a
    demo spawns, moves and grasps apart from the scene it stands in, and bakes only the
    latter into its bundle -- so the suffix crosses to the sim spelled with an
    underscore.

    :param name: The twin's name for the object.
    """
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", str(name))
    return sanitized if not sanitized[:1].isdigit() else "_" + sanitized


RESEND_INTERVAL = 1.0
"""Seconds between resends of an unacknowledged request.

See :meth:`SceneSyncClient.apply` for why resending is both needed and safe.
"""

DEFAULT_TIMEOUT = 10.0
"""Seconds :meth:`SceneSyncClient.apply` waits for the ack.

Generous: the work is a handful of prim writes between two physics steps, so an ack
that does not come back within a second means the sim is not listening rather than
busy.
"""


def encode_request(
    objects: Sequence[Dict],
    remove: Sequence[str] = (),
    request_id: Optional[str] = None,
    attach: Sequence[Dict] = (),
    detach: Sequence[str] = (),
) -> str:
    """Build the JSON one sync request carries.

    :param objects: each ``{"name", "position", "orientation"}`` plus, when the sim
        may have to *create* it, either ``"size"`` (x, y, z extents of a box) or
        ``"mesh"`` (an absolute path to a mesh file both sides can read), and
        optionally ``"mass"`` [kg] and ``"collider"``. An object with neither that
        the sim cannot find is reported as missing rather than invented -- silently
        guessing a shape would put geometry in the render that the twin never
        described.
    :param remove: prim names under :data:`SYNC_ROOT` to delete. Only ever things
        this interface created; it will not touch the apartment or the robot.
    :param attach: each ``{"name", "link"}`` -- carry the object on that prim.
    :param detach: names to hand back to physics.
    """
    return json.dumps(
        {
            "id": request_id or str(uuid.uuid4()),
            "objects": list(objects),
            "remove": list(remove),
            "attach": list(attach),
            "detach": list(detach),
        }
    )


def body_to_object(
    name: str,
    position,
    orientation,
    size=None,
    mass=None,
    track: bool = False,
    mesh: Optional[str] = None,
    collider: Optional[str] = None,
) -> Dict:
    """One entry of :func:`encode_request`'s ``objects``, from plain numbers.

    Quaternions are ``(x, y, z, w)`` -- ROS order, and the order every other twin
    boundary in this repo takes (``HomogeneousTransformationMatrix.from_xyz_quaternion``,
    ``robokudo``'s ``Pose.rotation``). The sim converts to Isaac's ``(w, x, y, z)``
    on the far side, once, so no caller has to remember which side it is on.
    """
    entry = {
        "name": prim_name(name),
        "position": [float(value) for value in position],
        "orientation": [float(value) for value in orientation],
        "track": bool(track),
    }
    if size is not None:
        entry["size"] = [float(value) for value in size]
    if mesh is not None:
        entry["mesh"] = str(mesh)
    if collider is not None:
        entry["collider"] = str(collider)
    if mass is not None:
        entry["mass"] = float(mass)
    return entry


class SceneSyncClient:
    """Publishes sync requests and waits for the sim to say it applied them.

    :param node: an rclpy node that is already being spun. The ack arrives on that
        node's executor, so a client on an unspun node will always time out.
    """

    def __init__(self, node, timeout: float = DEFAULT_TIMEOUT):
        self.node = node
        self.timeout = timeout
        self._acks: Dict[str, Dict] = {}
        self._pending: List[Dict] = []
        self._remove: List[str] = []
        self._attach: List[Dict] = []
        self._detach: List[str] = []
        self._synced: List[str] = []
        self._carried: Dict[str, str] = {}
        self._poses: Dict[str, Dict] = {}
        self._publisher = node.create_publisher(String, SCENE_SYNC_TOPIC, 10)
        self._subscription = node.create_subscription(
            String, SCENE_SYNC_ACK_TOPIC, self._on_ack, 10
        )
        self._pose_subscription = node.create_subscription(
            String, SCENE_SYNC_POSE_TOPIC, self._on_poses, 10
        )

    def _on_ack(self, message: String) -> None:
        ack = json.loads(message.data)
        self._acks[ack["id"]] = ack

    def _on_poses(self, message: String) -> None:
        """Cache the newest reported pose per tracked object.

        Cached rather than applied, so the twin is written on the thread that owns
        it, when :meth:`pull` is called -- and so a caller that stops pulling simply
        stops updating, instead of having the world change under a plan mid-motion.
        """
        body_names = {prim_name(name): name for name in self._synced}
        self._poses.update(
            {
                body_names.get(reported, reported): pose
                for reported, pose in json.loads(message.data).items()
            }
        )

    def tracked_poses(self) -> Dict[str, Dict]:
        """The newest pose the sim reported for each tracked object, in ``map``.

        ``{name: {"position": [x, y, z], "orientation": [x, y, z, w]}}``. Empty
        until something was placed with ``track=True``.
        """
        return dict(self._poses)

    def pull(self, world, names=None) -> Dict[str, float]:
        """Write the reported poses into ``world``; returns how far each body moved.

        The read half of the loop this module deliberately does not close: it writes
        the twin and stops. Nothing here republishes, so a pose travels
        Isaac -> twin and no further.

        Skips a body no longer parented to the world root. That is the carried case,
        and it is not an edge case: ``PickUpAction`` re-parents what it grasps onto
        the tool frame, at which point the *plan* owns where the object is and a pose
        from Isaac would fight that attachment every step. Ownership follows the
        attachment rather than a flag someone has to remember to flip.

        :param names: which tracked objects to write, or ``None`` for all of them.
        """
        from semantic_digital_twin.spatial_types.spatial_types import (
            HomogeneousTransformationMatrix,
        )

        import numpy as np

        source = (
            self._poses
            if names is None
            else {name: self._poses[name] for name in names if name in self._poses}
        )
        moved: Dict[str, float] = {}
        for name, pose in source.items():
            try:
                body = world.get_body_by_name(name)
            except Exception:
                continue
            if body.parent_kinematic_structure_entity is not world.root:
                continue
            before = np.asarray(body.global_pose.to_np())[:3, 3].ravel()
            # The sim reports where the *geometry* is; the twin stores where the body
            # frame is, and the two are the same thing only for a body whose origin sits
            # on its mesh. See :func:`body_T_shape`.
            map_T_body = _pose_matrix(
                pose["position"], pose["orientation"]
            ) @ np.linalg.inv(body_T_shape(body))
            body.parent_connection.origin = HomogeneousTransformationMatrix(
                data=map_T_body, reference_frame=world.root
            )
            after = np.asarray(body.global_pose.to_np())[:3, 3].ravel()
            moved[name] = float(np.linalg.norm(after - before))
        if moved:
            world.notify_state_change()
        return moved

    def place(
        self,
        name: str,
        position,
        orientation,
        size=None,
        mass=None,
        track: bool = False,
        mesh: Optional[str] = None,
        collider: Optional[str] = None,
    ) -> None:
        """Queue "put ``name`` here". Nothing is sent until :meth:`apply`.

        Batched on purpose: one request is one atomic update between two physics
        steps, so a scene the twin changed in several places never renders
        half-updated.

        :param mesh: an absolute path to a mesh file, for an object that is not a
            box. The **same file the twin loaded**, not a copy: that is what makes
            this able to stand in for a hardcoded prop table, because the render and
            the plan are then describing one file rather than two descriptions
            someone has to keep in step.
        :param collider: ``physics:approximation`` for a mesh -- ``convexHull``,
            ``convexDecomposition``, ``sdf``. Defaults to ``convexDecomposition``,
            which holds a cavity open where a single hull would fill it in; a thin
            concave shape like a spoon wants ``sdf`` instead. See
            :data:`~cram_vrb_lab.scenes.garmi_apartment.constants.TRANSPORT_PROPS`
            for why that choice is not cosmetic.
        :param track: whether the sim should report this object's pose back as
            physics moves it, for :meth:`pull` to write into the twin. **Off by
            default, and the default is the safe one**: with it on, Isaac owns
            where the object is and the twin follows; with it off the twin's belief
            is whatever a plan last said, and a thing set down 10 cm up stays there
            in the twin however far it falls in the render.

            Which you want is not a preference but a question of who is right. Turn
            it on for an object physics decides -- something released, dropped,
            knocked over. Leave it off for one a plan is *asserting*, above all
            anything in a gripper: ``PickUpAction`` re-parents what it grasps onto
            the tool frame, and a pose arriving from Isaac would fight that
            attachment every step.
        """
        self._pending.append(
            body_to_object(
                name, position, orientation, size, mass, track, mesh, collider
            )
        )
        if name not in self._synced:
            self._synced.append(name)

    def attach(self, name: str, link: str) -> None:
        """Queue "carry ``name`` rigidly on ``link``", an absolute prim path.

        Nothing in Isaac knows a plan picked something up: the object stays a free
        rigid body held between two fingers by friction, and the base is teleported
        rather than driven, so the hand arrives somewhere new having reported no
        motion and whatever it held is left behind. This asserts the carry instead --
        the object goes kinematic and rides the link at the offset it was grasped at.
        """
        self._attach.append({"name": prim_name(name), "link": str(link)})

    def detach(self, name: str) -> None:
        """Queue "hand ``name`` back to physics". Send it after the gripper opens."""
        self._detach.append(prim_name(name))

    def sync_attachments(self, world, prim_root: str) -> List[str]:
        """Queue the attach/detach that makes Isaac agree with what the twin holds.

        ``PickUpAction`` re-parents the body onto the tool frame and ``PlaceAction``
        puts it back, so this only has to read that. The tool frame itself has no
        geometry and need not have a prim, so it walks up to the nearest link that
        does; both are rigid, and the sim measures the offset from where they are.

        :return: the names whose carry changed. Empty means nothing to :meth:`apply`.
        """
        from semantic_digital_twin.robots.robot_parts import AbstractRobot

        robot_bodies = {
            id(link)
            for robot in world.get_semantic_annotations_by_type(AbstractRobot)
            for link in robot.bodies
        }
        carried: Dict[str, str] = {}
        for name in self._synced:
            try:
                body = world.get_body_by_name(name)
            except Exception:
                continue
            link = body.parent_kinematic_structure_entity
            while link is not None and id(link) in robot_bodies:
                if len(link.collision):
                    carried[name] = f"{prim_root}/{link.name.name}"
                    break
                link = link.parent_kinematic_structure_entity

        changed = []
        for name, link in carried.items():
            if self._carried.get(name) != link:
                self.attach(name, link)
                changed.append(name)
        for name in self._carried:
            if name not in carried:
                self.detach(name)
                changed.append(name)
        self._carried = carried
        return changed

    def remove(self, name: str) -> None:
        """Queue "delete ``name``". Only affects prims under :data:`SYNC_ROOT`."""
        self._remove.append(prim_name(name))

    def apply(self, timeout: Optional[float] = None) -> Dict:
        """Send everything queued and block until the sim acknowledges it.

        :return: the sim's report -- ``moved``, ``created``, ``removed``,
            ``missing`` (asked for, no prim, no ``size`` to build one), and
            ``error`` if the sim could not apply the request at all.
        :raises TimeoutError: if no ack arrives. That means the sim is not running
            this bridge, not that the update half-applied: the sim applies a request
            in one piece or not at all.
        """
        request_id = str(uuid.uuid4())
        payload = encode_request(
            self._pending, self._remove, request_id, self._attach, self._detach
        )
        self._pending, self._remove = [], []
        self._attach, self._detach = [], []
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)

        # Wait for the bridge's subscription to be matched before publishing. A topic
        # is not a queue: a message published before discovery has paired the two ends
        # is dropped, silently and with no error anywhere. That made the *first*
        # request of a session fail while every later one succeeded -- measured as
        # 180 s of nothing, then 0.14 s, 0.19 s, 0.17 s for the three that followed.
        while time.monotonic() < deadline:
            if self._publisher.get_subscription_count() > 0:
                break
            time.sleep(0.02)

        # Resent while waiting, because the other direction -- the bridge's ack
        # publisher against this node's subscription -- can still be matching when
        # the request lands. A resend is safe: a request names the state it wants
        # rather than a change to make, so applying it twice is applying it once.
        last_sent = 0.0
        while time.monotonic() < deadline:
            if request_id in self._acks:
                return self._acks.pop(request_id)
            now = time.monotonic()
            if now - last_sent > RESEND_INTERVAL:
                self._publisher.publish(String(data=payload))
                last_sent = now
            time.sleep(0.01)
        raise TimeoutError(
            f"no scene-sync ack in {self.timeout:g}s -- is the sim running with the "
            f"SceneSyncROS bridge, and is {SCENE_SYNC_ACK_TOPIC} reaching this node?"
        )


def spawn_in_both(world, sync, name, position, orientation, size, mass=None,
                  track=False, color=None):
    """Create one box in the twin *and* in Isaac, at the same pose; return the body.

    The thing a plan actually wants when it invents an object: without it the two
    sides disagree from the moment of creation, and every later symptom -- a grasp
    that closes on nothing, a plan that believes it placed something -- traces back
    here. ``add_boxes`` alone builds a body the render has never heard of;
    :meth:`SceneSyncClient.place` alone builds geometry the plan cannot reason about.

    The twin is written first and Isaac second on purpose: :meth:`SceneSyncClient.apply`
    blocks until the sim confirms, so when this returns both sides exist.

    **And if the sim cannot be reached, the twin body is taken back out again.**
    Without that the failure leaves behind exactly the disagreement this function
    exists to prevent -- and worse, a *durable* one: the world belongs to the
    giskard server, so a twin-only body outlives the process that made it and the
    next run spawning the same name gets ``DuplicateWorldEntityError: Multiple
    world entities match: ['perceived/free_box', 'perceived/free_box']``, thrown
    from whatever later touches it rather than from the call that failed.

    :param track: hand this object's pose to Isaac to own, and let
        :meth:`SceneSyncClient.pull` carry it back. See :meth:`SceneSyncClient.place`
        -- the short version is: on for something physics decides, off for something
        a plan is asserting.
    :param color: RGBA for the twin's box; the render uses its own colour for synced
        geometry so the two are told apart on sight.
    """
    from cram_vrb_lab.perception.twin_objects import add_boxes

    map_T_body = _pose_matrix(position, orientation)
    # movable=True: a spawned object is by definition one something may move --
    # physics if it is tracked, a plan if it is picked up -- and a FixedConnection
    # cannot carry a new pose at all.
    body = add_boxes(
        world, [(map_T_body, size)], clear_previous=False, names=[name], color=color,
        movable=True,
    )[0]

    sync.place(name, position, orientation, size=size, mass=mass, track=track)
    try:
        report = sync.apply()
    except Exception:
        _remove_body(world, body)
        raise
    if report.get("error") or name not in report.get("moved", []):
        # The sim answered and said no. Same rollback: an ack that reports a failure
        # is not a softer outcome than no ack at all.
        _remove_body(world, body)
        raise RuntimeError(f"sim refused to spawn {name!r}: {report}")
    return body


def _remove_body(world, body) -> None:
    """Take ``body`` back out of the twin, connection and all."""
    with world.modify_world():
        if body.parent_connection is not None:
            world.remove_connection(body.parent_connection)
        world.remove_kinematic_structure_entity(body)


def body_T_shape(body):
    """``body``'s frame to the frame of the mesh file that draws it, as a 4x4.

    Identity for the ordinary body, whose origin *is* its mesh's origin. Not identity
    for one whose frame was deliberately moved off its geometry -- onto the rim of a
    bowl, onto the handle of a spoon -- which is how the twin says where a gripper
    should reach, since ``PickUpAction`` always targets the body origin. Isaac knows
    nothing of that: it spawns a mesh file at the pose it is handed, so the pose that
    crosses this boundary is always the mesh's, and this is the conversion.
    """
    import numpy as np

    shapes = getattr(body, "collision", None)
    if not shapes:
        return np.eye(4)
    return np.asarray(shapes[0].origin.to_np())


def shape_pose_in_world(body):
    """``(position, xyzw)`` of ``body``'s geometry in ``map`` -- what :meth:`place` wants.

    The counterpart of the conversion :meth:`SceneSyncClient.pull` undoes on the way
    back, so a body with a moved frame stays in one place while crossing both ways.
    """
    import numpy as np
    from semantic_digital_twin.spatial_types.spatial_types import (
        HomogeneousTransformationMatrix,
    )

    map_T_shape = np.asarray(body.global_pose.to_np()) @ body_T_shape(body)
    orientation = np.asarray(
        HomogeneousTransformationMatrix(data=map_T_shape).to_quaternion().to_np()
    ).ravel()
    return map_T_shape[:3, 3].ravel(), orientation


def _pose_matrix(position, orientation):
    """``(position, xyzw)`` as a 4x4, without importing numpy at module scope."""
    import numpy as np
    from semantic_digital_twin.spatial_types.spatial_types import (
        HomogeneousTransformationMatrix,
    )

    return np.asarray(
        HomogeneousTransformationMatrix.from_xyz_quaternion(
            *position, *orientation
        ).to_np()
    )
