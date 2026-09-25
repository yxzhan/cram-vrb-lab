# %% [markdown]
# ## Launch

# %%
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Tuple

from IPython import get_ipython
in_notebook = get_ipython().__class__.__name__ == "ZMQInteractiveShell"

REPO =  Path.cwd().resolve().parent if in_notebook else Path.cwd().resolve()
sys.path.insert(0, str(REPO))


# os.environ.setdefault("ISAAC_PHYSICS", "newton")

os.environ.setdefault("ISAAC_HEADLESS", "1")
os.environ.setdefault("ISAAC_LIVESTREAM", "1")

# Browser viewer (cramera) for the plan: serves the live world, the plan tree and the
# executing motions on http://localhost:8765. "none" runs the demo without it; "rviz"
# and "rerun" are the other backends coraplex.visualization knows.
os.environ.setdefault("CORAPLEX_VISUALIZATION", "cramera")

# os.environ["ISAAC_WINDOW"] = "1920x1080"
# os.environ["ISAAC_WINDOW"] = "1280x720"
# os.environ["ISAAC_WINDOW"] = "960x540"
# os.environ["ISAAC_WINDOW"] = "854x480"
# os.environ["ISAAC_WINDOW"] = "768x432"
os.environ["ISAAC_WINDOW"] = "640x360"
# os.environ["ISAAC_WINDOW"] = "512x288"
# os.environ["DISPLAY"] = ":1"

# The rate giskard closes its QP loop at. Set here rather than passed to one of the
# launchers, because both subprocesses need it (cram_vrb_lab/control/rate.py): the
# server configures its QP with it, and the sim compares its own cycle rate against it
# and warns when the margin is gone. Telling only start_giskard_server leaves the sim
# on the default, and its warning then reports a rate nothing is running at.
#
# The two numbers move together: demos/sim.py steps at 33.3 Hz (rendering_dt = 6/200),
# which is what this machine sustains, and runner.FEEDBACK_MARGIN wants the sim 1.3x
# above the controller -- 25 * 1.3 = 32.5, just under it. 30 Hz would fit through the
# sim (33.3 > 30) but only by 1.11x, so a hitch leaves giskard closing its loop on a
# joint state that was not republished since it last looked. Watch the [sim] line: it
# has to read ~33 Hz at RTF ~1.00 with no WARNING.
os.environ["GISKARD_CONTROL_HZ"] = "15"


# Put the four kitchen objects -- cup, bowl, cereal box, milk box -- on the cabinet worktop
# os.environ["ISAAC_KITCHEN_PROPS"] = "1"
os.environ["ISAAC_KITCHEN_PROPS"] = "0"

# Three room-fixed 224x224 cameras framing the worktop workspace (front, left and
# right of (0, 7, 1) -- see cram_vrb_lab.scenes.garmi_apartment.constants.FIXED_CAMERAS).
# They render every cycle once they exist, so the other demos in this scene leave them
# off; set this to "0" if the [sim] line starts reporting a rate near GISKARD_CONTROL_HZ.
os.environ["ISAAC_FIXED_CAMERAS"] = "0"

RVIZ_CONFIG = REPO / "demos" / "rviz" / "garmi.rviz"
ROBOT, SCENE = "garmi", "garmi_apartment"
SPAWN_POSITION = (0.5, 6.0, 0.0259)
SPAWN_YAW = math.pi / 2

from launcher import (
    start_giskard_server,
    start_isaac_sim,
    start_rviz,
    start_streaming_client,
    stop,
)
from cram_vrb_lab.sim.isaac_app import livestream_enabled

if not in_notebook:
    # rviz_proc = start_rviz(rviz_config=RVIZ_CONFIG)
    sim_proc = start_isaac_sim(robot=ROBOT, scene=SCENE, camera="none",
                            spawn_position=SPAWN_POSITION, spawn_yaw=SPAWN_YAW)
    stream_proc = start_streaming_client() if livestream_enabled() else None
    giskard_proc = start_giskard_server(robot=ROBOT, scene=SCENE,
                                        spawn_position=SPAWN_POSITION, spawn_yaw=SPAWN_YAW)

# %% [markdown]
# ## CRAM context

# %%
GISKARD_READY_AT = time.monotonic()

import logging
import threading

import nest_asyncio
import numpy as np
import rclpy
import rclpy.signals
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from coraplex.datastructures.dataclasses import Context
from coraplex.visualization import WorldVisualization
from semantic_digital_twin.adapters.ros.world_fetcher import fetch_world_from_service
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.robots.garmi import Garmi

from cram_vrb_lab.robots.garmi.motions import GARMI_MOTION_MAPPINGS

nest_asyncio.apply()
logging.disable(logging.CRITICAL)

if not rclpy.ok():
    # No rclpy signal handlers: rclpy's own SIGINT handler shuts the ROS context down on
    # Ctrl-C, which stops the executor before the teleop goal can be cancelled -- the
    # cancel is sent, and its answer never comes back. Python's KeyboardInterrupt is
    # all this needs; quiet_shutdown takes rclpy down afterwards.
    rclpy.init(signal_handler_options=rclpy.signals.SignalHandlerOptions.NO)
node = rclpy.create_node("cram_garmi_node")
# The twin's /world_sync on a node of its own. Every callback of one node shares its
# default group, where rclpy runs them one at a time: a backlog of state updates --
# giskard publishes one every control cycle, and applying each takes the world lock --
# would otherwise queue giskard's action results behind it, and a goal that aborted
# would go unnoticed, with no restart, for as long as the backlog lasted.
sync_node = rclpy.create_node("cram_garmi_world_sync")
executor = MultiThreadedExecutor()
executor.add_node(node)
executor.add_node(sync_node)
spin_thread = threading.Thread(target=executor.spin, daemon=True, name="rclpy-executor")
spin_thread.start()

# TELEOP_STALL_TRACE=<file>: diagnose the pauses in the viewer's live updates -- up to
# two seconds in which the bridge publishes no new snapshot. Two watchdogs, for the two
# ways this process can stop:
#
# - one thread holding the GIL: faulthandler's own C thread dumps every stack once a
#   Python heartbeat has not pushed it back for TELEOP_STALL_SECONDS (0.5);
# - threads waiting on each other (the world lock): a sampler records every thread's
#   stack at 20 Hz, and when the bridge's last snapshot is older than the threshold it
#   writes out where each thread spent that time, most frequent first.
#
# Armed by watch_stalls() once the bridge exists.
STALL_TRACE = os.environ.get("TELEOP_STALL_TRACE")
STALL_SECONDS = float(os.environ.get("TELEOP_STALL_SECONDS", "0.5"))


def watch_stalls(bridge):
    import collections
    import faulthandler
    import traceback

    out = open(STALL_TRACE, "a", buffering=1)
    last_snapshot = [time.monotonic()]
    snapshot = bridge.snapshot

    def timed_snapshot():
        snapshot()
        last_snapshot[0] = time.monotonic()

    bridge.snapshot = timed_snapshot

    def heartbeat():
        while True:
            faulthandler.dump_traceback_later(STALL_SECONDS, repeat=False, file=out)
            time.sleep(0.1)

    def sampler():
        names = {}
        samples = collections.deque()          # (time, {thread name: stack text})
        reported = None
        while True:
            now = time.monotonic()
            for thread in threading.enumerate():
                names[thread.ident] = thread.name
            stacks = {
                names.get(ident, str(ident)): "".join(
                    traceback.format_list(traceback.extract_stack(frame)[-6:])
                )
                for ident, frame in sys._current_frames().items()
            }
            samples.append((now, stacks))
            while samples and now - samples[0][0] > 5.0:
                samples.popleft()
            age = now - last_snapshot[0]
            if age > STALL_SECONDS and reported != last_snapshot[0]:
                reported = last_snapshot[0]
                start = now - age
                counts = collections.Counter()
                for t, st in samples:
                    if t >= start:
                        for name, text in st.items():
                            counts[(name, text)] += 1
                out.write(f"\n===== no snapshot for {age:.2f}s (at {time.strftime('%X')}) =====\n")
                for (name, text), n in counts.most_common(12):
                    out.write(f"--- {name} x{n}\n{text}")
            time.sleep(0.05)

    threading.Thread(target=heartbeat, daemon=True, name="stall-heartbeat").start()
    threading.Thread(target=sampler, daemon=True, name="stall-sampler").start()
    print(f"stall trace: pauses over {STALL_SECONDS:g}s go to {STALL_TRACE}")


def quiet_shutdown():
    """Take everything this script started back down, in the order that stays quiet.

    Called before the interpreter exits rather than from ``atexit``: the thread pool the
    executor submits callbacks into is torn down through ``threading._register_atexit``,
    which runs ahead of every ordinary atexit handler, so a spinning daemon thread left
    to it ends the run with ``RuntimeError: cannot schedule new futures after shutdown``
    printed on top of whatever the script actually did -- which reads as a crash after a
    clean run.

    The order is what keeps it silent: the viewer first, whose sockets belong to this
    process; then the sim and the giskard server, which are other processes; then the
    executor and its thread; then the node and rclpy.

    Every step is guarded. The functions above run in a notebook too, where this can be
    reached before the cell that creates the viewer has run, and the whole
    thing is idempotent so calling it twice -- or after a cell already stopped one piece
    -- is harmless.
    """
    if globals().get("_has_shut_down"):
        return
    globals()["_has_shut_down"] = True

    visualization = globals().get("visualization")
    if visualization is not None:
        visualization.stop()
        globals()["visualization"] = None

    stop()  # isaac sim, the giskard server, rviz and the streaming client

    executor.shutdown()
    spin_thread.join(timeout=2.0)
    sync_node.destroy_node()
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

world = fetch_world_from_service(node=node, timeout_seconds=300)
WorldSynchronizer(_world=world, node=sync_node)

# Started after the world is fetched, so the viewer's first snapshot is the real
# apartment rather than an empty world. The backend comes from CORAPLEX_VISUALIZATION
# (set to "cramera" at the top of this file); RViz and the Isaac streaming client are
# launched separately above and are unaffected by it.
visualization = WorldVisualization.from_environment(world).start()

robot = world.get_semantic_annotations_by_type(Garmi)
robot = robot[0] if robot else Garmi.from_world(world)

context = Context(
    world=world,
    robot=robot,
    ros_node=node,
    evaluate_conditions=False,
    alternative_motion_mappings=GARMI_MOTION_MAPPINGS,
)

print(
    f"{time.monotonic() - GISKARD_READY_AT:.1f}s, CRAM context ready!"
    if "GISKARD_READY_AT" in globals()
    else ""
)
print(f"connected: {type(robot).__name__} | {len(world.bodies)} bodies")

# %% [markdown]
# ## Spawn bowl and spoon

# %%
from semantic_digital_twin.api import BodySpecification, Connection6DoFSpecification
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bowl,
    Bread,
    Cup,
    Knife,
    Milk,
    Spoon,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import Color

from cram_vrb_lab.paths import CRAM_SUBMODULE_DIR

OBJECT_RESOURCES = CRAM_SUBMODULE_DIR / "coraplex" / "resources" / "objects"


@dataclass(frozen=True)
class SceneObject:
    """One object this demo puts into the twin and into the render."""

    name: str
    """The twin's body name. Carries the mesh suffix on purpose: cramera's live viewer
    tells a demo object from the scene it stands in by that suffix, and skips it when
    computing the bundle signature. Without it, ``PickUpAction`` re-parenting the object
    changes the signature and the viewer reloads the page on every attach and detach
    (``live/live_bundle.py`` ``_is_overlay_body``, ``live/bridge.py``
    ``_refresh_bundle_signature``). The sim spells the suffix with an underscore in its
    prim names; see ``cram_vrb_lab.sim.scene_sync.prim_name``."""
    mesh: str
    annotation: type
    pose: Tuple[float, ...]
    """Where the *mesh* goes in the parent frame: (x, y, z[, roll, pitch, yaw])."""
    parent: str | None = None
    grasp_point: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    """The point of the mesh the body frame is put on, and so the point every grasp
    aims at -- ``PickUpAction`` always reaches for the body origin."""
    mass: float = 0.1
    collider: str = "convexDecomposition"
    color: Tuple[float, float, float] = (0.80, 0.80, 0.80)

    @property
    def mesh_path(self) -> str:
        """Absolute path of the mesh file, the same string both sides load."""
        return str(OBJECT_RESOURCES / self.mesh)

    @property
    def parent_T_mesh(self) -> HomogeneousTransformationMatrix:
        """:attr:`pose` as a matrix."""
        return HomogeneousTransformationMatrix.from_xyz_rpy(*self.pose)

    @property
    def self_T_mesh(self) -> HomogeneousTransformationMatrix:
        """The mesh's placement inside the body frame, i.e. ``BodySpecification.mesh``'s
        ``origin``: shifting the mesh by ``-grasp_point`` puts the body origin on it."""
        x, y, z = self.grasp_point
        return HomogeneousTransformationMatrix.from_xyz_rpy(-x, -y, -z)


SCENE_OBJECTS = (
    SceneObject(
        "bowl.stl", "bowl.stl", Bowl, (0.5, 7.2, 1.0),
        grasp_point=(0.0677, 0.0, 0.028), mass=0.058, color=(0.20, 0.45, 0.80),
    ),
    SceneObject(
        "spoon.stl", "spoon.stl", Spoon, (0.0, 0.0, -0.069), parent="drawer_1",
        grasp_point=(0.0, 0.0, 0.022), mass=0.05, color=(0.80, 0.80, 0.0),
    ),
    # SceneObject(
    #     "spoon2.stl", "spoon.stl", Spoon, (0.2, 7.2, 1.0),
    #     grasp_point=(0.0, 0.0, 0.022), mass=0.05, color=(0.25, 0.70, 0.40),
    # ),
    SceneObject(
        "jeroen_cup.stl", "jeroen_cup.stl", Cup, (-0.1, 7.35, 0.9650),
        mass=0.120, color=(0.25, 0.35, 0.80),
    ),
    SceneObject(
        "jeroen_cup2.stl", "jeroen_cup.stl", Cup, (0.1, 7.28, 0.9650),
        mass=0.120, color=(0.80, 0.30, 0.30),
    ),
    # SceneObject(
    #     "milk.stl", "milk.stl", Milk, (0.10, 7.58, 1.0527),
    #     mass=1.000, collider="convexHull", color=(0.88, 0.92, 0.96),
    # ),
    SceneObject(
        "bread.stl", "bread.stl", Bread, (0.38, 7.58, 0.9943),
        mass=0.400, collider="convexHull", color=(0.76, 0.55, 0.31),
    ),
    SceneObject(
        "big-knife.stl", "big-knife.stl", Knife, (0.23, 7.44, 0.9871),
        mass=0.100, color=(0.55, 0.57, 0.60),
    ),
)


def ensure_object(scene_object: SceneObject) -> str:
    """Spawn the object, or put an existing one back where it started.

    The second half matters after a reset: ``reset_context`` takes a carried object off
    the gripper with ``move_branch``, which preserves its world pose, so the body is
    left floating where the hand was.
    """
    name = scene_object.name
    self_T_mesh = scene_object.self_T_mesh
    parent_T_self = scene_object.parent_T_mesh @ self_T_mesh.inverse()
    color = Color(*scene_object.color)
    parent = (
        None
        if scene_object.parent is None
        else world.get_body_by_name(scene_object.parent)
    )
    if not world.is_kinematic_structure_entity_in_world_by_name(name):
        scene_object.annotation.get_annotation_specification(
            name,
            BodySpecification.mesh(
                name, scene_object.mesh_path, color=color, origin=self_T_mesh,
                parent_T_self=parent_T_self,
            ),
            parent_connection_specification=Connection6DoFSpecification(),
        ).spawn(world, parent=parent)
        return "spawned"
    body = world.get_body_by_name(name)
    # A reset leaves the body in the world, so this is the only path by which a changed
    # grasp point or colour reaches it. Shapes are model rather than state, hence
    # modify_world, and only when they really differ, since that republishes the world.
    shape = body.collision[0] if len(body.collision) else None
    if shape is not None and (
        not np.allclose(
            np.asarray(shape.origin.to_np()), np.asarray(self_T_mesh.to_np())
        )
        or shape.color != color
    ):
        with world.modify_world():
            shape.origin = self_T_mesh.copy_with_new_reference_frames(body, None)
            shape.color = color
    target_parent = world.root if parent is None else parent
    world.move_branch(body, target_parent)
    # Connection6DoF.origin runs the matrix through World.transform, which refuses one
    # without a reference frame.
    body.parent_connection.origin = parent_T_self.copy_with_new_reference_frames(
        target_parent, body
    )
    world.notify_state_change()
    return "put back"


SPAWN_PUBLISH_PAUSE = 0.5
"""Seconds between spawns, to keep the world-sync topic from overrunning giskard.

Not politeness -- without it this cell kills the server. Every spawn publishes its whole
geometry: ``Mesh.to_json`` embeds the vertices rather than the filename, so one object is
a ~1 MB message (bowl 0.7 MB, spoon 1.0 MB, knife 1.2 MB), and each is followed by a
state update. Seven objects is 14 messages and ~5 MB in a burst, against a subscription
whose QoS is ``depth=10`` KEEP_LAST -- so once ten samples sit unread while giskard is
busy with its control loop, the oldest are *overwritten*, which RELIABLE does not
protect against because it is a deliberate discard rather than a transport loss.

Lose a **model** block that way and the server dies on the next state update, which
carries degrees of freedom it has never heard of:

    StateUpdateContainsUnknownDegreesOfFreedomError: Received a WorldStateUpdate
    containing 7 DOF identifier(s) absent from the world state index

Seven, i.e. exactly the x/y/z/qx/qy/qz/qw of one 6DoF connection: one object's spawn.
"""


def spawn_objects():
    for scene_object in SCENE_OBJECTS:
        print(f"  {scene_object.name:12s} {ensure_object(scene_object)}")
        time.sleep(SPAWN_PUBLISH_PAUSE)


spawn_objects()

# %%
from cram_vrb_lab.sim.scene_sync import SceneSyncClient, shape_pose_in_world

scene_sync = SceneSyncClient(node)


def sync_objects():
    """Force the render to the poses the twin holds for :data:`SCENE_OBJECTS`, creating
    them in the sim if needed, and have the sim report them back from then on.

    The one direction :meth:`SceneSyncClient.follow` does not cover: this is the twin
    saying where the objects go. What physics then does to them -- settling, a knock, a
    drop -- comes back by itself.

    :return: the sim's report.
    """
    for scene_object in SCENE_OBJECTS:
        body = world.get_body_by_name(scene_object.name)
        # Isaac spawns the mesh file at the pose it is handed and knows nothing of the
        # twin's body frames, so what crosses is the mesh's pose, not the body's.
        # ``follow`` undoes the same offset on the way back.
        position, orientation = shape_pose_in_world(body)
        scene_sync.place(
            scene_object.name,
            position,
            orientation,
            mesh=scene_object.mesh_path,
            collider=scene_object.collider,
            mass=scene_object.mass,
            color=scene_object.color,
            track=True,
        )
    return scene_sync.apply()


# The twin follows Isaac for every object lying free, each sim cycle: see
# SceneSyncClient.follow. So nothing below pulls poses back by hand. The drawers and
# doors need nothing here either: giskard reads them from the sim itself and the twin
# hears them over /world_sync, like the robot's joints.
scene_sync.follow(world)

print("sync:", sync_objects())

# %% [markdown]
# ## Teleoperation
#
# No plan and no attachment: the robot stands by holding its pose, each hand follows an
# end-effector marker in the twin, and an object is held only because the fingers
# squeeze it in the sim. The twin sees what physics does to the objects through
# ``scene_sync.follow`` -- a held object is still a free body there, parented to the
# map, so it is followed like one lying on the worktop.
#
# The markers are ordinary bodies, so whatever moves a body in the twin moves a hand:
# a drag in cramera's 3D view, a VR controller grabbing one with the trigger, a script.
# Each marker has two fingers on one joint, and the hand's gripper follows how far they
# are open -- commanded straight to the sim, not through giskard. A VR controller's grip
# button pressed while it holds the marker opens or shuts them, one press each.

# %%
from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
    SelfCollisionAvoidance,
    UpdateTemporaryCollisionRules,
)
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import HoldPose
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionForEndEffector,
)
from semantic_digital_twin.datastructures.definitions import GripperState

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    PrismaticConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedomLimits
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

from cram_vrb_lab.control.teleop_tasks import FollowFrame, HoldJoints
from cram_vrb_lab.robots.garmi.joints import HEAD_JOINTS, LIFT_JOINTS
from cram_vrb_lab.sim.scene_reset import cancel_motion

ARMS = {
    "left": robot.get_left_arm_if_specified(),
    "right": robot.get_right_arm_if_specified(),
}

MARKER_NAMES = {name: f"teleop_{name}_ee" for name in ARMS}
"""The end-effector marker per arm, by body name."""

FINGER_NAMES = {
    name: (f"{marker}_finger_a", f"{marker}_finger_b") for name, marker in MARKER_NAMES.items()
}
"""The marker's two fingers per arm, by body name."""

GRIP_JOINTS = {name: f"{marker}_grip" for name, marker in MARKER_NAMES.items()}
"""The joint that opens a marker's fingers, per arm: what a VR controller's grip button
toggles, and what the hand's gripper is commanded to follow (see _queue_grip_move)."""

FINGER_TRAVEL = 0.04
"""[m] each marker finger slides, shut to wide open -- the FR3 hand's own travel, so the
marker opens as far as the hand it commands."""

MARKER_COLORS = {"left": Color(0.15, 0.45, 0.95), "right": Color(0.95, 0.45, 0.10)}

MARKER_SHIFT = 0.02
"""[m] the marker is drawn back from the tool frame, towards the wrist -- the T and the
fingers both. Only the drawing: the tool frame, what the hand follows, stays put."""


def _box(position, extents, color):
    return Box(
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(*position),
        scale=Scale(*extents),
        color=color,
    )


def marker_bodies(name: str, color: Color) -> Tuple[Body, Body, Body]:
    """A T, with the claw it drives: a bar across the tool frame's y -- the direction the
    fingers open in -- a stem from its middle back towards the wrist, and two fingers
    along +z that slide apart along y as the grip joint opens.

    All of it drawn :data:`MARKER_SHIFT` towards the wrist from the tool frame, which
    stays where it is -- it is what the hand is driven to. The T is what a hand takes
    it by -- the viewer tests reach against the marker's own body, not its fingers --
    so it sits behind the fingertips, where a hand holding a claw would be.

    Visual only. With no collision shape giskard's collision avoidance never sees it,
    and a hand can be driven right into the marker it follows.

    :return: the marker body and its two fingers.
    """
    palm = Body(
        name=PrefixedName(name, prefix="teleop"),
        visual=ShapeCollection([
            _box((0.0, 0.0, -MARKER_SHIFT), (0.012, 0.11, 0.012), color),
            # the stem, 7 cm on from the bar towards the wrist
            _box((0.0, 0.0, -MARKER_SHIFT - 0.035), (0.012, 0.012, 0.07), color),
        ]),
        collision=ShapeCollection([]),
    )
    fingers = tuple(
        Body(
            name=PrefixedName(finger, prefix="teleop"),
            # inner face on the finger's own frame, so shut means touching
            visual=ShapeCollection([
                _box((0.0, side * 0.005, 0.02 - MARKER_SHIFT), (0.012, 0.01, 0.04), color)
            ]),
            collision=ShapeCollection([]),
        )
        for finger, side in zip((f"{name}_finger_a", f"{name}_finger_b"), (1, -1))
    )
    return (palm,) + fingers


def hand_opening(arm) -> float:
    """[m] how far the robot's own fingers are open now, per finger."""
    wide = arm.end_effector.get_joint_state_by_type(GripperState.OPEN)
    return float(np.mean([connection.position for connection in wide.connections]))


def spawn_markers():
    """One marker per arm, on its hand, its fingers as open as the hand's. Before the
    goal is sent: adding a body while a giskard goal runs is a model change, and a model
    change aborts the goal.

    Paced like :func:`spawn_objects`, and for the same reason (see
    :data:`SPAWN_PUBLISH_PAUSE`): giskard deserializes each ``/world_sync`` message under
    its world lock, so while it is still applying the objects' ~1 MB meshes the markers'
    model blocks queue behind them, and a queue of ten drops the oldest. A marker whose
    model block is dropped kills the goal on the first state update that names it --
    ``StateUpdateContainsUnknownDegreesOfFreedomError`` with 8 DOFs: the palm's 7 and the
    grip's one."""
    time.sleep(SPAWN_PUBLISH_PAUSE)
    # Streamed like the demo objects, rather than baked into cramera's scene bundle --
    # that is what makes them draggable in the viewer. By name, since these are not
    # named like mesh files.
    try:
        from cramera.live.overlay import mark_overlay_bodies

        mark_overlay_bodies(
            *MARKER_NAMES.values(), *(n for pair in FINGER_NAMES.values() for n in pair)
        )
    except ImportError:
        pass
    for arm_name, arm in ARMS.items():
        name = MARKER_NAMES[arm_name]
        if not world.is_kinematic_structure_entity_in_world_by_name(name):
            palm, finger_a, finger_b = marker_bodies(name, MARKER_COLORS[arm_name])
            limits = DegreeOfFreedomLimits(
                lower=DerivativeMap(position=0.0, velocity=-0.2),
                upper=DerivativeMap(position=FINGER_TRAVEL, velocity=0.2),
            )
            with world.modify_world():
                for body in (palm, finger_a, finger_b):
                    world.add_kinematic_structure_entity(body)
                world.add_connection(Connection6DoF.create_with_dofs(
                    parent=world.root, child=palm, world=world,
                ))
                grip = PrismaticConnection.create_with_dofs(
                    world=world, parent=palm, child=finger_a,
                    name=PrefixedName(GRIP_JOINTS[arm_name], prefix="teleop"),
                    axis=Vector3.Y(reference_frame=palm), dof_limits=limits,
                )
                world.add_connection(grip)
                # the other finger on the same degree of freedom, the other way
                world.add_connection(PrismaticConnection(
                    name=PrefixedName(f"{GRIP_JOINTS[arm_name]}_mirror", prefix="teleop"),
                    parent=palm, child=finger_b, raw_dof=grip.raw_dof,
                    axis=Vector3.Y(reference_frame=palm), multiplier=-1.0,
                ))
            time.sleep(SPAWN_PUBLISH_PAUSE)
    markers_to_hands()


def markers_to_hands():
    """Put each marker back on its hand, its fingers as open as the hand's -- where the
    robot really is, so a goal following them starts by holding still."""
    with world._world_lock:
        for arm_name, arm in ARMS.items():
            palm = world.get_body_by_name(MARKER_NAMES[arm_name])
            palm.parent_connection.origin = HomogeneousTransformationMatrix(
                data=np.asarray(arm.end_effector.tool_frame.global_pose.to_np()),
                reference_frame=world.root,
            )
            world.get_connection_by_name(GRIP_JOINTS[arm_name]).position = min(
                max(hand_opening(arm), 0.0), FINGER_TRAVEL
            )
        world.notify_state_change()


spawn_markers()
MARKERS = {name: world.get_body_by_name(MARKER_NAMES[name]) for name in ARMS}


def teleop_chart() -> MotionStatechart:
    """One goal that runs until it is cancelled: both hands on their markers, the base
    held, and collisions avoided without ever aborting over one. The grippers are not
    in it: they are commanded straight to the sim (see _queue_grip_move)."""
    msc = MotionStatechart()
    for name, arm in ARMS.items():
        # Rooted at the arm mount, so each chain is its own arm alone: the two hands
        # never pull on the shared lift, and moving one leaves the other still.
        msc.add_node(FollowFrame(
            name=f"teleop_{name}",
            root_link=arm.root,
            tip_link=arm.end_effector.tool_frame,
            target_frame=MARKERS[name],
        ))
    # The base is teleported rather than driven, so anything collision avoidance asked
    # of it would jump the robot -- and leave behind whatever the hands are holding.
    msc.add_node(HoldPose(name="hold_base", root_link=world.root, tip_link=robot.root))
    # The lift and the head held too. The arm targets are rooted at the arm mounts and
    # never reach them, but collision avoidance may -- and a lift that moves carries
    # both mounts, so each hand's target shifts under the other arm's motion.
    msc.add_node(HoldJoints(name="hold_torso_and_head", connections=LIFT_JOINTS + HEAD_JOINTS))
    # The hands may touch anything: grasping is contact. Everything else keeps its
    # distance, and a violated distance only brakes -- it must not end the session.
    msc.add_node(UpdateTemporaryCollisionRules(temporary_rules=[
        AllowCollisionForEndEffector(end_effector=arm.end_effector)
        for arm in ARMS.values()
    ]))
    msc.add_node(ExternalCollisionAvoidance(cancel_if_collision_violated=False))
    msc.add_node(SelfCollisionAvoidance(cancel_if_collision_violated=False))
    return msc


# %%
# cramera writes a drag into the twin only on a plan's motion tick -- the one thread
# allowed to write the world while a plan runs -- and no plan runs here. So the moves it
# queues are applied by this timer instead, and only the markers' are taken at all --
# their poses and their grip joints: the objects are the sim's, and a drag of one in
# the twin would be undone by the next followed pose anyway; the robot's joints are
# the sim's too.
MOVE_APPLY_HZ = 30.0
_moves_pending = threading.Event()
bridge = (
    visualization.cramera_visualization.bridge
    if getattr(visualization, "cramera_visualization", None) is not None
    else None
)
if bridge is not None:
    _queue_move = bridge.queue_move
    _marker_keys = set(MARKER_NAMES.values())

    DRAG_IN_PROGRESS = 0.5
    """[s] since a marker's last intermediate move within which it counts as held."""

    _last_drag = {}            # marker -> monotonic time of its last intermediate move
    _held_off = set()          # markers whose drag is ignored until it is let go of

    def _queue_marker_move(request):
        key = request.object_key
        if key not in _marker_keys:
            return
        if key in _held_off:
            # the drag that was going when the goal failed: the marker stays on the hand
            # until it is let go of, instead of following the viewer straight back out
            if request.is_final:
                _held_off.discard(key)
            return
        if not request.is_final:
            _last_drag[key] = time.monotonic()
        _queue_move(request)
        _moves_pending.set()

    def hold_off_drags():
        """Ignore every marker drag in progress until it is let go of."""
        now = time.monotonic()
        _held_off.update(
            key for key, at in _last_drag.items() if now - at < DRAG_IN_PROGRESS
        )

    bridge.queue_move = _queue_marker_move

    _queue_joint_move = bridge.queue_joint_move
    # The hand the marker's fingers command, by the grip joint's full name, and where
    # its fingers go: straight to the sim's finger drives (gripper_topic), not through
    # giskard. A QP in the loop was a round trip through /world_sync plus a solve at
    # the control rate, and a reference velocity capping how fast the fingers close --
    # to open or shut a hand there is nothing for it to solve.
    from std_msgs.msg import Float64 as _Float64

    from cram_vrb_lab.robots.garmi.joints import MAX_FINGER_TRAVEL, gripper_topic

    _grip_sides = {
        str(world.get_connection_by_name(GRIP_JOINTS[side]).name): side for side in ARMS
    }
    _gripper_publishers = {
        side: node.create_publisher(_Float64, gripper_topic(side), 10) for side in ARMS
    }

    def _queue_grip_move(request):
        side = _grip_sides.get(request.connection_name)
        if side is None:
            return
        _queue_joint_move(request)           # the marker's own fingers, in the twin
        _moves_pending.set()
        travel = min(max(float(request.position), 0.0), MAX_FINGER_TRAVEL)
        _gripper_publishers[side].publish(_Float64(data=travel))

    bridge.queue_joint_move = _queue_grip_move

    def _apply_marker_moves():
        if not _moves_pending.is_set():
            return
        _moves_pending.clear()
        with world._world_lock:
            bridge.apply_moves()
            world.notify_state_change()

    move_timer = node.create_timer(
        1.0 / MOVE_APPLY_HZ, _apply_marker_moves, callback_group=ReentrantCallbackGroup()
    )

    if STALL_TRACE:
        watch_stalls(bridge)

    # TELEOP_LATENCY_TRACE=1: time the robot's state on its way to the viewer, and a
    # drag's on its way to the robot, stage by stage (see control.latency_trace).
    if os.environ.get("TELEOP_LATENCY_TRACE"):
        from cram_vrb_lab.control.latency_trace import LatencyTrace
        from cram_vrb_lab.robots.garmi.joints import (
            CONTROLLED_JOINTS,
            JOINT_STATES_TOPIC,
            VELOCITY_CMD_TOPIC,
        )

        latency_trace = LatencyTrace(
            node, world, bridge,
            joint=os.environ.get("TELEOP_LATENCY_JOINT", "right_fr3_joint1"),
            joint_states_topic=JOINT_STATES_TOPIC,
            velocity_topic=VELOCITY_CMD_TOPIC,
            controlled_joints=CONTROLLED_JOINTS,
            marker_keys=MARKER_NAMES.values(),
        )

    # Ghost hands: every viewer's controllers, reported to the bridge as their avatar,
    # go on to the sim, where a hand whose trigger is held with no marker in it takes
    # hold of whatever physical thing it is at (see cram_vrb_lab.sim.ghost_hands).
    from std_msgs.msg import String as _String

    from cram_vrb_lab.sim.ghost_hands import GHOST_HANDS_TOPIC, hand_key

    _ghost_publisher = node.create_publisher(_String, GHOST_HANDS_TOPIC, 10)

    def _forward_hands(viewer, parts):
        if parts is None:
            body = {"gone": [hand_key(viewer, hand) for hand in ("left", "right")]}
        else:
            body = {"hands": {
                hand_key(viewer, part["name"]): {
                    "position": part["position"],
                    "orientation": part["quaternion"],
                    "grab": part["grab"],
                    "scale": part.get("scale"),
                    "bar": part.get("bar"),
                    "color": part.get("color"),
                }
                for part in parts
                if part["name"] in ("left", "right")
            }}
            if not body["hands"]:
                return
        _ghost_publisher.publish(_String(data=json.dumps(body)))

    bridge.avatar_listeners.append(_forward_hands)

MODEL_SETTLE_SECONDS = 3.0
"""How long to leave giskard to apply the model changes above before sending the goal.

A goal starts by applying everything it has buffered, and it is the goal that dies if a
model block went missing in the rush -- so the rush has to be over first."""

time.sleep(MODEL_SETTLE_SECONDS)
giskard = context.giskard_wrapper
giskard.execute_async(teleop_chart())
print("teleop running: move", " / ".join(MARKER_NAMES.values()),
      "; open and close their fingers with", " / ".join(GRIP_JOINTS.values()))
for name, arm in ARMS.items():
    position = np.asarray(arm.end_effector.tool_frame.global_pose.to_np())[:3, 3].ravel()
    print(f"  {name} hand at {np.round(position, 3)} in map")

# %%
RESTART_PAUSE = 1.0
"""Seconds between putting the markers back and sending the goal again: enough for the
reset to reach giskard over /world_sync, so the new goal starts on the markers where the
hands are rather than where they were dragged to."""


def why_it_ended(result) -> str:
    """What giskard said about a goal that ended, as one line."""
    try:
        return f"{type(error := giskard._client.create_abort_exception(result)).__name__}: {error}"
    except Exception:
        return f"status {result.status}"


try:
    # Standing by. The goal never ends by itself, so a result means giskard aborted it --
    # an infeasible QP, a violated constraint, whatever it was. The session outlives it:
    # the markers snap back onto the hands, so whatever target made it fail is gone and
    # the new goal starts by holding still, and the goal is sent again.
    restarts = 0
    while True:
        while giskard._client.result is None:
            time.sleep(0.2)
        result, giskard._client.result = giskard._client.result, None
        restarts += 1
        print(f"teleop goal ended ({why_it_ended(result)}) -- markers back on the hands, "
              f"restarting (#{restarts})")
        if bridge is not None:
            # A marker still in a viewer's hand would be dragged straight back to where
            # it failed by the next move the viewer posts, 30 times a second -- so its
            # drag is ignored from here until it is let go of. First, so nothing slips
            # in behind the two steps below.
            hold_off_drags()
            # drags queued before the failure would carry the markers back out too
            with world._world_lock:
                bridge.apply_moves()
            # and the viewer is shown the last drag target over the world's pose for
            # as long as no plan ticks -- which here is always -- so it would go on
            # drawing the markers where they failed rather than back on the hands
            with bridge._moves_lock:
                for key in MARKER_NAMES.values():
                    bridge._last_moves.pop(key, None)
        markers_to_hands()
        time.sleep(RESTART_PAUSE)
        giskard.execute_async(teleop_chart())
except KeyboardInterrupt:
    print("stopping teleop")
finally:
    if globals().get("move_timer") is not None:
        node.destroy_timer(move_timer)
    print("giskard:", cancel_motion(context))
    quiet_shutdown()
