# %% [markdown]
# ## Launch

# %%
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
# os.environ.setdefault("ISAAC_LIVESTREAM", "1")

# Browser viewer (cramera) for the plan: serves the live world, the plan tree and the
# executing motions on http://localhost:8765. "none" runs the demo without it; "rviz"
# and "rerun" are the other backends coraplex.visualization knows.
os.environ.setdefault("CORAPLEX_VISUALIZATION", "cramera")

# os.environ["ISAAC_WINDOW"] = "1920x1080"
# os.environ["ISAAC_WINDOW"] = "1280x720"
# os.environ["ISAAC_WINDOW"] = "960x540"
# os.environ["ISAAC_WINDOW"] = "854x480"
# os.environ["ISAAC_WINDOW"] = "768x432"
os.environ["ISAAC_WINDOW"] = "224x224"
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
os.environ["ISAAC_FIXED_CAMERAS"] = "1"

RVIZ_CONFIG = REPO / "demos" / "rviz" / "garmi.rviz"
ROBOT, SCENE = "garmi", "garmi_apartment"
SPAWN_POSITION = (0, 5.5, 0.0259)
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
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms
from coraplex.execution_environment import real_robot, simulated_robot
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from coraplex.robot_plans.actions.core.navigation import LookAtAction, NavigateAction
from coraplex.visualization import WorldVisualization
from semantic_digital_twin.adapters.ros.world_fetcher import fetch_world_from_service
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.robots.garmi import Garmi
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Drawer,
    Handle,
)
from semantic_digital_twin.spatial_types import Point3, Quaternion
from semantic_digital_twin.spatial_types.spatial_types import Pose

from cram_vrb_lab.robots.garmi.motions import GARMI_MOTION_MAPPINGS

nest_asyncio.apply()
logging.disable(logging.CRITICAL)

if not rclpy.ok():
    rclpy.init()
node = rclpy.create_node("cram_garmi_node")
executor = MultiThreadedExecutor()
executor.add_node(node)
spin_thread = threading.Thread(target=executor.spin, daemon=True, name="rclpy-executor")
spin_thread.start()


def quiet_shutdown():
    """Take everything this script started back down, in the order that stays quiet.

    Called before the interpreter exits rather than from ``atexit``: the thread pool the
    executor submits callbacks into is torn down through ``threading._register_atexit``,
    which runs ahead of every ordinary atexit handler, so a spinning daemon thread left
    to it ends the run with ``RuntimeError: cannot schedule new futures after shutdown``
    printed on top of whatever the script actually did -- which reads as a crash after a
    clean run.

    The order is what keeps it silent: the carry timer first, so nothing new is queued
    onto the executor; then the viewer, whose sockets belong to this process; then the
    sim and the giskard server, which are other processes; then the executor and its
    thread; then the node and rclpy.

    Every step is guarded. The functions above run in a notebook too, where this can be
    reached before the cells that create the timer or the viewer have run, and the whole
    thing is idempotent so calling it twice -- or after a cell already stopped one piece
    -- is harmless.
    """
    if globals().get("_has_shut_down"):
        return
    globals()["_has_shut_down"] = True

    carry_timer = globals().get("carry_timer")
    if carry_timer is not None:
        node.destroy_timer(carry_timer)
        globals()["carry_timer"] = None

    visualization = globals().get("visualization")
    if visualization is not None:
        visualization.stop()
        globals()["visualization"] = None

    stop()  # isaac sim, the giskard server, rviz and the streaming client

    executor.shutdown()
    spin_thread.join(timeout=2.0)
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

world = fetch_world_from_service(node=node, timeout_seconds=300)
WorldSynchronizer(_world=world, node=node)

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

# The arm goals reach on their own, the base drives to them separately. With the
# base under giskard as well it pushes into the furniture, since the sim teleports
# it rather than driving it.
robot.mobile_base.full_body_controlled = False

# %% [markdown]
# ## Plan helpers

# %%
def run_plan(plan, collision_avoidance=True, real_mode=True):
    """Perform a CRAM plan, and report what ended it.

    :return: ``None`` if the plan ran to completion, otherwise
        ``"<ExceptionType>: <message>"`` -- the failure itself rather than the bare
        ``False`` this used to return. A run that misses its target misses it for a
        reason, and the reason is only knowable here: by the time the loop judges the
        settled poses the exception is gone, so "missed" and "missed because the base
        never arrived" would read the same. Returned as text rather than as the
        exception object because that is what the run report prints, and because the
        exception's traceback holds references into a world the reset is about to
        rebuild.
    """
    robot_mode = real_robot if real_mode else simulated_robot
    # publishes the plan tree to the viewer and lights its nodes up as they run; a
    # no-op when the backend shows no plan (NONE / RVIZ). execute_single() and
    # sequential() hand back the node, so the viewer gets the Plan that owns it.
    if plan.plan is not None:
        visualization.attach_plan(plan.plan)
    try:
        with robot_mode(collision_avoidance=collision_avoidance):
            plan.perform()
    except KeyboardInterrupt:
        # Re-raised, so that ctrl-c reaches the loop below and ends the run rather than
        # only failing the plan that happened to be running.
        from cram_vrb_lab.sim.scene_reset import cancel_motion
        print("  interrupted --", cancel_motion(context))
        raise
    except Exception as failure:
        # Everything, not the handful of giskard types this used to name: the failures
        # that end a plan come from several families that share no base --
        # GiskardException, the motion statechart's DataclassExceptions
        # (NoProgressError, CollisionViolatedError), the world's own -- and a loop that
        # is meant to keep running must not stop at whichever one it has not met yet.
        #
        # Printed here as well as returned, and deliberately: this line lands the moment
        # the plan gives up, in among the chatter of the run it belongs to, whereas the
        # report prints it once the run is over and judged. The rules the report draws
        # around itself are what keep the two readable as the separate things they are.
        print(f"  plan failed -- {type(failure).__name__}: {failure}")
        return f"{type(failure).__name__}: {failure}"
    finally:
        # Physics owns where the objects ended up: a place lets them settle, a knock
        # moves them. So the twin follows the sim here rather than asserting the poses a
        # plan last believed. globals(), because scene_sync belongs to a later cell, and
        # a failed pull must not be what ends a run.
        sync = globals().get("scene_sync")
        if sync is not None:
            try:
                moved = sync.pull(world)
                if moved:
                    print("  pulled:", {n: round(d, 4) for n, d in moved.items()})
            except Exception as failure:
                print(f"  object pull failed -- {type(failure).__name__}: {failure}")
    return None


def annotate(view_type, name):
    """Annotate a container and return its ``Handle`` annotation.

    The annotation rather than the handle body: an action is given a designator that can
    say where it may be grasped (``HasGraspPoses``), and reads the body off it itself.
    The motions that pull the container still take the body -- ``handle.root``.
    """
    body = world.get_body_by_name(name)
    handle_body = world.get_body_by_name(f"{name}_handle")
    existing = next(
        (view for view in world.get_semantic_annotations_by_type(view_type)
         if view.root is body),
        None,
    )
    if existing is not None:
        return existing.handle
    view = view_type(root=body, handle=Handle(root=handle_body))
    with world.modify_world():
        world.add_semantic_annotation_recursively(view)
    return view.handle


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
    #     "spoon2.stl", "spoon.stl", Spoon, (0.5, 7.2, 1.0),
    #     grasp_point=(0.0, 0.0, 0.022), mass=0.05, color=(0.25, 0.70, 0.40),
    # ),
    # SceneObject(
    #     "jeroen_cup.stl", "jeroen_cup.stl", Cup, (-0.05, 7.58, 0.9650),
    #     mass=0.120, color=(0.90, 0.90, 0.92),
    # ),
    # SceneObject(
    #     "milk.stl", "milk.stl", Milk, (0.10, 7.58, 1.0527),
    #     mass=1.000, collider="convexHull", color=(0.88, 0.92, 0.96),
    # ),
    # SceneObject(
    #     "bread.stl", "bread.stl", Bread, (0.38, 7.58, 0.9943),
    #     mass=0.400, collider="convexHull", color=(0.76, 0.55, 0.31),
    # ),
    # SceneObject(
    #     "big-knife.stl", "big-knife.stl", Knife, (0.23, 7.44, 0.9871),
    #     mass=0.100, color=(0.55, 0.57, 0.60),
    # ),
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
_robot_bodies = {id(body) for body in robot.bodies}


def sync_objects(settle=None):
    """Force the render to the poses the twin holds for :data:`SCENE_OBJECTS`.

    Skips whatever the plan is carrying: the twin parents a grasped object onto the hand
    and the sim welds it there, so a pose written here would be overridden on the next
    step.

    :param settle: seconds to let physics settle, after which the settled poses are
        pulled back into the twin. None to only push.
    :return: the sim's report, and what ``pull`` moved in the twin.
    """
    for scene_object in SCENE_OBJECTS:
        body = world.get_body_by_name(scene_object.name)
        if id(body.parent_kinematic_structure_entity) in _robot_bodies:
            continue
        # Isaac spawns the mesh file at the pose it is handed and knows nothing of the
        # twin's body frames, so what crosses is the mesh's pose, not the body's.
        # ``pull`` undoes the same offset on the way back.
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
    report = scene_sync.apply()
    if settle is None:
        return report, {}
    time.sleep(settle)
    return report, scene_sync.pull(world)


def print_object_poses(label):
    print(label)
    for scene_object in SCENE_OBJECTS:
        body = world.get_body_by_name(scene_object.name)
        print(f"  {scene_object.name:15s} "
              f"{np.round(np.asarray(body.global_pose.to_np())[:3, 3].ravel(), 4)}")


report, moved = sync_objects(settle=2)
print("sync:", report)
print_object_poses("Object Settled:")

# %%
# Weld a grasped object to the hand in Isaac for as long as the twin says it is held.
# A timer rather than a call after the pick, because TransportAction picks up, drives
# and puts down inside one perform().
GARMI_PRIM_ROOT = "/garmi"
_carry_lock = threading.Lock()


def _carry_tick():
    if not _carry_lock.acquire(blocking=False):
        return
    try:
        if scene_sync.sync_attachments(world, GARMI_PRIM_ROOT):
            print("carry:", scene_sync.apply())
    except Exception as failure:
        print(f"  carry sync failed -- {type(failure).__name__}: {failure}")
    finally:
        _carry_lock.release()


_previous = globals().get("carry_timer")
if _previous is not None:
    node.destroy_timer(_previous)
# Its own callback group: apply() blocks on an ack that arrives on this node, and
# rclpy puts a node's callbacks in one mutually exclusive group by default.
carry_timer = node.create_timer(
    0.1, _carry_tick, callback_group=ReentrantCallbackGroup()
)

# %% [markdown]
# ## Transport task

# %%
# The annotation, not the body: TransportAction reads .root off its object_designator.
from coraplex.robot_plans.actions.composite.transporting import TransportAction

DRAWER = "drawer_1"

# TransportAction opens the drawer the spoon is in itself, but only if it finds a Drawer
# annotation for it: inside_container() names the body, and _make_open_container_actions
# looks up `an(entity(Drawer).where(drawer.root == container))` -- no annotation, no
# query result, and the plan walks up to a shut drawer and reaches for the spoon inside
# it. The reasoner's own drawers_with_a_handle() rule does not supply one here.
drawer_handle = annotate(Drawer, DRAWER)

# PlaceAction puts the body origin here. The grasp itself is whatever the object
# offers -- Bowl traces its rim wall, Cuttlery reaches down across the piece -- so
# nothing here says how to take hold of it.
BOWL_TARGET_POINT = Point3.from_iterable([1.6, 5.1, 0.88])
SPOON_TARGET_POINT = Point3.from_iterable([1.6, 5.3, 0.85])

SUCCESS_TOLERANCE = 0.2
"""How far off its target an object may come to rest, on each axis, and still count."""

TASK_TARGETS = {"bowl.stl": BOWL_TARGET_POINT, "spoon.stl": SPOON_TARGET_POINT}


class Delivery(NamedTuple):
    """One object's verdict, and the distance behind it.

    The distance travels with the verdict because a bare ``True``/``False`` cannot be
    read: 0.21 m out on one axis and 2 m across the room both print as "missed", and
    only the first says the tolerance is what the run fell foul of.
    """

    reached: bool
    """Whether the object came to rest within :data:`SUCCESS_TOLERANCE` of its target."""

    offset: np.ndarray
    """Per-axis distance [m] from the target it was asked to reach."""


def delivered():
    """Which of :data:`TASK_TARGETS` the objects actually reached.

    Read off the twin, which ``run_plan`` has just pulled the settled poses into, so
    this judges where an object came to rest rather than where the plan believed it put
    it: a place that drops it on the way counts as a miss, and a plan that raised
    halfway can still have delivered the one it had already put down.

    Called once per run, from the loop below rather than from ``run_task``, so that the
    runs ``run_task`` never returns from -- the ones :data:`TASK_TIMEOUT` cuts short --
    are judged by the same call as the rest. It prints nothing itself; the loop prints
    every verdict of a run together, in one table.

    :return: a :class:`Delivery` per object in :data:`TASK_TARGETS`, by name.
    """
    outcome = {}
    for name, target in TASK_TARGETS.items():
        position = np.asarray(
            world.get_body_by_name(name).global_pose.to_np()
        )[:3, 3].ravel()
        offset = np.abs(position - np.asarray(target.to_np()).ravel()[:3])
        outcome[name] = Delivery(
            reached=bool(np.all(offset <= SUCCESS_TOLERANCE)), offset=offset
        )
    return outcome


def run_task(arm=Arms.RIGHT):
    """Carry the bowl and then the spoon to the dining table.

    :return: what ``run_plan`` returned -- ``None`` if the plan ran to completion, the
        failure that ended it otherwise. Where the objects ended up is a separate
        question, answered by :func:`delivered` once the loop has the run back.
    """
    return run_plan(sequential([
        # ParkArmsAction(arm=Arms.BOTH),
        TransportAction(
            object_designator=world.get_semantic_annotations_by_type(Bowl)[0],
            arm=arm,
            target_location=Pose(
                position=BOWL_TARGET_POINT, reference_frame=world.root
            ),
        ),
    # ], context=context), collision_avoidance=True)

    # run_plan(sequential([
        # ParkArmsAction(arm=Arms.BOTH),
        # NavigateAction(Pose(
        #     Point3.from_iterable(
        #         [0, 5.5, 0]
        #     ),
        #     Quaternion.from_iterable(
        #         [0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)]
        #     ),
        #     reference_frame=world.root,
        # )),
        TransportAction(
            object_designator=world.get_semantic_annotations_by_type(Spoon)[0],
            arm=arm,
            target_location=Pose(
                position=SPOON_TARGET_POINT, reference_frame=world.root
            ),
        ),
    ], context=context), collision_avoidance=False)


# %% [markdown]
# ## Reset All

# %%
from cram_vrb_lab.sim.scene_reset import SceneResetClient, cancel_motion, reset_context

scene_reset = SceneResetClient(node)


def reset_all():
    """Put the robot, the sim and the twin back to how the run started."""
    # First, or the goal giskard is still executing keeps commanding the robot and
    # carries on with whatever the interrupted plan was doing, out of the pose the
    # reset restores.
    print("giskard:", cancel_motion(context))
    print("sim:    ", scene_reset())
    print("context:", reset_context(world))
    time.sleep(2.0)
    print("base:   ",
          np.round(np.asarray(robot.root.global_pose.to_np())[:3, 3].ravel(), 4))


# %% [markdown]
# ## Recording

# %%
from cram_vrb_lab.sim.episode_recording import EpisodeRecorderClient, describe

RECORD_EPISODES = True
"""Whether to record each run as a demonstration episode.

Costs disk, not control rate: the capture is 10 Hz against the sim's ~20 Hz cycle
and the PNG encoding happens on the sim's writer thread, so what this pays for is
disk rather than a slower robot. Measured over three five-minute runs: ~120 MB of
PNGs per minute of run, 0 frames dropped, which
:mod:`cram_vrb_lab.datasets.lerobot_export` then turns into ~7 MB/min of dataset.
Set it to False for a run that is only being watched.
"""

TASK_INSTRUCTION = "put the bowl and the spoon on the dining table"
"""What the recorded episodes demonstrate, in the words a policy is conditioned on.

Recorded per episode rather than derived from the plan, because it is the one field
nothing can reconstruct afterwards: the plan is a TransportAction over two object
designators, and no amount of replaying it produces the sentence a VLA is asked to
follow.
"""

recorder = EpisodeRecorderClient(node) if RECORD_EPISODES else None


def episode_outcome(failure, outcome):
    """One word for how a run ended, for the episode's ``meta.json``.

    The plan's verdict and the twin's disagree in both directions -- a plan can
    complete and still have put the spoon down out of tolerance, and a plan that
    raised halfway can still have delivered the bowl it had already placed (see
    :func:`print_run_report`) -- so the label is built from both, and the one that
    decides is the twin: an episode is a demonstration of objects ending up
    somewhere, not of a plan returning cleanly.
    """
    if all(delivery.reached for delivery in outcome.values()):
        return "success"
    if any(delivery.reached for delivery in outcome.values()):
        return "partial"
    return "timeout" if failure and "exceeded" in str(failure) else "failed"


# %% [markdown]
# ## Loop

# %%
import signal

RUNS = 10
"""How many times to run the task. 0 repeats until interrupted."""

TASK_TIMEOUT = 600.0
"""Seconds a run may take before it is given up on and the loop moves on.

A plan that stops making progress does not always fail: it can sit in a giskard goal
that never ends, or in a location search probing candidate after candidate, and
nothing below has a deadline of its own.
"""


class TaskTimeout(BaseException):
    """Raised in the main thread when a run outstays :data:`TASK_TIMEOUT`.

    Off ``BaseException`` rather than ``Exception`` on purpose: ``run_plan`` swallows
    every ``Exception`` to keep the loop going, which would turn the deadline into a
    single failed plan and let the run carry on into the next one.
    """


def give_up(signal_number, frame):
    raise TaskTimeout(f"run exceeded {TASK_TIMEOUT:.0f}s")


# Delivered by SIGALRM, which is why this interrupts a blocked call at all; it is also
# why the timeout only works from the main thread.
signal.signal(signal.SIGALRM, give_up)

run = 0
deliveries = {name: 0 for name in TASK_TARGETS}


def success_rate():
    return {name: f"{count}/{run}" for name, count in deliveries.items()}


REPORT_ROW = "  {name:<{width}}  {offset:<23}  {result:<9}  {rate:<5}"
"""One line of the per-run table; see :func:`print_run_report`."""


def print_run_report(index, duration, failure, outcome):
    """Print one run as a table: how long it took, what ended it, what each object did,
    and the rate.

    Everything a run is judged on in one block, rather than the three prints from three
    places this replaces (``delivered``'s own per-object lines, the timeout message, and
    a separate success-rate dict). The two halves belong together: ``failure`` is the
    plan's own verdict and ``outcome`` is the twin's, and they disagree in both
    directions -- a plan can complete and still have put the spoon down out of
    tolerance, and a plan that raised halfway can still have delivered the bowl it had
    already placed.

    :param index: The run's number, i.e. the denominator of the rate column.
    :param duration: Seconds the run took, measured around ``run_task`` alone -- so it
        is directly comparable with :data:`TASK_TIMEOUT`, and excludes the reset and the
        settle that follow it, which are fixed overhead rather than something the run
        earned.
    :param failure: What ``run_task`` returned, or the timeout that cut the run short;
        ``None`` if the plan ran to completion.
    :param outcome: The :class:`Delivery` per object, from :func:`delivered`.
    """
    width = max(len(name) for name in outcome)
    header = REPORT_ROW.format(
        width=width,
        name="object",
        offset="off by x/y/z [m]",
        result="result",
        rate="rate",
    )
    # Ruled off top and bottom, because the report is not the only thing a run prints:
    # the plan's own failure line, the object pull and the reset all print as they
    # happen, and ``failure`` deliberately appears twice -- once as it happened, once
    # here. The rules are what say which of the two this is.
    rule = "  " + "-" * (len(header) - 2)
    print(rule)
    print(f"  run {index} in {duration:.1f}s: {failure or 'plan completed'}")
    print(header)
    print(rule)
    for name, delivery in outcome.items():
        print(REPORT_ROW.format(
            width=width,
            name=name,
            offset=" ".join(f"{axis:7.3f}" for axis in delivery.offset),
            result="delivered" if delivery.reached else "missed",
            # Read after the counters below have taken this run in, so the column is
            # the rate including it rather than the one before it.
            rate=f"{deliveries[name]}/{index}",
        ))
    print(rule)


try:
    while RUNS == 0 or run < RUNS:
        run += 1
        print(f"=== run {run} ===")
        # Started before the timer, so the episode covers the whole attempt including
        # whatever the plan does in its first second. A start while an episode is
        # somehow still open closes that one as "superseded" rather than refusing, so
        # a run that died without stopping costs one episode and not the rest.
        if recorder is not None:
            recorder.start(task=TASK_INSTRUCTION, episode=f"run_{run:03d}")
        signal.setitimer(signal.ITIMER_REAL, TASK_TIMEOUT)
        # Monotonic, not wall clock: this is a duration, and a run long enough to matter
        # is long enough for an NTP step to land in the middle of it.
        started_at = time.monotonic()
        failure = None
        try:
            failure = run_task()
        except TaskTimeout as expired:
            # Kept and reported rather than only printed: a run that was cut short still
            # put the objects somewhere, and every run has to contribute one verdict or
            # the rate means nothing. reset_all() below cancels the goal that was still
            # executing.
            failure = str(expired)
        finally:
            # In the finally, so the run that raised is timed like the one that returned.
            duration = time.monotonic() - started_at
            signal.setitimer(signal.ITIMER_REAL, 0)
        # One judgement per run, here rather than inside run_task, so that a run the
        # timeout cut short is judged exactly like one that returned. run_plan's finally
        # has pulled the settled poses into the twin by now either way.
        outcome = delivered()
        for name, delivery in outcome.items():
            deliveries[name] += delivery.reached
        # Stopped after the verdict rather than after the plan, for two reasons: the
        # label the episode is filed under is only known here, and the extra second
        # of frames is the objects at rest -- the state the attempt actually ended in,
        # which a policy trained on this has to recognise as done.
        if recorder is not None:
            print("record: ", describe(recorder.stop(
                outcome=episode_outcome(failure, outcome),
                notes={
                    "failure": None if failure is None else str(failure),
                    "seconds": round(duration, 2),
                    "offsets": {
                        name: [round(float(axis), 4) for axis in delivery.offset]
                        for name, delivery in outcome.items()
                    },
                },
            )))
        print_run_report(run, duration, failure, outcome)
        print("Reset in 5 seconds...")
        time.sleep(5)
        reset_all()
        spawn_objects()
        print("sync:", sync_objects(settle=2)[0])
        print_object_poses("Object Settled:")
except KeyboardInterrupt:
    print("stopped after run", run)
finally:
    print("success rate:", success_rate())
    quiet_shutdown()
