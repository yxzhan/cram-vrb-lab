# %% [markdown]
# ## Launch

# %%
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from IPython import get_ipython
in_notebook = get_ipython().__class__.__name__ == "ZMQInteractiveShell"

REPO =  Path.cwd().resolve().parent if in_notebook else Path.cwd().resolve()
sys.path.insert(0, str(REPO))

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
os.environ["DISPLAY"] = ":1"


# Put the four kitchen objects -- cup, bowl, cereal box, milk box -- on the cabinet worktop
# os.environ["ISAAC_KITCHEN_PROPS"] = "1"
os.environ["ISAAC_KITCHEN_PROPS"] = "0"

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
    giskard_proc = start_giskard_server(robot=ROBOT, scene=SCENE, control_hz=15,
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
from giskardpy.data_types.exceptions import GiskardException
from giskardpy.motion_statechart.exceptions import CollisionViolatedError
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
    robot_mode = real_robot if real_mode else simulated_robot
    # publishes the plan tree to the viewer and lights its nodes up as they run; a
    # no-op when the backend shows no plan (NONE / RVIZ). execute_single() and
    # sequential() hand back the node, so the viewer gets the Plan that owns it.
    if plan.plan is not None:
        visualization.attach_plan(plan.plan)
    try:
        with robot_mode(collision_avoidance=collision_avoidance):
            plan.perform()
    except (GiskardException, CollisionViolatedError) as failure:
        print(f"Catch giskard failed -- {type(failure).__name__}: {failure}")
        return False
    except KeyboardInterrupt:
        # Re-raised, so that ctrl-c reaches the loop below and ends the run rather than
        # only failing the plan that happened to be running.
        from cram_vrb_lab.sim.scene_reset import cancel_motion
        print("  interrupted --", cancel_motion(context))
        raise
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
    return True


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
        "bowl.stl", "bowl.stl", Bowl, (0.0, 7.2, 1.0),
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


def run_task():
    """Carry the bowl and then the spoon to the dining table."""
    bowl_done = run_plan(sequential([
        ParkArmsAction(arm=Arms.BOTH),
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
            object_designator=world.get_semantic_annotations_by_type(Bowl)[0],
            arm=Arms.RIGHT,
            target_location=Pose(
                position=BOWL_TARGET_POINT, reference_frame=world.root
            ),
        ),
    ], context=context), collision_avoidance=False)

    spoon_done = run_plan(sequential([
        ParkArmsAction(arm=Arms.BOTH),
        NavigateAction(Pose(
            Point3.from_iterable(
                [0, 5.5, 0]
            ),
            Quaternion.from_iterable(
                [0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)]
            ),
            reference_frame=world.root,
        )),
        # NavigateAction(Pose(
        #     Point3.from_iterable(
        #         [-1, 7.2, 0]
        #     ),
        #     Quaternion.from_iterable(
        #         [0.0, 0.0, 0.0, 1.0]
        #     ),
        #     reference_frame=world.root,
        # )),
        TransportAction(
            object_designator=world.get_semantic_annotations_by_type(Spoon)[0],
            arm=Arms.RIGHT,
            target_location=Pose(
                position=SPOON_TARGET_POINT, reference_frame=world.root
            ),
        ),
    ], context=context), collision_avoidance=True)
    return bowl_done and spoon_done


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
# ## Loop

# %%
RUNS = 10
"""How many times to run the task. 0 repeats until interrupted."""

run = 0
try:
    while RUNS == 0 or run < RUNS:
        run += 1
        print(f"=== run {run} ===")
        print("task:", run_task())
        reset_all()
        spawn_objects()
        print("sync:", sync_objects(settle=2)[0])
        print_object_poses("Object Settled:")
except KeyboardInterrupt:
    print("stopped after run", run)
finally:
    quiet_shutdown()
