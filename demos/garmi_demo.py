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

# os.environ.setdefault("ISAAC_HEADLESS", "1")
# os.environ.setdefault("ISAAC_LIVESTREAM", "1")

# os.environ["ISAAC_WINDOW"] = "1920x1080"
# os.environ["ISAAC_WINDOW"] = "1280x720"
# os.environ["ISAAC_WINDOW"] = "960x540"
# os.environ["ISAAC_WINDOW"] = "854x480"
# os.environ["ISAAC_WINDOW"] = "768x432"
# os.environ["ISAAC_WINDOW"] = "640x360"
os.environ["ISAAC_WINDOW"] = "512x288"
# os.environ["DISPLAY"] = ":0"


# Put the four kitchen objects -- cup, bowl, cereal box, milk box -- on the cabinet worktop
# os.environ["ISAAC_KITCHEN_PROPS"] = "1"
os.environ["ISAAC_KITCHEN_PROPS"] = "0"

RVIZ_CONFIG = REPO / "demos" / "rviz" / "garmi.rviz"
ROBOT, SCENE = "garmi", "garmi_apartment"
SPAWN_POSITION = (0, 6.0, 0.0259)
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
    rviz_proc = start_rviz(rviz_config=RVIZ_CONFIG)
    sim_proc = start_isaac_sim(robot=ROBOT, scene=SCENE, camera="both",
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
from coraplex.datastructures.enums import ApproachDirection, Arms, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.execution_environment import real_robot, simulated_robot
from coraplex.plans.factories import execute_single, sequential
from coraplex.robot_plans.actions.core.navigation import LookAtAction, NavigateAction
from coraplex.robot_plans.actions.core.pick_up import GraspingAction, PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import (
    MoveTorsoAction,
    ParkArmsAction,
    SetGripperAction,
)
from coraplex.robot_plans.motions.container import ClosingMotion, OpeningMotion
from coraplex.robot_plans.motions.gripper import (
    MoveGripperMotion,
    MoveTCPWaypointsMotion,
    MoveToolCenterPointMotion,
)
from coraplex.robot_plans.motions.robot_body import MoveJointsMotion
from coraplex.view_manager import ViewManager
from giskardpy.data_types.exceptions import GiskardException
from giskardpy.motion_statechart.exceptions import CollisionViolatedError
from semantic_digital_twin.adapters.ros.world_fetcher import fetch_world_from_service
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.datastructures.definitions import GripperState, TorsoState
from semantic_digital_twin.robots.garmi import Garmi
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bowl,
    Door,
    Drawer,
    Handle,
)
from semantic_digital_twin.spatial_types import Point3, Quaternion
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose,
)

from cram_vrb_lab.robots.garmi.motions import GARMI_MOTION_MAPPINGS

nest_asyncio.apply()
logging.disable(logging.CRITICAL)

if not rclpy.ok():
    rclpy.init()
node = rclpy.create_node("cram_garmi_node")
executor = MultiThreadedExecutor()
executor.add_node(node)
threading.Thread(target=executor.spin, daemon=True, name="rclpy-executor").start()

world = fetch_world_from_service(node=node, timeout_seconds=300)
WorldSynchronizer(_world=world, node=node)

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
# ## Plan helpers

# %%
ARRIVED = 0.05
GRASPED = 0.01
RETREAT = 0.08
ARM_PREFIX = {Arms.LEFT: "left", Arms.RIGHT: "right"}

def run_plan(plan, collision_avoidance=True, real_mode=True):
    robot_mode = real_robot if real_mode else simulated_robot
    try:
        with robot_mode(collision_avoidance=collision_avoidance):
            plan.perform()
    except (GiskardException, CollisionViolatedError) as failure:
        print(f"Catch giskard failed -- {type(failure).__name__}: {failure}")
        return False
    except KeyboardInterrupt:
        from cram_vrb_lab.sim.scene_reset import cancel_motion
        print("  interrupted --", cancel_motion(context))
        raise
    return True


def body_position(name):
    return np.asarray(world.get_body_by_name(name).global_pose.to_np())[:3, 3].ravel()


def annotate(view_type, name):
    body = world.get_body_by_name(name)
    handle = world.get_body_by_name(f"{name}_handle")
    if not any(view.root is body
               for view in world.get_semantic_annotations_by_type(view_type)):
        with world.modify_world():
            world.add_semantic_annotation_recursively(
                view_type(root=body, handle=Handle(root=handle))
            )
    return handle


def drive_to(handle_name, standoff, lateral, attempts=1):
    base_z = float(np.asarray(robot.root.global_pose.to_np())[2, 3])
    target = Pose(
        Point3.from_iterable(
            [float(body_position(handle_name)[0]) + lateral, 7.12 - standoff, base_z]
        ),
        Quaternion.from_iterable(
            [0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)]
        ),
        reference_frame=world.root,
    )
    goal = np.asarray(target.to_np())[:2, 3].ravel()
    for _ in range(attempts):
        run_plan(execute_single(NavigateAction(target), context=context))
        error = float(np.linalg.norm(body_position("base_link")[:2] - goal))
        if error <= ARRIVED:
            print(f"  at {handle_name}, error {error:.3f} m")
            return True
    print(f"  WARNING: {error:.3f} m from the station after {attempts} tries")
    return False


def nudge_base(forward=0.0, left=0.0, turn=0.0):
    base = np.asarray(robot.root.global_pose.to_np())
    yaw = math.atan2(base[1, 0], base[0, 0]) + turn
    position = base[:3, 3] + forward * base[:3, 0] + left * base[:3, 1]
    target = Pose(
        Point3.from_iterable(position),
        Quaternion.from_iterable([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]),
        reference_frame=world.root,
    )
    return run_plan(execute_single(NavigateAction(target), context=context))


def grasp_handle(handle, arm, attempts=3):
    grasp = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        ViewManager.get_end_effector_view(arm, robot),
        manipulation_offset=RETREAT,
    )
    _, commanded, _ = grasp.grasp_pose_sequence(handle)
    goal_frame = np.asarray(handle.global_pose.to_np()) @ np.asarray(commanded.to_np())
    goal = goal_frame[:3, 3].ravel()
    for attempt in range(1, attempts + 1):
        run_plan(execute_single(GraspingAction(handle, arm, grasp), context=context))
        tool = ViewManager.get_end_effector_view(arm, robot).tool_frame
        error = float(np.linalg.norm(
            np.asarray(tool.global_pose.to_np())[:3, 3].ravel() - goal
        ))
        print(f"  grasp {attempt}: {error * 1000:.1f} mm")
        if error <= GRASPED:
            return True
    return False


def retreat(arm, distance=RETREAT):
    tool = np.asarray(
        ViewManager.get_end_effector_view(arm, robot).tool_frame.global_pose.to_np()
    )
    # Column 2 is the tool frame's +z, which points out between the fingers
    # (see _TOOL_FRAME_RPY in cram_vrb_lab/robots/garmi/joints.py); backing off
    # is -z. It used to be column 0 because the approach axis used to be x.
    tool[:3, 3] -= distance * tool[:3, 2]
    target = Pose(
        Point3.from_iterable(tool[:3, 3]),
        HomogeneousTransformationMatrix(data=tool).to_quaternion(),
        reference_frame=world.root,
    )
    return run_plan(
        execute_single(MoveToolCenterPointMotion(target, arm), context=context),
        collision_avoidance=False,
    )


def work_container(motion, handle, arm, attempts=3):
    grasp_handle(handle, arm, attempts)
    run_plan(execute_single(motion(handle, arm), context=context))
    run_plan(execute_single(MoveGripperMotion(GripperState.OPEN, arm), context=context))
    retreat(arm)


def reset_pos():
    run_plan(sequential([
        MoveTorsoAction(TorsoState.LOW),
        SetGripperAction(Arms.LEFT, GripperState.OPEN),
        SetGripperAction(Arms.RIGHT, GripperState.OPEN),
        ParkArmsAction(arm=Arms.BOTH),
    ], context=context))


# %% [markdown]
# ## Spawn bowl and spoon

# %%
from semantic_digital_twin.api import BodySpecification, Connection6DoFSpecification
from semantic_digital_twin.semantic_annotations.semantic_annotations import Bowl, Spoon
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix

from cram_vrb_lab.paths import CRAM_SUBMODULE_DIR

OBJECT_RESOURCES = CRAM_SUBMODULE_DIR / "coraplex" / "resources" / "objects"
BOWL_NAME, SPOON_NAME, SPOON2_NAME = "bowl", "spoon", "spoon2"
BOWL_STL = str(OBJECT_RESOURCES / "bowl.stl")
SPOON_STL = str(OBJECT_RESOURCES / "spoon.stl")
BOWL_POSE = HomogeneousTransformationMatrix.from_xyz_rpy(0.0, 7.2, 1.0)
SPOON_DRAWER_NAME = "drawer_1"
SPOON_IN_DRAWER_POSE = HomogeneousTransformationMatrix.from_xyz_rpy(0.00, 0.0, -0.02)
SPOON2_POSE = HomogeneousTransformationMatrix.from_xyz_rpy(0.3, 7.2, 1.0)

# Where the gripper should take hold, in *mesh* coordinates [m].
#
# The gripper always reaches for the body's own origin: ``PickUpAction`` hands its reach
# ``Pose(reference_frame=object.root)`` and ``GraspDescription.grasp_pose_sequence``
# does the same, so nothing downstream offers a grasp offset to pass. Both meshes are
# centred on their bounding box, which puts that origin in the middle of the bowl's
# cavity and halfway through the spoon's thickness -- the fingers close on air 3 cm
# above the table, or into the table. So the *body frame* is what has to move: it is
# placed on the point below and the mesh is hung off it, and every grasp then aims
# there.
#
#   bowl:  the rim on the +x side (the lip is x 0.065..0.070 at z 0.033), a few mm below
#          the top edge so the fingers straddle the wall instead of the tip.
#   spoon: the middle of the handle, which is 11 mm wide there; x > 0.03 is the scoop.
GRASP_POINTS = {
    BOWL_NAME: (0.0, 0.0677, 0.028),
    SPOON_NAME: (0.0, 0.0, 0.02),
    SPOON2_NAME: (0.0, 0.0, 0.02),
}


def body_T_mesh(name):
    """The mesh's placement inside the body frame, i.e. ``BodySpecification.mesh``'s
    ``origin``.

    Shifting the mesh by ``-grasp_point`` is the same thing as moving the body origin
    onto ``grasp_point``, and the second is what we mean.
    """
    x, y, z = GRASP_POINTS.get(name, (0.0, 0.0, 0.0))
    return HomogeneousTransformationMatrix.from_xyz_rpy(-x, -y, -z)


def grasp_target(point, name):
    """``point`` -- where the mesh should end up -- as the body-origin target an action
    takes, since ``PlaceAction`` puts the *origin* at the pose it is given.

    Only valid for a target with no rotation, which is what the transports below use.
    """
    offset = GRASP_POINTS.get(name, (0.0, 0.0, 0.0))
    return Point3.from_iterable([p + o for p, o in zip(point, offset)])


def ensure_object(annotation, name, stl, pose, parent=None):
    """Spawn the object, or put an existing one back where it started.

    Both halves matter after a reset. ``reset_context`` takes a carried object off
    the gripper with ``move_branch``, which *preserves its world pose* -- so the body
    is still floating where the hand was, and a spawn guarded on the name alone skips
    it and then syncs that pose into Isaac. That is the bowl reappearing in the
    gripper.

    ``pose`` says where the *mesh* goes, as it did before there were grasp points, so
    the body is placed at ``pose`` composed with the offset that separates the two.
    """
    self_T_mesh = body_T_mesh(name)
    parent_T_self = pose @ self_T_mesh.inverse()
    if not world.is_kinematic_structure_entity_in_world_by_name(name):
        annotation.get_annotation_specification(
            name,
            BodySpecification.mesh(
                name, stl, origin=self_T_mesh, parent_T_self=parent_T_self
            ),
            parent_connection_specification=Connection6DoFSpecification(),
        ).spawn(world, parent=parent)
        return "spawned"
    body = world.get_body_by_name(name)
    # Re-apply the mesh offset as well: a reset leaves the body in the world, so this is
    # the only path by which a changed grasp point reaches an object already spawned.
    # A shape origin is model rather than state, so it goes through modify_world, and
    # only when it really differs, since that republishes the world. The visual and the
    # collision shape are one object, so the single write moves both.
    shape = body.collision[0] if len(body.collision) else None
    if shape is not None and not np.allclose(
        np.asarray(shape.origin.to_np()), np.asarray(self_T_mesh.to_np())
    ):
        with world.modify_world():
            shape.origin = self_T_mesh.copy_with_new_reference_frames(body, None)
    target_parent = world.root if parent is None else parent
    world.move_branch(body, target_parent)
    # ``pose`` is a plain parent-relative matrix, but Connection6DoF.origin now runs it
    # through ``World.transform``, which refuses a matrix without a reference frame.
    body.parent_connection.origin = parent_T_self.copy_with_new_reference_frames(
        target_parent, body
    )
    world.notify_state_change()
    return "put back"


print("bowl:  ", ensure_object(Bowl, BOWL_NAME, BOWL_STL, BOWL_POSE))
print("spoon: ", ensure_object(
    Spoon, SPOON_NAME, SPOON_STL, SPOON_IN_DRAWER_POSE,
    parent=world.get_body_by_name(SPOON_DRAWER_NAME),
))
print("spoon2:", ensure_object(Spoon, SPOON2_NAME, SPOON_STL, SPOON2_POSE))

# %%
from cram_vrb_lab.sim.scene_sync import SceneSyncClient, shape_pose_in_world

scene_sync = SceneSyncClient(node)
for name, stl, collider, mass in (
    (BOWL_NAME, BOWL_STL, "convexDecomposition", 0.058),
    (SPOON_NAME, SPOON_STL, "convexDecomposition", 0.05),
    (SPOON2_NAME, SPOON_STL, "convexDecomposition", 0.05),
):
    body = world.get_body_by_name(name)
    # Isaac spawns the mesh file at the pose it is handed and knows nothing of the twin's
    # body frames, so what crosses is the mesh's pose, not the body's. ``pull`` undoes
    # the same offset on the way back.
    position, orientation = shape_pose_in_world(body)
    scene_sync.place(
        name,
        position,
        orientation,
        mesh=stl,
        collider=collider,
        mass=mass,
        track=True,
    )
print("sync:", scene_sync.apply())


for name in (BOWL_NAME, SPOON_NAME, SPOON2_NAME):
    body = world.get_body_by_name(name)
    print(f"  {name:6s} {np.round(np.asarray(body.global_pose.to_np())[:3, 3].ravel(), 4)}")


from time import sleep

sleep(1)
moved = scene_sync.pull(world)
print("Object Settled:")
for name in (BOWL_NAME, SPOON_NAME, SPOON2_NAME):
    body = world.get_body_by_name(name)
    print(f"  {name:6s} {np.round(np.asarray(body.global_pose.to_np())[:3, 3].ravel(), 4)}")


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
# ## Open the drawer

# %%
DRAWER = "drawer_1"
DRAWER_ARM = Arms.LEFT
STANDOFF = {Drawer: (1.0, 0.6), Door: (1.2, 0)}

robot.mobile_base.full_body_controlled = False

drawer_handle = annotate(Drawer, DRAWER)
drawer_joint = world.get_connection_by_name(f"{DRAWER}_joint")

# reset_pos()
# drive_to(f"{DRAWER}_handle", *STANDOFF[Drawer])

# run_plan(sequential([
#     # LookAtAction(drawer_handle.global_pose),
#     GraspingAction(drawer_handle, DRAWER_ARM, GraspDescription(
#         ApproachDirection.FRONT,
#         VerticalAlignment.NoAlignment,
#         ViewManager.get_end_effector_view(DRAWER_ARM, robot),
#         manipulation_offset=0.1,
#     )),
#     OpeningMotion(drawer_handle, DRAWER_ARM),
#     # ClosingMotion(drawer_handle, DRAWER_ARM),
#     MoveGripperMotion(GripperState.OPEN, DRAWER_ARM)
# ], context=context), collision_avoidance=False)


# run_plan(sequential([
#     # ClosingMotion(drawer_handle, DRAWER_ARM),
#     MoveGripperMotion(GripperState.OPEN, DRAWER_ARM)
# ], context=context), collision_avoidance=False)

# retreat(DRAWER_ARM)
# reset_pos()

# run_plan(execute_single(LookAtAction(drawer_handle.global_pose), context=context))
# work_container(OpeningMotion, drawer_handle, DRAWER_ARM)
# work_container(ClosingMotion, drawer_handle, DRAWER_ARM)
# print("opened:", drawer_joint.position)

# %%
# The annotation, not the body: TransportAction reads .root off its object_designator.
from coraplex.robot_plans.actions.composite.transporting import TransportAction

end_effector = context.robot.get_right_arm_if_specified().end_effector

# bowl = world.get_semantic_annotations_by_type(Bowl)[0]
# spoon = world.get_semantic_annotations_by_type(Spoon)[1]

BOWL_TARGET_POINT = grasp_target([1.6, 5.2, 0.88], BOWL_NAME)
SPOON_TARGET_POINT = grasp_target([1.6, 5.3, 0.83], SPOON_NAME)

done = run_plan(sequential([
    ParkArmsAction(arm=Arms.BOTH),
    # # Note: always need TorsoState.HIGH or next(iter(self)) of CostmapLocation fails
    # TransportAction(
    #     object_designator=world.get_semantic_annotations_by_type(Spoon)[1],
    #     arm=Arms.RIGHT,
    #     grasp_description=GraspDescription(
    #         ApproachDirection.RIGHT,
    #         VerticalAlignment.TOP,
    #         rotate_gripper=True,
    #         end_effector=end_effector,
    #     ),
    #     target_location=Pose(
    #         position=SPOON_TARGET_POINT, reference_frame=world.root
    #     ),
    # ),
    TransportAction(
        object_designator=world.get_semantic_annotations_by_type(Bowl)[0],
        arm=Arms.RIGHT,
        grasp_description=GraspDescription(
            ApproachDirection.RIGHT,
            VerticalAlignment.TOP,
            end_effector,
            rotate_gripper=True,
        ),
        target_location=Pose(
            position=BOWL_TARGET_POINT, reference_frame=world.root
        ),
    ),
    NavigateAction(Pose(
        Point3.from_iterable(
            [-1, 6.0, 0]
        ),
        Quaternion.from_iterable(
            [0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)]
        ),
        reference_frame=world.root,
    )),
    TransportAction(
        object_designator=world.get_semantic_annotations_by_type(Spoon)[0],
        arm=Arms.RIGHT,
        grasp_description=GraspDescription(
            ApproachDirection.RIGHT,
            VerticalAlignment.TOP,
            rotate_gripper=True,
            end_effector=end_effector,
        ),
        target_location=Pose(
            position=SPOON_TARGET_POINT, reference_frame=world.root
        ),
    ),
], context=context), collision_avoidance=False)

# %%
time.sleep(10)
stop()

# %% [markdown]
# ## Reset All
#
# %%
from cram_vrb_lab.sim.scene_reset import SceneResetClient, cancel_motion, reset_context

scene_reset = SceneResetClient(node)
# First, or the goal giskard is still executing keeps commanding the robot and carries
# on with whatever the interrupted plan was doing, out of the pose the reset restores.
print("giskard:", cancel_motion(context))
print("sim:    ", scene_reset())
print("context:", reset_context(world))
time.sleep(2.0)
print("base:   ", np.round(np.asarray(robot.root.global_pose.to_np())[:3, 3].ravel(), 4))

# %% [markdown]
# ## Perception

# %%
from cram_vrb_lab.perception import pipeline as rk
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName

from cram_vrb_lab.perception.twin_objects import (
    DETECTION_PREFIX,
    add_boxes,
    detection_pose_in_map,
    ensure_camera_body,
)
from cram_vrb_lab.robots.garmi.joints import (
    CAMERA_IN_HEAD,
    CAMERA_OPTICAL_IN_HEAD_QUAT,
    CAMERA_PARENT_LINK,
)

LOOK_AT = (0.3, 7.32, 1.0)
COUNTERTOP = ((-0.30, 0.65), (7.05, 7.45), (0.90, 1.35))
EXPECTED_OBJECTS = 4
LOOK_ATTEMPTS = 10
GRIP_BELOW_TOP = 0.00

REAL_PERCEPTION = False
"""Whether to run robokudo, or replay :data:`CANNED_BODIES`.

``True`` ticks the real pipeline: it needs the camera topics live, costs a
subscription thread plus a tf lookup per RGB-D pair, and retries up to
:data:`LOOK_ATTEMPTS` times until it finds :data:`EXPECTED_OBJECTS` on the worktop.
``False`` replays a recorded result, so the plan below can be worked on without any
of that.

Flip it to ``True`` and run the perception cell to **re-record**: it prints what it
saw, already in ``map``, as :func:`as_canned_source` source -- ready to paste back
over :data:`CANNED_BODIES`.
"""

@dataclass(frozen=True)
class CannedBody:
    """One perceived box, **in ``map``** -- what the canned path spawns.

    Deliberately not a :class:`~cram_vrb_lab.perception.pipeline.Detection`, which is
    in ``camera_color_optical_frame``. A camera-frame stand-in is only correct while
    the robot stands exactly where the recording was made, because
    ``detection_pose_in_map`` reads ``map_T_camera`` out of the twin's *current*
    forward kinematics -- park somewhere else and the boxes follow the camera off the
    worktop. Freezing the result of that transform instead makes the canned path
    independent of where the robot is standing, which is the whole point of it.
    """

    position: Tuple[float, float, float]
    """Centre of the box in ``map`` [m]."""

    orientation: Tuple[float, float, float, float]
    """``(x, y, z, w)``, the box's rotation in ``map``."""

    extents: Tuple[float, float, float]
    """Side lengths [m], in the box's own frame."""


def canned_pose(canned):
    """``map_T_box`` as a 4x4 numpy array, which is what :func:`add_boxes` takes."""
    return np.asarray(
        HomogeneousTransformationMatrix.from_xyz_quaternion(
            *canned.position, *canned.orientation
        ).to_np()
    )


CANNED_BODIES = [
    CannedBody((-0.03, 7.23, 0.9752), (0.0, 0.0, 0.0, 1.0), (0.132, 0.131, 0.0603)),
    CannedBody((0.15, 7.20, 0.9849), (0.0, 0.0, 0.0, 1.0), (0.1051, 0.0897, 0.0799)),
    CannedBody((0.32, 7.24, 0.9745), (0.0, 0.0, 0.0, 1.0), (0.132, 0.132, 0.0590)),
    CannedBody((0.50, 7.18, 0.9856), (0.0, 0.0, 0.0, 1.0), (0.1063, 0.0895, 0.0811)),
]
"""Four boxes on the worktop, standing in for a perception run.

**A seed, not a recording.** Every number is measured or derived, but no camera ever
reported this exact set -- it is somewhere sane to start, not ground truth:

- x and y are :data:`~cram_vrb_lab.scenes.garmi_apartment.constants.KITCHEN_PROPS`'
  front row -- ``bowl_left``, ``cup_left``, ``bowl_right``, ``cup_right``, left to
  right. The same constants Isaac spawns the props from, so the twin and the render
  agree by construction rather than through a transform that can drift.
- z is the worktop (``KITCHEN_WORKTOP[2]`` = 0.945) plus half the box height, i.e. an
  object *standing on* the surface rather than hovering. Checked against the one
  settled centre the sim has actually reported: ``demo_ori.py``'s
  ``BOWL_CENTRE_IN_MAP`` is 0.9783 for a 0.0665 m bowl, and 0.945 + 0.0665/2 =
  0.97825.
- ``extents`` are what robokudo really measured on this worktop -- two bowls and two
  cups, in this order.
- ``orientation`` is **assumed** identity. ``ClusterPoseBBAnnotator`` only ever fits a
  rotation about z, and these four are near enough axis-symmetric that a yaw hardly
  moves the grasp -- but this is the one field a real recording is worth having.

Run once with :data:`REAL_PERCEPTION` ``= True`` and paste what it prints over this.
"""


def to_canned(detections):
    """Camera-frame ``detections`` -> :class:`CannedBody` in ``map``.

    Where the camera frame is left behind, and the only place the robot's current
    pose is allowed to matter: from here on the boxes are pinned to the world.
    """
    canned = []
    for detection in detections:
        map_T_box = HomogeneousTransformationMatrix(
            data=detection_pose_in_map(world, detection)
        )
        canned.append(
            CannedBody(
                position=tuple(
                    float(v) for v in np.asarray(map_T_box.to_np())[:3, 3].ravel()
                ),
                orientation=tuple(
                    float(v)
                    for v in np.asarray(map_T_box.to_quaternion().to_np()).ravel()
                ),
                extents=tuple(float(v) for v in detection.extents),
            )
        )
    return canned


def as_canned_source(canned_bodies, name="CANNED_BODIES"):
    """``canned_bodies`` as pastable source for :data:`CANNED_BODIES`.

    Full ``repr`` precision, not the rounded numbers the ``keep``/``DROP`` lines
    print: those are for reading, and pasting them back would quantise the poses the
    grasp is computed from.
    """
    def tup(values):
        return "(" + ", ".join(repr(float(v)) for v in values) + ")"

    lines = [f"{name} = ["]
    for canned in canned_bodies:
        lines.append(
            f"    CannedBody({tup(canned.position)}, {tup(canned.orientation)}, "
            f"{tup(canned.extents)}),"
        )
    lines.append("]")
    return "\n".join(lines)


def look_countertop():
    node_for_pipeline = rk.make_pipeline_node()
    try:
        detections = rk.detect(node_for_pipeline, descriptor)
    finally:
        node_for_pipeline.destroy_node()

    on_top = []
    for i, d in enumerate(detections):
        position = detection_pose_in_map(world, d)[:3, 3]
        inside = all(lo <= v <= hi for v, (lo, hi) in zip(position, COUNTERTOP))
        print(f"    [{i}] map {np.round(position, 3)} "
              f"extent {np.round(d.extents, 3)}  {'keep' if inside else 'DROP'}")
        if inside:
            on_top.append(d)
    return on_top


def grasp_offset(canned):
    """Where the body's origin sits inside its box: the +x face, level with the top.

    Takes a :class:`CannedBody` rather than a ``Detection`` because both paths are in
    ``map`` by the time this is called. Only ``extents`` is read either way, and those
    mean the same thing in both.
    """
    return (canned.extents[0] / 2, 0.0, canned.extents[2] / 2 - GRIP_BELOW_TOP)
    # return (0.0, 0.0, 0.0)
# 

def annotate_bowls(bodies):
    """Annotate every perceived body as a :class:`Bowl`; return the annotations.

    ``add_detections`` creates plain :class:`Body` objects, deliberately: the pipeline
    is geometric and says nothing about *what* it saw. ``TransportAction`` takes an
    ``object_designator: HasRootBody`` and reads ``.root`` off it, which a bare body
    does not have -- so something has to put a class on the box before it can be
    transported.

    Everything gets ``Bowl`` here, which is a lie about the two cups and does not
    matter yet: nothing in this plan branches on the class, and ``Bowl`` carries no
    geometry of its own -- the grasp still comes from the detected box and
    :func:`grasp_offset`. Replace this with a real classifier, or hand-pick the
    indices, the moment a plan starts caring which is which.
    """
    # Named after the body rather than left to default: every Bowl would otherwise be
    # called "Bowl", and four annotations sharing a name make get_semantic_annotation_by_name
    # a coin toss and the RViz/log output unreadable. They are not *dropped* -- Bowl is
    # eq=False, so the world dedupes by identity -- just indistinguishable.
    bowls = [
        Bowl(
            root=body,
            name=PrefixedName(body.name.name, prefix=DETECTION_PREFIX),
            class_label="bowl",
        )
        for body in bodies
    ]
    with world.modify_world():
        for bowl in bowls:
            world.add_semantic_annotation(bowl)
    return bowls


def bowl_of(body):
    """The :class:`Bowl` annotation whose root is ``body``.

    ``PickUpAction.object_designator`` is a ``HasRootBody``, so a bare body out of
    ``add_detections`` fails it with ``AttributeError: 'Body' object has no attribute
    'root'``. Looked up from the world rather than read off :func:`annotate_bowls`'
    return value, so a cell that only picks can be re-run without re-detecting.

    .. warning::
       ``PlaceAction`` wants the **opposite**: its ``object_designator`` is typed
       ``Body`` (``coraplex/robot_plans/actions/core/placing.py``), and it uses it as
       one -- ``DetachNode(body=...)`` and
       ``pose_sequence(..., self.object_designator)``, which reads ``.collision`` off
       it. Hand it the annotation and it fails the mirror-image way, ``AttributeError:
       'Bowl' object has no attribute 'collision'``.

       Not a guess about the API: ``TransportAction`` itself passes the annotation to
       ``PickUpAction`` and ``self.object_designator.root`` to ``PlaceAction``, two
       lines apart (``composite/transporting.py``). So the two calls below are
       deliberately asymmetric.
    """
    return next(
        bowl
        for bowl in world.get_semantic_annotations_by_type(Bowl)
        if bowl.root is body
    )


# Still needed with the detections canned: GARMI's URDF has no camera link, and
# detection_pose_in_map cannot get out of the camera frame without one.
ensure_camera_body(
    world, CAMERA_PARENT_LINK, CAMERA_IN_HEAD, CAMERA_OPTICAL_IN_HEAD_QUAT
)
# Only look_countertop() needs this, and it costs a subscription thread plus a tf
# lookup per RGB-D pair -- so it is not paid for on the canned path.
descriptor = rk.camera_descriptor() if REAL_PERCEPTION else None

# %%
# Performed even though the detections are canned -- CANNED_DETECTIONS are in the
# camera frame, so this is what puts the head where they were recorded from.
run_plan(execute_single(
    LookAtAction(Pose(Point3.from_iterable(LOOK_AT), reference_frame=world.root)),
    context=context,
))

if REAL_PERCEPTION:
    for attempt in range(1, LOOK_ATTEMPTS + 1):
        print(f"look {attempt}/{LOOK_ATTEMPTS}:")
        kept = look_countertop()
        print(f"  {len(kept)}/{EXPECTED_OBJECTS} on the countertop")
        if len(kept) == EXPECTED_OBJECTS:
            break
    canned = to_canned(kept)
    # Re-recording is the main reason to come down this branch, so always offer the
    # paste rather than making it a separate call to remember.
    print(f"\n{as_canned_source(canned)}\n")
else:
    canned = list(CANNED_BODIES)
    print(f"canned: replaying {len(canned)} bodies (REAL_PERCEPTION=False)")

# Both paths meet here, already in map: add_boxes is add_detections with the camera
# frame left behind, so nothing below depends on where the robot is standing.
bodies = add_boxes(
    world,
    [(canned_pose(c), c.extents) for c in canned],
    origin_offsets=[grasp_offset(c) for c in canned],
)
bowls = annotate_bowls(bodies)
for body, c in zip(bodies, canned):
    print(f"  {body.name.name:14s} h {c.extents[2]:.3f}  map "
          f"{np.round(np.asarray(body.global_pose.to_np())[:3, 3].ravel(), 3)}")


# %%
# The annotation, not the body: TransportAction reads .root off its object_designator.
from coraplex.robot_plans.actions.composite.transporting import TransportAction

end_effector = context.robot.get_right_arm_if_specified().end_effector

# %%

bowl = world.get_semantic_annotations_by_type(Bowl)[3]
BOWL_TARGET_POINT = Point3.from_iterable([1.6, 5.2, 0.85])

done = run_plan(sequential([
    TransportAction(
        object_designator=bowl,
        arm=Arms.RIGHT,
        grasp_description=GraspDescription(
            ApproachDirection.RIGHT,
            VerticalAlignment.TOP,
            end_effector,
            rotate_gripper=False,
        ),
        target_location=Pose(
            position=BOWL_TARGET_POINT, reference_frame=world.root
        ),
    ),
], context=context))


# %% [markdown]
# ## Pick and place

# %%
# PICK_HINT = (0.5, 7.2)
PICK_HINT = (0.3, 7.3)
# PICK_HINT = (0.1, 7.3)

PICK_ARM = Arms.RIGHT
PICK_APPROACH = ApproachDirection.FRONT
PICK_ALIGNMENT = VerticalAlignment.TOP
PICK_CLEARANCE = 0.02
ROTATE_GRIPPER = True
PLACE_BODY = world.get_body_by_name(DRAWER)
LIFT_AFTER_PLACE = 0.15

robot.mobile_base.full_body_controlled = False

target_body = min(
    bodies,
    key=lambda b: float(np.linalg.norm(
        np.asarray(b.global_pose.to_np())[:3, 3].ravel()[:2] - np.asarray(PICK_HINT)
    )),
)
# The body carries the pose the waypoints below are built from; the annotation is what
# the actions take. Same object either way -- target_bowl.root is target_body.
target_bowl = bowl_of(target_body)
print("picking", target_body.name.name)

# %%
nudge_base(turn=math.pi / 6)
# nudge_base(turn=math.pi / 9)

# %%

pick_grasp = GraspDescription(
    PICK_APPROACH,
    PICK_ALIGNMENT,
    ViewManager.get_end_effector_view(PICK_ARM, robot),
    rotate_gripper=ROTATE_GRIPPER,
    manipulation_offset=PICK_CLEARANCE,
)

pick_T = HomogeneousTransformationMatrix(
    data=np.asarray(target_body.global_pose.to_np())
)
pick_xyz = np.asarray(pick_T.to_np())[:3, 3].ravel()

place_T = HomogeneousTransformationMatrix(
    data=np.asarray(PLACE_BODY.global_pose.to_np())
)
place_xyz = np.asarray(place_T.to_np())[:3, 3].ravel()

carry_waypoints = [
    (pick_xyz[0], pick_xyz[1], pick_xyz[2] + PICK_CLEARANCE),
    (pick_xyz[0], place_xyz[1], pick_xyz[2] + PICK_CLEARANCE),
    (place_xyz[0], place_xyz[1], place_xyz[2] + LIFT_AFTER_PLACE),
    # (place_xyz[0], place_xyz[1], place_xyz[2] + 0.2),
]
carry_quaternion = (
    pick_T.to_rotation_matrix() @ pick_grasp.grasp_orientation().to_rotation_matrix()
).to_quaternion()

done = run_plan(sequential([
    LookAtAction(Pose(Point3.from_iterable(pick_xyz), reference_frame=world.root)),
    PickUpAction(
        object_designator=target_bowl,
        arm=PICK_ARM,
        grasp_description=pick_grasp,
    ),
    LookAtAction(
        Pose(Point3.from_iterable(carry_waypoints[2]), reference_frame=world.root)
    ),
    MoveTCPWaypointsMotion(
        waypoints=[
            Pose(Point3.from_iterable(point), carry_quaternion,
                 reference_frame=world.root)
            for point in carry_waypoints
        ],
        arm=PICK_ARM,
    ),
    # Body, not the annotation -- PlaceAction is the odd one out; see bowl_of.
    PlaceAction(
        object_designator=target_body,
        target_location=Pose(
            Point3.from_iterable(place_xyz),
            pick_T.to_quaternion(),
            reference_frame=world.root,
        ),
        arm=PICK_ARM,
    ),
    MoveToolCenterPointMotion(
        Pose(
            Point3.from_iterable(
                (place_xyz[0], place_xyz[1],
                 place_xyz[2] + LIFT_AFTER_PLACE)
            ),
            carry_quaternion,
            reference_frame=world.root,
        ),
        PICK_ARM,
    ),
    ParkArmsAction(PICK_ARM),
], context=context))
print("pick and place:", done)

# %% [markdown]
# ## Close the drawer

# %%
robot.mobile_base.full_body_controlled = False

work_container(ClosingMotion, drawer_handle, DRAWER_ARM)
nudge_base(-0.3)
reset_pos()
print("closed:", drawer_joint.position)

# %% [markdown]
# ## Cabinet door

# %%
DOOR = "cabinet_door_1"
DOOR_ARM = Arms.RIGHT

robot.mobile_base.full_body_controlled = True

door_handle = annotate(Door, DOOR)
door_joint = world.get_connection_by_name(f"{DOOR}_joint")

drive_to(f"{DOOR}_handle", *STANDOFF[Door])
reset_pos()
run_plan(execute_single(LookAtAction(door_handle.global_pose), context=context))

work_container(OpeningMotion, door_handle, DOOR_ARM)
print("opened:", door_joint.position)

work_container(ClosingMotion, door_handle, DOOR_ARM)
print("closed:", door_joint.position)

# %%
nudge_base(-0.2)
reset_pos()

# %% [markdown]
# ## Shutdown

# %%
time.sleep(10)
stop()
