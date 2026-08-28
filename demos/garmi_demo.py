# %%
import math
import os
import sys
import time
from pathlib import Path

# REPO = Path.cwd().resolve().parent
REPO = Path.cwd().resolve()
sys.path.insert(0, str(REPO))

# os.environ["DISPLAY"] = ":0"
# os.environ["ISAAC_RENDER"] = "0"

# No local Isaac window; watch the viewport over WebRTC instead (port 49100).
os.environ["ISAAC_HEADLESS"] = "1"
os.environ["ISAAC_LIVESTREAM"] = "1"

# os.environ["ISAAC_WINDOW"] = "1280x720"
# os.environ["ISAAC_WINDOW"] = "960x540"
# os.environ["ISAAC_WINDOW"] = "854x480"
# os.environ["ISAAC_WINDOW"] = "768x432"
os.environ["ISAAC_WINDOW"] = "640x360"
# os.environ["ISAAC_WINDOW"] = "512x288"

# Put the four kitchen objects -- cup, bowl, cereal box, milk box -- on the cabinet worktop
os.environ["ISAAC_KITCHEN_PROPS"] = "1"

from launcher import (
    start_giskard_server,
    start_isaac_sim,
    start_rviz,
    start_streaming_client,
    stop,
)
from cram_vrb_lab.sim.isaac_app import livestream_enabled

RVIZ_CONFIG = REPO / "demos" / "rviz" / "garmi.rviz"
ROBOT, SCENE = "garmi", "garmi_apartment"
SPAWN_POSITION = (0, 5.0, 0.0259)
SPAWN_YAW = -math.pi / 2

rviz_proc = start_rviz(rviz_config=RVIZ_CONFIG)
sim_proc = start_isaac_sim(robot=ROBOT, scene=SCENE, camera="both",
                           spawn_position=SPAWN_POSITION, spawn_yaw=SPAWN_YAW)
stream_proc = start_streaming_client() if livestream_enabled() else None
giskard_proc = start_giskard_server(robot=ROBOT, scene=SCENE,
                                    spawn_position=SPAWN_POSITION, spawn_yaw=SPAWN_YAW)

GISKARD_READY_AT = time.monotonic()

# %%
import threading
import logging
import nest_asyncio
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

nest_asyncio.apply()
logging.disable(logging.CRITICAL)

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ApproachDirection, Arms, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.execution_environment import real_robot
from coraplex.plans.factories import execute_single, sequential
from coraplex.robot_plans.actions.core.navigation import LookAtAction, NavigateAction
from coraplex.robot_plans.actions.core.pick_up import GraspingAction, PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction, SetGripperAction, MoveTorsoAction
from coraplex.robot_plans.motions.container import ClosingMotion, OpeningMotion
from coraplex.robot_plans.motions.gripper import (
    MoveGripperMotion,
    MoveTCPWaypointsMotion,
    MoveToolCenterPointMotion,
)
from coraplex.robot_plans.motions.robot_body import MoveJointsMotion
from coraplex.view_manager import ViewManager
from giskardpy.data_types.exceptions import GiskardException
from semantic_digital_twin.adapters.ros.world_fetcher import fetch_world_from_service
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.datastructures.definitions import GripperState, TorsoState
from semantic_digital_twin.robots.garmi import Garmi
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
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

since_giskard = (
    f" | {time.monotonic() - GISKARD_READY_AT:.1f}s CRAM context ready in "
    if "GISKARD_READY_AT" in globals()
    else ""
)
print(f"connected: {type(robot).__name__} | {len(world.bodies)} bodies{since_giskard}")

# %%
# robot.mobile_base.full_body_controlled = True

# STANDOFF = {Drawer: (1.5, 0.0), Door: (1.2, -0.3)}
STANDOFF = {Drawer: (1.3, 0.0), Door: (1.3, -0.4)}
ARRIVED = 0.05
GRASPED = 0.01
RETREAT = 0.12

ARM_PREFIX = {Arms.LEFT: "arm_0", Arms.RIGHT: "arm_1"}
TUCK_JOINTS = ["fr3_joint3", "fr3_joint4"]
TUCK_POSITIONS = {Arms.LEFT: [-2, -1.5], Arms.RIGHT: [2, -1.5]}
HOME = "cabinet_door_1"

# %%
def run_plan(plan, collision_avoidance=True):
    try:
        with real_robot(collision_avoidance=collision_avoidance):
            plan.perform()
    except GiskardException as failure:
        print(f"  giskard failed -- {type(failure).__name__}: {failure}")
        return False
    return True


def body_position(name):
    return np.asarray(world.get_body_by_name(name).global_pose.to_np())[:3, 3].ravel()


def tool_position(arm):
    tool = ViewManager.get_end_effector_view(arm, robot).tool_frame
    return np.asarray(tool.global_pose.to_np())[:3, 3].ravel()


def other_arm(arm):
    return Arms.RIGHT if arm == Arms.LEFT else Arms.LEFT


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


def base_height():
    return float(np.asarray(robot.root.global_pose.to_np())[2, 3])


def station_facing(handle_x, standoff, lateral):
    return Pose(
        Point3.from_iterable([handle_x + lateral, 7.12 - standoff, base_height()]),
        Quaternion.from_iterable(
            [0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)]
        ),
        reference_frame=world.root,
    )


def drive_to(handle_name, standoff, lateral, attempts=10):
    target = station_facing(float(body_position(handle_name)[0]), standoff, lateral)
    goal = np.asarray(target.to_np())[:2, 3].ravel()
    for attempt in range(1, attempts + 1):
        run_plan(execute_single(NavigateAction(target), context=context))
        error = float(np.linalg.norm(body_position("base_link")[:2] - goal))
        if error <= ARRIVED:
            print(f"  at {handle_name}, error {error:.3f} m")
            return True
    print(f"  WARNING: {error:.3f} m from the station after {attempts} tries")
    return False


def nudge_base(forward=0.0, left=0.0, turn=0.0):
    """Move the base in its own frame; height unchanged.

    ``forward`` runs along base_link +x, ``left`` along +y, ``turn`` [rad] is about
    +z (positive = counter-clockwise, i.e. towards ``left``). Negative is back,
    right and clockwise. All three at once is fine -- the drive is an OmniDrive.

    The translation is measured in the frame the base is in *now*, so a call with
    both is "go there, then face that way" rather than an arc.
    """
    base = np.asarray(robot.root.global_pose.to_np())
    yaw = math.atan2(base[1, 0], base[0, 0]) + turn
    position = base[:3, 3] + forward * base[:3, 0] + left * base[:3, 1]
    target = Pose(
        Point3.from_iterable(position),
        Quaternion.from_iterable([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]),
        reference_frame=world.root,
    )
    return run_plan(execute_single(NavigateAction(target), context=context))


def tuck_arm(arm, positions=None):
    positions = TUCK_POSITIONS[arm] if positions is None else positions
    names = [f"{ARM_PREFIX[arm]}_{joint}" for joint in TUCK_JOINTS]
    return run_plan(
        execute_single(MoveJointsMotion(names, list(positions)), context=context)
    )


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
        error = float(np.linalg.norm(tool_position(arm) - goal))
        print(f"  grasp {attempt}: {error * 1000:.1f} mm")
        if error <= GRASPED:
            return True
    return False


def retreat(arm, distance=RETREAT):
    tool = np.asarray(
        ViewManager.get_end_effector_view(arm, robot).tool_frame.global_pose.to_np()
    )
    tool[:3, 3] -= distance * tool[:3, 0]
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


def open_container(handle, arm, attempts=3, tuck=True):
    if tuck:
        tuck_arm(other_arm(arm))
    work_container(OpeningMotion, handle, arm, attempts)
    retreat(arm)


# RELEASE_TRAVEL = 0.008
# """Per-finger travel [m] that just lets go of a handle rod, without opening the hand.

# The rod is 12 mm across, so the fingers sit near 0.006 when gripping it and 0.008 is
# loose. Why so little matters: the drawer fronts are ``convexDecomposition`` colliders
# (they have to be, or the hull fills the drawer and nothing can go in it), and a
# decomposed flat panel is not flat -- PhysX voxelises the mesh and the hull faces come
# out stair-stepped. A fingertip resting on that face and sweeping the full 0.04 m to
# GripperState.OPEN climbs a step and jams. 2 mm does not.
# """


# def release_handle(arm, travel=RELEASE_TRAVEL):
#     """Let go of the handle without sweeping the fingers across the door panel."""
#     names = [f"{ARM_PREFIX[arm]}_gripper_fr3_finger_joint{i}" for i in (1, 2)]
#     return run_plan(
#         execute_single(MoveJointsMotion(names, [travel, travel]), context=context),
#         collision_avoidance=False,
#     )


def close_container(handle, arm, attempts=3, tuck=True):
    if tuck:
        tuck_arm(other_arm(arm))
    work_container(ClosingMotion, handle, arm, attempts)
    # grasp_handle(handle, arm, attempts)
    # run_plan(execute_single(ClosingMotion(handle, arm), context=context))
    # Not work_container's plain open-then-leave: crack the fingers, back the hand
    # out of the recess, and only then open properly. Opening first is what catches
    # on the panel. Retreating first is not an option either -- the hand is still
    # holding the rod, so it would pull the drawer back out.
    # release_handle(arm)
    retreat(arm)
    # run_plan(execute_single(MoveGripperMotion(GripperState.OPEN, arm), context=context))


def reset_pos():
    run_plan(sequential([
        MoveTorsoAction(TorsoState.LOW),
        SetGripperAction(Arms.LEFT, GripperState.OPEN),
        SetGripperAction(Arms.RIGHT, GripperState.OPEN),
        ParkArmsAction(arm=Arms.BOTH),
    ], context=context))


# %%
ROUNDS = 1
TASKS = [
    # (Drawer, "drawer_1", Arms.LEFT),
    # (Door, "cabinet_door_1", Arms.RIGHT),
    (Drawer, "drawer_2", Arms.LEFT),
    # (Drawer, "drawer_3", Arms.LEFT),
    # (Drawer, "drawer_4", Arms.RIGHT),
]

STANDOFF = {Drawer: (1.1, 0.5), Door: (1.2, -0.6)}
robot.mobile_base.full_body_controlled = False
# TUCK_ARM = robot.mobile_base.full_body_controlled
TUCK_ARM = False


for round_id in range(1, ROUNDS + 1):
    print(f"===== round {round_id}/{ROUNDS} =====")
    for view_type, name, arm in TASKS:
        handle = annotate(view_type, name)
        joint = world.get_connection_by_name(f"{name}_joint")
        print(f"{name} with {arm.name} arm")
        # drive_to(f"{HOME}_handle", *STANDOFF[Door])
        drive_to(f"{name}_handle", *STANDOFF[view_type])
        reset_pos()
        run_plan(execute_single(LookAtAction(handle.global_pose), context=context))
        open_container(handle, arm, tuck=TUCK_ARM)
        print(f"  opened: {joint.position}")
        # robot.mobile_base.full_body_controlled = True
        # close_container(handle, arm, tuck=TUCK_ARM)
        # print(f"  closed: {joint.position}")
    # run_plan(execute_single(MoveTorsoAction(TorsoState.MID), context=context))
    # reset_pos()
    # drive_to(f"{HOME}_handle", *STANDOFF[Door])


# %% ===================================================================
# Perception on the worktop, then pick and place one of the objects.
# Needs ISAAC_KITCHEN_PROPS=1 and camera="both", both set at the top of this file.
# Rough numbers, all here so they can be tuned in one place.
# ======================================================================

LOOK_AT = (0.3, 7.32, 1.0)
PICK_HINT = (0.55, 7.2)
PLACE_POSITION = (0.17, 7.42, 0.98)
PICK_ARM = Arms.LEFT

# (x, y, z) bounds in map. Detections outside this are dropped.
COUNTERTOP = ((-0.30, 0.65), (7.05, 7.45), (0.90, 1.35))

PICK_APPROACH = ApproachDirection.FRONT
PICK_ALIGNMENT = VerticalAlignment.TOP
PICK_CLEARANCE = 0.12   # pre-pose height above the object, and the lift after it
ROTATE_GRIPPER = False  # roll the gripper 90 deg about its approach axis

# %%
# GARMI's URDF has no camera link, so give the twin one at the offset the sim
# publishes as static tf; without it detections cannot leave the camera frame.
from cram_vrb_lab.perception.twin_objects import (
    DETECTION_PREFIX,
    add_detections,
    clear_detections,
    detection_pose_in_map,
    ensure_camera_body,
)
from cram_vrb_lab.robots.garmi.joints import (
    CAMERA_IN_HEAD,
    CAMERA_OPTICAL_IN_HEAD_QUAT,
    CAMERA_PARENT_LINK,
)

camera_body = ensure_camera_body(
    world, CAMERA_PARENT_LINK, CAMERA_IN_HEAD, CAMERA_OPTICAL_IN_HEAD_QUAT
)
print("twin camera frame:", camera_body.name)

# %%
# Never hand rk.detect this file's `node` -- see the warning on make_pipeline_node.
from cram_vrb_lab.perception import pipeline as rk

descriptor = rk.camera_descriptor()

# %%
# drive_to(f"{HOME}_handle", 1.1, -0.2)
# reset_pos()
run_plan(execute_single(
    LookAtAction(Pose(Point3.from_iterable(LOOK_AT), reference_frame=world.root)),
    context=context,
))

# %%
# Only add to the twin once all four are there: a short count means two objects
# merged into one cluster, or one was missed.
EXPECTED_OBJECTS = 4
LOOK_ATTEMPTS = 10


def look_countertop():
    # A fresh pipeline node per detect, or py_trees_ros re-declares its parameters.
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


for attempt in range(1, LOOK_ATTEMPTS + 1):
    print(f"look {attempt}/{LOOK_ATTEMPTS}:")
    kept = look_countertop()
    print(f"  {len(kept)}/{EXPECTED_OBJECTS} on the countertop")
    if len(kept) == EXPECTED_OBJECTS:
        break
    # rk.stop_camera(descriptor)

# %%
# Per detection, not one constant: the four objects are 0.067 to 0.300 m tall, so
# a fixed z means something different for each. The fingers are 0.045 m long, so
# grip a little below the top face instead of at the centre.
GRIP_BELOW_TOP = 0.00

def grasp_offset(detection):
    """Offset from the detected box's centre, in that box's frame (z is world up)."""
    return (detection.extents[0] / 2, 0.0, detection.extents[2] / 2 - GRIP_BELOW_TOP)


bodies = add_detections(world, kept, origin_offset=grasp_offset)
print(f"{len(bodies)} body(ies) added with the {DETECTION_PREFIX!r} prefix "
      f"(pose = grip point, {GRIP_BELOW_TOP} m below each box's top):")
for body, d in zip(bodies, kept):
    print(f"  {body.name.name:14s} h {d.extents[2]:.3f}  "
          f"offset {grasp_offset(d)[2]:+.3f}  map "
          f"{np.round(np.asarray(body.global_pose.to_np())[:3, 3].ravel(), 3)}")

# %%
# Drop every perceived body again -- they go out over /world_sync too, so giskard
# stops colliding against them. add_detections already does this before each look;
# this is for clearing up without looking again.
# print(f"removed {clear_detections(world)} {DETECTION_PREFIX!r} body(ies), "
#       f"{len([b for b in world.bodies if b.name.prefix == DETECTION_PREFIX])} left")

# Done perceiving -- stops the per-frame tf warnings for the rest of the session.
# Left off while you are still re-running the look cell: once this runs, looking again
# needs a new descriptor.
# rk.stop_camera(descriptor)

# %%
target_body = min(
    bodies,
    key=lambda b: float(np.linalg.norm(
        np.asarray(b.global_pose.to_np())[:3, 3].ravel()[:2] - np.asarray(PICK_HINT)
    )),
)
print("picking", target_body.name.name, "at",
      np.round(np.asarray(target_body.global_pose.to_np())[:3, 3].ravel(), 3))

# %%

PICK_ARM = Arms.RIGHT
PICK_APPROACH = ApproachDirection.FRONT
PICK_ALIGNMENT = VerticalAlignment.TOP
PICK_CLEARANCE = 0.02
ROTATE_GRIPPER = True

robot.mobile_base.full_body_controlled = False

# PLACE_POSITION = (0.85, 7.18, 0.17)

# Drawer 2
PLACE_POSITION = (-0.06, 6.85, 0.6)

# PLACE_POSITION = (0.55, 7.25, 1.0)
# target_body=bodies[3]

# %%

nudge_base(turn=math.pi / 9)
# reset_pos()
# tuck_arm(Arms.LEFT)

# %%
# Top-down grasp: VerticalAlignment.TOP is what makes it overhead (ApproachDirection
# has no TOP). PICK_CLEARANCE has to exceed the object's half height, because
# GraspDescription sizes the pre-pose off the approach direction's bbox axis, not z.
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


CARRY_WAYPOINTS = [
    (pick_xyz[0], pick_xyz[1], 1.05),
    (pick_xyz[0], PLACE_POSITION[1], 1.05),
    (PLACE_POSITION[0], PLACE_POSITION[1], 0.95),
    # (0.8, 6.9, 0.2),
]

carry_quaternion = (
    pick_T.to_rotation_matrix() @ pick_grasp.grasp_orientation().to_rotation_matrix()
).to_quaternion()

place_pose = Pose(
    Point3.from_iterable(PLACE_POSITION),
    pick_T.to_quaternion(),
    reference_frame=world.root,
)

carry = MoveTCPWaypointsMotion(
    waypoints=[
        Pose(Point3.from_iterable(point), carry_quaternion, reference_frame=world.root)
        for point in CARRY_WAYPOINTS
    ],
    arm=PICK_ARM,
)

look_at_object = Pose(Point3.from_iterable(pick_xyz), reference_frame=world.root)
look_at_target = Pose(
    Point3.from_iterable(
        CARRY_WAYPOINTS[2]
    ),
    reference_frame=world.root,
)
LIFT_AFTER_PLACE = 0.25
lift_after_place = Pose(
    Point3.from_iterable(
        (PLACE_POSITION[0], PLACE_POSITION[1], PLACE_POSITION[2] + LIFT_AFTER_PLACE)
    ),
    carry_quaternion,
    reference_frame=world.root,
)

done = run_plan(sequential([
    LookAtAction(look_at_object),
    PickUpAction(
        object_designator=target_body,
        arm=PICK_ARM,
        grasp_description=pick_grasp,
    ),
    LookAtAction(look_at_target),
    carry,
    PlaceAction(
        object_designator=target_body,
        target_location=place_pose,
        arm=PICK_ARM,
    ),
    MoveToolCenterPointMotion(lift_after_place, PICK_ARM),
    ParkArmsAction(PICK_ARM),
], context=context))
print("pick and place:", done)

# %%

# handle = annotate(Door, f"{HOME}_handle")
# nudge_base(turn=-math.pi / 6)
robot.mobile_base.full_body_controlled = False
close_container(handle, Arms.LEFT, tuck=False)
nudge_base(-0.5)
reset_pos()

# %%

ROUNDS = 1
STANDOFF = {Drawer: (1.1, 0.5), Door: (1.2, -0.6)}
TASKS = [
    (Door, "cabinet_door_1", Arms.RIGHT),
]
robot.mobile_base.full_body_controlled = True
TUCK_ARM=True

for round_id in range(1, ROUNDS + 1):
    print(f"===== round {round_id}/{ROUNDS} =====")
    for view_type, name, arm in TASKS:
        handle = annotate(view_type, name)
        joint = world.get_connection_by_name(f"{name}_joint")
        print(f"{name} with {arm.name} arm")
        # drive_to(f"{HOME}_handle", *STANDOFF[Door])
        drive_to(f"{name}_handle", *STANDOFF[view_type])
        reset_pos()
        run_plan(execute_single(LookAtAction(handle.global_pose), context=context))
        open_container(handle, arm, tuck=TUCK_ARM)
        print(f"  opened: {joint.position}")
        close_container(handle, arm, tuck=TUCK_ARM)
        print(f"  closed: {joint.position}")


# %%

nudge_base(-0.2)
reset_pos()

# %%
from time import sleep
sleep(10)

stop()

# %%
