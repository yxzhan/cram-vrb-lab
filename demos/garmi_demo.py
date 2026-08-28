# %% [markdown]
# ## Launch

# %%
import math
import os
import sys
import time
from pathlib import Path

REPO = Path.cwd().resolve()
sys.path.insert(0, str(REPO))

os.environ.setdefault("ISAAC_HEADLESS", "1")
os.environ.setdefault("ISAAC_LIVESTREAM", "1")

# os.environ["ISAAC_WINDOW"] = "1280x720"
# os.environ["ISAAC_WINDOW"] = "960x540"
# os.environ["ISAAC_WINDOW"] = "854x480"
# os.environ["ISAAC_WINDOW"] = "768x432"
os.environ["ISAAC_WINDOW"] = "640x360"
# os.environ["ISAAC_WINDOW"] = "512x288"

# Put the four kitchen objects -- cup, bowl, cereal box, milk box -- on the cabinet worktop
os.environ["ISAAC_KITCHEN_PROPS"] = "1"

RVIZ_CONFIG = REPO / "demos" / "rviz" / "garmi.rviz"
ROBOT, SCENE = "garmi", "garmi_apartment"
SPAWN_POSITION = (0, 5.0, 0.0259)
SPAWN_YAW = -math.pi / 2

from launcher import (
    start_giskard_server,
    start_isaac_sim,
    start_rviz,
    start_streaming_client,
    stop,
)
from cram_vrb_lab.sim.isaac_app import livestream_enabled

rviz_proc = start_rviz(rviz_config=RVIZ_CONFIG)
sim_proc = start_isaac_sim(robot=ROBOT, scene=SCENE, camera="both",
                           spawn_position=SPAWN_POSITION, spawn_yaw=SPAWN_YAW)
stream_proc = start_streaming_client() if livestream_enabled() else None
giskard_proc = start_giskard_server(robot=ROBOT, scene=SCENE,
                                    spawn_position=SPAWN_POSITION, spawn_yaw=SPAWN_YAW)

# %% [markdown]
# ## CRAM context

# %%
import logging
import threading

import nest_asyncio
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ApproachDirection, Arms, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.execution_environment import real_robot
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
print(f"connected: {type(robot).__name__} | {len(world.bodies)} bodies")

# %% [markdown]
# ## Plan helpers

# %%
ARRIVED = 0.05
GRASPED = 0.01
RETREAT = 0.12
STANDOFF = {Drawer: (1.1, 0.5), Door: (1.2, -0.6)}
ARM_PREFIX = {Arms.LEFT: "arm_0", Arms.RIGHT: "arm_1"}
TUCK_JOINTS = ["fr3_joint3", "fr3_joint4"]
TUCK_POSITIONS = {Arms.LEFT: [-2, -1.5], Arms.RIGHT: [2, -1.5]}


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


def drive_to(handle_name, standoff, lateral, attempts=10):
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


def work_container(motion, handle, arm, attempts=3, tuck=True):
    if tuck:
        tuck_arm(Arms.RIGHT if arm == Arms.LEFT else Arms.LEFT)
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
# ## Open the drawer

# %%
DRAWER = "drawer_2"
DRAWER_ARM = Arms.LEFT

robot.mobile_base.full_body_controlled = False

drawer_handle = annotate(Drawer, DRAWER)
drawer_joint = world.get_connection_by_name(f"{DRAWER}_joint")

drive_to(f"{DRAWER}_handle", *STANDOFF[Drawer])
reset_pos()
run_plan(execute_single(LookAtAction(drawer_handle.global_pose), context=context))
work_container(OpeningMotion, drawer_handle, DRAWER_ARM, tuck=False)
print("opened:", drawer_joint.position)

# %% [markdown]
# ## Perception

# %%
from cram_vrb_lab.perception import pipeline as rk
from cram_vrb_lab.perception.twin_objects import (
    add_detections,
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


def grasp_offset(detection):
    return (detection.extents[0] / 2, 0.0, detection.extents[2] / 2 - GRIP_BELOW_TOP)


ensure_camera_body(
    world, CAMERA_PARENT_LINK, CAMERA_IN_HEAD, CAMERA_OPTICAL_IN_HEAD_QUAT
)
descriptor = rk.camera_descriptor()

# %%
run_plan(execute_single(
    LookAtAction(Pose(Point3.from_iterable(LOOK_AT), reference_frame=world.root)),
    context=context,
))

for attempt in range(1, LOOK_ATTEMPTS + 1):
    print(f"look {attempt}/{LOOK_ATTEMPTS}:")
    kept = look_countertop()
    print(f"  {len(kept)}/{EXPECTED_OBJECTS} on the countertop")
    if len(kept) == EXPECTED_OBJECTS:
        break

bodies = add_detections(world, kept, origin_offset=grasp_offset)
for body, d in zip(bodies, kept):
    print(f"  {body.name.name:14s} h {d.extents[2]:.3f}  map "
          f"{np.round(np.asarray(body.global_pose.to_np())[:3, 3].ravel(), 3)}")

# %% [markdown]
# ## Pick and place

# %%
PICK_HINT = (0.55, 7.2)
PICK_ARM = Arms.RIGHT
PICK_APPROACH = ApproachDirection.FRONT
PICK_ALIGNMENT = VerticalAlignment.TOP
PICK_CLEARANCE = 0.02
ROTATE_GRIPPER = True
PLACE_POSITION = (-0.06, 6.85, 0.6)
LIFT_AFTER_PLACE = 0.25

robot.mobile_base.full_body_controlled = False

target_body = min(
    bodies,
    key=lambda b: float(np.linalg.norm(
        np.asarray(b.global_pose.to_np())[:3, 3].ravel()[:2] - np.asarray(PICK_HINT)
    )),
)
print("picking", target_body.name.name)

nudge_base(turn=math.pi / 9)

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

carry_waypoints = [
    (pick_xyz[0], pick_xyz[1], 1.05),
    (pick_xyz[0], PLACE_POSITION[1], 1.05),
    (PLACE_POSITION[0], PLACE_POSITION[1], 0.95),
]
carry_quaternion = (
    pick_T.to_rotation_matrix() @ pick_grasp.grasp_orientation().to_rotation_matrix()
).to_quaternion()

done = run_plan(sequential([
    LookAtAction(Pose(Point3.from_iterable(pick_xyz), reference_frame=world.root)),
    PickUpAction(
        object_designator=target_body,
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
    PlaceAction(
        object_designator=target_body,
        target_location=Pose(
            Point3.from_iterable(PLACE_POSITION),
            pick_T.to_quaternion(),
            reference_frame=world.root,
        ),
        arm=PICK_ARM,
    ),
    MoveToolCenterPointMotion(
        Pose(
            Point3.from_iterable(
                (PLACE_POSITION[0], PLACE_POSITION[1],
                 PLACE_POSITION[2] + LIFT_AFTER_PLACE)
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

work_container(ClosingMotion, drawer_handle, DRAWER_ARM, tuck=False)
nudge_base(-0.5)
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

work_container(OpeningMotion, door_handle, DOOR_ARM, tuck=True)
print("opened:", door_joint.position)

work_container(ClosingMotion, door_handle, DOOR_ARM, tuck=True)
print("closed:", door_joint.position)

# %%
nudge_base(-0.2)
reset_pos()

# %% [markdown]
# ## Shutdown

# %%
time.sleep(10)
stop()
