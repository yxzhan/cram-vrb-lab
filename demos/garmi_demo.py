# %%
import math
import os
import sys
from pathlib import Path

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
sim_proc = start_isaac_sim(robot=ROBOT, scene=SCENE, camera="none",
                           spawn_position=SPAWN_POSITION, spawn_yaw=SPAWN_YAW)
stream_proc = start_streaming_client() if livestream_enabled() else None
giskard_proc = start_giskard_server(robot=ROBOT, scene=SCENE,
                                    spawn_position=SPAWN_POSITION, spawn_yaw=SPAWN_YAW)

# %%
import threading

import nest_asyncio
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

nest_asyncio.apply()

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ApproachDirection, Arms, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.execution_environment import real_robot
from coraplex.plans.factories import execute_single, sequential
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import GraspingAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction, SetGripperAction
from coraplex.robot_plans.motions.container import ClosingMotion, OpeningMotion
from coraplex.robot_plans.motions.gripper import MoveGripperMotion
from coraplex.robot_plans.motions.robot_body import MoveJointsMotion
from coraplex.view_manager import ViewManager
from giskardpy.data_types.exceptions import GiskardException
from semantic_digital_twin.adapters.ros.world_fetcher import fetch_world_from_service
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.robots.garmi import Garmi
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Door,
    Drawer,
    Handle,
)
from semantic_digital_twin.spatial_types import Point3, Quaternion
from semantic_digital_twin.spatial_types.spatial_types import Pose

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
print("connected:", type(robot).__name__, "|", len(world.bodies), "bodies")

# %%
robot.mobile_base.full_body_controlled = True

# STANDOFF = {Drawer: (1.5, 0.0), Door: (1.2, -0.3)}
STANDOFF = {Drawer: (1.5, 0.0), Door: (1.3, -0.4)}
ARRIVED = 0.05
GRASPED = 0.01

ARM_PREFIX = {Arms.LEFT: "arm_0", Arms.RIGHT: "arm_1"}
TUCK_JOINTS = ["fr3_joint2", "fr3_joint3"]
TUCK_POSITIONS = {Arms.LEFT: [-2, -1], Arms.RIGHT: [-2, 1]}
HOME = "cabinet_door_1"


# %%
def run_plan(plan):
    try:
        with real_robot(collision_avoidance=True):
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


def station_facing(handle_x, standoff, lateral):
    return Pose(
        Point3.from_iterable([handle_x + lateral, 7.12 - standoff, 0.0]),
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


def work_container(motion, handle, arm, attempts=3):
    tuck_arm(other_arm(arm))
    grasp_handle(handle, arm, attempts)
    run_plan(execute_single(motion(handle, arm), context=context))
    run_plan(execute_single(MoveGripperMotion(GripperState.OPEN, arm), context=context))


def open_container(handle, arm, attempts=3):
    work_container(OpeningMotion, handle, arm, attempts)


def close_container(handle, arm, attempts=3):
    work_container(ClosingMotion, handle, arm, attempts)


def reset_pos():
    run_plan(sequential([
        SetGripperAction(Arms.LEFT, GripperState.OPEN),
        SetGripperAction(Arms.RIGHT, GripperState.OPEN),
        ParkArmsAction(Arms.LEFT),
        ParkArmsAction(Arms.RIGHT),
    ], context=context))

# %%
drive_to(f"{HOME}_handle", *STANDOFF[Door])
reset_pos()

# %%
ROUNDS = 10
TASKS = [
    (Drawer, "drawer_1", Arms.LEFT),
    (Door, "cabinet_door_1", Arms.RIGHT),
    (Drawer, "drawer_2", Arms.LEFT),
    (Drawer, "drawer_3", Arms.RIGHT),
    # (Drawer, "drawer_4", Arms.RIGHT),
]

# %%
for round_id in range(1, ROUNDS + 1):
    print(f"===== round {round_id}/{ROUNDS} =====")
    for view_type, name, arm in TASKS:
        handle = annotate(view_type, name)
        joint = world.get_connection_by_name(f"{name}_joint")
        print(f"{name} with {arm.name} arm")
        drive_to(f"{name}_handle", *STANDOFF[view_type])
        reset_pos()
        open_container(handle, arm)
        print(f"  opened: {joint.position}")
        close_container(handle, arm)
        print(f"  closed: {joint.position}")
    drive_to(f"{HOME}_handle", *STANDOFF[Door])
    reset_pos()


