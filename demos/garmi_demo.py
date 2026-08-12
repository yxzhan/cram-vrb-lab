# %%
import math
import sys
from pathlib import Path

REPO = Path.cwd().resolve()
sys.path.insert(0, str(REPO))

# import os
# os.environ["DISPLAY"] = ":0"

from launcher import start_giskard_server, start_isaac_sim, start_rviz, stop

RVIZ_CONFIG = REPO / "demos" / "rviz" / "garmi.rviz"
ROBOT, SCENE = "garmi", "garmi_apartment"

# Where GARMI *starts*, not where it works from -- the base drives, so the demo
# navigates to the cabinet below. Anywhere with clear floor will do; this is the
# middle of the living room, facing map +y. z is BASE_LINK_HEIGHT: the base is
# teleported rather than rolled (see undrive_wheels), so nothing settles the
# wheels onto the floor by itself.
# Only the sim is told this. Giskard learns the pose from /odom and the static
# map->odom, the way a real robot's localization would deliver it.
SPAWN_POSITION = (0.5, 6.0, 0.0259)
SPAWN_YAW = math.pi / 2

rviz_proc = start_rviz(rviz_config=RVIZ_CONFIG)
sim_proc = start_isaac_sim(robot=ROBOT, scene=SCENE, camera="none",
                           spawn_position=SPAWN_POSITION, spawn_yaw=SPAWN_YAW)
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
from semantic_digital_twin.adapters.ros.world_fetcher import fetch_world_from_service
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer

from cram_vrb_lab.robots.garmi.motions import GARMI_MOTION_MAPPINGS
from semantic_digital_twin.robots.garmi import Garmi

if not rclpy.ok():
    rclpy.init()
node = rclpy.create_node('cram_garmi_node')
executor = MultiThreadedExecutor()
executor.add_node(node)
threading.Thread(target=executor.spin, daemon=True, name='rclpy-executor').start()

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
print('connected, robot:', type(robot).__name__)
print('bodies in the twin:', len(world.bodies), '-- GARMI plus the whole flat')

# %%
from coraplex.datastructures.enums import Arms
from coraplex.execution_environment import real_robot
from coraplex.plans.factories import execute_single, sequential
from coraplex.view_manager import ViewManager

ARM = Arms.LEFT


def run_plan(plan, collision_avoidance=True):
    with real_robot(collision_avoidance=collision_avoidance):
        plan.perform()
    print('done')


def tool_position(arm=ARM):
    tool = ViewManager.get_end_effector_view(arm, robot).tool_frame
    return np.asarray(tool.global_pose.to_np())[:3, 3].ravel()


def tool_axis(arm=ARM):
    tool = ViewManager.get_end_effector_view(arm, robot).tool_frame
    return np.asarray(tool.global_pose.to_np())[:3, 2].ravel()


def body_position(name):
    return np.asarray(world.get_body_by_name(name).global_pose.to_np())[:3, 3].ravel()

# %%
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction, SetGripperAction
from semantic_digital_twin.datastructures.definitions import GripperState

run_plan(sequential([
        ParkArmsAction(Arms.LEFT),
        ParkArmsAction(Arms.RIGHT),
        SetGripperAction(Arms.LEFT, GripperState.OPEN),
        SetGripperAction(Arms.RIGHT, GripperState.OPEN),
    ], context=context))

# %%
# Driving is what the mobile base buys: the kitchen run is 2.5 m wide, so no one
# standing position reaches all of it, and the demo drives to each handle instead
# of being spawned in front of one. The base is an OmniDrive, so giskard is free
# to solve this sideways as well as forwards; the sim consumes linear.y.
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from semantic_digital_twin.spatial_types import Point3, Quaternion
from semantic_digital_twin.spatial_types.spatial_types import Pose

STANDOFF = 1.1
"""How far south of the cabinet fronts (y = 7.12) to stand [m].

The home pose holds the hands 0.79 m in front of base_link, so anything closer
parks them inside the cabinet.
"""

SHOULDER_OFFSET = 0.061
"""The left shoulder's y offset in the base frame [m]. Facing map +y that is an
offset in -x, so a base standing at ``handle_x + this`` puts the shoulder on the
handle."""


def station_facing(handle_x):
    """A base pose in front of ``handle_x`` on the kitchen run, facing map +y.

    ``reference_frame`` is not optional: without it the pose reaches giskard with
    no frame to resolve against and the Cartesian goal is built with a null tip,
    which fails inside the solver rather than at the call.
    """
    return Pose(
        Point3.from_iterable([handle_x + SHOULDER_OFFSET, 7.12 - STANDOFF, 0.0]),
        Quaternion.from_iterable([0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)]),
        reference_frame=world.root,
    )


ARRIVED = 0.05
"""How close to the station counts as arrived [m]."""


def drive_to(handle_name, attempts=3):
    """Drive to the handle, repeating until the base is actually there.

    The first NavigateAction routinely ends short -- it drives most of the way
    and then the goal finishes -- while a second one closes the remaining
    distance to a millimetre. Rather than paper over that with a longer motion,
    the demo just asks again until the base has arrived; a call that is already
    there returns in about a second, so the retry costs nothing when it is not
    needed.
    """
    target = station_facing(float(body_position(handle_name)[0]))
    goal = np.asarray(target.to_np())[:2, 3].ravel()
    for attempt in range(1, attempts + 1):
        run_plan(execute_single(NavigateAction(target), context=context))
        error = float(np.linalg.norm(body_position('base_link')[:2] - goal))
        print(f'  navigate {attempt}: base {np.round(body_position("base_link"), 3)}'
              f' error {error:.3f} m')
        if error <= ARRIVED:
            break
    else:
        print(f'  WARNING: still {error:.3f} m from the station after {attempts} tries')
    print('at', handle_name, np.round(body_position(handle_name), 3))

# %%
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Drawer,
    Handle,
    Door,
)

from coraplex.robot_plans.actions.core.container import OpenAction, CloseAction

drawer_id = "1"
drawer_body = world.get_body_by_name(f"drawer_{drawer_id}")
handle_body = world.get_body_by_name(f"drawer_{drawer_id}_handle")

if not world.get_semantic_annotations_by_type(Drawer):
    with world.modify_world():
        world.add_semantic_annotation_recursively(
            Drawer(root=drawer_body, handle=Handle(root=handle_body))
        )
print("drawer annotated:", drawer_body.name, "with handle", handle_body.name)
print("handle at", np.round(body_position(f"drawer_{drawer_id}_handle"), 3))


door_id = "1"
door_body = world.get_body_by_name(f"cabinet_door_{door_id}")
door_handle_body = world.get_body_by_name(f"cabinet_door_{door_id}_handle")

if not world.get_semantic_annotations_by_type(Door):
    with world.modify_world():
        world.add_semantic_annotation_recursively(
            Door(root=door_body, handle=Handle(root=door_handle_body))
        )
print("Door annotated:", door_body.name, "with handle", door_handle_body.name)


# %%
drive_to(f"drawer_{drawer_id}_handle")

run_plan(
    execute_single(OpenAction(handle_body, Arms.LEFT), context=context),
    collision_avoidance=True,
)
print("Opened: drawer joint:", world.get_connection_by_name(f"drawer_{drawer_id}_joint").position)

# %%
# run_plan(
#     execute_single(CloseAction(handle_body, Arms.LEFT), context=context),
#     collision_avoidance=False,
# )
# print("Closed: drawer joint:", world.get_connection_by_name(f"drawer_{drawer_id}_joint").position)

run_plan(sequential([
        ParkArmsAction(Arms.LEFT),
        ParkArmsAction(Arms.RIGHT),
        SetGripperAction(Arms.LEFT, GripperState.OPEN),
        SetGripperAction(Arms.RIGHT, GripperState.OPEN),
    ], context=context))

drive_to(f"cabinet_door_{door_id}_handle")

run_plan(
    execute_single(OpenAction(door_handle_body, Arms.RIGHT), context=context),
    collision_avoidance=True,
)
print("Opened, Door joint:", world.get_connection_by_name(f"cabinet_door_{door_id}_joint").position)

# %%
# run_plan(
#     execute_single(CloseAction(door_handle_body, Arms.LEFT), context=context),
#     collision_avoidance=False,
# )
# print("Closed, Door joint:", world.get_connection_by_name(f"cabinet_door_{door_id}_joint").position)


run_plan(sequential([
        ParkArmsAction(Arms.LEFT),
        ParkArmsAction(Arms.RIGHT),
        SetGripperAction(Arms.LEFT, GripperState.OPEN),
        SetGripperAction(Arms.RIGHT, GripperState.OPEN),
    ], context=context))

# %%
stop()
