# %%
import math
import sys
from pathlib import Path

REPO = Path.cwd().resolve()  # this notebook lives in demos/
sys.path.insert(0, str(REPO))

# import os
# os.environ["DISPLAY"] = ":0"

from launcher import start_giskard_server, start_isaac_sim, start_rviz, stop

RVIZ_CONFIG = REPO / "demos" / "rviz" / "garmi.rviz"
ROBOT, SCENE = "panda", "garmi_apartment"

# Bolted to the floor in front of the kitchen run, facing map +y at the cabinet
# fronts 0.5 m away. Clear floor: nearest MJCF body is the cabinet, 0.77 m off.
# SPAWN_POSITION = (1.2, 6.8, 0.0)
SPAWN_POSITION = (1.3, 6.6, 0.0)
SPAWN_YAW = math.pi

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

nest_asyncio.apply()  # CRAM's REAL execution calls GiskardWrapper.execute,
                      # which run_until_completes inside the already-running kernel loop.

from coraplex.datastructures.dataclasses import Context
from semantic_digital_twin.adapters.ros.world_fetcher import fetch_world_from_service
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer

from cram_vrb_lab.robots.panda.motions import PANDA_MOTION_MAPPINGS
from cram_vrb_lab.robots.panda.semantic_model import Panda

if not rclpy.ok():
    rclpy.init()
node = rclpy.create_node('cram_panda_garmi_node')
executor = MultiThreadedExecutor()
executor.add_node(node)
threading.Thread(target=executor.spin, daemon=True, name='rclpy-executor').start()

world = fetch_world_from_service(node=node, timeout_seconds=300)
WorldSynchronizer(_world=world, node=node)

robot = world.get_semantic_annotations_by_type(Panda)
robot = robot[0] if robot else Panda.from_world(world)

context = Context(
    world=world,
    robot=robot,
    ros_node=node,
    evaluate_conditions=False,
    alternative_motion_mappings=PANDA_MOTION_MAPPINGS,
)
print('connected, robot:', type(robot).__name__)
print('bodies in the twin:', len(world.bodies), '-- the arm plus the whole flat')

# %%
from coraplex.datastructures.enums import Arms
from coraplex.execution_environment import real_robot
from coraplex.plans.factories import execute_single, sequential
from coraplex.view_manager import ViewManager


def run_plan(plan, collision_avoidance=True):
    """Perform a CRAM plan on the real (sim) robot via giskard."""
    with real_robot(collision_avoidance=collision_avoidance):
        plan.perform()
    print('done')


def tool_position():
    """The gripper's tool frame, in map."""
    tool = ViewManager.get_end_effector_view(Arms.LEFT, robot).tool_frame
    return np.asarray(tool.global_pose.to_np())[:3, 3].ravel()


def tool_axis():
    """Where the gripper points: the tool frame's z-axis, in map."""
    tool = ViewManager.get_end_effector_view(Arms.LEFT, robot).tool_frame
    return np.asarray(tool.global_pose.to_np())[:3, 2].ravel()


def body_position(name):
    """Any body of the flat, in map -- straight out of the MJCF twin."""
    return np.asarray(world.get_body_by_name(name).global_pose.to_np())[:3, 3].ravel()

# %%
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction, SetGripperAction
from semantic_digital_twin.datastructures.definitions import GripperState

# run_plan(execute_single(ParkArmsAction(Arms.LEFT), context=context))
# run_plan(execute_single(SetGripperAction(Arms.LEFT, GripperState.OPEN), context=context))

# %%
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Drawer,
    Handle,
    Door,
)

from coraplex.robot_plans.actions.core.container import OpenAction, CloseAction

drawer_id = "4"
drawer_body = world.get_body_by_name(f"drawer_{drawer_id}")
handle_body = world.get_body_by_name(f"drawer_{drawer_id}_handle")

if not world.get_semantic_annotations_by_type(Drawer):
    with world.modify_world():
        world.add_semantic_annotation_recursively(
            Drawer(root=drawer_body, handle=Handle(root=handle_body))
        )
print("drawer annotated:", drawer_body.name, "with handle", handle_body.name)


door_id = "2"
door_body = world.get_body_by_name(f"cabinet_door_{door_id}")
door_handle_body = world.get_body_by_name(f"cabinet_door_{door_id}_handle")

if not world.get_semantic_annotations_by_type(Door):
    with world.modify_world():
        world.add_semantic_annotation_recursively(
            Door(root=door_body, handle=Handle(root=door_handle_body))
        )
print("Door annotated:", door_body.name, "with handle", door_handle_body.name)


# %%
run_plan(
    execute_single(OpenAction(handle_body, Arms.LEFT), context=context),
    collision_avoidance=False,
)
print("Opened: drawer joint:", world.get_connection_by_name(f"drawer_{drawer_id}_joint").position)

# %%
run_plan(
    execute_single(CloseAction(handle_body, Arms.LEFT), context=context),
    collision_avoidance=False,
)
print("Closed: drawer joint:", world.get_connection_by_name(f"drawer_{drawer_id}_joint").position)

# %%
run_plan(execute_single(ParkArmsAction(Arms.LEFT), context=context))
run_plan(execute_single(SetGripperAction(Arms.LEFT, GripperState.OPEN), context=context))

# %%
run_plan(
    execute_single(OpenAction(door_handle_body, Arms.LEFT), context=context),
    collision_avoidance=False,
)
print("Opened, Door joint:", world.get_connection_by_name(f"cabinet_door_{door_id}_joint").position)

# %%
run_plan(
    execute_single(CloseAction(door_handle_body, Arms.LEFT), context=context),
    collision_avoidance=False,
)
print("Closed, Door joint:", world.get_connection_by_name(f"cabinet_door_{door_id}_joint").position)

# %% [markdown]
# ## Shutdown

# %%
stop()  # stops the isaac sim + giskard server + rviz started above
