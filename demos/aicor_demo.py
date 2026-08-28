# %% [markdown]
# ## Launch

# %%
import os
import sys
from pathlib import Path

REPO = Path.cwd().resolve()
sys.path.insert(0, str(REPO))

os.environ["ISAAC_WINDOW"] = "960x540"

SPAWN_POSITION = (-1.5, 0.0, 0.05)

from launcher import (
    start_giskard_server,
    start_isaac_sim,
    start_rviz,
    start_streaming_client,
    stop,
)
from cram_vrb_lab.sim.isaac_app import livestream_enabled

rviz_proc = start_rviz(vgl=True)
sim_proc = start_isaac_sim(spawn_position=SPAWN_POSITION, camera="both")
stream_proc = start_streaming_client() if livestream_enabled() else None
giskard_proc = start_giskard_server(spawn_position=SPAWN_POSITION)

# %% [markdown]
# ## CRAM context

# %%
import threading

import nest_asyncio
import rclpy
from rclpy.executors import MultiThreadedExecutor

from coraplex.alternative_motion_mappings.stretch_motion_mapping import (
    StretchClose,
    StretchMoveReal,
    StretchMoveSim,
    StretchMoveToolCenterPoint,
)
from coraplex.datastructures.dataclasses import Context
from semantic_digital_twin.adapters.ros.world_fetcher import fetch_world_from_service
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.robots.stretch import Stretch

STRETCH_MOTION_MAPPINGS = [
    StretchMoveToolCenterPoint,
    StretchMoveSim,
    StretchMoveReal,
    StretchClose,
]

nest_asyncio.apply()

if not rclpy.ok():
    rclpy.init()
node = rclpy.create_node("cram_perception_node")
executor = MultiThreadedExecutor()
executor.add_node(node)
threading.Thread(target=executor.spin, daemon=True, name="rclpy-executor").start()

world = fetch_world_from_service(node=node, timeout_seconds=300)
WorldSynchronizer(_world=world, node=node)

robot = world.get_semantic_annotations_by_type(Stretch)
robot = robot[0] if robot else Stretch.from_world(world)

context = Context(
    world=world,
    robot=robot,
    ros_node=node,
    evaluate_conditions=False,
    alternative_motion_mappings=STRETCH_MOTION_MAPPINGS,
)
print("connected, robot:", type(robot).__name__)

# %% [markdown]
# ## Perception

# %%
import numpy as np

from coraplex.execution_environment import real_robot
from coraplex.plans.factories import execute_single
from coraplex.robot_plans.actions.core.navigation import LookAtAction, NavigateAction
from semantic_digital_twin.spatial_types import Point3, Pose, Quaternion

from cram_vrb_lab.perception import pipeline as rk
from cram_vrb_lab.perception.twin_objects import add_detections

COUNTERTOP = "island_countertop"
COUNTERTOP_WAYPOINTS = [
    ([0.0, -0.5, 0.0], [0, 0, 0, 1]),
]

for position, orientation in COUNTERTOP_WAYPOINTS:
    target = Pose(
        Point3.from_iterable(position),
        Quaternion.from_iterable(orientation),
        reference_frame=world.root,
    )
    with real_robot(collision_avoidance=True):
        execute_single(NavigateAction(target), context=context).perform()

countertop = world.get_body_by_name(COUNTERTOP)
with real_robot(collision_avoidance=True):
    execute_single(LookAtAction(countertop.global_pose), context=context).perform()

# %%
descriptor = rk.camera_descriptor()
rk_node = rk.make_pipeline_node()

# %%
detections = rk.detect(rk_node, descriptor)
bodies = add_detections(world, detections)

print(f"{len(bodies)} body(ies) added to the twin:")
for body, detection in zip(bodies, detections):
    position = np.asarray(body.global_pose.to_np())[:3, 3].ravel()
    print(f"  {body.name.name:16s} map {np.round(position, 3)}  "
          f"extent {np.round(detection.extents, 3)}")

# %% [markdown]
# ## Drawer

# %%
from coraplex.datastructures.enums import Arms
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.core.container import OpenAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Drawer,
    Handle,
)

DRAWER = "cabinet9_drawer_middle"
DRAWER_HANDLE = "handle_cab9_m"
DRAWER_JOINT = "cabinet9_drawer_middle_joint"
DRAWER_WAYPOINTS = [
    ([0.0, 2.0, 0.0], [0, 0, 0, 1]),
    ([1.8, 2.0, 0.0], [0, 0, 0, 1]),
    ([1.8, 0.0, 0.0], [0, 0, -1, 0]),
]

with real_robot(collision_avoidance=True):
    sequential([
        ParkArmsAction(Arms.LEFT),
        MoveTorsoAction(TorsoState.HIGH),
    ], context=context).perform()

for position, orientation in DRAWER_WAYPOINTS:
    target = Pose(
        Point3.from_iterable(position),
        Quaternion.from_iterable(orientation),
        reference_frame=world.root,
    )
    with real_robot(collision_avoidance=True):
        execute_single(NavigateAction(target), context=context).perform()

# %%
drawer_body = world.get_body_by_name(DRAWER)
handle_body = world.get_body_by_name(DRAWER_HANDLE)

if not world.get_semantic_annotations_by_type(Drawer):
    with world.modify_world():
        world.add_semantic_annotation_recursively(
            Drawer(root=drawer_body, handle=Handle(root=handle_body))
        )

with real_robot(collision_avoidance=True):
    execute_single(LookAtAction(handle_body.global_pose), context=context).perform()

with real_robot(collision_avoidance=False):
    execute_single(OpenAction(handle_body, Arms.LEFT), context=context).perform()

print("drawer joint:", world.get_connection_by_name(DRAWER_JOINT).position)

# %% [markdown]
# ## Shutdown

# %%
stop()
