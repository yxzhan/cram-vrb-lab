
import sys
from pathlib import Path

REPO = Path.cwd().resolve()  # this notebook lives in demos/
sys.path.insert(0, str(REPO))

# import os
# os.environ["DISPLAY"] = ":0"

from launcher import start_isaac_sim, start_giskard_server, start_rviz, stop
SPAWN_POSITION = (-1.5, 0.0, 0.05)

rviz_proc = start_rviz()
sim_proc = start_isaac_sim(spawn_position=SPAWN_POSITION, camera="both")   # both RGB and depth
giskard_proc = start_giskard_server(spawn_position=SPAWN_POSITION)

# %%
import threading

import nest_asyncio
import rclpy
from rclpy.executors import MultiThreadedExecutor

nest_asyncio.apply()  # CRAM's REAL execution calls GiskardWrapper.execute,
                      # which run_until_completes inside the already-running kernel loop.

from coraplex.datastructures.dataclasses import Context
from coraplex.alternative_motion_mappings.stretch_motion_mapping import (
    StretchMoveToolCenterPoint,
    StretchMoveSim,
    StretchMoveReal,
    StretchClose,
)
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.adapters.ros.world_fetcher import fetch_world_from_service
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer

STRETCH_MOTION_MAPPINGS = [
    StretchMoveToolCenterPoint,
    StretchMoveSim,
    StretchMoveReal,
    StretchClose,
]

if not rclpy.ok():
    rclpy.init()
node = rclpy.create_node('cram_perception_node')
executor = MultiThreadedExecutor()
executor.add_node(node)
threading.Thread(target=executor.spin, daemon=True, name='rclpy-executor').start()

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
print('connected, robot:', type(robot).__name__)

# %% [markdown]
# ## A small run helper

# %%
from coraplex.execution_environment import real_robot
from coraplex.plans.factories import sequential, execute_single


def run_plan(plan, collision_avoidance=True):
    """Perform a CRAM plan on the real (sim) robot via giskard."""
    with real_robot(collision_avoidance=collision_avoidance):
        plan.perform()
    print('done')

# %%
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from semantic_digital_twin.spatial_types import Pose, Quaternion, Point3

waypoints = [
    ([0.0, -0.5, 0.0], [0, 0, 0, 1]),
    # ([0.0, 2, 0.0], [0, 0, 0, 1]),
    # ([1.8, 2, 0.0], [0, 0, 0, 1]),
    # ([1.8, 0, 0.0], [0, 0, -1, 0])
]

for _waypoint in waypoints:
    target = Pose(
        Point3.from_iterable(_waypoint[0]),
        Quaternion.from_iterable(_waypoint[1]),
        reference_frame=world.root
    )
    run_plan(execute_single(NavigateAction(target), context=context))

# %%
from coraplex.robot_plans.actions.core.navigation import LookAtAction
from semantic_digital_twin.spatial_types import Pose, Point3


countertop = world.get_body_by_name("island_countertop")

run_plan(execute_single(LookAtAction(countertop.global_pose), context=context))

from cram_vrb_lab.perception.twin_objects import camera_pose_in_map
import numpy as np

print('camera in map:\n', np.round(camera_pose_in_map(world), 3))


# %%
import matplotlib.pyplot as plt

from sensor_msgs.msg import Image

from cram_vrb_lab.robots.stretch.joints import DEPTH_IMAGE_TOPIC, RGB_IMAGE_TOPIC


def grab(topic, timeout=15.0):
    """Return the next message on `topic`, using the executor already spinning."""
    import time
    box = {}
    sub = node.create_subscription(Image, topic, lambda m: box.setdefault('m', m), 1)
    try:
        deadline = time.time() + timeout
        while 'm' not in box and time.time() < deadline:
            time.sleep(0.05)
    finally:
        node.destroy_subscription(sub)
    if 'm' not in box:
        raise TimeoutError(f'nothing published on {topic} within {timeout}s '
                           '-- was the sim started with camera="both"?')
    return box['m']


rgb_msg, depth_msg_ = grab(RGB_IMAGE_TOPIC), grab(DEPTH_IMAGE_TOPIC)
rgb = np.frombuffer(rgb_msg.data, np.uint8).reshape(rgb_msg.height, rgb_msg.width, 3)
depth = np.frombuffer(depth_msg_.data, np.float32).reshape(depth_msg_.height,
                                                           depth_msg_.width)

# print(f'rgb   {rgb.shape} {rgb_msg.encoding}  frame={rgb_msg.header.frame_id}')
# print(f'depth {depth.shape} {depth_msg_.encoding}  '
#       f'valid {np.isfinite(depth).mean():.0%}  '
#       f'range {np.nanmin(depth):.2f}..{np.nanmax(depth[np.isfinite(depth)]):.2f} m')

# fig, axes = plt.subplots(1, 2, figsize=(13, 4))
# axes[0].imshow(rgb); axes[0].set_title('rgb8'); axes[0].axis('off')
# im = axes[1].imshow(np.where(np.isfinite(depth), depth, np.nan), cmap='viridis')
# axes[1].set_title('depth [m]'); axes[1].axis('off')
# fig.colorbar(im, ax=axes[1], shrink=0.8)
# plt.tight_layout()
# plt.show()

# %%
from cram_vrb_lab.perception import pipeline as rk

# Built once and reused: constructing the descriptor spins up robokudo's own camera
# node and its subscriptions, and a second one would just duplicate them.
descriptor = rk.camera_descriptor()

rk_node = rk.make_pipeline_node()


# %%
detections = rk.detect(rk_node, descriptor)

print(f'{len(detections)} cluster(s), poses in camera_color_optical_frame:')
for i, d in enumerate(detections):
    print(f'  [{i}] pos {np.round(d.position, 3)}  '
          f'extent {np.round(d.extents, 3)}  volume {d.volume * 1e3:.2f} L')

# %%
from cram_vrb_lab.perception.twin_objects import add_detections, DETECTION_PREFIX

bodies = add_detections(world, detections)

print(f'{len(bodies)} body(ies) added to the twin:')
for body in bodies:
    position = np.asarray(body.global_pose.to_np())[:3, 3].ravel()
    print(f'  {body.name.name:16s} map {np.round(position, 3)}')

print(f'\nbodies now carrying the {DETECTION_PREFIX!r} prefix:',
      len([b for b in world.bodies if b.name.prefix == DETECTION_PREFIX]))


# %%
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.datastructures.grasp import GraspDescription
from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from coraplex.view_manager import ViewManager
from coraplex.robot_plans.actions.core.robot_body import (
    ParkArmsAction,
    MoveTorsoAction,
)
from coraplex.datastructures.enums import Arms
from semantic_digital_twin.datastructures.definitions import TorsoState

run_plan(sequential([
    ParkArmsAction(Arms.LEFT),
    MoveTorsoAction(TorsoState.HIGH),
], context=context))

grasp = GraspDescription(
    ApproachDirection.FRONT,   # approach along map -y = along the arm
    VerticalAlignment.NoAlignment,
    ViewManager.get_end_effector_view(Arms.LEFT, robot),
)

# %%
waypoints = [
    ([0.0, 2, 0.0], [0, 0, 0, 1]),
    ([1.8, 2, 0.0], [0, 0, 0, 1]),
    ([1.8, 0, 0.0], [0, 0, -1, 0])
]

for _waypoint in waypoints:
    target = Pose(
        Point3.from_iterable(_waypoint[0]),
        Quaternion.from_iterable(_waypoint[1]),
        reference_frame=world.root
    )
    run_plan(execute_single(NavigateAction(target), context=context))

# %%
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Drawer,
    Handle,
)

drawer_body = world.get_body_by_name("cabinet9_drawer_middle")
handle_body = world.get_body_by_name("handle_cab9_m")

if not world.get_semantic_annotations_by_type(Drawer):
    with world.modify_world():
        world.add_semantic_annotation_recursively(
            Drawer(root=drawer_body, handle=Handle(root=handle_body))
        )
print("drawer annotated:", drawer_body.name, "with handle", handle_body.name)

# %%
from coraplex.robot_plans.actions.core.navigation import LookAtAction

run_plan(execute_single(LookAtAction(handle_body.global_pose), context=context))

# %%
from coraplex.robot_plans.actions.core.container import OpenAction, CloseAction

run_plan(
    execute_single(OpenAction(handle_body, Arms.LEFT), context=context),
    collision_avoidance=False,
)
print("drawer joint:", world.get_connection_by_name("cabinet10_drawer_middle_joint").position)

# %% [markdown]
# ## Shutdown

# %%
stop()  # stops the isaac sim + giskard server + rviz started above


