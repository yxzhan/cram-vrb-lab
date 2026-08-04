
import sys
from pathlib import Path

REPO = Path.cwd().resolve().parent  # this notebook lives in demos/
sys.path.insert(0, str(REPO))

from launcher import start_isaac_sim, start_giskard_server, start_rviz, stop

RVIZ_CONFIG = REPO / "demos" / "garmi.rviz"
SIM_SCRIPT = REPO / "demos" / "stretch_garmi_apartment_sim.py"
SERVER_SCRIPT = REPO / "demos" / "stretch_garmi_apartment_giskard_server.py"

rviz_proc = start_rviz(rviz_config=RVIZ_CONFIG)
sim_proc = start_isaac_sim(sim_script=SIM_SCRIPT, camera="both")
giskard_proc = start_giskard_server(server_script=SERVER_SCRIPT)

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
node = rclpy.create_node('cram_garmi_perception_node')
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
print('bodies in the twin:', len(world.bodies))

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
from coraplex.robot_plans.actions.core.robot_body import SetGripperAction
from semantic_digital_twin.datastructures.definitions import GripperState
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

run_plan(execute_single(SetGripperAction(Arms.LEFT, GripperState.OPEN), context=context))

# %%
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from semantic_digital_twin.spatial_types import Pose, Quaternion, Point3

waypoints = [
    # ([0.5, 6.5, 0.0], [0, 0, 0, 1]),
    ([0.5, 6.5, 0.0], [0, 0, 1, 1]),

    # ([-1.0, 4.5, 0.0], [0, 0, -1, 0]),
    # ([1.8, 5, 0.0], [0, 0, 0, 1]),
    # ([-1.5, 5, 0.0], [0, 0, -1, 0])
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

from cram_vrb_lab.scenes.garmi_apartment.constants import (
    FLOOR_CLUSTER_TUNING,
    FLOOR_CROP,
    FURNITURE_IN_VIEW,
    LIVING_ROOM_FLOOR,
    KITCHEN_WORKTOP,
    STRETCH_SPAWN_POSITION,
)


# %%

# print('robot spawned at', STRETCH_SPAWN_POSITION, '-> looking at', LIVING_ROOM_FLOOR)

run_plan(execute_single(
    LookAtAction(Pose(Point3.from_iterable(KITCHEN_WORKTOP), reference_frame=world.root)),
    context=context,
))


# %%

from cram_vrb_lab.perception.twin_objects import camera_pose_in_map
import numpy as np

print('camera in map:\n', np.round(camera_pose_in_map(world), 3))

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

print(f'rgb   {rgb.shape} {rgb_msg.encoding}  frame={rgb_msg.header.frame_id}')
print(f'depth {depth.shape} {depth_msg_.encoding}  '
      f'valid {np.isfinite(depth).mean():.0%}  '
      f'range {np.nanmin(depth):.2f}..{np.nanmax(depth[np.isfinite(depth)]):.2f} m')

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].imshow(rgb); axes[0].set_title('rgb8'); axes[0].axis('off')
im = axes[1].imshow(np.where(np.isfinite(depth), depth, np.nan), cmap='viridis')
axes[1].set_title('depth [m]'); axes[1].axis('off')
fig.colorbar(im, ax=axes[1], shrink=0.8)
plt.tight_layout()
plt.show()


from cram_vrb_lab.perception import pipeline as rk

# Built once and reused: constructing the descriptor spins up robokudo's own camera
# node and its subscriptions, and a second one would just duplicate them.
descriptor = rk.camera_descriptor()

if 'rk_node' in globals():
    rk_node.destroy_node()
rk_node = rk.make_pipeline_node()

# A fresh pipeline too: py_trees keeps blackboard state on the object.
pipeline = rk.build_pipeline(descriptor, crop=FLOOR_CROP, tuning=FLOOR_CLUSTER_TUNING)
detections = rk.detect(rk_node, descriptor, pipeline=pipeline)

print(f'{len(detections)} cluster(s), poses in camera_color_optical_frame:')
for i, d in enumerate(detections):
    print(f'  [{i}] pos {np.round(d.position, 3)}  '
          f'extent {np.round(d.extents, 3)}  volume {d.volume * 1e3:.2f} L')


from cram_vrb_lab.perception.twin_objects import add_detections, DETECTION_PREFIX

bodies = add_detections(world, detections)

print(f'{len(bodies)} body(ies) added to the twin:')
for body in bodies:
    position = np.asarray(body.global_pose.to_np())[:3, 3].ravel()
    print(f'  {body.name.name:16s} map {np.round(position, 3)}')

print(f'\nbodies now carrying the {DETECTION_PREFIX!r} prefix:',
      len([b for b in world.bodies if b.name.prefix == DETECTION_PREFIX]))


def body_position_in_map(name):
    body = world.get_body_by_name(name)
    return np.asarray(body.global_pose.to_np())[:3, 3].ravel()


detected_positions = [
    np.asarray(b.global_pose.to_np())[:3, 3].ravel() for b in bodies
]

print(f'{len(detected_positions)} detection(s) vs {len(FURNITURE_IN_VIEW)} '
      'expected piece(s) of furniture\n')
print(f'{"ground truth":16s} {"map (x, y)":>18s}  {"nearest detection":>18s}  {"dxy [m]":>8s}')
for name in FURNITURE_IN_VIEW:
    truth = body_position_in_map(name)
    if not detected_positions:
        print(f'{name:16s} {np.round(truth[:2], 3)!s:>18s}  {"-- none --":>18s}  {"":>8s}')
        continue
    distances = [np.linalg.norm(truth[:2] - found[:2]) for found in detected_positions]
    best = int(np.argmin(distances))
    print(f'{name:16s} {np.round(truth[:2], 3)!s:>18s}  '
          f'{np.round(detected_positions[best][:2], 3)!s:>18s}  {distances[best]:8.3f}')


print(f'\n{"detection":12s} {"map (x, y)":>18s}  {"extents (m)":>22s}  '
      f'{"nearest truth":16s} {"dxy [m]":>8s}')
for index, (body, detection) in enumerate(zip(bodies, detections)):
    found = np.asarray(body.global_pose.to_np())[:3, 3].ravel()
    distances = {name: np.linalg.norm(body_position_in_map(name)[:2] - found[:2])
                 for name in FURNITURE_IN_VIEW}
    nearest = min(distances, key=distances.get)
    print(f'{body.name.name:12s} {np.round(found[:2], 3)!s:>18s}  '
          f'{np.round(detection.extents, 3)!s:>22s}  '
          f'{nearest:16s} {distances[nearest]:8.3f}')


from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Drawer,
    Handle,
)

drawer_body = world.get_body_by_name("drawer_1")
handle_body = world.get_body_by_name("drawer_1_handle")

if not world.get_semantic_annotations_by_type(Drawer):
    with world.modify_world():
        world.add_semantic_annotation_recursively(
            Drawer(root=drawer_body, handle=Handle(root=handle_body))
        )
print("drawer annotated:", drawer_body.name, "with handle", handle_body.name)

# %%
from coraplex.robot_plans.actions.core.navigation import LookAtAction

# run_plan(execute_single(LookAtAction(handle_body.global_pose), context=context))

# %%
from coraplex.robot_plans.actions.core.container import OpenAction, CloseAction

run_plan(
    execute_single(OpenAction(handle_body, Arms.LEFT), context=context),
    collision_avoidance=False,
)
# print("drawer joint:", world.get_connection_by_name("drawer_1").position)

# stop(patterns=(SIM_SCRIPT.name, SERVER_SCRIPT.name, 'rviz2'))


