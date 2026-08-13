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


SPAWN_POSITION = (0, 5.0, 0.0259)
SPAWN_YAW = -math.pi / 2

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
from giskardpy.data_types.exceptions import GiskardException


def run_plan(plan, collision_avoidance=True, strict=False):
    try:
        with real_robot(collision_avoidance=collision_avoidance):
            plan.perform()
    except GiskardException as failure:
        if strict:
            raise
        # str() on these carries giskard's own error_message() and, where it has
        # one, its suggested correction.
        print(f'giskard failed -- {type(failure).__name__}: {failure}')
        return False
    print('done')
    return True


def tool_frame_matrix(arm=Arms.LEFT):
    """The tool frame's 4x4 pose in map."""
    tool = ViewManager.get_end_effector_view(arm, robot).tool_frame
    return np.asarray(tool.global_pose.to_np())


def tool_position(arm=Arms.LEFT):
    return tool_frame_matrix(arm)[:3, 3].ravel()


def tool_axis(arm=Arms.LEFT):
    """Where the gripper points: the tool frame's **x**-axis, in map.

    Column 0, not 2. The Franka Hand points its z out between the fingers, but
    CRAM reads the approach direction off the x column
    (``EndEffector.__post_init__``), so ``load_patched_urdf`` turns the TCP
    frame a quarter turn about y -- see ``_TOOL_FRAME_RPY`` in
    ``cram_vrb_lab/robots/garmi/joints.py``. Reading z here would report the
    hand's -x and quietly send any grasp debugging off in the wrong direction.
    """
    return tool_frame_matrix(arm)[:3, 0].ravel()


def closing_axis(arm=Arms.LEFT):
    """The axis the fingers close along, in map -- the tool frame's y."""
    return tool_frame_matrix(arm)[:3, 1].ravel()


def body_position(name):
    return np.asarray(world.get_body_by_name(name).global_pose.to_np())[:3, 3].ravel()

# %%
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction, SetGripperAction
from semantic_digital_twin.datastructures.definitions import GripperState

# run_plan(sequential([
#         ParkArmsAction(Arms.LEFT),
#         ParkArmsAction(Arms.RIGHT),
#         SetGripperAction(Arms.LEFT, GripperState.OPEN),
#         SetGripperAction(Arms.RIGHT, GripperState.OPEN),
#     ], context=context))

# run_plan(sequential([
#         SetGripperAction(Arms.LEFT, GripperState.CLOSE),
#         SetGripperAction(Arms.RIGHT, GripperState.CLOSE),
#     ], context=context))

# run_plan(sequential([
#         SetGripperAction(Arms.LEFT, GripperState.OPEN),
#         SetGripperAction(Arms.RIGHT, GripperState.OPEN),
#     ], context=context))

from coraplex.robot_plans.actions.core.navigation import NavigateAction
from semantic_digital_twin.spatial_types import Point3, Quaternion
from semantic_digital_twin.spatial_types.spatial_types import Pose

STANDOFF = 1.3
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


def drive_to(handle_name, attempts=10):
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
# Opening a container, decomposed so the reach can be measured and repeated.
#
# This mirrors OpenAction._action_plan
# (coraplex/robot_plans/actions/core/container.py): grasp the handle, drive the
# container's own joint to its limit, let go. OpenAction runs those three as one
# sequence and exposes nothing in between, so a grasp that lands slightly off --
# which is what stops the drawer opening -- is invisible and unrepeatable. Keep
# this in step if OpenAction changes upstream.
from coraplex.datastructures.enums import ApproachDirection, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.robot_plans.actions.core.pick_up import GraspingAction
from coraplex.robot_plans.motions.container import ClosingMotion, OpeningMotion
from coraplex.robot_plans.motions.gripper import MoveGripperMotion
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Drawer,
    Handle,
    Door,
)

from coraplex.robot_plans.actions.core.container import OpenAction, CloseAction

GRASPED = 0.01
"""How close the tool frame has to land to the commanded grasp pose [m]."""


def _grasp_description(arm):
    """The grasp OpenAction uses internally: approach the handle head on."""
    return GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        ViewManager.get_end_effector_view(arm, robot),
    )


def report_grasp_geometry(handle_body, arm=Arms.LEFT):
    """Print what CRAM aims at versus what is actually there.

    Worth having in front of you because the two are *not* the same point: CRAM
    grasps at the handle body's origin (``grasp_pose_sequence`` builds its pose
    from ``Pose(reference_frame=body)``), while the collision geometry -- the rod
    the fingers actually have to close around -- sits a centimetre and a half in
    front of it.
    """
    box = handle_body.collision.as_bounding_box_collection_in_frame(
        handle_body
    ).bounding_box()
    # min_*/max_* are relative to the box's own origin, not to the body frame.
    centre = np.array([
        float(box.origin.x) + (box.min_x + box.max_x) / 2,
        float(box.origin.y) + (box.min_y + box.max_y) / 2,
        float(box.origin.z) + (box.min_z + box.max_z) / 2,
    ])
    print(f'  handle origin (grasp target): {np.round(body_position(handle_body.name.name), 4)}')
    print(f'  collision box in body frame : centre {np.round(centre, 4)}'
          f' size {np.round(np.array(box.dimensions), 4)}')
    print(f'  -> CRAM aims {np.linalg.norm(centre) * 1000:.1f} mm off the collision centre')


def grasp_handle(handle_body, arm=Arms.LEFT, attempts=6):
    """Reach and close on the handle, repeating while the tool lands short.

    Same shape as :func:`drive_to`, and for the same reason: a Cartesian goal in
    this stack can finish without having converged, and the residual is only
    visible if you measure it. Reported **in the tool frame** -- approach,
    closing and lift -- because that is what says whether the gripper stopped too
    early, too high, or off to one side; a single distance would not.
    """
    grasp = _grasp_description(arm)
    _, commanded, _ = grasp.grasp_pose_sequence(handle_body)
    # grasp_pose_sequence works in the *handle's* frame (it starts from
    # Pose(reference_frame=body)), so it has to be lifted into map before it can
    # be compared with where the tool actually is.
    goal_frame = (np.asarray(handle_body.global_pose.to_np())
                  @ np.asarray(commanded.to_np()))
    goal = goal_frame[:3, 3].ravel()

    for attempt in range(1, attempts + 1):
        run_plan(execute_single(GraspingAction(handle_body, arm, grasp),
                                context=context),
                 collision_avoidance=True)
        residual = tool_position(arm) - goal
        # Resolved along the axes of the pose that was *asked* for, so the three
        # numbers keep meaning the same thing however the arm ended up oriented.
        approach, closing, lift = (
            float(residual @ goal_frame[:3, i]) for i in range(3)
        )
        error = float(np.linalg.norm(residual))
        print(f'  grasp {attempt}: residual {error * 1000:6.1f} mm'
              f'  (approach {approach * 1000:+6.1f}, closing {closing * 1000:+6.1f},'
              f' lift {lift * 1000:+6.1f} mm)')
        if error <= GRASPED:
            return True
    print(f'  WARNING: tool still {error * 1000:.1f} mm off after {attempts} tries')
    return False


def _work_container(motion, handle_body, arm, attempts):
    report_grasp_geometry(handle_body, arm)
    grasp_handle(handle_body, arm, attempts)
    run_plan(execute_single(motion(handle_body, arm), context=context),
             collision_avoidance=True)
    run_plan(execute_single(MoveGripperMotion(GripperState.OPEN, arm), context=context),
             collision_avoidance=True)


def open_container(handle_body, arm=Arms.LEFT, attempts=3):
    """OpenAction, with the reach measured and retried. See the note above."""
    _work_container(OpeningMotion, handle_body, arm, attempts)


def close_container(handle_body, arm=Arms.LEFT, attempts=3):
    """CloseAction, likewise -- it is the same three steps with ClosingMotion."""
    _work_container(ClosingMotion, handle_body, arm, attempts)

from itertools import cycle
for i in list(cycle([1, 2, 3]))[:30]:
    print(i)
    drawer_body = world.get_body_by_name(f"drawer_{drawer_id}")
    handle_body = world.get_body_by_name(f"drawer_{drawer_id}_handle")

    if not world.get_semantic_annotations_by_type(Drawer):
        with world.modify_world():
            world.add_semantic_annotation_recursively(
                Drawer(root=drawer_body, handle=Handle(root=handle_body))
            )
    print("drawer annotated:", drawer_body.name, "with handle", handle_body.name)
    print("handle at", np.round(body_position(f"drawer_{drawer_id}_handle"), 3))

    drive_to(f"drawer_{drawer_id}_handle")

    run_plan(sequential([
            ParkArmsAction(Arms.LEFT),
            ParkArmsAction(Arms.RIGHT),
            SetGripperAction(Arms.LEFT, GripperState.OPEN),
            SetGripperAction(Arms.RIGHT, GripperState.OPEN),
        ], context=context))

    open_container(handle_body, Arms.LEFT)
    print("Opened: drawer joint:", world.get_connection_by_name(f"drawer_{drawer_id}_joint").position)

    close_container(handle_body, Arms.LEFT)
    print("Closed: drawer joint:", world.get_connection_by_name(f"drawer_{drawer_id}_joint").position)

# %%

# door_id = "1"
# door_body = world.get_body_by_name(f"cabinet_door_{door_id}")
# door_handle_body = world.get_body_by_name(f"cabinet_door_{door_id}_handle")

# if not world.get_semantic_annotations_by_type(Door):
#     with world.modify_world():
#         world.add_semantic_annotation_recursively(
#             Door(root=door_body, handle=Handle(root=door_handle_body))
#         )
# print("Door annotated:", door_body.name, "with handle", door_handle_body.name)

# drive_to(f"cabinet_door_{door_id}_handle")
# open_container(door_handle_body, Arms.RIGHT)
# # run_plan(
# #     execute_single(OpenAction(door_handle_body, Arms.RIGHT), context=context),
# #     collision_avoidance=False,
# # )
# print("Opened, Door joint:", world.get_connection_by_name(f"cabinet_door_{door_id}_joint").position)

# close_container(door_handle_body, Arms.RIGHT)
# print("Closed, Door joint:", world.get_connection_by_name(f"cabinet_door_{door_id}_joint").position)


# run_plan(sequential([
#         ParkArmsAction(Arms.LEFT),
#         ParkArmsAction(Arms.RIGHT),
#         SetGripperAction(Arms.LEFT, GripperState.OPEN),
#         SetGripperAction(Arms.RIGHT, GripperState.OPEN),
#     ], context=context))

