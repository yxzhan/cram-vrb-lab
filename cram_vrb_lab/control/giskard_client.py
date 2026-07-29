"""Shared client boilerplate for the demo scripts.

Import this first: it initializes the rclpy node used by giskardpy's rospy
shim and returns a connected GiskardWrapper.
"""

import atexit

import nest_asyncio

from giskardpy.middleware.ros2 import rospy

# GiskardWrapper.execute blocks via asyncio's run_until_complete; inside
# jupyter the kernel's event loop is already running, which raises
# "Cannot run the event loop while another loop is running" without this.
nest_asyncio.apply()


def add_end_conditions(msc, task, timeout_seconds=60.0, settle_seconds=1.0):
    """Finish `msc` when `task` has been reached and held for `settle_seconds`,
    or when the timeout expires. Returns the timeout node.

    The settle phase matters for this sim: giskard's tree keeps publishing the
    last commanded velocities for ~1 s between goal completion and its
    terminate-zero message, and the sim latches twists forever. Ending the
    motion while velocities are still large lets the robot coast ~0.2 m past
    the goal; holding the goal for a second lets the QP wind them down to
    zero first.
    """
    from giskardpy.motion_statechart.graph_node import EndMotion
    from giskardpy.motion_statechart.monitors.payload_monitors import CountSeconds

    settle = CountSeconds(seconds=settle_seconds)
    settle.start_condition = task.observation_variable
    timeout = CountSeconds(seconds=timeout_seconds)
    msc.add_nodes([settle, timeout])
    msc.add_node(EndMotion.when_all_true([task, settle]))
    msc.add_node(EndMotion.when_true(timeout))
    return timeout


def add_external_collision_avoidance(
    msc, max_velocity=0.2, cancel_if_collision_violated=True
):
    """Add whole-body external collision avoidance to the motion statechart `msc`.

    The robot then keeps a safety distance from every other collision-enabled body
    in giskard's world -- the apartment walls and furniture, once the environment
    is loaded (see `cram_vrb_lab.scenes.apartment.giskard_world`).
    The robot is auto-detected from the world, and the distances come from the
    `AvoidExternalCollisions` rule the semantic Stretch model registers.

    `cancel_if_collision_violated` aborts the motion if a body is already inside the
    violated distance (rather than pushing through it); set it False to keep
    commanding the goal while merely braking near obstacles. Returns the added node.
    """
    from giskardpy.motion_statechart.goals.collision_avoidance import (
        ExternalCollisionAvoidance,
    )

    node = ExternalCollisionAvoidance(
        max_velocity=max_velocity,
        cancel_if_collision_violated=cancel_if_collision_violated,
    )
    msc.add_node(node)
    return node


def add_pointing(
    msc,
    world,
    goal_point,
    tip_link: str = "camera_color_optical_frame",
    root_link: str = "base_link",
    pointing_axis=None,
    max_velocity: float = 0.3,
):
    """Add a persistent camera-pointing goal to the motion statechart `msc`.

    Unlike CRAM's REAL execution (which sequences every motion), a giskard task
    added straight to the chart runs *concurrently* with the other goals -- the
    same way :func:`add_external_collision_avoidance` does -- so the camera keeps
    tracking `goal_point` for the whole motion instead of only before it.

    `goal_point` is a ``Point3`` in some world frame. Only the chain between
    `root_link` and `tip_link` moves to satisfy it: the default ``base_link`` root
    keeps the motion to the head (pan/tilt); pass ``world.root`` to let the base
    turn too. `pointing_axis` is the tip axis aimed at the point (default: the
    camera's +Z optical axis). Returns the added node.
    """
    from giskardpy.motion_statechart.tasks.pointing import Pointing
    from semantic_digital_twin.spatial_types import Vector3

    tip = world.get_body_by_name(tip_link)
    if pointing_axis is None:
        pointing_axis = Vector3.Z()
    # Pointing.build transforms the axis into the tip frame, so it must carry a
    # reference frame -- the axis is expressed in the tip link (matching giskard's
    # own LookingMotion, which sets forward_facing_axis.reference_frame = camera root).
    pointing_axis.reference_frame = tip
    node = Pointing(
        root_link=world.get_body_by_name(root_link),
        tip_link=tip,
        goal_point=goal_point,
        pointing_axis=pointing_axis,
        max_velocity=max_velocity,
    )
    msc.add_node(node)
    return node


def connect(node_name: str = "giskard_demo_client"):
    rospy.init_node(node_name)
    # Join the spinner thread and destroy the node at interpreter exit;
    # otherwise the process aborts with "terminate called without an active
    # exception" while tearing down live executor threads.
    atexit.register(rospy.shutdown)
    from giskardpy.middleware.ros2.python_interface import GiskardWrapper

    giskard = GiskardWrapper(node_handle=rospy.node, giskard_node_name="giskard")
    return giskard
