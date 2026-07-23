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


def connect(node_name: str = "giskard_demo_client"):
    rospy.init_node(node_name)
    # Join the spinner thread and destroy the node at interpreter exit;
    # otherwise the process aborts with "terminate called without an active
    # exception" while tearing down live executor threads.
    atexit.register(rospy.shutdown)
    from giskardpy.middleware.ros2.python_interface import GiskardWrapper

    giskard = GiskardWrapper(node_handle=rospy.node, giskard_node_name="giskard")
    return giskard
