"""Shared client boilerplate for the demo scripts.

Import this first: it initializes the rclpy node used by giskardpy's rospy
shim and returns a connected GiskardWrapper.
"""

import atexit

from giskardpy.middleware.ros2 import rospy


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


def connect(node_name: str = "giskard_demo_client"):
    rospy.init_node(node_name)
    # Join the spinner thread and destroy the node at interpreter exit;
    # otherwise the process aborts with "terminate called without an active
    # exception" while tearing down live executor threads.
    atexit.register(rospy.shutdown)
    from giskardpy.middleware.ros2.python_interface import GiskardWrapper

    giskard = GiskardWrapper(node_handle=rospy.node, giskard_node_name="giskard")
    return giskard
