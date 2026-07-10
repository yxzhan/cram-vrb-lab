#!/usr/bin/env python
"""Joint-space goal demo: move the lift (and optionally other joints).

Usage:
    python demo_joint_goal.py [lift_height]     # default 0.9 m
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from giskard_client import add_end_conditions, connect

from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from semantic_digital_twin.datastructures.joint_state import JointState


def main():
    lift_height = float(sys.argv[1]) if len(sys.argv) > 1 else 0.9
    giskard = connect()

    goal = {
        giskard.world.get_connection_by_name("joint_lift"): lift_height,
        giskard.world.get_connection_by_name("joint_wrist_yaw"): 0.0,
    }

    msc = MotionStatechart()
    # 0.02 tolerance: the velocity-bridge pipeline settles within
    # 0.01-0.02 of the goal, so the default 0.01 is flaky.
    task = JointPositionList(goal_state=JointState.from_mapping(goal), threshold=0.02)
    msc.add_node(task)
    add_end_conditions(msc, task, timeout_seconds=30.0)

    print(f"moving joint_lift to {lift_height} ...")
    giskard.execute(msc)
    from giskardpy.motion_statechart.data_types import ObservationStateValues

    reached = msc.observation_state[task] == ObservationStateValues.TRUE
    print(f"done. goal reached: {reached}")


if __name__ == "__main__":
    main()
