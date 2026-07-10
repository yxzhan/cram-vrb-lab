#!/usr/bin/env python
"""Gripper demo: open or close the fingers through giskard.

Usage:
    python demo_gripper.py open|close
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from giskard_client import add_end_conditions, connect

from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from semantic_digital_twin.datastructures.joint_state import JointState

FINGER_OPEN = 0.109
FINGER_CLOSED = 0.0


def main():
    action = sys.argv[1] if len(sys.argv) > 1 else "open"
    position = FINGER_OPEN if action == "open" else FINGER_CLOSED

    giskard = connect()
    goal = {
        giskard.world.get_connection_by_name("joint_gripper_finger_left"): position,
        giskard.world.get_connection_by_name("joint_gripper_finger_right"): position,
    }

    msc = MotionStatechart()
    # 0.02 tolerance: the velocity-bridge pipeline settles within
    # 0.01-0.02 of the goal, so the default 0.01 is flaky.
    task = JointPositionList(goal_state=JointState.from_mapping(goal), threshold=0.02)
    msc.add_node(task)
    add_end_conditions(msc, task, timeout_seconds=30.0)

    print(f"gripper {action} ...")
    giskard.execute(msc)
    reached = msc.observation_state[task] == ObservationStateValues.TRUE
    print(f"done. goal reached: {reached}")


if __name__ == "__main__":
    main()
