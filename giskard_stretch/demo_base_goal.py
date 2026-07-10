#!/usr/bin/env python
"""Base motion demo: drive the base to a pose given relative to its current
pose (exercises giskard's diff-drive control through /stretch/cmd_vel).

Uses DifferentialDriveBaseGoal (orient -> drive -> orient), the proper idiom
for a non-holonomic base; a plain CartesianPose on base_link tends to
oscillate around the goal.

Usage:
    python demo_base_goal.py [dx dy]      # default 0.5 0.0 (meters, base frame)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from giskard_client import add_end_conditions, connect

from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.goals.cartesian_goals import DifferentialDriveBaseGoal
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from semantic_digital_twin.spatial_types.spatial_types import Pose, Vector3


def main():
    args = sys.argv[1:]
    dx, dy = (float(a) for a in args) if len(args) == 2 else (0.5, 0.0)

    giskard = connect()
    world = giskard.world
    base = world.get_kinematic_structure_entity_by_name("base_link")

    goal_pose = Pose(position=Vector3(dx, dy, 0.0), reference_frame=base)

    msc = MotionStatechart()
    # 5 cm tolerance: with the velocity-bridge/kinematic-base pipeline
    # the default 1 cm threshold sits inside the controller's limit
    # cycle and never latches.
    task = DifferentialDriveBaseGoal(goal_pose=goal_pose, threshold=0.05)
    msc.add_node(task)
    add_end_conditions(msc, task, timeout_seconds=60.0)

    print(f"driving base by ({dx}, {dy}) in the base frame ...")
    giskard.execute(msc)
    reached = msc.observation_state[task] == ObservationStateValues.TRUE
    print(f"done. goal reached: {reached}")


if __name__ == "__main__":
    main()
