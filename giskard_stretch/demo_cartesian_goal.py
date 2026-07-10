#!/usr/bin/env python
"""Cartesian end-effector goal demo for link_grasp_center.

Moves the grasp center by an offset given in its own frame (keeps current
orientation).

Usage:
    python demo_cartesian_goal.py [dx dy dz]      # default 0.0 0.0 0.15, arm only
    python demo_cartesian_goal.py --full-body [dx dy dz]   # base may move too
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from giskard_client import add_end_conditions, connect

from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from semantic_digital_twin.spatial_types.spatial_types import Pose, Vector3


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    full_body = "--full-body" in sys.argv
    dx, dy, dz = (float(a) for a in args) if len(args) == 3 else (0.0, 0.0, 0.15)

    giskard = connect()
    world = giskard.world
    tip = world.get_kinematic_structure_entity_by_name("link_grasp_center")
    root = world.root if full_body else world.get_kinematic_structure_entity_by_name("base_link")

    goal_pose = Pose(position=Vector3(dx, dy, dz), reference_frame=tip)

    msc = MotionStatechart()
    # 3 cm / 0.03 rad tolerance: velocity-level whole-body control on
    # this pipeline does not settle to the default 1 cm reliably.
    task = CartesianPose(
        root_link=root, tip_link=tip, goal_pose=goal_pose, threshold=0.03
    )
    msc.add_node(task)
    add_end_conditions(msc, task, timeout_seconds=60.0)

    mode = "full-body" if full_body else "arm-only"
    print(f"moving link_grasp_center by ({dx}, {dy}, {dz}) in its own frame ({mode}) ...")
    giskard.execute(msc)
    reached = msc.observation_state[task] == ObservationStateValues.TRUE
    print(f"done. goal reached: {reached}")


if __name__ == "__main__":
    main()
