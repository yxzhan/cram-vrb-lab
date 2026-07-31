#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Franka Panda opens drawer_1 of the garmi-apartment using IK (KinematicsSolver).

Key references from .backup/demo.py
-------------------------------------
  QUAT_TOP (drawer handle, horizontal bar, fingers close in Z):
      [sqrt(0.5), -sqrt(0.5), 0, 0]  hand_z -> world +Y, hand_y -> world -Z
  WRIST_TO_FINGERTIP = 0.11 m  ->  IK target = handle_pos - 0.11 m in +Y
  APPROACH_OFFSET    = 0.10 m standoff in world -Y before advancing
  HOME_Q = [0, 0, 0, -pi/2, 0, pi/2, -pi/4]  ->  IK warm-start

State machine
-------------
  PRE_GRASP  - EE 0.10 m in front of wrist-target, gripper open
  APPROACH   - EE at wrist-target (fingertips at handle), gripper open
  GRASP      - hold, close gripper (60 steps)
  PULL       - retract EE 0.25 m in -Y, drawer slides open
  DONE       - hold

Usage
-----
  cd /isaac-sim
  ./python.sh /mnt/dev-tools/garmi-scene/garmi-apartment/franka_open_drawer.py
"""

import os
os.environ["DISPLAY"] = ":0"

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})


import os
import sys
import carb
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot.manipulators.examples.franka import Franka, KinematicsSolver
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils import viewports
from pxr import UsdGeom



# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_USD  = os.path.join(SCRIPT_DIR, "world.usda")

# ---------------------------------------------------------------------------
# Arm configuration (from demo.py)
# ---------------------------------------------------------------------------
HOME_Q         = np.array([0.0, 0.0, 0.0, -np.pi / 2, 0.0, np.pi / 2, -np.pi / 4])
GRIPPER_OPEN   = np.array([0.04,  0.04 ])
GRIPPER_CLOSED = np.array([0.008, 0.008])   # handle bar ~12 mm thick

# Franka base: aligned with drawer_1 center X, facing +Y (90 deg around Z)
ROBOT_POS = np.array([-0.090, 6.70, 0.0])
ROBOT_ORI = np.array([1, 0.0, 0.0, 0.0])   # (w,x,y,z)

# ---------------------------------------------------------------------------
# Target positions  (world space, metres)
#
#   drawer_1_handle center = (-0.090, 7.132, 0.800)
#   WRIST_TO_FINGERTIP     = 0.11 m  ->  right_gripper IK target
#                            = handle_center - [0, 0.11, 0]
#                            = (-0.090, 7.022, 0.800)
#   APPROACH_OFFSET        = 0.10 m standoff in -Y
#   PULLED (0.25 m open)   = wrist_closed - [0, 0.25, 0]
# ---------------------------------------------------------------------------
WRIST_TARGET  = np.array([-0.090, 6.3, 0.800])
PRE_GRASP_POS = WRIST_TARGET + np.array([0.0, -0.10, 0.0])
PULLED_POS    = WRIST_TARGET + np.array([0.0, -0.25, 0.0])

# EE orientation: R_x(-90 deg)  right_gripper Z -> world +Y (toward cabinet)
# Same as demo.py QUAT_TOP, targeted at the right_gripper Lula frame.
EE_ORI = np.array([0.7071068, -0.7071068, 0.0, 0.0])   # (w,x,y,z)

# ---------------------------------------------------------------------------
# Thresholds / timing
# ---------------------------------------------------------------------------
POS_THRESH  = 0.02   # m
GRASP_STEPS = 60     # simulation steps to close gripper

# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------
PRE_GRASP, APPROACH, GRASP, PULL, DONE = range(5)
STATE_NAME = ["PRE_GRASP", "APPROACH", "GRASP", "PULL", "DONE"]

PHASE_TARGET = {
    PRE_GRASP: PRE_GRASP_POS,
    APPROACH:  WRIST_TARGET,
    GRASP:     WRIST_TARGET,
    PULL:      PULLED_POS,
    DONE:      PULLED_POS,
}

# ---------------------------------------------------------------------------
# Scene / robot setup
# ---------------------------------------------------------------------------
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

my_world = World(stage_units_in_meters=1.0)

add_reference_to_stage(usd_path=SCENE_USD, prim_path="/World/Scene")

# Invisible physics ground plane (apartment floor mesh has no physics)
my_world.scene.add_default_ground_plane()
UsdGeom.Imageable(
    my_world.stage.GetPrimAtPath("/World/defaultGroundPlane")
).MakeInvisible()

# Use the official Franka class — required for KinematicsSolver compatibility
my_franka = my_world.scene.add(
    Franka(
        prim_path="/World/Franka",
        name="franka",
        position=ROBOT_POS,
        orientation=ROBOT_ORI,
    )
)

# Set HOME_Q as default so IK warm-start is non-singular
my_franka.set_joints_default_state(
    positions=np.concatenate([HOME_Q, GRIPPER_OPEN])
)

my_world.reset()

viewports.set_camera_view(
    eye=WRIST_TARGET + np.array([2.0, -2.0, 1.5]),
    target=WRIST_TARGET,
)

# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------
ik_solver         = KinematicsSolver(my_franka)
articulation_ctrl = my_franka.get_articulation_controller()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ee_pos_error(target: np.ndarray) -> float:
    pos, _ = my_franka.end_effector.get_world_pose()
    return float(np.linalg.norm(pos - target))


def transition(new_state: int) -> int:
    print(f"  -> {STATE_NAME[new_state]}")
    return new_state


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------
state        = PRE_GRASP
step_count   = 0
reset_needed = False

print(f"Starting: {STATE_NAME[state]}")

while simulation_app.is_running():
    my_world.step(render=True)

    if my_world.is_stopped() and not reset_needed:
        reset_needed = True

    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            state      = PRE_GRASP
            step_count = 0
            reset_needed = False
            print(f"Reset -> {STATE_NAME[state]}")

        # ---- IK ----
        target_pos = PHASE_TARGET[state]
        actions, succ = ik_solver.compute_inverse_kinematics(
            target_position=target_pos,
            target_orientation=EE_ORI,
        )
        if succ:
            articulation_ctrl.apply_action(actions)
        else:
            carb.log_warn(f"IK did not converge (state={STATE_NAME[state]})")

        # ---- Gripper ----
        if state in (GRASP, PULL, DONE):
            my_franka.gripper.close()
        else:
            my_franka.gripper.open()

        # ---- State transitions ----
        if state == PRE_GRASP:
            if ee_pos_error(PRE_GRASP_POS) < POS_THRESH:
                state = transition(APPROACH)

        elif state == APPROACH:
            if ee_pos_error(WRIST_TARGET) < POS_THRESH:
                state      = transition(GRASP)
                step_count = 0

        elif state == GRASP:
            step_count += 1
            if step_count >= GRASP_STEPS:
                state = transition(PULL)

        elif state == PULL:
            if ee_pos_error(PULLED_POS) < POS_THRESH:
                state = transition(DONE)
                print("  drawer_1 open.")

simulation_app.close()
