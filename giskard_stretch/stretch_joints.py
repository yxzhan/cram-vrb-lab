"""Shared constants for the giskard <-> Isaac Sim stretch integration.

The joint order of CONTROLLED_JOINTS is the contract between giskard's
joint-group velocity controller and the sim's velocity-command integrator
(VELOCITY_CONTROLLED_JOINTS in examples/apartment.py): the Float64MultiArray
velocity command carries no joint names, only values in this order.
"""

import os

CONTROLLED_JOINTS = [
    "joint_lift",
    "joint_arm_l3",
    "joint_arm_l2",
    "joint_arm_l1",
    "joint_arm_l0",
    "joint_wrist_yaw",
    "joint_wrist_pitch",
    "joint_wrist_roll",
    "joint_head_pan",
    "joint_head_tilt",
    "joint_gripper_finger_left",
    "joint_gripper_finger_right",
]

VELOCITY_CMD_TOPIC = "/stretch/joint_velocity_cmd"
JOINT_COMMAND_TOPIC = "/stretch/joint_command"
JOINT_STATES_TOPIC = "/stretch/joint_states"
CMD_VEL_TOPIC = "/stretch/cmd_vel"
ODOM_TOPIC = "/odom"

_STRETCH_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "usd", "stretch"
)
SIM_URDF_PATH = os.path.join(_STRETCH_DIR, "stretch.urdf")

# giskardpy's semantic Stretch model expects link_straight_gripper as the root
# of the gripper subtree; the sim URDF attaches the S3 gripper body directly to
# link_wrist_roll. Reparent it through an intermediate link.
_S3_JOINT_OLD = """  <joint name="joint_gripper_s3_body" type="fixed">
    <origin rpy="0 0 -3.14159265358975" xyz="0 0 0.0209999999993159"/>
    <parent link="link_wrist_roll"/>
    <child link="link_gripper_s3_body"/>
    <axis xyz="0 0 0"/>
  </joint>"""

_S3_JOINT_NEW = """  <link name="link_straight_gripper"/>
  <joint name="joint_straight_gripper" type="fixed">
    <origin rpy="0 0 -3.14159265358975" xyz="0 0 0.0209999999993159"/>
    <parent link="link_wrist_roll"/>
    <child link="link_straight_gripper"/>
    <axis xyz="0 0 0"/>
  </joint>
  <joint name="joint_gripper_s3_body" type="fixed">
    <origin rpy="0 0 0" xyz="0 0 0"/>
    <parent link="link_straight_gripper"/>
    <child link="link_gripper_s3_body"/>
    <axis xyz="0 0 0"/>
  </joint>"""


def load_patched_urdf() -> str:
    with open(SIM_URDF_PATH) as f:
        urdf = f.read()

    assert urdf.count(_S3_JOINT_OLD) == 1, "joint_gripper_s3_body block not found"
    urdf = urdf.replace(_S3_JOINT_OLD, _S3_JOINT_NEW)

    mesh_dir = os.path.join(_STRETCH_DIR, "meshes")
    urdf = urdf.replace('filename="./meshes/', f'filename="{mesh_dir}/')
    return urdf
