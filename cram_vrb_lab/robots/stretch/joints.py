"""Shared constants for the giskard <-> Isaac Sim stretch integration.

The joint order of CONTROLLED_JOINTS is the contract between giskard's
joint-group velocity controller and the sim's velocity-command integrator
(``StretchROS`` in ``cram_vrb_lab.robots.stretch.isaac_node``): the
Float64MultiArray velocity command carries no joint names, only values in
this order.
"""

import os

from cram_vrb_lab.paths import ASSETS_DIR

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
GRIPPER_CMD_TOPIC = "/stretch/gripper_command"
ODOM_TOPIC = "/odom"

RGB_IMAGE_TOPIC = "/head_camera/image_raw"
RGB_INFO_TOPIC = "/head_camera/camera_info"
DEPTH_IMAGE_TOPIC = "/head_camera/depth/image_raw"
DEPTH_INFO_TOPIC = "/head_camera/depth/camera_info"
# Camera messages are stamped in this frame; it exists in the tf tree because
# the giskard server publishes it from the (patched) URDF.
CAMERA_FRAME_ID = "camera_color_optical_frame"

# Official URDF from the hello-robot/stretch_urdf submodule (SE3 with the DW3
# dex wrist and SG3 gripper -- the variant matching assets/stretch/stretch.usd).
_STRETCH_URDF_DIR = str(ASSETS_DIR / "stretch_urdf" / "stretch_urdf" / "SE3")
SIM_URDF_PATH = os.path.join(
    _STRETCH_URDF_DIR, "stretch_description_SE3_eoa_wrist_dw3_tool_sg3.urdf"
)

# giskardpy's semantic Stretch model expects link_straight_gripper as the root
# of the gripper subtree; the official URDF attaches the S3 gripper body
# directly to link_wrist_roll. Reparent it through an intermediate link.
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

# The official URDF ships the two gripper finger joints with zeroed limits
# (effort/lower/upper/velocity all 0), which would make them immovable for
# giskard. Give them their real range (matches the articulation in the sim).
_FINGER_LIMIT_OLD = '<limit effort="0" lower="0" upper="0" velocity="0"/>'
_FINGER_LIMIT_NEW = '<limit effort="100" lower="-0.6" upper="0.6" velocity="1.0"/>'


def load_patched_urdf() -> str:
    with open(SIM_URDF_PATH) as f:
        urdf = f.read()

    assert urdf.count(_S3_JOINT_OLD) == 1, "joint_gripper_s3_body block not found"
    urdf = urdf.replace(_S3_JOINT_OLD, _S3_JOINT_NEW)

    assert urdf.count(_FINGER_LIMIT_OLD) == 2, "expected exactly the 2 finger limits"
    urdf = urdf.replace(_FINGER_LIMIT_OLD, _FINGER_LIMIT_NEW)

    mesh_dir = os.path.join(_STRETCH_URDF_DIR, "meshes")
    urdf = urdf.replace('filename="./meshes/', f'filename="{mesh_dir}/')
    return urdf
