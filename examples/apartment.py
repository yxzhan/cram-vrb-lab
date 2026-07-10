#!/usr/bin/env python
"""Isaac Sim apartment scene with a Stretch robot, controlled over ROS 2.

Minimal simulation side of the giskard demo (see ../giskard_stretch/):
- publishes /stretch/joint_states, /odom, TF, /head_camera/image_raw
- subscribes /stretch/cmd_vel + /cmd_vel (Twist, kinematic base with a 1 s
  watchdog), /stretch/joint_command (position targets),
  /stretch/joint_velocity_cmd (giskard's streamed velocities, integrated into
  position targets each sim step), /stretch/gripper_command (Float64)

Run with the Isaac Sim python:
    ~/.local/bin/isaacsim_python_wrapper.sh examples/apartment.py
"""

import math
import os
import shutil
import sys
import time
from pathlib import Path

# ROS 2 environment for the Isaac ROS2 bridge (must be set before the bridge
# extension loads).
os.environ.setdefault("ROS_DOMAIN_ID", "0")
os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
os.environ.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path(os.getcwd())

# Copy the precompiled kit cache if it is not present yet (binder image mounts
# it at /mnt/isaacsim-cache; drastically shortens the first startup).
target_dir = "/isaac-sim/kit/cache"
source_dir = "/mnt/isaacsim-cache/cache"
if os.path.isdir(source_dir) and not os.path.isdir(target_dir):
    shutil.copytree(source_dir, target_dir)


# ---------------------------------------------------------------- SimulationApp
from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": False,
    "hide_ui": False,
    "width": 1280,
    "height": 960,
    "renderer": "RaytracedLighting",
    "display_options": 3286,  # show the default grid
})
print("SimulationApp Ready!")


# ------------------------------------------------------------------- The scene
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.utils.prims import define_prim, create_prim
from isaacsim.core.utils import viewports
from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.ros2.bridge")

my_world = World(stage_units_in_meters=1.0, physics_dt=1 / 200, rendering_dt=8 / 200)
my_world.reset()

# Ground
define_prim("/World/Ground", "Xform").GetReferences().AddReference(
    f"{BASE_DIR}/../usd/Grid/default_environment.usd"
)

# Apartment
create_prim(
    usd_path=f"{BASE_DIR}/../usd/apartment/apartmentICRA.usda",
    prim_path="/World/Apartment",
    position=np.array([-6, 5, 0.0701]),
)

# Lights so the raytraced scene is not black
for i in range(1, 4):
    create_prim(
        prim_path=f"/World/Ground/Light_{i}",
        prim_type="SphereLight",
        attributes={"inputs:intensity": 10000},
        position=(-4 * i, 0, 2),
    )

viewports.set_camera_view(eye=np.array([-6.5, -2, 2]), target=np.array([-1, 1, 1]))

for _ in range(30):
    my_world.step(render=True)


# ---------------------------------------------------------------- The Stretch
from isaacsim.core.prims import Articulation

create_prim(
    usd_path=f"{BASE_DIR}/../usd/stretch/stretch.usd",
    prim_path="/World/stretch",
    position=np.array([-1.5, 0, 0.05]),
    orientation=np.array([0, 0, 0, 1]),
)

stretch = Articulation(prim_paths_expr="/World/stretch", name="stretch")
my_world.reset()

for _ in range(10):
    my_world.step(render=True)

# --- Fix the telescoping arm joint gains ---
# The arm extends through four serially-chained prismatic joints
# (joint_arm_l0..l3, 0.13 m each). Isaac auto-generates their drive gains from
# the tiny link masses (~20 N/m stiffness), so a commanded extension springs
# back instead of holding. Raise the PD gains to values comparable to the
# hand-tuned joint_lift (stiffness 50000 / damping 300).
arm_joints = ["joint_arm_l0", "joint_arm_l1", "joint_arm_l2", "joint_arm_l3"]
arm_dof = np.array([stretch.get_dof_index(n) for n in arm_joints])

kps = np.full((1, len(arm_dof)), 1.0e4)   # stiffness [N/m]
kds = np.full((1, len(arm_dof)), 2.0e2)   # damping  [N/(m/s)]
stretch.set_gains(kps=kps, kds=kds, joint_indices=arm_dof)

# A larger force budget so the drive can actually reach the target.
stretch.set_max_efforts(np.full((1, len(arm_dof)), 200.0), joint_indices=arm_dof)

# Cap the speed so the arm extends/retracts gently (URDF default is 1.0 m/s).
ARM_MAX_VEL = 0.1   # m/s per segment
stretch.set_max_joint_velocities(
    np.full((1, len(arm_dof)), ARM_MAX_VEL), joint_indices=arm_dof)

# --- Make the base kinematic ---
# The differential-drive wheels have an in-place-rotation deadzone in
# simulation (the caster is merged into base_link and the centre of mass sits
# ahead of the axle, so pivoting needs sideways wheel scrub that static
# friction blocks until |angular.z| ~0.9). Rather than fight the contact
# physics, the ROS node integrates the commanded twist into the base pose and
# teleports base_link there every step (StretchROS.integrate_base). Undrive the
# wheels (kp=kd=0, zero friction) so their cosmetic spin adds no reaction.
wheel_dof = np.array([stretch.get_dof_index("joint_left_wheel"),
                      stretch.get_dof_index("joint_right_wheel")])
stretch.set_gains(kps=np.zeros((1, 2)), kds=np.zeros((1, 2)), joint_indices=wheel_dof)
stretch.set_friction_coefficients(np.zeros((1, 2)), joint_indices=wheel_dof)

for _ in range(3):
    my_world.step(render=True)
print("Stretch ready: arm gains raised, base kinematic.")


# ------------------------------------------------------------------ Head camera
import omni
from pxr import UsdGeom
from isaacsim.sensors.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils

stage = omni.usd.get_context().get_stage()

head_cam_prim = ("/World/stretch/link_head_tilt/camera_bottom_screw_frame/"
                 "camera_link/camera_color_frame/camera_color_optical_frame")
UsdGeom.Camera.Define(stage, head_cam_prim)
head_cam = Camera(
    prim_path=head_cam_prim,
    frequency=30,
    resolution=(640, 360),
    orientation=rot_utils.euler_angles_to_quats(np.array([-90, 0, 0]), degrees=True),
)
head_cam.initialize()
head_cam.set_focal_length(1.5)
head_cam.set_clipping_range(near_distance=0.01, far_distance=20)

for _ in range(20):
    my_world.step(render=True)

# Gripper finger DOFs for /stretch/gripper_command.
finger_dof = np.array([stretch.get_dof_index("joint_gripper_finger_left"),
                       stretch.get_dof_index("joint_gripper_finger_right")])


# ------------------------------------------------------------- ROS 2 bridge node
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from sensor_msgs.msg import JointState, Image
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64, Float64MultiArray
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster

# Joints accepting streamed velocity commands on /<prefix>/joint_velocity_cmd
# (Float64MultiArray carries no names, so this ORDER is the contract with the
# publisher -- keep it in sync with giskard_stretch/stretch_joints.py).
VELOCITY_CONTROLLED_JOINTS = [
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

if not rclpy.ok():
    rclpy.init(args=None)


# --- tiny quaternion helpers (scalar-last x, y, z, w), no external deps ---
def _qconj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def _qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def _qrot(q, v):
    u = q[:3]
    s = q[3]
    return 2 * np.dot(u, v) * u + (s * s - np.dot(u, u)) * v + 2 * s * np.cross(u, v)


def _as_np(x):
    if hasattr(x, "numpy"):  # warp array or torch CPU tensor
        try:
            return x.numpy()
        except Exception:
            return x.detach().cpu().numpy()
    return np.asarray(x)


class StretchROS(Node):
    def __init__(self, robot, head_cam=None, finger_dof=None, prefix="stretch"):
        super().__init__(f"{prefix}_ros")
        self.robot = robot
        self.head_cam = head_cam
        self.finger_dof = finger_dof

        # Differential base geometry (for the cosmetic wheel spin only -- the
        # base is driven kinematically, see integrate_base).
        self.wheel_base = 0.3407
        self.wheel_radius = 0.051
        self.wheel_dof = np.array([robot.get_dof_index("joint_left_wheel"),
                                   robot.get_dof_index("joint_right_wheel")])

        # Internal commanded base pose for kinematic dead-reckoning, seeded from
        # the robot's current world pose; integrate_base advances it from cmd_vel.
        _p, _q = robot.get_world_poses()
        self._bx, self._by, self._bz = (float(_p[0][0]), float(_p[0][1]),
                                        float(_p[0][2]))
        _w, _x, _y, _z = _q[0]
        self._byaw = math.atan2(2.0 * (_w * _z + _x * _y),
                                1.0 - 2.0 * (_y * _y + _z * _z))
        self._cmd_v = 0.0
        self._cmd_w = 0.0
        # cmd_vel watchdog: a streamed twist (giskard/Nav2) that stops arriving
        # is zeroed after this many seconds instead of staying latched forever.
        self.CMD_VEL_TIMEOUT = 1.0
        self._cmd_time = None

        # Streamed joint velocity commands (giskard closed-loop control):
        # integrated into position targets each sim step, see
        # integrate_joint_velocities.
        self.VEL_CMD_TIMEOUT = 0.5   # s without a message -> hold position
        self.VEL_MAX_LEAD = 0.02     # rad/m: max target lead over measured
        self.vel_cmd_dof = np.array(
            [robot.get_dof_index(j) for j in VELOCITY_CONTROLLED_JOINTS])
        dof_limits = robot.get_dof_limits()[0]
        self._vel_dof_lower = dof_limits[self.vel_cmd_dof, 0]
        self._vel_dof_upper = dof_limits[self.vel_cmd_dof, 1]
        self._vel_cmd = None
        self._vel_cmd_time = None
        self._vel_targets = None

        # TF: link names and the base link index
        self.body_names = list(robot.body_names)
        self.base_idx = next(
            (i for i, n in enumerate(self.body_names) if "base_link" in n), 0
        )
        self._odom_prev = None          # (t, x, y, yaw) for finite-difference twist

        self.create_subscription(Twist, f"/{prefix}/cmd_vel", self.cmd_vel_cb, 10)
        self.create_subscription(Twist, "/cmd_vel", self.cmd_vel_cb, 10)
        self.create_subscription(JointState, f"/{prefix}/joint_command",
                                 self.joint_cmd_cb, 10)
        self.create_subscription(Float64MultiArray, f"/{prefix}/joint_velocity_cmd",
                                 self.joint_vel_cmd_cb, 10)
        self.create_subscription(Float64, f"/{prefix}/gripper_command",
                                 self.gripper_cmd_cb, 10)
        self.pub_js = self.create_publisher(JointState, f"/{prefix}/joint_states", 10)
        self.pub_head_img = self.create_publisher(Image, "/head_camera/image_raw", 10)
        self.pub_odom = self.create_publisher(Odometry, "/odom", 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf = StaticTransformBroadcaster(self)
        self._publish_static_tf()

    def cmd_vel_cb(self, msg):
        # Just latch the twist; integrate_base (called every sim step) applies it.
        self._cmd_v = float(msg.linear.x)
        self._cmd_w = float(msg.angular.z)
        self._cmd_time = time.time()

    def joint_vel_cmd_cb(self, msg):
        if len(msg.data) != len(self.vel_cmd_dof):
            self.get_logger().warning(
                f"joint_velocity_cmd has {len(msg.data)} values, expected "
                f"{len(self.vel_cmd_dof)}; dropping", throttle_duration_sec=5.0)
            return
        self._vel_cmd = np.asarray(msg.data, dtype=float)
        self._vel_cmd_time = time.time()

    def integrate_joint_velocities(self, dt):
        """Integrate streamed joint velocities into position targets (called
        every sim step, like integrate_base). Silence beyond VEL_CMD_TIMEOUT
        holds position (the drives latch the last targets); zero-velocity
        joints stay anchored to their measured position; the lead clamp keeps
        targets near reality so a blocked joint (contact) is not wound up."""
        if self._vel_cmd is None:
            return
        if time.time() - self._vel_cmd_time > self.VEL_CMD_TIMEOUT:
            self._vel_targets = None
            return
        measured = self.robot.get_joint_positions()[0][self.vel_cmd_dof]
        if self._vel_targets is None:
            self._vel_targets = measured.copy()
        target = self._vel_targets + self._vel_cmd * dt
        target = np.where(self._vel_cmd == 0.0, measured, target)
        target = np.clip(target, measured - self.VEL_MAX_LEAD,
                         measured + self.VEL_MAX_LEAD)
        target = np.clip(target, self._vel_dof_lower, self._vel_dof_upper)
        self._vel_targets = target
        self.robot.set_joint_position_targets(
            target.reshape(1, -1), joint_indices=self.vel_cmd_dof)

    def integrate_base(self, dt):
        """Kinematic base: dead-reckon the latched twist into the base pose and
        teleport base_link there. Avoids the differential-wheel in-place-rotation
        deadzone (wheels scrub/stick below ~0.9 rad/s) and is exact at any speed.
        The wheels are spun cosmetically (undriven) for TF/visual consistency."""
        if (self._cmd_time is not None and (self._cmd_v or self._cmd_w)
                and time.time() - self._cmd_time > self.CMD_VEL_TIMEOUT):
            # watchdog: a silent stream must not keep driving the base forever
            self._cmd_v = 0.0
            self._cmd_w = 0.0
        v, w = self._cmd_v, self._cmd_w
        yaw = self._byaw
        self._bx += v * math.cos(yaw) * dt
        self._by += v * math.sin(yaw) * dt
        self._byaw = yaw + w * dt
        ny = self._byaw
        quat = np.array([[math.cos(ny / 2.0), 0.0, 0.0, math.sin(ny / 2.0)]])
        self.robot.set_world_poses(
            np.array([[self._bx, self._by, self._bz]], dtype=float), quat)
        # zero root velocity so physics does not add a second displacement
        self.robot.set_velocities(np.zeros((1, 6)))
        # cosmetic wheel spin (velocity STATE, undriven -> no chassis reaction)
        vl = (v - w * self.wheel_base / 2.0) / self.wheel_radius
        vr = (v + w * self.wheel_base / 2.0) / self.wheel_radius
        self.robot.set_joint_velocities(
            np.array([[vl, vr]]), joint_indices=self.wheel_dof)

    def joint_cmd_cb(self, msg):
        target = self.robot.get_joint_positions()[0]
        for name, pos in zip(msg.name, msg.position):
            target[self.robot.get_dof_index(name)] = pos
        self.robot.set_joint_position_targets([target])

    def gripper_cmd_cb(self, msg):
        tgt = self.robot.get_joint_positions()[0].copy()
        for i in self.finger_dof:
            tgt[i] = float(msg.data)
        self.robot.set_joint_position_targets([tgt])

    def publish_joint_states(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.robot.dof_names)
        msg.position = self.robot.get_joint_positions()[0].tolist()
        self.pub_js.publish(msg)

    def _image_msg(self, rgb):
        msg = Image()
        msg.height = rgb.shape[0]
        msg.width = rgb.shape[1]
        msg.encoding = "rgb8"
        msg.step = rgb.shape[1] * 3
        msg.data = np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()
        return msg

    def publish_camera(self):
        if self.head_cam is None:
            return
        rgba = self.head_cam.get_rgba()
        if rgba is None or len(rgba) == 0:
            return
        self.pub_head_img.publish(self._image_msg(rgba[:, :, :3]))

    def publish_odom(self):
        """Publish odom->base_link as nav_msgs/Odometry (ground-truth = perfect
        odometry). Twist is estimated by finite differences in the base frame."""
        lt = _as_np(self.robot._physics_view.get_link_transforms()).reshape(-1, 7)
        p, q = lt[self.base_idx, :3], lt[self.base_idx, 3:7]
        now = self.get_clock().now()

        msg = Odometry()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = "odom"
        msg.child_frame_id = "base_link"
        msg.pose.pose.position.x = float(p[0])
        msg.pose.pose.position.y = float(p[1])
        msg.pose.pose.position.z = float(p[2])
        msg.pose.pose.orientation.x = float(q[0])
        msg.pose.pose.orientation.y = float(q[1])
        msg.pose.pose.orientation.z = float(q[2])
        msg.pose.pose.orientation.w = float(q[3])

        t = now.nanoseconds * 1e-9
        yaw = math.atan2(2.0 * (q[3] * q[2] + q[0] * q[1]),
                         1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]))
        if self._odom_prev is not None:
            pt, px, py, pyaw = self._odom_prev
            dt = t - pt
            if dt > 1e-6:
                dx, dy = float(p[0]) - px, float(p[1]) - py
                cos_y, sin_y = math.cos(yaw), math.sin(yaw)
                msg.twist.twist.linear.x = (dx * cos_y + dy * sin_y) / dt
                msg.twist.twist.linear.y = (-dx * sin_y + dy * cos_y) / dt
                dyaw = math.atan2(math.sin(yaw - pyaw), math.cos(yaw - pyaw))
                msg.twist.twist.angular.z = dyaw / dt
        self._odom_prev = (t, float(p[0]), float(p[1]), yaw)
        self.pub_odom.publish(msg)

    def _publish_static_tf(self):
        # base_link -> laser, from the URDF joint_laser (xyz 0.004 0 0.1664,
        # yaw pi). The fixed "laser" link is merged out of the articulation on
        # URDF import, so it is not in body_names -- publish it here.
        now = self.get_clock().now().to_msg()
        t = self._make_tf(now, "base_link", "laser",
                          np.array([0.004, 0.0, 0.1664]),
                          np.array([0.0, 0.0, 1.0, 0.0]))  # x,y,z,w -> yaw pi
        self.static_tf.sendTransform([t])

    def publish_tf(self):
        # World pose of every link: (num_links, 7) = x, y, z, qx, qy, qz, qw.
        # _physics_view is private but the only non-OmniGraph way to read all link poses.
        lt = _as_np(self.robot._physics_view.get_link_transforms()).reshape(-1, 7)
        pb, qb = lt[self.base_idx, :3], lt[self.base_idx, 3:7]
        qb_inv = _qconj(qb)
        now = self.get_clock().now().to_msg()

        # odom -> base_link (ground-truth pose as perfect odometry).
        tfs = [self._make_tf(now, "odom", "base_link", pb, qb)]
        for i, name in enumerate(self.body_names):  # base_link -> every other link
            if i == self.base_idx:
                continue
            p_rel = _qrot(qb_inv, lt[i, :3] - pb)
            q_rel = _qmul(qb_inv, lt[i, 3:7])
            tfs.append(self._make_tf(now, "base_link", name, p_rel, q_rel))

        self.tf_broadcaster.sendTransform(tfs)

    @staticmethod
    def _make_tf(stamp, parent, child, p, q):
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = parent
        t.child_frame_id = child
        t.transform.translation.x = float(p[0])
        t.transform.translation.y = float(p[1])
        t.transform.translation.z = float(p[2])
        t.transform.rotation.x = float(q[0])
        t.transform.rotation.y = float(q[1])
        t.transform.rotation.z = float(q[2])
        t.transform.rotation.w = float(q[3])
        return t


stretch_node = StretchROS(stretch, head_cam=head_cam, finger_dof=finger_dof,
                          prefix="stretch")
print("StretchROS node ready.")


# -------------------------------------------------------------------- Main loop
try:
    while simulation_app.is_running():
        dt = my_world.get_rendering_dt()
        stretch_node.integrate_base(dt)
        stretch_node.integrate_joint_velocities(dt)
        my_world.step(render=True)
        rclpy.spin_once(stretch_node, timeout_sec=0.0)
        stretch_node.publish_joint_states()
        stretch_node.publish_tf()
        stretch_node.publish_odom()
        stretch_node.publish_camera()
except KeyboardInterrupt:
    pass
finally:
    stretch_node.destroy_node()
    rclpy.shutdown()
    simulation_app.close()
