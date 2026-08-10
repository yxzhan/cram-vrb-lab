"""Stretch robot in Isaac Sim: spawning, drive tuning, and the ROS 2 bridge node.

- :func:`spawn_stretch` loads the USD, fixes the arm joint gains and undrives
  the wheels (the base is driven kinematically, see ``StretchROS.integrate_base``).
- :func:`create_head_camera` sets up the RGBD head camera.
- :class:`StretchROS` publishes joint states, /odom, TF and the camera streams,
  and consumes cmd_vel / streamed joint velocities / gripper commands on the
  topics defined in :mod:`cram_vrb_lab.robots.stretch.joints`.

.. warning::
   Import only after :func:`cram_vrb_lab.sim.isaac_app.create_simulation_app`
   has run -- this module imports ``isaacsim.core`` at module scope.
"""

import math
import tempfile
import time

import numpy as np
import omni.kit.commands
from geometry_msgs.msg import Twist
from isaacsim.core.prims import Articulation, XFormPrim
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Float64, Float64MultiArray
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from cram_vrb_lab.sim.velocity_integrator import (
    StreamedVelocityIntegrator,
    dof_indices,
)
from cram_vrb_lab.sim.ros_utils import (
    SimBridge,
    as_np,
    build_camera_info,
    depth_msg,
    image_msg,
    make_tf,
    qconj,
    qmul,
    qrot,
)
from .joints import (
    CAMERA_FRAME_ID,
    CMD_VEL_TOPIC,
    CONTROLLED_JOINTS,
    DEPTH_IMAGE_TOPIC,
    DEPTH_INFO_TOPIC,
    FINGER_JOINTS,
    GRIPPER_CMD_TOPIC,
    JOINT_STATES_TOPIC,
    ODOM_TOPIC,
    RGB_IMAGE_TOPIC,
    RGB_INFO_TOPIC,
    VELOCITY_CMD_TOPIC,
    head_camera_static_transforms,
    load_patched_urdf,
)

JOINT_DRIVE_STIFFNESS = 1.0e5
JOINT_DRIVE_DAMPING = 1.0e4
"""Drive gains for every joint the robot is commanded through.

One number for all of them, which only makes sense once you know that the URDF
importer authors **acceleration** drives: PhysX reads the stiffness as
(m/s^2)/m or (rad/s^2)/rad and scales the force by the joint's effective mass, so
what a gravity load costs in droop is ``g / stiffness`` *whatever the link
weighs*. A 2 kg lift and a 50 g finger therefore want the same gain, and at 1e5
both sit within about 0.1 mm of their target. (Same reasoning, and the same
measurement, as :data:`cram_vrb_lab.robots.panda.isaac_node.FINGER_MASS`.)

The importer derives its own gains from link inertias and they are far too soft
in these units: the lift sags to the bottom of its 1.1 m travel and the wrist
droops onto its limit. That is what used to make this robot unusable when
imported from URDF, and why it was loaded from a hand-tuned USD instead.

The stiffness also sets the grip: with ``StreamedVelocityIntegrator.MAX_LEAD`` of
0.02 rad the finger drive develops 1e5 * I_finger * 0.02 of torque.
"""

DRIVEN_JOINTS = [joint for joint in CONTROLLED_JOINTS if "wheel" not in joint]
"""Everything given a position drive: the controlled joints minus the wheels,
which are undriven so their cosmetic spin adds no reaction to the base."""

STRETCH_PRIM_PATH = "/stretch"
"""Where the importer puts the robot: ``/`` plus the URDF's ``<robot name=...>``."""

# The camera sensor hangs off the very frame its images are stamped in, so it
# needs no pose maths of its own. The importer lays the links out flat under the
# robot prim rather than nested as the link tree is, but the frame is a real part
# of the articulation: measured across a head_pan/head_tilt move, it travels with
# link_head_tilt and keeps a constant 5.46 cm offset from it -- the same offset
# the URDF's fixed chain gives. It only exists because merge_fixed_joints is off,
# which is also what keeps joints.head_camera_static_transforms() a valid chain.
HEAD_CAM_FRAME_PRIM = f"{STRETCH_PRIM_PATH}/{CAMERA_FRAME_ID}"
HEAD_CAM_PRIM = f"{HEAD_CAM_FRAME_PRIM}/head_camera"
CAMERA_RESOLUTION = (640, 360)


def spawn_stretch(world, render, position=(0.0, 0.0, 0.0), yaw=0.0):
    """Import the Stretch URDF into the open stage, tune its drives, and return
    its Articulation.

    :param position: base position in the Isaac world frame (= giskard's ``map``).
    :param yaw: heading about z [rad].

    Built from the very URDF giskard and the twin plan against
    (:func:`cram_vrb_lab.robots.stretch.joints.load_patched_urdf`), as the Panda
    is, so no converted USD has to be kept in step with it. What a URDF does not
    carry is **drives**: the importer derives its own, and they cannot hold this
    robot up -- see :data:`JOINT_DRIVE_STIFFNESS` for what replaces them and why
    the numbers look the way they do.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".urdf", prefix="stretch_patched_", delete=False
    ) as urdf_file:
        urdf_file.write(load_patched_urdf())
        urdf_path = urdf_file.name

    _, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    # A mobile robot: the base is free, and StretchROS.integrate_base teleports it.
    import_config.fix_base = False
    import_config.import_inertia_tensor = True
    import_config.distance_scale = 1.0
    # Keep the fixed joints: the head-camera frame chain hangs off them, and the
    # semantic model looks bodies up by the names they carry.
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    # Nothing here needs the robot to avoid itself -- giskard plans the motions --
    # and the gripper's hulls overlap the wrist they are mounted on.
    import_config.self_collision = False

    articulation_root = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=urdf_path,
        import_config=import_config,
        get_articulation_root=True,
    )[1]
    print(f"Stretch imported from {urdf_path} to {articulation_root}")

    # On the prim, before physics runs: this is the pose a later world.reset()
    # restores, and where the base starts dead-reckoning from.
    XFormPrim(STRETCH_PRIM_PATH).set_world_poses(
        np.array([position], dtype=float),
        # Isaac's (w, x, y, z) order; a pure yaw, so only w and z are non-zero.
        np.array([[math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]], dtype=float),
    )

    # Reset before wrapping: the freshly imported prims are not in the physics
    # scene yet, and Articulation reads its link metadata in its constructor.
    world.reset()
    for _ in range(5):
        world.step(render=render)

    stretch = Articulation(prim_paths_expr=articulation_root, name="stretch")
    world.reset()
    for _ in range(10):
        world.step(render=render)

    print(f"Stretch spawned at {tuple(round(float(v), 4) for v in position)} "
          f"yaw {math.degrees(yaw):.1f} deg")

    # --- Replace the importer's drives ---
    # Every joint the robot is commanded through, in one go: see
    # JOINT_DRIVE_STIFFNESS for why one number covers a 2 kg lift and a 50 g
    # finger alike.
    driven_dof = dof_indices(stretch, DRIVEN_JOINTS)
    stretch.set_gains(
        kps=np.full((1, len(driven_dof)), JOINT_DRIVE_STIFFNESS),
        kds=np.full((1, len(driven_dof)), JOINT_DRIVE_DAMPING),
        joint_indices=driven_dof,
    )
    # A force budget the drives can actually spend (the URDF says 100 for every
    # joint, including the lift that carries the whole arm).
    stretch.set_max_efforts(
        np.full((1, len(driven_dof)), 200.0), joint_indices=driven_dof
    )

    # Cap the telescoping joints' speed so the arm extends gently (URDF: 1.0 m/s).
    telescope_dof = dof_indices(
        stretch, ["joint_arm_l0", "joint_arm_l1", "joint_arm_l2", "joint_arm_l3"]
    )
    stretch.set_max_joint_velocities(
        np.full((1, len(telescope_dof)), 0.1), joint_indices=telescope_dof
    )

    # --- Make the base kinematic ---
    # The differential-drive wheels have an in-place-rotation deadzone in
    # simulation (the caster is merged into base_link and the centre of mass sits
    # ahead of the axle, so pivoting needs sideways wheel scrub that static
    # friction blocks until |angular.z| ~0.9). Rather than fight the contact
    # physics, the ROS node integrates the commanded twist into the base pose and
    # teleports base_link there every step (StretchROS.integrate_base). Undrive the
    # wheels (kp=kd=0, zero friction) so their cosmetic spin adds no reaction.
    wheel_dof = dof_indices(stretch, ["joint_left_wheel", "joint_right_wheel"])
    stretch.set_gains(kps=np.zeros((1, 2)), kds=np.zeros((1, 2)),
                      joint_indices=wheel_dof)
    stretch.set_friction_coefficients(np.zeros((1, 2)), joint_indices=wheel_dof)

    for _ in range(3):
        world.step(render=render)
    print("Stretch ready: drives replaced, base kinematic.")
    return stretch


def create_head_camera(world, render, want_depth=False):
    """One RGBD head camera sensor; the caller picks which streams StretchROS
    publishes. The camera does RTX raytraced rendering, so skipping it
    (``--camera none`` / ISAAC_NO_CAMERA=1) also lets machines whose GPU/display
    cannot render it run the control path, which does not use the camera."""
    import omni
    from pxr import UsdGeom
    from isaacsim.sensors.camera import Camera
    import isaacsim.core.utils.numpy.rotations as rot_utils 

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath(HEAD_CAM_FRAME_PRIM):
        raise RuntimeError(f"no {HEAD_CAM_FRAME_PRIM} to mount the head camera on")

    UsdGeom.Camera.Define(stage, HEAD_CAM_PRIM)
    head_cam = Camera(
        prim_path=HEAD_CAM_PRIM,
        frequency=30,
        resolution=CAMERA_RESOLUTION,
    )
    # The orientation is set on the prim, NOT through Camera(orientation=...):
    # that argument goes through Isaac's own ROS<->USD camera axis conversion, and
    # passing the identity there lands the prim looking straight at the floor
    # (measured: prim quat (0.708, 0, 0, -0.706) against a frame at
    # (0.707, 0, 0.707, 0), while Camera.get_world_pose() cheerfully reports the
    # frame's). Here the parent is a REP-103 optical frame -- +z is the view
    # direction, +y is down -- and a USD camera looks down its own -z with +y up,
    # so half a turn about x is what aligns them.
    XFormPrim(HEAD_CAM_PRIM).set_local_poses(
        translations=np.zeros((1, 3)),
        orientations=np.array([[0.0, 1.0, 0.0, 0.0]]),  # (w, x, y, z)
    )
    head_cam.initialize()
    head_cam.set_focal_length(1.0)
    head_cam.set_clipping_range(near_distance=0.05, far_distance=20)
    if want_depth:
        # distance_to_image_plane = metric depth (m) read back by get_depth().
        head_cam.add_distance_to_image_plane_to_frame()

    for _ in range(20):
        world.step(render=render)
    return head_cam


class StretchROS(SimBridge):
    def __init__(self, robot, head_cam=None, publish_rgb=True, publish_depth=False):
        super().__init__("stretch_ros")
        self.robot = robot
        self.head_cam = head_cam
        self.publish_rgb = publish_rgb
        self.publish_depth = publish_depth

        # Differential base geometry (for the cosmetic wheel spin only -- the
        # base is driven kinematically, see integrate_base).
        self.wheel_base = 0.3407
        self.wheel_radius = 0.051
        self.wheel_dof = dof_indices(
            robot, ["joint_left_wheel", "joint_right_wheel"])

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
        self.integrator = StreamedVelocityIntegrator(
            robot, CONTROLLED_JOINTS, holding_joints=FINGER_JOINTS)

        # TF: link names and the base link index
        self.body_names = list(robot.body_names)
        self.base_idx = next(
            (i for i, n in enumerate(self.body_names) if "base_link" in n), 0
        )
        self._odom_prev = None          # (t, x, y, yaw) for finite-difference twist

        # Streamed command topics use queue depth 1: only the LATEST command
        # matters, and letting a queue build up makes the sim execute commands
        # that are hundreds of milliseconds old -- the controller answers the
        # perceived lag with overshoot and eventually a limit cycle.
        self.create_subscription(Twist, CMD_VEL_TOPIC, self.cmd_vel_cb, 1)
        self.create_subscription(Float64MultiArray, VELOCITY_CMD_TOPIC,
                                 self.joint_vel_cmd_cb, 1)
        self.create_subscription(Float64, GRIPPER_CMD_TOPIC,
                                 self.gripper_cmd_cb, 10)
        self.pub_js = self.create_publisher(JointState, JOINT_STATES_TOPIC, 10)
        # Intrinsics are constant; build the CameraInfo once and just restamp it.
        self._camera_info = (
            build_camera_info(self.head_cam, *CAMERA_RESOLUTION, CAMERA_FRAME_ID)
            if self.head_cam is not None else None)
        if self.publish_rgb:
            self.pub_head_img = self.create_publisher(Image, RGB_IMAGE_TOPIC, 10)
            self.pub_head_info = self.create_publisher(CameraInfo, RGB_INFO_TOPIC, 10)
        if self.publish_depth:
            self.pub_head_depth = self.create_publisher(
                Image, DEPTH_IMAGE_TOPIC, 10)
            self.pub_head_depth_info = self.create_publisher(
                CameraInfo, DEPTH_INFO_TOPIC, 10)
        self.pub_odom = self.create_publisher(Odometry, ODOM_TOPIC, 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        self.publish_camera_static_tf()

    def cmd_vel_cb(self, msg):
        # Just latch the twist; integrate_base (called every sim step) applies it.
        self._cmd_v = float(msg.linear.x)
        self._cmd_w = float(msg.angular.z)
        self._cmd_time = time.time()

    def joint_vel_cmd_cb(self, msg):
        if not self.integrator.accept(msg.data):
            self.get_logger().warning(
                f"joint_velocity_cmd has {len(msg.data)} values, expected "
                f"{len(CONTROLLED_JOINTS)}; dropping", throttle_duration_sec=5.0)

    def apply_commands(self, dt):
        self.integrate_base(dt)
        self.integrate_joint_velocities(dt)

    def publish(self):
        self.publish_joint_states()
        self.publish_tf()
        self.publish_odom()
        self.publish_camera()

    def integrate_joint_velocities(self, dt):
        """Integrate streamed joint velocities into position targets, called
        every sim step like :meth:`integrate_base`. See
        :class:`~cram_vrb_lab.sim.velocity_integrator.StreamedVelocityIntegrator`
        for why this is not a plain Euler step."""
        self.integrator.step(dt)

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

    def gripper_cmd_cb(self, msg):
        """Command both fingers to a travel [m] directly, bypassing giskard.

        Through the integrator rather than straight onto the drives: it owns the
        targets of every controlled joint, the fingers included, so a direct write
        would be overwritten within one sim step (see
        :meth:`~cram_vrb_lab.sim.velocity_integrator.StreamedVelocityIntegrator.hold_at`).
        """
        self.integrator.hold_at(FINGER_JOINTS, msg.data)

    def publish_joint_states(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.robot.dof_names)
        msg.position = self.robot.get_joint_positions()[0].tolist()
        self.pub_js.publish(msg)

    def _publish_camera_info(self, pub, stamp):
        self._camera_info.header.stamp = stamp
        pub.publish(self._camera_info)

    def publish_camera(self):
        if self.head_cam is None:
            return
        stamp = self.get_clock().now().to_msg()
        if self.publish_rgb:
            rgba = self.head_cam.get_rgba()
            if rgba is not None and len(rgba) > 0:
                self.pub_head_img.publish(
                    image_msg(rgba[:, :, :3], stamp, CAMERA_FRAME_ID))
                self._publish_camera_info(self.pub_head_info, stamp)
        if self.publish_depth:
            depth = self.head_cam.get_depth()
            if depth is not None and len(depth) > 0:
                self.pub_head_depth.publish(
                    depth_msg(depth, stamp, CAMERA_FRAME_ID))
                self._publish_camera_info(self.pub_head_depth_info, stamp)

    def publish_odom(self):
        """Publish odom->base_link as nav_msgs/Odometry (ground-truth = perfect
        odometry). Twist is estimated by finite differences in the base frame."""
        lt = as_np(self.robot._physics_view.get_link_transforms()).reshape(-1, 7)
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

    def publish_camera_static_tf(self):
        """Publish the fixed head-camera frame chain as static tf.

        The frames (camera_link ... camera_color_optical_frame /
        camera_depth_optical_frame) hang off link_head_tilt, which the per-step
        :meth:`publish_tf` already emits; the fixed frames below it normally come
        from the giskard server, which drops them while executing a goal. Publishing
        them here once (latched) keeps camera_color_optical_frame -- the frame the
        head-camera images are stamped in -- available continuously.

        The transforms come from the same patched URDF giskard parses, NOT from the
        runtime USD prims: ``create_head_camera`` turns the camera_color_optical_frame
        prim into a ``UsdGeom.Camera`` and rewrites its orientation, so its live
        transform no longer carries the ROS optical rotation.
        """
        now = self.get_clock().now().to_msg()
        tfs = [
            make_tf(now, parent, child, xyz, quat)
            for parent, child, xyz, quat in head_camera_static_transforms()
        ]
        self.static_tf_broadcaster.sendTransform(tfs)

    def publish_tf(self):
        # World pose of every link: (num_links, 7) = x, y, z, qx, qy, qz, qw.
        # _physics_view is private but the only non-OmniGraph way to read all link poses.
        lt = as_np(self.robot._physics_view.get_link_transforms()).reshape(-1, 7)
        pb, qb = lt[self.base_idx, :3], lt[self.base_idx, 3:7]
        qb_inv = qconj(qb)
        now = self.get_clock().now().to_msg()

        # odom -> base_link (ground-truth pose as perfect odometry).
        tfs = [make_tf(now, "odom", "base_link", pb, qb)]
        for i, name in enumerate(self.body_names):  # base_link -> every other link
            if i == self.base_idx:
                continue
            p_rel = qrot(qb_inv, lt[i, :3] - pb)
            q_rel = qmul(qb_inv, lt[i, 3:7])
            tfs.append(make_tf(now, "base_link", name, p_rel, q_rel))

        self.tf_broadcaster.sendTransform(tfs)
