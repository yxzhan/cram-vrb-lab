"""GARMI in Isaac Sim: URDF import, drive tuning, and the ROS 2 bridge node.

Two robots' worth of pattern, both already in this repo:

- the arms are FR3s with Franka Hands, so the drive tuning is
  :mod:`cram_vrb_lab.robots.panda.isaac_node`'s;
- the base is a mobile base driven kinematically, so it is
  :mod:`cram_vrb_lab.robots.stretch.isaac_node`'s -- ``integrate_base``,
  ``publish_odom`` and ``publish_tf`` below are that node's, widened from a
  differential drive's two degrees of freedom to an omni drive's three.

.. warning::
   Import only after :func:`cram_vrb_lab.sim.isaac_app.create_simulation_app`
   has run -- this module imports ``isaacsim.core`` and ``omni`` at module scope.
"""

import math
import tempfile
import time

import numpy as np
import omni.kit.commands
from geometry_msgs.msg import Twist
from isaacsim.core.prims import Articulation, XFormPrim
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray
from tf2_ros import TransformBroadcaster

from cram_vrb_lab.sim.ros_utils import SimBridge, as_np, make_tf, qconj, qmul, qrot
from cram_vrb_lab.sim.velocity_integrator import (
    MAX_LEAD,
    StreamedVelocityIntegrator,
    dof_indices,
)

from .joints import (
    ARM_JOINTS,
    CMD_VEL_TOPIC,
    CONTROLLED_JOINTS,
    FINGER_JOINTS,
    FINGER_MASS,
    GRIPPER_CMD_TOPIC,
    GRIPPER_OPEN_TRAVEL,
    HEAD_JOINTS,
    JOINT_STATES_TOPIC,
    LIFT_JOINTS,
    ODOM_TOPIC,
    PARK_CONFIGURATION,
    ROBOT_NAME,
    SIDES,
    VELOCITY_CMD_TOPIC,
    WHEEL_JOINTS,
    arm_joints,
    load_patched_urdf,
)

GARMI_PRIM_PATH = f"/{ROBOT_NAME}"
"""Where the importer puts the robot: ``/`` plus the URDF's ``<robot name=...>``,
which :func:`~cram_vrb_lab.robots.garmi.joints.load_patched_urdf` rewrites to a
name that survives the importer's prim-path sanitising."""

BASE_LINK_HEIGHT = 0.0259
"""Height [m] of ``base_link`` above the floor when the wheels are on it.

The wheel centres sit 0.05 m above ``base_link`` and the wheels have a 0.0759 m
radius; the description's own MuJoCo ``home`` keyframe opens with the same
number. A demo that spawns GARMI at z = 0 sinks it into the floor by this much.
"""

WHEEL_RADIUS = 0.0759
"""[m], from the wheels' collision cylinders. Only used for the cosmetic spin."""

ARM_DRIVE_STIFFNESS = 1.0e5
"""[N m/rad]. The importer derives drive gains from link inertias, which leaves
the wrist joints soft enough to sag under the hand."""

ARM_DRIVE_DAMPING = 1.0e4

COLUMN_DRIVE_STIFFNESS = 1.0e5
COLUMN_DRIVE_DAMPING = 1.0e4
"""Gains for the lift and the head.

Same reasoning and the same number as the arm's, which is only sensible because
the importer authors **acceleration** drives: PhysX scales the force by the
joint's effective mass, so a 20 kg lift column and a 1 kg head want the same gain
(see the Stretch's ``JOINT_DRIVE_STIFFNESS`` for the full account). Without them
the lift sinks under the torso it carries and the head tips forward, which is
what freezing these joints used to hide.
"""

FINGER_DRIVE_STIFFNESS = 5000.0
FINGER_DRIVE_DAMPING = 142.0
"""Finger drive gains as a **force** drive would take them, [N/m] and [N/(m/s)].

The stiffness *is* the grip, and there is no other knob for it: the integrator
lets a target lead the measured position by
:data:`~cram_vrb_lab.sim.velocity_integrator.MAX_LEAD` and no further, so a
blocked finger settles at exactly ``stiffness * MAX_LEAD`` -- 8 N at NVIDIA's own
Franka value of 400 (which the Panda still runs on), 100 N here. That bound is
not a tuning artefact but the point of the clamp, so a finger will always yield
to anything that pushes harder; what the gain buys is *how much* harder.

100 N is the ceiling this description allows: it is the ``effort`` the URDF gives
both finger joints, and above it :data:`FINGER_MAX_EFFORT` would start clipping
instead. It is also past the real FR3 hand's 70 N continuous grasping force, so
there is nothing left to raise. If the fingers still lose their hold, the cause
is geometric -- the arm's drives are three orders of magnitude stronger, so a
gripper closing anywhere but *around* the rod is prised open no matter the gain.

8 N was sized for a different job. The Panda's note says as much: it is "far
above the 0.5 N the cube's weight needs", i.e. enough to hold a light object
against gravity. Opening a drawer is not that -- the arm has to hold a 12 mm
handle rod against the pull of the container coming out. Slowing the container
down is the other half of the fix; see ``PRISMATIC_VELOCITY_LIMIT`` in
:mod:`cram_vrb_lab.scenes.garmi_apartment.giskard_world`.

Damping is scaled by sqrt of the stiffness ratio rather than with it, which is
what keeps the damping ratio ``d / (2 * sqrt(k * m))`` where it was (5.9, i.e.
firmly overdamped, as NVIDIA's 400/40 pair already was): scaling it with the
stiffness would leave the fingers slower to close with every raise, leaving it
alone would let them ring.

Both are divided by :data:`~cram_vrb_lab.robots.garmi.joints.FINGER_MASS` before
they are handed to ``set_gains``, because the URDF importer authors *acceleration*
drives -- see the Panda's ``FINGER_MASS`` for the full account of what that costs
when it is missed.
"""

FINGER_MAX_EFFORT = FINGER_DRIVE_STIFFNESS * MAX_LEAD / FINGER_MASS
"""Force budget [N] handed to the finger drives, so ``maxForce`` never binds.

Raising the stiffness alone does nothing if the drive's force limit is reached
first, and the importer authors that limit from the URDF's ``effort="100"``
without saying which units it means it in. That matters here and nowhere else,
because these are the only **acceleration** drives whose limit is anywhere near
their output: PhysX may read ``maxForce`` in force units (100 N, exactly the
grip asked for above, so saturating) or in the drive's own acceleration units
(100 m/s^2, which after the finger's 29 g is **2.9 N** -- limp, and completely
insensitive to the gain, which is the symptom that led here).

Rather than pin down which, size the budget so it is not the binding constraint
under either reading: ``stiffness * MAX_LEAD`` is what the drive wants to spend,
and dividing by the mass expresses it in the pessimistic units. The grip stays
capped at ``stiffness * MAX_LEAD`` by the integrator's clamp regardless, so a
generous budget cannot make the fingers stronger than intended -- it can only
stop them being weaker.

The Stretch does the same thing for the same reason (``set_max_efforts(200.0)``,
against a URDF that also says 100 for every joint); the Panda does not, and gets
away with it because a cube needs 0.5 N.
"""


def spawn_garmi(
    world,
    render,
    position=(0.0, 0.0, 0.0),
    orientation=(1.0, 0.0, 0.0, 0.0),
):
    """Import GARMI into the open stage and return its Articulation.

    The base is placed on the *prim*, before physics ever runs. Unlike the
    Panda's, this pose is not shared with giskard: the robot drives, so giskard
    learns where it is from the odometry
    :meth:`GarmiROS.publish_odom` publishes from exactly here.

    :param orientation: quaternion in Isaac's ``(w, x, y, z)`` order.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".urdf", prefix="garmi_patched_", delete=False
    ) as urdf_file:
        urdf_file.write(load_patched_urdf())
        urdf_path = urdf_file.name

    _, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    # The base drives, so it is not welded to the world. It is still not moved by
    # its wheels: integrate_base teleports it, see undrive_wheels.
    import_config.fix_base = False
    import_config.import_inertia_tensor = True
    import_config.distance_scale = 1.0
    # Keep the fixed joints: the hands, the TCP frames and the mount frames the
    # semantic model looks bodies up by would otherwise be merged away, and the
    # twin and the render would no longer describe the same link tree.
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    # The fingers' collision hulls overlap the hand they are mounted on, and the
    # two arms are mounted close together on the torso; with self-collision on,
    # the solver spends every step pushing overlapping hulls apart. Giskard plans
    # the motions, against the garmi.srdf self-collision matrix the upstream
    # model loads.
    import_config.self_collision = False

    # Run from wherever the demo was launched: the importer copies the textures
    # the .obj meshes reference (body.mtl wants fabric.jpg, head.mtl wants
    # face.jpg) into `materials/textures` relative to the working directory AND
    # records them relative too, so write and lookup only agree as long as the
    # cwd is left alone. The drop is gitignored -- see /materials/ in .gitignore.
    articulation_root = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=urdf_path,
        import_config=import_config,
        get_articulation_root=True,
    )[1]
    print(f"GARMI imported from {urdf_path} to {articulation_root}")

    XFormPrim(GARMI_PRIM_PATH).set_world_poses(
        np.array([position], dtype=float), np.array([orientation], dtype=float)
    )
    print(f"GARMI placed at {tuple(round(v, 4) for v in position)} "
          f"quat(wxyz) {orientation}")

    # Reset before wrapping: the freshly imported prims are not in the physics
    # scene yet, and Articulation reads its link metadata in its constructor.
    world.reset()
    for _ in range(5):
        world.step(render=render)

    garmi = Articulation(prim_paths_expr=articulation_root, name=ROBOT_NAME)
    world.reset()
    for _ in range(10):
        world.step(render=render)

    print("GARMI imported and wrapped; call move_to_park once the scene is built.")
    return garmi


def undrive_wheels(garmi):
    """Take all force out of the four mecanum wheels.

    The base is driven kinematically -- :meth:`GarmiROS.integrate_base` teleports
    ``base_link`` -- and these wheels are then only cosmetic, so they must add no
    reaction to the chassis.

    Not a shortcut taken to save effort: **a mecanum wheel cannot be simulated
    from this description**. Each one's collision geometry is a plain cylinder;
    the angled rollers that produce sideways motion are not modelled at all, and
    the whole effect comes from their anisotropic friction. Gazebo fakes it with
    a wheel-slip plugin and a per-wheel friction direction (both are in the
    description, in the ``<gazebo>`` blocks this repo strips); PhysX has no
    equivalent. Driving these cylinders would give a base that can only go
    forwards, and giskard commands sideways velocity.
    """
    wheel_dof = dof_indices(garmi, WHEEL_JOINTS)
    zeros = np.zeros((1, len(wheel_dof)))
    garmi.set_gains(kps=zeros, kds=zeros, joint_indices=wheel_dof)
    garmi.set_friction_coefficients(zeros, joint_indices=wheel_dof)


def move_to_park(garmi, world, render):
    """Set the drive gains, undrive the wheels, and put the robot in its home pose.

    Both arms to :data:`~cram_vrb_lab.robots.garmi.joints.PARK_CONFIGURATION`,
    hands open, lift down and head level.

    .. warning::
       Call this **last**, after everything else in the scene has been spawned.
       ``world.reset()`` restores both the state physics started from and the
       drive parameters authored on the prims, so gains and poses set before
       anything that resets are silently thrown away.
    """
    arm_dof = dof_indices(garmi, ARM_JOINTS)
    finger_dof = dof_indices(garmi, FINGER_JOINTS)
    column_dof = dof_indices(garmi, LIFT_JOINTS + HEAD_JOINTS)

    garmi.set_gains(
        kps=np.full((1, len(arm_dof)), ARM_DRIVE_STIFFNESS),
        kds=np.full((1, len(arm_dof)), ARM_DRIVE_DAMPING),
        joint_indices=arm_dof,
    )
    garmi.set_gains(
        kps=np.full((1, len(column_dof)), COLUMN_DRIVE_STIFFNESS),
        kds=np.full((1, len(column_dof)), COLUMN_DRIVE_DAMPING),
        joint_indices=column_dof,
    )
    garmi.set_gains(
        kps=np.full((1, len(finger_dof)), FINGER_DRIVE_STIFFNESS / FINGER_MASS),
        kds=np.full((1, len(finger_dof)), FINGER_DRIVE_DAMPING / FINGER_MASS),
        joint_indices=finger_dof,
    )
    # The gain above is only worth what the drive is allowed to spend: see
    # FINGER_MAX_EFFORT for why the importer's limit cannot be trusted here.
    garmi.set_max_efforts(
        np.full((1, len(finger_dof)), FINGER_MAX_EFFORT), joint_indices=finger_dof
    )
    undrive_wheels(garmi)

    positions = garmi.get_joint_positions()
    for side in SIDES:
        positions[0, dof_indices(garmi, arm_joints(side))] = PARK_CONFIGURATION
    positions[0, finger_dof] = GRIPPER_OPEN_TRAVEL
    positions[0, column_dof] = 0.0
    garmi.set_joint_positions(positions)
    garmi.set_joint_position_targets(positions)
    for _ in range(30):
        world.step(render=render)

    print(f"GARMI parked, both arms at {PARK_CONFIGURATION}, lift down, head level")


class GarmiROS(SimBridge):
    """ROS 2 bridge for the simulated GARMI.

    Publishes joint states, odometry and TF for giskard to close its loop on, and
    accepts the streamed joint velocities and base Twist it sends back plus a
    direct gripper command.
    """

    CMD_VEL_TIMEOUT = 1.0
    """Seconds without a Twist before the base is stopped.

    A streamed command (giskard, Nav2) that stops arriving must not keep driving
    the robot across the flat forever.
    """

    def __init__(self, robot):
        super().__init__("garmi_ros")
        self.robot = robot
        self.integrator = StreamedVelocityIntegrator(
            robot, CONTROLLED_JOINTS, holding_joints=FINGER_JOINTS
        )

        # Queue depth 1 on the streamed commands: only the LATEST value matters,
        # and a backlog makes the sim execute stale commands.
        self.create_subscription(
            Float64MultiArray, VELOCITY_CMD_TOPIC, self.joint_velocity_cmd_cb, 1
        )
        self.create_subscription(Twist, CMD_VEL_TOPIC, self.cmd_vel_cb, 1)
        self.create_subscription(Float64, GRIPPER_CMD_TOPIC, self.gripper_cmd_cb, 10)
        self.pub_joint_states = self.create_publisher(
            JointState, JOINT_STATES_TOPIC, 10
        )
        self.pub_odom = self.create_publisher(Odometry, ODOM_TOPIC, 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Commanded base pose for kinematic dead-reckoning, seeded from where the
        # robot was spawned; integrate_base advances it from cmd_vel.
        position, quaternion = robot.get_world_poses()
        self._bx, self._by, self._bz = (float(v) for v in position[0])
        w, x, y, z = quaternion[0]
        self._byaw = math.atan2(
            2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)
        )
        self._cmd_x = self._cmd_y = self._cmd_yaw = 0.0
        self._cmd_time = None

        self.wheel_dof = dof_indices(robot, WHEEL_JOINTS)
        self.body_names = list(robot.body_names)
        self.base_idx = next(
            (i for i, name in enumerate(self.body_names) if "base_link" in name), 0
        )
        self._odom_prev = None

    # --- commands ----------------------------------------------------------

    def joint_velocity_cmd_cb(self, msg):
        if not self.integrator.accept(msg.data):
            self.get_logger().warning(
                f"{VELOCITY_CMD_TOPIC} carried {len(msg.data)} values, expected "
                f"{len(CONTROLLED_JOINTS)}; dropping",
                throttle_duration_sec=5.0,
            )

    def cmd_vel_cb(self, msg):
        """Latch the base twist; :meth:`integrate_base` applies it every sim step.

        ``linear.y`` is read, unlike on the differential-drive Stretch: giskard
        fills it for an ``OmniDrive`` connection, and dropping it would silently
        turn every sideways solution into a stall.
        """
        self._cmd_x = float(msg.linear.x)
        self._cmd_y = float(msg.linear.y)
        self._cmd_yaw = float(msg.angular.z)
        self._cmd_time = time.time()

    def gripper_cmd_cb(self, msg):
        """Command every finger to a travel [m] directly, bypassing giskard.

        Through the integrator rather than straight onto the drives: it owns the
        targets of every controlled joint, so a direct write would be overwritten
        within one sim step.
        """
        self.integrator.hold_at(FINGER_JOINTS, msg.data)

    def apply_commands(self, dt):
        self.integrate_base(dt)
        self.integrator.step(dt)

    def integrate_base(self, dt):
        """Dead-reckon the latched twist into the base pose and teleport there.

        The omni-drive version of the Stretch's ``integrate_base``: the commanded
        velocity is expressed in the *base* frame, so both components are rotated
        into the world frame before they are integrated. See
        :func:`undrive_wheels` for why the wheels do not do this themselves.
        """
        if self._cmd_time is not None and (
            self._cmd_x or self._cmd_y or self._cmd_yaw
        ):
            if time.time() - self._cmd_time > self.CMD_VEL_TIMEOUT:
                self._cmd_x = self._cmd_y = self._cmd_yaw = 0.0

        cos_yaw, sin_yaw = math.cos(self._byaw), math.sin(self._byaw)
        self._bx += (self._cmd_x * cos_yaw - self._cmd_y * sin_yaw) * dt
        self._by += (self._cmd_x * sin_yaw + self._cmd_y * cos_yaw) * dt
        self._byaw += self._cmd_yaw * dt

        half = self._byaw / 2.0
        self.robot.set_world_poses(
            np.array([[self._bx, self._by, self._bz]], dtype=float),
            np.array([[math.cos(half), 0.0, 0.0, math.sin(half)]], dtype=float),
        )
        # Zero the root velocity so physics does not add a second displacement on
        # top of the teleport.
        self.robot.set_velocities(np.zeros((1, 6)))

        # Cosmetic wheel spin (velocity STATE on undriven joints -> no chassis
        # reaction). A mecanum wheel's true spin depends on the roller geometry
        # this description does not model, so this only has to look plausible:
        # roll with the forward component and counter-rotate to turn.
        speed = self._cmd_x / WHEEL_RADIUS
        turn = self._cmd_yaw
        self.robot.set_joint_velocities(
            np.array([[speed - turn, speed + turn, speed - turn, speed + turn]]),
            joint_indices=self.wheel_dof,
        )

    # --- telemetry ---------------------------------------------------------

    def publish(self):
        self.publish_joint_states()
        self.publish_odom()
        self.publish_tf()

    def publish_joint_states(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.robot.dof_names)
        msg.position = self.robot.get_joint_positions()[0].tolist()
        self.pub_joint_states.publish(msg)

    def publish_odom(self):
        """Publish ``odom -> base_link`` as ground-truth (= perfect) odometry.

        The twist is estimated by finite differences and reported in the base
        frame, which is what an omni drive's odometry means.
        """
        transforms = as_np(
            self.robot._physics_view.get_link_transforms()
        ).reshape(-1, 7)
        position = transforms[self.base_idx, :3]
        quaternion = transforms[self.base_idx, 3:7]
        now = self.get_clock().now()

        msg = Odometry()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = "odom"
        msg.child_frame_id = "base_link"
        msg.pose.pose.position.x = float(position[0])
        msg.pose.pose.position.y = float(position[1])
        msg.pose.pose.position.z = float(position[2])
        msg.pose.pose.orientation.x = float(quaternion[0])
        msg.pose.pose.orientation.y = float(quaternion[1])
        msg.pose.pose.orientation.z = float(quaternion[2])
        msg.pose.pose.orientation.w = float(quaternion[3])

        stamp = now.nanoseconds * 1e-9
        yaw = math.atan2(
            2.0 * (quaternion[3] * quaternion[2] + quaternion[0] * quaternion[1]),
            1.0 - 2.0 * (quaternion[1] ** 2 + quaternion[2] ** 2),
        )
        if self._odom_prev is not None:
            previous_stamp, previous_x, previous_y, previous_yaw = self._odom_prev
            dt = stamp - previous_stamp
            if dt > 1e-6:
                dx = float(position[0]) - previous_x
                dy = float(position[1]) - previous_y
                cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
                msg.twist.twist.linear.x = (dx * cos_yaw + dy * sin_yaw) / dt
                msg.twist.twist.linear.y = (-dx * sin_yaw + dy * cos_yaw) / dt
                dyaw = math.atan2(
                    math.sin(yaw - previous_yaw), math.cos(yaw - previous_yaw)
                )
                msg.twist.twist.angular.z = dyaw / dt
        self._odom_prev = (stamp, float(position[0]), float(position[1]), yaw)
        self.pub_odom.publish(msg)

    def publish_tf(self):
        """``odom -> base_link -> every other link``, from the physics view.

        ``_physics_view`` is private but the only non-OmniGraph way to read every
        link pose at once.
        """
        transforms = as_np(
            self.robot._physics_view.get_link_transforms()
        ).reshape(-1, 7)
        base_position = transforms[self.base_idx, :3]
        base_quaternion = transforms[self.base_idx, 3:7]
        base_quaternion_inv = qconj(base_quaternion)
        now = self.get_clock().now().to_msg()

        tfs = [make_tf(now, "odom", "base_link", base_position, base_quaternion)]
        for index, name in enumerate(self.body_names):
            if index == self.base_idx:
                continue
            tfs.append(
                make_tf(
                    now,
                    "base_link",
                    name,
                    qrot(base_quaternion_inv, transforms[index, :3] - base_position),
                    qmul(base_quaternion_inv, transforms[index, 3:7]),
                )
            )
        self.tf_broadcaster.sendTransform(tfs)
