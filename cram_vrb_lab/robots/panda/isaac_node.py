"""Panda in Isaac Sim: URDF import, drive tuning, and the ROS 2 bridge node.

- :func:`spawn_panda` imports the patched URDF straight into the open stage, so
  no converted USD has to be checked in, tunes the drives and puts the arm in
  its park pose.
- :class:`PandaROS` publishes joint states and consumes streamed joint
  velocities and gripper commands on the topics in
  :mod:`cram_vrb_lab.robots.panda.joints`.

Nothing here publishes odometry or tf: the Panda does not move, and its base
pose is a shared constant rather than something measured, so giskard's
:class:`~cram_vrb_lab.robots.panda.giskard_config.WorldWithPandaConfig` already
knows where the arm stands.

.. warning::
   Import only after :func:`cram_vrb_lab.sim.isaac_app.create_simulation_app`
   has run -- this module imports ``isaacsim.core`` and ``omni`` at module scope.
"""

import tempfile

import numpy as np
import omni.kit.commands
from isaacsim.core.prims import Articulation, XFormPrim
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray

from cram_vrb_lab.scenes.apartment.constants import (
    PANDA_BASE_ORIENTATION_WXYZ,
    PANDA_BASE_POSITION_IN_MAP,
)
from cram_vrb_lab.sim.velocity_integrator import (
    StreamedVelocityIntegrator,
    dof_indices,
)

from .joints import (
    ARM_JOINTS,
    CONTROLLED_JOINTS,
    FINGER_JOINTS,
    GRIPPER_CMD_TOPIC,
    GRIPPER_OPEN_TRAVEL,
    JOINT_STATES_TOPIC,
    PARK_CONFIGURATION,
    VELOCITY_CMD_TOPIC,
    load_patched_urdf,
)

PANDA_PRIM_PATH = "/panda"
"""Where the importer puts the robot: ``/`` plus the URDF's ``<robot name=...>``."""

ARM_DRIVE_STIFFNESS = 1.0e5
"""[N m/rad]. The importer derives drive gains from link inertias, which leaves
the wrist joints soft enough to sag under the hand; these are stiff enough that
a commanded pose is actually held."""

ARM_DRIVE_DAMPING = 1.0e4

FINGER_DRIVE_STIFFNESS = 400.0
FINGER_DRIVE_DAMPING = 40.0
"""Finger drive gains, [N/m] and [N/(m/s)].

Taken from NVIDIA's own Franka setup
(``omni.physxdemos.utils.franka_helpers.get_default_franka_parameters``), which
is what the official Franka USD is built with. Reproducing them is the whole
reason the imported robot behaves like the shipped asset.

A URDF describes geometry, mass and limits; it says **nothing about drives**. So
the importer derives gains from link inertias, and a Franka finger weighs 14 g:
whatever it derives, and anything sized for the arm, leaves the fingers
oscillating like a spring. Sizing the damping analytically does not rescue it
either -- critical damping for a 14 g mass at this stiffness works out around 5,
and at that value the fingers still ring, because the drive is fighting the
articulation's effective inertia rather than the bare link's.

Together with ``StreamedVelocityIntegrator.MAX_LEAD`` the stiffness also sets
the grip force: the target may lead the measured position by 0.02 m, so a
blocked finger pushes with about 8 N -- inside the joint's own 20 N effort
limit, and far above the 0.5 N the cube's weight needs.
"""


def spawn_panda(
    world,
    render,
    position=PANDA_BASE_POSITION_IN_MAP,
    orientation=PANDA_BASE_ORIENTATION_WXYZ,
):
    """Import the Panda into the open stage and return its Articulation.

    The base is placed on the *prim*, before physics ever runs, and matches the
    posed ``map -> panda_link0`` connection
    :class:`~cram_vrb_lab.robots.panda.giskard_config.WorldWithPandaConfig`
    builds -- both read the same constants, so the arm giskard plans for stands
    where Isaac renders it.

    :param orientation: quaternion in Isaac's ``(w, x, y, z)`` order.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".urdf", prefix="panda_patched_", delete=False
    ) as urdf_file:
        urdf_file.write(load_patched_urdf())
        urdf_path = urdf_file.name

    _, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    import_config.fix_base = True
    import_config.import_inertia_tensor = True
    import_config.distance_scale = 1.0
    # Keep the fixed joints: panda_hand and the tool/fingertip frames the
    # semantic model looks bodies up by would otherwise be merged away, and the
    # twin and the render would no longer describe the same link tree.
    import_config.merge_fixed_joints = False
    # The collision meshes that ship with the description are already convex
    # hulls per link, so nothing needs decomposing.
    import_config.convex_decomp = False
    # A finger's collision hull overlaps the hand it is mounted on, so with
    # self-collision enabled the solver spends every step pushing the two apart
    # and the fingers sit in a spring-like tremor. Nothing here needs the robot
    # to avoid itself: giskard plans the motions, and the demos' collision
    # avoidance runs against the environment.
    import_config.self_collision = False

    articulation_root = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=urdf_path,
        import_config=import_config,
        get_articulation_root=True,
    )[1]
    print(f"Panda imported from {urdf_path} to {articulation_root}")

    # Placed on the prim rather than through the physics view: the base is fixed
    # to the world where the prim stands when physics starts, and that is also
    # the pose a later world.reset() restores.
    XFormPrim(PANDA_PRIM_PATH).set_world_poses(
        np.array([position], dtype=float), np.array([orientation], dtype=float)
    )
    print(f"Panda placed at {tuple(round(v, 4) for v in position)} "
          f"quat(wxyz) {orientation}")

    # Reset before wrapping: the freshly imported prims are not in the physics
    # scene yet, and Articulation reads its link metadata in its constructor --
    # against an unregistered articulation that metadata is None.
    world.reset()
    for _ in range(5):
        world.step(render=render)

    panda = Articulation(prim_paths_expr=articulation_root, name="panda")
    world.reset()
    for _ in range(10):
        world.step(render=render)

    print("Panda imported and wrapped; call move_to_park once the scene is built.")
    return panda


def move_to_park(panda, world, render):
    """Set the drive gains and put the arm in its park pose with the hand open.

    .. warning::
       Call this **last**, after everything else in the scene has been spawned.
       ``world.reset()`` re-initializes the physics view: it restores both the
       state physics started from (every joint at zero, which for a Panda means
       folded flat with the wrist on a limit) *and* the drive parameters
       authored on the prims. Anything that spawns a body resets --
       ``spawn_props`` does -- so gains and poses set before it are silently
       thrown away, and the robot runs on the importer's derived gains while
       looking like it is running on yours.
    """
    arm_dof = dof_indices(panda, ARM_JOINTS)
    finger_dof = dof_indices(panda, FINGER_JOINTS)

    panda.set_gains(
        kps=np.full((1, len(arm_dof)), ARM_DRIVE_STIFFNESS),
        kds=np.full((1, len(arm_dof)), ARM_DRIVE_DAMPING),
        joint_indices=arm_dof,
    )
    panda.set_gains(
        kps=np.full((1, len(finger_dof)), FINGER_DRIVE_STIFFNESS),
        kds=np.full((1, len(finger_dof)), FINGER_DRIVE_DAMPING),
        joint_indices=finger_dof,
    )

    positions = panda.get_joint_positions()
    positions[0, arm_dof] = PARK_CONFIGURATION
    positions[0, finger_dof] = GRIPPER_OPEN_TRAVEL
    panda.set_joint_positions(positions)
    panda.set_joint_position_targets(positions)
    for _ in range(30):
        world.step(render=render)

    print(f"Panda parked at {PARK_CONFIGURATION}")


class PandaROS(Node):
    """ROS 2 bridge for the simulated Panda.

    Publishes joint states for giskard to close its loop on, and accepts the
    streamed joint velocities it sends back plus a direct gripper command.
    """

    def __init__(self, robot):
        super().__init__("panda_ros")
        self.robot = robot
        self.finger_dof = dof_indices(robot, FINGER_JOINTS)
        self.integrator = StreamedVelocityIntegrator(
            robot, CONTROLLED_JOINTS, holding_joints=FINGER_JOINTS
        )

        # Queue depth 1 on the streamed command: only the LATEST velocity
        # matters, and a backlog makes the sim execute commands that are
        # hundreds of milliseconds old.
        self.create_subscription(
            Float64MultiArray, VELOCITY_CMD_TOPIC, self.joint_velocity_cmd_cb, 1
        )
        self.create_subscription(Float64, GRIPPER_CMD_TOPIC, self.gripper_cmd_cb, 10)
        self.pub_joint_states = self.create_publisher(
            JointState, JOINT_STATES_TOPIC, 10
        )

    def joint_velocity_cmd_cb(self, msg):
        if not self.integrator.accept(msg.data):
            self.get_logger().warning(
                f"{VELOCITY_CMD_TOPIC} carried {len(msg.data)} values, expected "
                f"{len(CONTROLLED_JOINTS)}; dropping",
                throttle_duration_sec=5.0,
            )

    def gripper_cmd_cb(self, msg):
        """Command both fingers to a travel [m] directly, bypassing giskard."""
        targets = self.robot.get_joint_positions()[0].copy()
        for index in self.finger_dof:
            targets[index] = float(msg.data)
        self.robot.set_joint_position_targets([targets])

    def integrate_joint_velocities(self, dt):
        self.integrator.step(dt)

    def publish_joint_states(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.robot.dof_names)
        msg.position = self.robot.get_joint_positions()[0].tolist()
        self.pub_joint_states.publish(msg)
