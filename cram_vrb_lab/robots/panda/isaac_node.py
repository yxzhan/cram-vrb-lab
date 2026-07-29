"""Panda in Isaac Sim: URDF import, drive tuning, and the ROS 2 bridge node.

- :func:`spawn_panda` imports the patched URDF straight into the open stage, so
  no converted USD has to be checked in, tunes the drives and puts the arm in
  its park pose.
- :class:`PandaROS` publishes joint states and consumes streamed joint
  velocities and gripper commands on the topics in
  :mod:`cram_vrb_lab.robots.panda.joints`.

Nothing here publishes odometry or tf: the Panda is bolted to the world origin,
which is exactly what giskard's :class:`~cram_vrb_lab.robots.panda.giskard_config.WorldWithPandaConfig`
assumes, so ``map`` and the robot base are the same frame by construction.

.. warning::
   Import only after :func:`cram_vrb_lab.sim.isaac_app.create_simulation_app`
   has run -- this module imports ``isaacsim.core`` and ``omni`` at module scope.
"""

import tempfile

import numpy as np
import omni.kit.commands
from isaacsim.core.prims import Articulation
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray

from cram_vrb_lab.sim.velocity_integrator import StreamedVelocityIntegrator

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

FINGER_DRIVE_STIFFNESS = 1.0e4
"""[N/m]. Together with ``StreamedVelocityIntegrator.MAX_LEAD`` this sets the
grip force: the target may lead the measured position by 0.02 m, so a closing
finger pushes with up to ~200 N -- ample for the 50 g cube, and the finger stops
on contact rather than crushing through."""

FINGER_DRIVE_DAMPING = 1.0e3


def spawn_panda(world, render):
    """Import the Panda into the open stage and return its Articulation.

    The robot lands at the world origin with its base fixed, matching the
    identity ``map -> panda_link0`` connection the giskard world config builds.
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

    articulation_root = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=urdf_path,
        import_config=import_config,
        get_articulation_root=True,
    )[1]
    print(f"Panda imported from {urdf_path} to {articulation_root}")

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

    arm_dof = np.array([panda.get_dof_index(name) for name in ARM_JOINTS])
    panda.set_gains(
        kps=np.full((1, len(arm_dof)), ARM_DRIVE_STIFFNESS),
        kds=np.full((1, len(arm_dof)), ARM_DRIVE_DAMPING),
        joint_indices=arm_dof,
    )
    finger_dof = np.array([panda.get_dof_index(name) for name in FINGER_JOINTS])
    panda.set_gains(
        kps=np.full((1, len(finger_dof)), FINGER_DRIVE_STIFFNESS),
        kds=np.full((1, len(finger_dof)), FINGER_DRIVE_DAMPING),
        joint_indices=finger_dof,
    )

    print("Panda ready")
    return panda


def move_to_park(panda, world, render):
    """Put the arm in its park pose with the hand open.

    .. warning::
       Call this **after** everything else in the scene has been spawned. A
       zero-joint Panda stands folded flat with its wrist on a limit, so the
       pose has to be set -- but ``world.reset()`` restores the state physics
       started from, and anything else that spawns a body (``spawn_props``)
       resets too. Setting the pose earlier would simply be undone.
    """
    arm_dof = np.array([panda.get_dof_index(name) for name in ARM_JOINTS])
    finger_dof = np.array([panda.get_dof_index(name) for name in FINGER_JOINTS])

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
        self.finger_dof = np.array(
            [robot.get_dof_index(name) for name in FINGER_JOINTS]
        )
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
