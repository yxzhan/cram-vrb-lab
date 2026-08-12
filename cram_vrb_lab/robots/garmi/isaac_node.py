"""GARMI in Isaac Sim: URDF import, drive tuning, and the ROS 2 bridge node.

Mirrors :mod:`cram_vrb_lab.robots.panda.isaac_node` -- the arms are FR3s and the
hands are Franka Hands, so the drive tuning is the Panda's -- with the base
frozen in the patched URDF and both arms controlled.

.. warning::
   Import only after :func:`cram_vrb_lab.sim.isaac_app.create_simulation_app`
   has run -- this module imports ``isaacsim.core`` and ``omni`` at module scope.
"""

import tempfile

import numpy as np
import omni.kit.commands
from isaacsim.core.prims import Articulation, XFormPrim
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray

from cram_vrb_lab.sim.ros_utils import SimBridge
from cram_vrb_lab.sim.velocity_integrator import (
    StreamedVelocityIntegrator,
    dof_indices,
)

from .joints import (
    ARM_JOINTS,
    CONTROLLED_JOINTS,
    FINGER_JOINTS,
    FINGER_MASS,
    GRIPPER_CMD_TOPIC,
    GRIPPER_OPEN_TRAVEL,
    JOINT_STATES_TOPIC,
    PARK_CONFIGURATION,
    ROBOT_NAME,
    SIDES,
    VELOCITY_CMD_TOPIC,
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
radius. The base is imported fixed, so nothing settles this by itself: a demo
that spawns GARMI at z = 0 sinks it into the floor by this much.
"""

ARM_DRIVE_STIFFNESS = 1.0e5
"""[N m/rad]. The importer derives drive gains from link inertias, which leaves
the wrist joints soft enough to sag under the hand."""

ARM_DRIVE_DAMPING = 1.0e4

FINGER_DRIVE_STIFFNESS = 400.0
FINGER_DRIVE_DAMPING = 40.0
"""Finger drive gains as a **force** drive would take them, [N/m] and [N/(m/s)].

NVIDIA's own Franka numbers. Divided by
:data:`~cram_vrb_lab.robots.garmi.joints.FINGER_MASS` before they are handed to
``set_gains``, because the URDF importer authors *acceleration* drives -- see the
Panda's ``FINGER_MASS`` for the full account of what that costs when it is
missed.
"""


def spawn_garmi(
    world,
    render,
    position=(0.0, 0.0, 0.0),
    orientation=(1.0, 0.0, 0.0, 0.0),
):
    """Import GARMI into the open stage and return its Articulation.

    The base is placed on the *prim*, before physics ever runs, and has to match
    the posed ``map -> base_link`` connection
    :class:`~cram_vrb_lab.robots.garmi.giskard_config.WorldWithGarmiConfig`
    builds. Both sides are given the same
    :class:`~cram_vrb_lab.specs.SpawnPose` by the demo.

    :param orientation: quaternion in Isaac's ``(w, x, y, z)`` order.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".urdf", prefix="garmi_patched_", delete=False
    ) as urdf_file:
        urdf_file.write(load_patched_urdf())
        urdf_path = urdf_file.name

    _, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    import_config.fix_base = True
    import_config.import_inertia_tensor = True
    import_config.distance_scale = 1.0
    # Keep the fixed joints: the hands, the TCP frames and the fingertip frames
    # the semantic model looks bodies up by would otherwise be merged away, and
    # the twin and the render would no longer describe the same link tree.
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    # The fingers' collision hulls overlap the hand they are mounted on, and the
    # two arms are mounted close together on the torso; with self-collision on,
    # the solver spends every step pushing overlapping hulls apart. Giskard plans
    # the motions and the demos avoid the environment, not the robot itself.
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

    # Placed on the prim rather than through the physics view: the base is fixed
    # to the world where the prim stands when physics starts, and that is also
    # the pose a later world.reset() restores.
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


def move_to_park(garmi, world, render):
    """Set the drive gains and put both arms in their park pose, hands open.

    .. warning::
       Call this **last**, after everything else in the scene has been spawned.
       ``world.reset()`` restores both the state physics started from and the
       drive parameters authored on the prims, so gains and poses set before
       anything that resets are silently thrown away.
    """
    arm_dof = dof_indices(garmi, ARM_JOINTS)
    finger_dof = dof_indices(garmi, FINGER_JOINTS)

    garmi.set_gains(
        kps=np.full((1, len(arm_dof)), ARM_DRIVE_STIFFNESS),
        kds=np.full((1, len(arm_dof)), ARM_DRIVE_DAMPING),
        joint_indices=arm_dof,
    )
    garmi.set_gains(
        kps=np.full((1, len(finger_dof)), FINGER_DRIVE_STIFFNESS / FINGER_MASS),
        kds=np.full((1, len(finger_dof)), FINGER_DRIVE_DAMPING / FINGER_MASS),
        joint_indices=finger_dof,
    )

    positions = garmi.get_joint_positions()
    for side in SIDES:
        positions[0, dof_indices(garmi, arm_joints(side))] = PARK_CONFIGURATION
    positions[0, finger_dof] = GRIPPER_OPEN_TRAVEL
    garmi.set_joint_positions(positions)
    garmi.set_joint_position_targets(positions)
    for _ in range(30):
        world.step(render=render)

    print(f"GARMI parked, both arms at {PARK_CONFIGURATION}")


class GarmiROS(SimBridge):
    """ROS 2 bridge for the simulated GARMI.

    Publishes joint states for giskard to close its loop on, and accepts the
    streamed joint velocities it sends back plus a direct gripper command.
    """

    def __init__(self, robot):
        super().__init__("garmi_ros")
        self.robot = robot
        self.integrator = StreamedVelocityIntegrator(
            robot, CONTROLLED_JOINTS, holding_joints=FINGER_JOINTS
        )

        # Queue depth 1 on the streamed command: only the LATEST velocity
        # matters, and a backlog makes the sim execute stale commands.
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
        """Command every finger to a travel [m] directly, bypassing giskard.

        Through the integrator rather than straight onto the drives: it owns the
        targets of every controlled joint, so a direct write would be overwritten
        within one sim step.
        """
        self.integrator.hold_at(FINGER_JOINTS, msg.data)

    def integrate_joint_velocities(self, dt):
        self.integrator.step(dt)

    def apply_commands(self, dt):
        self.integrate_joint_velocities(dt)

    def publish(self):
        self.publish_joint_states()

    def publish_joint_states(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.robot.dof_names)
        msg.position = self.robot.get_joint_positions()[0].tolist()
        self.pub_joint_states.publish(msg)
