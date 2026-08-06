"""Pick-and-place props on the Isaac Sim side: the physics the gripper has to beat.

:func:`spawn_props` builds the cube as a rigid body with friction and mass, so
nothing about the grasp is faked -- whether the cube comes off its surface is
decided by contact forces between the robot's fingers and the cube.

:class:`PropsROS` publishes the cube's true pose. The digital twin tracks where
CRAM *believes* the cube is; after a grasp attempt the belief and the physics can
disagree (the fingers closed on air, the cube slipped mid-carry), and that
disagreement is the measurement this demo is after.

.. warning::
   Import only after :func:`cram_vrb_lab.sim.isaac_app.create_simulation_app`
   has run -- this module imports ``isaacsim.core`` at module scope.
"""

import numpy as np
from geometry_msgs.msg import PoseStamped
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.prims import define_prim
from tf2_ros import TransformBroadcaster

from cram_vrb_lab.sim.ros_utils import SimBridge, make_tf

from .constants import (
    APARTMENT_LAYOUT,
    CUBE_COLOR,
    CUBE_FRICTION,
    CUBE_GROUND_TRUTH_FRAME,
    CUBE_MASS,
    CUBE_POSE_TOPIC,
    CUBE_SIZE,
    PROPS_FRAME_ID,
    PropLayout,
)

PROPS_ROOT = "/World/Props"


def spawn_props(world, render, layout: PropLayout = APARTMENT_LAYOUT):
    """Spawn the graspable cube and return its prim.

    The cube is released just above the surface the scene provides and falls onto
    it, then settled for enough physics steps to start the demo at rest rather
    than mid-bounce. The pose it actually settled at is printed: the surface
    belongs to the scene, so that is the only way to learn its real height.
    """
    define_prim(PROPS_ROOT, "Xform")

    # Isaac averages the two materials in a contact, so the cube alone cannot set
    # the friction of the finger/cube pair -- it can only pull the average up.
    # Restitution 0 keeps the cube from bouncing when it lands, and from hopping
    # away when a finger brushes it on approach.
    cube_material = PhysicsMaterial(
        prim_path=f"{PROPS_ROOT}/pick_cube_material",
        static_friction=CUBE_FRICTION,
        dynamic_friction=CUBE_FRICTION,
        restitution=0.0,
    )
    cube = DynamicCuboid(
        prim_path=f"{PROPS_ROOT}/pick_cube",
        name="pick_cube",
        position=np.array(layout.cube_start_position),
        scale=np.array([CUBE_SIZE] * 3),
        color=np.array(CUBE_COLOR),
        mass=CUBE_MASS,
        physics_material=cube_material,
    )

    world.reset()
    # Long enough for a dropped cube to land and stop bouncing, not just for one
    # that was spawned already resting.
    for _ in range(120):
        world.step(render=render)

    print(f"Props ready: cube spawned at {layout.cube_start_position}, "
          f"settled at {tuple(round(float(v), 4) for v in cube.get_world_pose()[0])} "
          f"(expected surface z={layout.surface_z}).")
    return cube


class PropsROS(SimBridge):
    """Publishes the cube's ground-truth pose -- a stand-in for perception.

    Both a :class:`geometry_msgs.msg.PoseStamped` on
    :data:`~cram_vrb_lab.scenes.props.constants.CUBE_POSE_TOPIC` (for the twin to
    read back) and a tf frame (so RViz can show the true cube next to the twin's
    believed one).
    """

    receives_commands = False  # nothing to subscribe to: the props only report

    def __init__(self, cube):
        super().__init__("props_ros")
        self.cube = cube
        self.pub_cube_pose = self.create_publisher(PoseStamped, CUBE_POSE_TOPIC, 10)
        self.tf_broadcaster = TransformBroadcaster(self)

    def publish(self):
        self.publish_props()

    def publish_props(self):
        position, orientation = self.cube.get_world_pose()  # quat is (w, x, y, z)
        stamp = self.get_clock().now().to_msg()

        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = PROPS_FRAME_ID
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])
        msg.pose.orientation.w = float(orientation[0])
        msg.pose.orientation.x = float(orientation[1])
        msg.pose.orientation.y = float(orientation[2])
        msg.pose.orientation.z = float(orientation[3])
        self.pub_cube_pose.publish(msg)

        # make_tf takes the ROS (x, y, z, w) order.
        self.tf_broadcaster.sendTransform([
            make_tf(stamp, PROPS_FRAME_ID, CUBE_GROUND_TRUTH_FRAME, position,
                    (orientation[1], orientation[2], orientation[3], orientation[0]))
        ])
