"""Pick-and-place props on the Isaac Sim side: the physics the gripper has to beat.

:func:`spawn_props` builds the two pedestals as static colliders and the cube as
a rigid body with friction and mass, so nothing about the grasp is faked --
whether the cube comes off the pedestal is decided by contact forces between the
Stretch's fingers and the cube.

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
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.utils.prims import define_prim
from rclpy.node import Node
from tf2_ros import TransformBroadcaster

from cram_vrb_lab.sim.ros_utils import make_tf

from .constants import (
    CUBE_COLOR,
    CUBE_FRICTION,
    CUBE_GROUND_TRUTH_FRAME,
    CUBE_MASS,
    CUBE_POSE_TOPIC,
    CUBE_SIZE,
    CUBE_START_POSITION,
    PEDESTAL_COLOR,
    PEDESTAL_SIZE,
    PICK_PEDESTAL_POSITION,
    PLACE_PEDESTAL_POSITION,
    PROPS_FRAME_ID,
)

PROPS_ROOT = "/World/Props"


def _pedestal(name, xy):
    """One static pedestal, centred on ``xy``, standing on the floor."""
    return FixedCuboid(
        prim_path=f"{PROPS_ROOT}/{name}",
        name=name,
        position=np.array([xy[0], xy[1], PEDESTAL_SIZE[2] / 2]),
        scale=np.array(PEDESTAL_SIZE),
        color=np.array(PEDESTAL_COLOR),
    )


def spawn_props(world, render):
    """Spawn the two pedestals and the graspable cube; return the cube prim.

    The cube is dropped in resting exactly on the pick pedestal's top face and
    then settled for a few physics steps, so it starts the demo at rest rather
    than mid-bounce.
    """
    define_prim(PROPS_ROOT, "Xform")

    _pedestal("pick_pedestal", PICK_PEDESTAL_POSITION)
    _pedestal("place_pedestal", PLACE_PEDESTAL_POSITION)

    # Isaac averages the two materials in a contact, so the cube alone cannot set
    # the friction of the finger/cube pair -- it can only pull the average up.
    # Restitution 0 keeps the cube from hopping off the pedestal when a finger
    # brushes it on approach.
    cube_material = PhysicsMaterial(
        prim_path=f"{PROPS_ROOT}/pick_cube_material",
        static_friction=CUBE_FRICTION,
        dynamic_friction=CUBE_FRICTION,
        restitution=0.0,
    )
    cube = DynamicCuboid(
        prim_path=f"{PROPS_ROOT}/pick_cube",
        name="pick_cube",
        position=np.array(CUBE_START_POSITION),
        scale=np.array([CUBE_SIZE] * 3),
        color=np.array(CUBE_COLOR),
        mass=CUBE_MASS,
        physics_material=cube_material,
    )

    world.reset()
    for _ in range(30):  # let the cube settle onto the pedestal
        world.step(render=render)

    print(f"Props ready: cube at {CUBE_START_POSITION}, pedestals at "
          f"{PICK_PEDESTAL_POSITION} and {PLACE_PEDESTAL_POSITION}.")
    return cube


class PropsROS(Node):
    """Publishes the cube's ground-truth pose -- a stand-in for perception.

    Both a :class:`geometry_msgs.msg.PoseStamped` on
    :data:`~cram_vrb_lab.scenes.props.constants.CUBE_POSE_TOPIC` (for the twin to
    read back) and a tf frame (so RViz can show the true cube next to the twin's
    believed one).
    """

    def __init__(self, cube):
        super().__init__("props_ros")
        self.cube = cube
        self.pub_cube_pose = self.create_publisher(PoseStamped, CUBE_POSE_TOPIC, 10)
        self.tf_broadcaster = TransformBroadcaster(self)

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
