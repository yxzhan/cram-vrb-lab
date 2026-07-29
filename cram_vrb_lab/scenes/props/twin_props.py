"""Pick-and-place props on the digital-twin side.

:func:`add_props_to_twin` puts the same cube and pedestals CRAM plans against
into the shared ``semantic_digital_twin`` world, built from the constants the
Isaac side uses, so the twin and the render start out identical.

:func:`sync_cube_from_sim` pulls the cube's *actual* pose out of the sim and
writes it back into the twin. That is the only place the twin learns the grasp
did not go as planned -- CRAM's ``AttachNode`` moves the cube with the gripper in
the twin whether or not the physical fingers ever caught it.
"""

import time

from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Color, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

from .constants import (
    CUBE_BODY_NAME,
    CUBE_COLOR,
    CUBE_POSE_TOPIC,
    CUBE_SIZE,
    CUBE_START_POSITION,
    PEDESTAL_COLOR,
    PEDESTAL_SIZE,
    PICK_PEDESTAL_BODY_NAME,
    PICK_PEDESTAL_POSITION,
    PLACE_PEDESTAL_BODY_NAME,
    PLACE_PEDESTAL_POSITION,
)


def _box_body(name, scale, color):
    """A single-box body whose origin is the box centre.

    Collision and visual get their own ``Box``: :class:`Body` transforms the
    shapes of each collection into its own frame in place, so the two collections
    must not share one shape object.
    """
    def box():
        return Box(
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(),
            scale=Scale(*scale),
            color=Color(*color, 1.0),
        )

    return Body(
        name=PrefixedName(name, prefix="props"),
        collision=ShapeCollection([box()]),
        visual=ShapeCollection([box()]),
    )


def add_props_to_twin(world):
    """Add the cube and the two pedestals to ``world``; return the cube body.

    Idempotent: if the cube is already there (a re-run of the notebook cell
    against a still-running server) the existing body is returned untouched.

    The bodies are attached to the world root with fixed connections. CRAM's
    ``AttachNode`` replaces whatever connection a grasped body has, so nothing is
    gained by making the cube's connection movable up front; poses are updated by
    swapping the connection instead (:func:`set_cube_pose`).
    """
    existing = [b for b in world.bodies if b.name.name == CUBE_BODY_NAME]
    if existing:
        return existing[0]

    cube = _box_body(CUBE_BODY_NAME, (CUBE_SIZE,) * 3, CUBE_COLOR)
    pedestals = [
        (_box_body(PICK_PEDESTAL_BODY_NAME, PEDESTAL_SIZE, PEDESTAL_COLOR),
         (*PICK_PEDESTAL_POSITION, PEDESTAL_SIZE[2] / 2)),
        (_box_body(PLACE_PEDESTAL_BODY_NAME, PEDESTAL_SIZE, PEDESTAL_COLOR),
         (*PLACE_PEDESTAL_POSITION, PEDESTAL_SIZE[2] / 2)),
    ]

    with world.modify_world():
        for body, position in [(cube, CUBE_START_POSITION)] + pedestals:
            world.add_kinematic_structure_entity(body)
            world.add_connection(FixedConnection(
                parent=world.root,
                child=body,
                parent_T_connection_expression=(
                    HomogeneousTransformationMatrix.from_xyz_rpy(*position)
                ),
            ))
    return cube


def set_cube_pose(world, cube, root_T_cube):
    """Move ``cube`` to ``root_T_cube`` (a transform in the world root frame).

    Re-attaches the body to the world root with a new fixed connection, the same
    way CRAM's ``ModelChangeExecutable`` re-parents a grasped object -- which also
    means this detaches the cube from the gripper if it is currently held.
    """
    with world.modify_world():
        world.remove_connection(cube.parent_connection)
        world.add_connection(FixedConnection(
            parent=world.root,
            child=cube,
            parent_T_connection_expression=root_T_cube,
        ))


class CubePoseSensor:
    """The cube's ground-truth pose from the sim -- this demo's stand-in for
    perception.

    Subscribes once and holds the newest message. The sim publishes every step,
    so the cached pose is at most one sim tick old; subscribing per read would
    instead race the executor, which only picks a new subscription up on its next
    wait-set rebuild.

    :param node: a node that is already being spun by an executor.
    """

    def __init__(self, node: Node):
        self._latest = None
        self._subscription = node.create_subscription(
            PoseStamped, CUBE_POSE_TOPIC, self._on_pose, 1
        )

    def _on_pose(self, message: PoseStamped):
        self._latest = message

    def pose(self, timeout: float = 10.0) -> PoseStamped:
        """The newest pose, waiting for the first message if none has arrived.

        :raises TimeoutError: if the sim is not publishing -- it was started
            without ``--props``, or its topics are not visible from this ROS
            domain.
        """
        deadline = time.time() + timeout
        while self._latest is None and time.time() < deadline:
            time.sleep(0.05)
        if self._latest is None:
            raise TimeoutError(
                f"nothing published on {CUBE_POSE_TOPIC} within {timeout}s -- is "
                f"the sim running with --props?"
            )
        return self._latest

    def position(self, timeout: float = 10.0):
        """The cube's true centre as ``(x, y, z)``."""
        p = self.pose(timeout).pose.position
        return (p.x, p.y, p.z)


def sync_cube_from_sim(world, cube, sensor: CubePoseSensor, timeout: float = 10.0):
    """Snap the twin's cube onto the pose Isaac's physics actually has it at.

    Use it before a grasp (the cube may have settled or been nudged since it was
    spawned) and after one (to see whether the gripper really took it along).
    Returns the position that was written, as ``(x, y, z)``.
    """
    message = sensor.pose(timeout)
    position = message.pose.position
    orientation = message.pose.orientation
    root_T_cube = HomogeneousTransformationMatrix.from_xyz_quaternion(
        position.x, position.y, position.z,
        orientation.x, orientation.y, orientation.z, orientation.w,
    )
    set_cube_pose(world, cube, root_T_cube)
    return (position.x, position.y, position.z)
