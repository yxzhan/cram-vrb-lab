"""Put camera detections into the digital twin.

robokudo reports each cluster in the camera's optical frame. This module moves them
into ``map`` and adds one body per detection to the shared
``semantic_digital_twin`` world, so the twin ends up knowing about objects it has no
description for -- which is the point of the perception demo.

The transform out of the camera frame is taken from the twin's own forward
kinematics rather than from tf. There is no ``tf2_ros`` listener anywhere in this
repo, and giskard stops publishing the fixed camera frames while it is executing a
goal (which is exactly why ``StretchROS.publish_camera_static_tf`` latches them on
the sim side). ``world.compute_forward_kinematics_np`` reads the same joint state
giskard is controlling and cannot go stale between the look and the lookup.
"""

from __future__ import annotations

import numpy as np
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Color, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

from cram_vrb_lab.robots.stretch.joints import CAMERA_FRAME_ID

DETECTION_PREFIX = "perceived"
"""Name prefix every detected body gets, so they can be told apart from the modelled
apartment and cleared as a group."""

DETECTION_COLOR = (0.1, 0.7, 0.9, 0.6)
"""RGBA. Translucent and unlike anything in the apartment, so in RViz it is obvious
which boxes came from perception rather than from the URDF."""


def camera_pose_in_map(world):
    """``map_T_camera`` as a 4x4 numpy array, from the twin's forward kinematics."""
    camera_body = world.get_body_by_name(CAMERA_FRAME_ID)
    return np.asarray(world.compute_forward_kinematics_np(world.root, camera_body))


def detection_pose_in_map(world, detection):
    """Move one :class:`~cram_vrb_lab.perception.pipeline.Detection` into ``map``.

    robokudo's quaternions are ``(x, y, z, w)``
    (``robokudo/types/tf.py`` ``Pose.rotation``), which is also the order
    :meth:`HomogeneousTransformationMatrix.from_xyz_quaternion` takes, so the
    components pass straight through.
    """
    camera_T_object = np.asarray(
        HomogeneousTransformationMatrix.from_xyz_quaternion(
            *detection.position, *detection.orientation
        ).to_np()
    )
    return camera_pose_in_map(world) @ camera_T_object


def clear_detections(world):
    """Remove every previously perceived body; returns how many went.

    Makes the notebook's detect cell re-runnable: looking twice should replace what
    was seen, not pile up a second set of boxes.
    """
    stale = [
        body for body in world.bodies if body.name.prefix == DETECTION_PREFIX
    ]
    if not stale:
        return 0
    with world.modify_world():
        for body in stale:
            if body.parent_connection is not None:
                world.remove_connection(body.parent_connection)
            world.remove_kinematic_structure_entity(body)
    return len(stale)


def add_detections(world, detections, clear_previous=True):
    """Add one body per detection to ``world``; return the bodies created.

    Each is a box the size of the detected bounding box, fixed to the world root at
    the detected pose. The world is broadcast to the giskard server over
    ``/world_sync`` by the notebook's ``WorldSynchronizer``, so these appear in
    giskard's collision world too -- the robot can then plan around what it saw.

    Nothing here claims to know *what* was detected. The pipeline is geometric; a
    class label would need a detector this venv has no weights for.
    """
    if clear_previous:
        clear_detections(world)

    bodies = []
    with world.modify_world():
        for index, detection in enumerate(detections):
            body = _detection_body(f"{DETECTION_PREFIX}_{index}", detection.extents)
            world.add_kinematic_structure_entity(body)
            world.add_connection(
                FixedConnection(
                    parent=world.root,
                    child=body,
                    parent_T_connection_expression=(
                        HomogeneousTransformationMatrix(
                            data=detection_pose_in_map(world, detection)
                        )
                    ),
                )
            )
            bodies.append(body)
    return bodies


def _detection_body(name, extents):
    """A box body of the given side lengths, centred on its own origin.

    Collision and visual get separate ``Box`` objects: :class:`Body` transforms the
    shapes of each collection into its own frame in place, so sharing one instance
    between the two collections would apply the transform twice.
    """

    def box():
        return Box(
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(),
            scale=Scale(*(float(value) for value in extents)),
            color=Color(*DETECTION_COLOR),
        )

    return Body(
        name=PrefixedName(name, prefix=DETECTION_PREFIX),
        collision=ShapeCollection([box()]),
        visual=ShapeCollection([box()]),
    )
