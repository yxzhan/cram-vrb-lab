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


def ensure_camera_body(world, parent_link, translation, quaternion_xyzw,
                       name=CAMERA_FRAME_ID):
    """Make sure ``world`` has a body called ``name`` rigidly attached to
    ``parent_link``; return it. A no-op if one is already there.

    The Stretch needs none of this: its URDF carries the whole RealSense frame
    chain, so :func:`camera_pose_in_map` finds ``camera_color_optical_frame`` in the
    twin the moment the world is fetched. GARMI's URDF has **no camera link at all**
    -- 63 joints, two IMUs, two 2D lidars, and nothing optical -- so a demo that
    mounts a camera on GARMI in Isaac has to tell the twin where that camera sits, or
    every detection is stuck in the camera frame with no way into ``map``.

    Give it the same offset the sim publishes as static tf (for GARMI, the
    ``CAMERA_*`` constants in :mod:`cram_vrb_lab.robots.garmi.joints`) and the two
    descriptions agree by construction.

    :param parent_link: name of the body to hang the camera off, e.g. ``"head"``.
    :param translation: ``(x, y, z)`` [m] in the parent's frame.
    :param quaternion_xyzw: the parent -> optical-frame rotation, ROS order.
    """
    existing = [body for body in world.bodies if body.name.name == name]
    if existing:
        return existing[0]

    parent = world.get_body_by_name(parent_link)
    camera = Body(name=PrefixedName(name))
    with world.modify_world():
        world.add_kinematic_structure_entity(camera)
        world.add_connection(
            FixedConnection(
                parent=parent,
                child=camera,
                parent_T_connection_expression=(
                    HomogeneousTransformationMatrix.from_xyz_quaternion(
                        *translation, *quaternion_xyzw
                    )
                ),
            )
        )
    return camera


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


def add_detections(world, detections, clear_previous=True, origin_offset=(0.0, 0.0, 0.0)):
    """Add one body per detection to ``world``; return the bodies created.

    Each is a box the size of the detected bounding box, fixed to the world root at
    the detected pose. The world is broadcast to the giskard server over
    ``/world_sync`` by the notebook's ``WorldSynchronizer``, so these appear in
    giskard's collision world too -- the robot can then plan around what it saw.

    Nothing here claims to know *what* was detected. The pipeline is geometric; a
    class label would need a detector this venv has no weights for.

    :param origin_offset: ``(x, y, z)`` [m] in the detected box's own frame, moving
        each body's **origin** away from the box's centre -- or a callable
        ``(detection) -> (x, y, z)`` when the offset has to depend on the box, which
        it usually does. The geometry does not move: the shapes are shifted back by
        the same amount, so the box still stands where it was seen.

        A constant only means the same thing to every object if they are all the
        same size, and detections are not: "grip 0.03 m below the top face" is
        ``lambda d: (0, 0, d.extents[2] / 2 - 0.03)``, which is one rule for a
        carton, a bowl and a cup alike, while a fixed +0.06 puts the origin near the
        top of the carton and well above the bowl.

        This is the knob for where a grasp lands. ``PickUpAction`` aims the tool
        centre point at the object designator's *origin* -- ``ReachAction(
        target_pose=self.object_designator.global_pose, ...)`` -- and there is no
        offset parameter anywhere along that path, so the origin is the only place
        the choice can be made. Aiming at the centre is fine for something small
        and wrong for anything tall: a top grasp on a 0.20 m carton would put the
        fingers 0.10 m inside it, and the Franka's are 0.045 m long.

        The box's z axis is world up -- ``ClusterPoseBBAnnotator`` only ever fits a
        rotation about z -- so ``(0, 0, h/2 - d)`` means "grip ``d`` below the top
        face" whichever way the object happens to be turned.

        Note what this also does to placing: ``PlaceAction`` puts the *origin* at
        its ``target_location``, so that target becomes where the grip point should
        end up, not where the box's centre should.
    """
    if clear_previous:
        clear_detections(world)

    resolve_offset = (
        origin_offset if callable(origin_offset) else lambda detection: origin_offset
    )

    bodies = []
    with world.modify_world():
        for index, detection in enumerate(detections):
            offset = np.asarray(resolve_offset(detection), dtype=float)
            box_T_origin = np.eye(4)
            box_T_origin[:3, 3] = offset

            body = _detection_body(
                f"{DETECTION_PREFIX}_{index}", detection.extents, offset
            )
            world.add_kinematic_structure_entity(body)
            world.add_connection(
                FixedConnection(
                    parent=world.root,
                    child=body,
                    parent_T_connection_expression=(
                        HomogeneousTransformationMatrix(
                            data=detection_pose_in_map(world, detection) @ box_T_origin
                        )
                    ),
                )
            )
            bodies.append(body)
    return bodies


def _detection_body(name, extents, origin_offset=(0.0, 0.0, 0.0)):
    """A box body of the given side lengths, ``-origin_offset`` from its own origin.

    The shape is shifted by the negative of what the body frame was shifted by, so
    the two cancel and the box lands where it was detected however the origin was
    moved -- see :func:`add_detections`.

    Collision and visual get separate ``Box`` objects: :class:`Body` transforms the
    shapes of each collection into its own frame in place, so sharing one instance
    between the two collections would apply the transform twice.
    """
    x, y, z = (float(value) for value in origin_offset)

    def box():
        return Box(
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(x=-x, y=-y, z=-z),
            scale=Scale(*(float(value) for value in extents)),
            color=Color(*DETECTION_COLOR),
        )

    return Body(
        name=PrefixedName(name, prefix=DETECTION_PREFIX),
        collision=ShapeCollection([box()]),
        visual=ShapeCollection([box()]),
    )
