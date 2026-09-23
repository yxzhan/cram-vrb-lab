"""The Isaac half of :mod:`cram_vrb_lab.sim.scene_joints`: measure the scene's joints.

.. warning::
   Import only after :func:`cram_vrb_lab.sim.isaac_app.create_simulation_app` has
   run -- this module imports ``isaacsim.core`` at module scope.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np
from isaacsim.core.prims import RigidPrim
from pxr import Usd, UsdGeom, UsdPhysics
from sensor_msgs.msg import JointState

from cram_vrb_lab.sim.numpy_bridge import numpy_view
from cram_vrb_lab.sim.ros_utils import SimBridge, qconj, qmul, qrot
from cram_vrb_lab.sim.scene_joints import SCENE_JOINT_STATES_TOPIC, SceneJoint

AXES = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}


class _Joint:
    """One USD joint, resolved: its two bodies and its frames on each."""

    def __init__(self, spec: SceneJoint, joint: UsdPhysics.Joint, cache):
        self.spec = spec
        self.revolute = joint.GetPrim().IsA(UsdPhysics.RevoluteJoint)
        typed = (UsdPhysics.RevoluteJoint if self.revolute else UsdPhysics.PrismaticJoint)(
            joint.GetPrim()
        )
        self.axis = np.array(AXES[typed.GetAxisAttr().Get()])
        self.bodies = [str(joint.GetBody0Rel().GetTargets()[0]),
                       str(joint.GetBody1Rel().GetTargets()[0])]
        stage = joint.GetPrim().GetStage()
        # The joint frame on each body, as (position, x-y-z-w rotation) in the body
        # frame. The position is authored in the body's *scaled* space -- the room
        # doors carry a 1.09 y scale -- while the pose physics reports is unscaled,
        # so the scale is folded in here, once: physics never changes it.
        self.frames = []
        for body, position, rotation in (
            (self.bodies[0], joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get()),
            (self.bodies[1], joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get()),
        ):
            world = np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(body))).T
            scale = np.linalg.norm(world[:3, :3], axis=0)
            imaginary = rotation.GetImaginary()
            self.frames.append((
                scale * np.array(position, dtype=float),
                np.array([*imaginary, rotation.GetReal()], dtype=float),
            ))

    def value(self, poses) -> float:
        """The joint coordinate, from the two bodies' world poses.

        Measured the way PhysX defines it -- body1's joint frame relative to
        body0's -- so it is the number the joint's own limits are written in, and
        converted into the twin's units and sign.

        :param poses: ``{body path: (position, x-y-z-w quaternion)}``.
        """
        frames = []
        for body, (local_position, local_rotation) in zip(self.bodies, self.frames):
            position, rotation = poses[body]
            frames.append((position + qrot(rotation, local_position),
                           qmul(rotation, local_rotation)))
        (p0, q0), (p1, q1) = frames
        if self.revolute:
            relative = qmul(qconj(q0), q1)
            # The twist about the joint axis, which is the whole rotation for a joint
            # that has no other freedom -- and the part that means something for one
            # PhysX lets flex a little under load.
            coordinate = 2.0 * math.atan2(float(np.dot(relative[:3], self.axis)),
                                          float(relative[3]))
            coordinate = math.atan2(math.sin(coordinate), math.cos(coordinate))
        else:
            coordinate = float(np.dot(qrot(qconj(q0), p1 - p0), self.axis))
        return self.spec.sign * coordinate


class SceneJointsROS(SimBridge):
    """Publishes :data:`~cram_vrb_lab.sim.scene_joints.SCENE_JOINT_STATES_TOPIC`
    every cycle.

    Computed from body poses rather than read off an articulation, because under
    PhysX there is none: the doors and the cabinet are loose joints between a
    kinematic frame and the dynamic leaves (see
    :mod:`cram_vrb_lab.sim.newton_scene`, which gives them a root only for Newton).
    Body poses are what both engines answer, so this works on either.
    """

    receives_commands = False

    def __init__(self, world, root: str, joints: Sequence[SceneJoint]):
        """
        :param root: the prim the scene was loaded under; the joint paths in
            ``joints`` are relative to it.
        """
        super().__init__("scene_joints_ros")
        self.world = world
        self._publisher = self.create_publisher(JointState, SCENE_JOINT_STATES_TOPIC, 10)
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        self.joints: List[_Joint] = []
        missing = []
        for spec in joints:
            prim = world.stage.GetPrimAtPath(f"{root}/{spec.usd_joint}")
            if not prim.IsValid() or not prim.IsA(UsdPhysics.Joint):
                missing.append(spec.usd_joint)
                continue
            self.joints.append(_Joint(spec, UsdPhysics.Joint(prim), cache))
        self._paths = sorted({body for joint in self.joints for body in joint.bodies})
        self._view: Optional[RigidPrim] = None
        self._names = [joint.spec.connection for joint in self.joints]
        print(
            f"[scene joints] publishing {len(self.joints)} joint(s) on "
            f"{SCENE_JOINT_STATES_TOPIC}"
            + (f"; not in the stage: {missing}" if missing else ""),
            flush=True,
        )

    def _poses(self):
        """World poses of every body a joint hangs off, from a cached view.

        Cached for the reason :meth:`SceneSyncROS._release_view` gives, and rebuilt
        when its physics handle has gone: the sync bridge restarts physics to delete
        a prim, and a view from before the restart no longer answers.
        """
        valid = getattr(self._view, "is_physics_handle_valid", None)
        if self._view is None or (callable(valid) and not valid()):
            view = numpy_view(
                RigidPrim(self._paths, name="scene_joints", reset_xform_properties=False)
            )
            # else get_world_poses silently falls back to the stale stage
            view.initialize()
            self._view = view
        positions, orientations = self._view.get_world_poses()
        return {
            path: (
                np.asarray(positions[index], dtype=float),
                # Isaac's (w, x, y, z) to the (x, y, z, w) the helpers take
                np.asarray(orientations[index], dtype=float)[[1, 2, 3, 0]],
            )
            for index, path in enumerate(self._paths)
        }

    def publish(self) -> None:
        if not self.joints:
            return
        try:
            poses = self._poses()
            positions = [joint.value(poses) for joint in self.joints]
        except Exception as failure:  # noqa: BLE001 - a bad read must not end the sim
            # Dropped, not fatal: the next cycle builds a new view. This runs on the
            # sim loop thread, where an exception ends the simulator.
            self._view = None
            self.get_logger().error(f"scene joint read failed: {failure}")
            return
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = self._names
        message.position = positions
        self._publisher.publish(message)
