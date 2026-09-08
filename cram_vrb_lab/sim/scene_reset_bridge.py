"""The Isaac half of :mod:`cram_vrb_lab.sim.scene_reset`: snapshot, then restore.

.. warning::
   Import only after :func:`cram_vrb_lab.sim.isaac_app.create_simulation_app` has
   run -- this module imports ``isaacsim.core`` at module scope.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from isaacsim.core.prims import RigidPrim
from pxr import Usd, UsdPhysics
from std_srvs.srv import Trigger

from cram_vrb_lab.sim.ros_utils import SimBridge
from cram_vrb_lab.sim.scene_reset import RESET_SERVICE


class SceneResetROS(SimBridge):
    """Restores the scene to how it looked once it was built, on request.

    The snapshot is taken lazily, on the first sim step rather than in
    ``__init__``: the bridge is constructed inside ``runner.build``, and the last
    thing that runs there is the robot's park pose. Waiting one step means the
    snapshot is of a scene that is finished -- props settled, arms parked -- which
    is the state a demo actually starts from and therefore the one worth returning
    to.

    The restore runs **inside the service callback**, which is not the hazard it
    would be anywhere else: :func:`cram_vrb_lab.sim.runner.run` spins these nodes
    itself, from the sim loop thread, between ``world.step`` and ``publish``. So the
    callback is already on the only thread allowed to touch USD and PhysX, and
    already in the gap between two steps.

    It was written the other way first -- queue in the callback, apply in
    :meth:`apply_commands` -- which **deadlocks**: the callback waited on work that
    could only run on the thread it was blocking. Measured as a reset that failed
    after exactly the 20 s the wait allowed.
    """

    receives_commands = True  # the service has to be spun

    def __init__(
        self, world, robot=None, integrator=None, robot_bridge=None, sync_bridge=None
    ):
        """
        :param robot: the robot ``Articulation``, restored through its own API
            rather than by teleporting links -- an articulation's links are bound by
            joints, so writing their poses individually fights the solver.
        :param integrator: the streamed-velocity integrator driving that robot, if
            any. **Load-bearing**: it holds the position targets the drives chase,
            so a reset that leaves them alone puts the robot back and is then
            immediately pulled towards wherever the last plan left it.
        :param robot_bridge: the robot's own ROS node, if it keeps state that
            competes with a teleport. Anything with a ``resync_base`` is told to
            re-read the robot after the restore; a kinematically driven base
            *teleports itself* to a pose the bridge dead-reckons, so without this
            the base is the one thing a reset cannot move. See
            :meth:`~cram_vrb_lab.robots.garmi.isaac_node.GarmiROS.resync_base`.
        :param sync_bridge: the scene-sync bridge, if there is one. Objects a plan
            spawned through it postdate the snapshot, so restoring that snapshot
            says nothing about them -- they would survive a reset as leftovers of a
            run that is supposed to be over. Cleared, not restored.
        """
        super().__init__("scene_reset_ros")
        self.world = world
        self.robot = robot
        self.integrator = integrator
        self.robot_bridge = robot_bridge
        self.sync_bridge = sync_bridge

        self._snapshot: Optional[dict] = None
        self.create_service(Trigger, RESET_SERVICE, self._on_request)

    # %% snapshot

    def _rigid_body_paths(self) -> List[str]:
        """Every dynamic prim in the stage that is not part of the robot.

        Found by walking for ``PhysicsRigidBodyAPI`` rather than from a list of prop
        roots, so a scene that grows new objects -- props, the perceived boxes a
        plan spawns, anything dropped in by hand -- is covered without this module
        being told about it.
        """
        robot_prefix = None
        if self.robot is not None and len(self.robot.prim_paths) > 0:
            robot_prefix = str(self.robot.prim_paths[0]).rsplit("/", 1)[0]

        paths = []
        for prim in Usd.PrimRange(self.world.stage.GetPseudoRoot()):
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            path = str(prim.GetPath())
            if robot_prefix and path.startswith(robot_prefix):
                continue
            paths.append(path)
        return paths

    @staticmethod
    def _is_kinematic(prim) -> bool:
        """Whether PhysX drives this body by pose rather than by force.

        A kinematic body has no velocity to speak of -- it does not respond to
        forces, and PhysX refuses ``setLinearVelocity`` on one outright. Most of
        this apartment is kinematic (walls, cabinets, the furniture that must not
        fall over), so clearing velocities blindly is 255 rejected calls per reset
        and a thousand log lines of ``Body must be non-kinematic!`` -- harmless in
        effect, but enough noise to bury a real error.
        """
        attribute = UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr()
        return bool(attribute and attribute.Get())

    def _take_snapshot(self) -> None:
        paths = self._rigid_body_paths()
        bodies = RigidPrim(paths) if paths else None
        positions, orientations = (
            bodies.get_world_poses() if bodies is not None else (None, None)
        )
        stage = self.world.stage
        dynamic = [
            index
            for index, path in enumerate(paths)
            if not self._is_kinematic(stage.GetPrimAtPath(path))
        ]
        self._snapshot = {
            "paths": paths,
            "dynamic": np.array(dynamic, dtype=int),
            "positions": None if positions is None else np.array(positions),
            "orientations": None if orientations is None else np.array(orientations),
            "joints": (
                None if self.robot is None else self.robot.get_joint_positions().copy()
            ),
            "base": (
                None
                if self.robot is None
                else tuple(np.array(value) for value in self.robot.get_world_poses())
            ),
        }
        print(
            f"[reset] snapshot: {len(paths)} rigid bodies "
            f"({len(dynamic)} dynamic)"
            + ("" if self.robot is None else " + the robot"),
            flush=True,
        )

    # %% the service

    def _on_request(self, request, response):
        """Restore the snapshot; see the class docstring for why that is safe here."""
        if self._snapshot is None:
            response.success = False
            response.message = (
                "no snapshot yet -- the sim has not completed a step since it started"
            )
            return response
        # Guarded for the same reason SceneSyncROS guards its requests: this runs on
        # the sim loop thread, so an exception here does not fail the reset, it ends
        # the simulator -- and silently, since simulation_app.close() beats Python to
        # reporting it. A failed reset must be a failed reset.
        try:
            response.message = self._restore()
            response.success = True
        except Exception as failure:  # noqa: BLE001 - a failed reset must not be fatal
            response.success = False
            response.message = f"{type(failure).__name__}: {failure}"
            self.get_logger().error(f"scene reset failed: {response.message}")
        return response

    def apply_commands(self, dt: float) -> None:
        """Take the snapshot once, on the first step. See the class docstring."""
        if self._snapshot is None:
            self._take_snapshot()

    # %% the work

    def _restore(self) -> str:
        snapshot = self._snapshot
        restored = 0

        if snapshot["paths"]:
            bodies = RigidPrim(snapshot["paths"])
            bodies.set_world_poses(snapshot["positions"], snapshot["orientations"])
            # Velocity, not just pose. A rigid body keeps the motion it had before it
            # was moved, so restoring the pose alone hands PhysX a scene that is back
            # where it started and still travelling -- props drift off the worktop in
            # the first few steps after a reset that looked correct in the viewport.
            #
            # Only the dynamic ones: see _is_kinematic for what asking the rest costs.
            dynamic = snapshot["dynamic"]
            if len(dynamic):
                bodies.set_velocities(np.zeros((len(dynamic), 6)), indices=dynamic)
            restored = len(snapshot["paths"])

        if self.robot is not None:
            self.robot.set_joint_positions(snapshot["joints"])
            self.robot.set_joint_velocities(np.zeros_like(snapshot["joints"]))
            self.robot.set_joint_position_targets(snapshot["joints"])
            positions, orientations = snapshot["base"]
            self.robot.set_world_poses(positions, orientations)

        if self.integrator is not None:
            # Dropping the held targets makes the integrator re-seed from the
            # measured position on its next step, instead of continuing to chase
            # wherever the last plan left them.
            self.integrator.forget_targets()

        # Same problem one level up, for a base that is teleported rather than
        # driven. Asked for by capability rather than by robot type: a robot whose
        # base is simulated properly has nothing to resync and simply lacks this.
        resync = getattr(self.robot_bridge, "resync_base", None)
        if callable(resync):
            resync()

        # Deleted before the snapshot is restored below would be wrong: deleting
        # restarts physics, and a restart discards exactly what the restore just
        # wrote. So this runs last, and re-applies the robot state after it.
        dropped = []
        delete_spawned = getattr(self.sync_bridge, "delete_spawned", None)
        if callable(delete_spawned):
            dropped = delete_spawned()
            if dropped and self.robot is not None:
                # The stop/play inside the delete reverts the robot too, so put it
                # back where the snapshot says a second time.
                self.robot.set_joint_positions(snapshot["joints"])
                self.robot.set_joint_velocities(np.zeros_like(snapshot["joints"]))
                self.robot.set_joint_position_targets(snapshot["joints"])
                positions, orientations = snapshot["base"]
                self.robot.set_world_poses(positions, orientations)
                resync = getattr(self.robot_bridge, "resync_base", None)
                if callable(resync):
                    resync()

        return (
            f"restored {restored} bodies"
            + ("" if self.robot is None else " and the robot")
            + (f", deleted {len(dropped)} synced-in" if dropped else "")
        )
