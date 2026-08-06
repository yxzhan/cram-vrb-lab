"""What a robot, a scene and a robot-in-a-scene *are*, as data.

These are the types the demo entry points (``demos/sim.py``,
``demos/giskard_server.py``) are written against; the instances live in
``robots/<robot>/spec.py`` / ``scenes/<scene>/spec.py`` and are collected in
:mod:`cram_vrb_lab.setups`.

.. note::
   Nothing here may import Isaac Sim, giskardpy, semantic_digital_twin or ROS at
   module scope, and neither may the spec modules that build these objects. The
   sim runs under the Isaac python and the giskard server under the CRAM venv,
   and neither interpreter has the other's packages -- but both processes import
   the registry to look their combination up. Every field is therefore a
   *callable* that does its own imports, in the process that has them. That is
   also why the annotations below are strings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple


@dataclass(frozen=True)
class SpawnPose:
    """Where a robot starts, in the giskard ``map`` frame (= the Isaac world frame).

    Passed in from the demo (``--spawn-position`` / ``--spawn-yaw``, i.e. the
    notebook's ``spawn_position=`` / ``spawn_yaw=``) rather than baked into a
    scene, because where a robot belongs in a room is a property of the demo, not
    of the room. The default is the origin, unrotated: a robot the demo says
    nothing about stands at ``map``'s origin.

    Flat on the floor by construction -- position plus a heading, no roll or
    pitch. Nothing here mounts a robot tilted, and a scalar yaw is what both
    sides want anyway (Isaac takes a quaternion, giskard a transformation matrix,
    and the prop layouts rotate their offsets by it).
    """

    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    """(x, y, z) in metres."""

    yaw: float = 0.0
    """Heading about z, in radians."""

    @property
    def quaternion_wxyz(self) -> Tuple[float, float, float, float]:
        """The heading as a quaternion in Isaac's ``(w, x, y, z)`` order."""
        return (math.cos(self.yaw / 2.0), 0.0, 0.0, math.sin(self.yaw / 2.0))

    def to_transformation_matrix(self):
        """The pose as a semantic_digital_twin
        ``HomogeneousTransformationMatrix``, for the giskard side."""
        from semantic_digital_twin.spatial_types.spatial_types import (
            HomogeneousTransformationMatrix,
        )

        return HomogeneousTransformationMatrix.from_xyz_rpy(
            *self.position, yaw=self.yaw
        )

    def __str__(self) -> str:
        x, y, z = self.position
        return f"({x:.3f}, {y:.3f}, {z:.3f}) yaw {math.degrees(self.yaw):.1f} deg"


ORIGIN = SpawnPose()
"""The default spawn pose: the map origin, unrotated."""


@dataclass(frozen=True)
class EnvironmentSpec:
    """An environment description to merge into the giskard world, and its pose.

    The two callables are deliberately not a description *string*: an apartment
    URDF is pre-edited as text before parsing while an MJCF has to go through
    MuJoCo's compiler from a file on disk, so the one thing both scenes can hand
    over is the parsed :class:`~semantic_digital_twin.world.World`.

    Consumed by :func:`cram_vrb_lab.control.giskard_world.build_world_config`.
    """

    load: Callable[[], "World"]
    """Parse the environment description into its own world."""

    pose: Callable[[], "HomogeneousTransformationMatrix"]
    """Where that world's root sits in ``map``, i.e. the measured alignment
    between the description and the scene Isaac renders."""


@dataclass(frozen=True)
class RobotSpec:
    """A robot, on both sides of the demo: giskard's and Isaac's."""

    name: str
    """The ``--robot`` value."""

    giskard_world: Callable[[Optional[EnvironmentSpec], SpawnPose], "WorldConfig"]
    """``(environment, spawn_pose) -> WorldConfig``: this robot, plus the
    environment it is given (``None`` for a scene giskard should not know about).
    Robot-only, so the same function serves every scene.

    A robot bolted to ``map`` is built *at* the spawn pose, because that is the
    only thing that tells giskard where it stands. A robot with a base ignores the
    argument: it learns its pose from odometry and localization, which is what a
    real one does too."""

    giskard_interface: Callable[[], "RobotInterfaceConfig"]
    """The giskard interface config wiring this robot's ROS topics."""

    spawn: Callable[..., object]
    """``(world, render, spawn_pose) -> handle``: put the robot into the Isaac
    stage at that pose."""

    ros_node: Callable[..., "SimBridge"]
    """``(world, render, handle, args) -> SimBridge``: the sim-side ROS bridge,
    including whatever sensors it publishes (the ``args`` namespace carries the
    scene flags, e.g. ``--camera``)."""

    park: Optional[Callable[..., None]] = None
    """``(handle, world, render)``, run once the scene is fully built. For an arm
    whose drive gains and park pose must be set *after* the last
    ``world.reset()``; ``None`` when the robot needs nothing."""


@dataclass(frozen=True)
class SceneSpec:
    """A scene, on both sides: what Isaac renders and what giskard collides with."""

    name: str
    """The ``--scene`` value."""

    load: Callable[..., None]
    """``(world, render, camera_eye=None, camera_target=None)``: build the Isaac
    stage."""

    environment: Optional[Callable[[], EnvironmentSpec]] = None
    """Lazily built description of the same scenery for giskard's world.
    ``None`` when the scene is only scenery to look at (an empty stage), in which
    case giskard plans against the robot alone."""


@dataclass(frozen=True)
class PropsSpec:
    """The pick-and-place props, where the scene has somewhere to put them."""

    layout: Callable[[SpawnPose], "PropLayout"]
    """``(spawn_pose) -> PropLayout``: where the cube starts and where it is
    carried to, in ``map``.

    A function of where the robot stands, because for a bolted-down arm it has to
    be: the props sit on the surface the arm is mounted on, within its reach, so
    moving the arm moves them. For a mobile robot the props stay where the room
    puts them and the argument is ignored."""

    by_default: bool = False
    """Spawn them without being asked. False means ``--props`` opts in: the cube
    is physics the demo does not always want (and a demo that ignores it would
    still have it lying on the floor)."""


@dataclass(frozen=True)
class Viewport:
    """A non-default framing for the Isaac viewport camera."""

    eye: Tuple[float, float, float]
    target: Tuple[float, float, float]


@dataclass(frozen=True)
class Setup:
    """One runnable combination: a robot, a scene, and what only the *pair* knows.

    Everything below is pair-level on purpose. Whether there are props to grasp
    depends on whether the scene offers a surface to put them on; how the viewport
    is framed depends on what the robot does there.

    Where the robot *stands* is deliberately not here: it comes from the demo as a
    :class:`SpawnPose`, and both fields below are given it.
    """

    robot: RobotSpec
    scene: SceneSpec

    props: Optional[PropsSpec] = None
    """``None`` when this scene has no prop layout, which also makes ``--props``
    an error rather than a flag that silently does nothing."""

    viewport: Optional[Callable[[SpawnPose], Viewport]] = None
    """``(spawn_pose) -> Viewport``, or ``None`` to use the scene's own default
    view. Takes the pose because a close-up is framed on the robot."""

    @property
    def name(self) -> str:
        return f"{self.robot.name} x {self.scene.name}"

    def wants_props(self, requested: bool) -> bool:
        """Whether to spawn the props, given the ``--props`` flag.

        :raises SystemExit: if props were asked for and this pair has no layout.
        """
        if self.props is None:
            if requested:
                raise SystemExit(
                    f"--props: {self.name} has no prop layout. The cube's "
                    "positions are defined per scene in "
                    "cram_vrb_lab.scenes.props.constants."
                )
            return False
        return self.props.by_default or requested
