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

from dataclasses import dataclass
from typing import Callable, Optional, Tuple


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

    giskard_world: Callable[[Optional[EnvironmentSpec]], "WorldConfig"]
    """Build the giskard world config: this robot, plus the environment it is
    given (``None`` for a scene giskard should not know about). Robot-only, so
    the same function serves every scene."""

    giskard_interface: Callable[[], "RobotInterfaceConfig"]
    """The giskard interface config wiring this robot's ROS topics."""

    spawn: Callable[..., object]
    """``(world, render, position=None) -> handle``. ``position=None`` means the
    robot's own default placement (see the ``isaac_node`` module)."""

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

    layout: "PropLayout"
    """Where the cube starts and where it is carried to, in ``map``."""

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

    Everything below is pair-level on purpose. Where a robot spawns depends on the
    room it spawns in; whether there are props to grasp depends on whether the
    scene offers a surface to put them on; how the viewport is framed depends on
    what the robot does there.
    """

    robot: RobotSpec
    scene: SceneSpec

    spawn_position: Optional[Tuple[float, float, float]] = None
    """Where the robot stands in this scene, in ``map``. ``None`` uses the
    robot's own default."""

    props: Optional[PropsSpec] = None
    """``None`` when this scene has no prop layout, which also makes ``--props``
    an error rather than a flag that silently does nothing."""

    viewport: Optional[Viewport] = None
    """``None`` uses the scene's own default view."""

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
