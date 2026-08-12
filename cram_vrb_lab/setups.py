"""Which robot in which scene: the registry both demo entry points look up.

``demos/sim.py`` and ``demos/giskard_server.py`` are the same two programs for
every demo; ``--robot`` and ``--scene`` pick a row out of this module. Adding a
combination means adding an entry to :data:`SETUPS` -- no new entry scripts, and
usually no new code at all.

Three kinds of fact:

- :data:`ROBOTS` / :data:`SCENES` -- what a robot or a scene *is*, collected from
  ``robots/<robot>/spec.py`` and ``scenes/<scene>/spec.py``, so everything about
  one of them still lives in its own directory.
- :data:`SETUPS` -- what is true only of a *pair*: whether the scene has props to
  pick up, and how the viewport frames what the robot is doing.
- :func:`get_setup`, :func:`add_setup_arguments` and :func:`spawn_pose_from_args`
  -- the lookup and the command-line flags, shared by both entry points.

Where the robot *stands* is not in here: it is a :class:`~cram_vrb_lab.specs.SpawnPose`
the demo passes in (``--spawn-position`` / ``--spawn-yaw``), defaulting to the map
origin. A notebook that wants its robot somewhere else says so; nothing in the
package decides it on the demo's behalf.

.. note::
   Imported by both the Isaac python and the CRAM venv, so nothing here (or in
   the spec modules it imports) may pull in isaacsim, giskardpy or ROS at module
   scope -- see :mod:`cram_vrb_lab.specs`. The constants below are plain python.
"""

import math
from typing import Dict, Tuple

from cram_vrb_lab.robots.garmi.spec import GARMI
from cram_vrb_lab.robots.panda.spec import PANDA
from cram_vrb_lab.robots.stretch.spec import STRETCH
from cram_vrb_lab.scenes.apartment.spec import APARTMENT
from cram_vrb_lab.scenes.empty.spec import EMPTY
from cram_vrb_lab.scenes.garmi_apartment.spec import GARMI_APARTMENT
from cram_vrb_lab.scenes.props.constants import APARTMENT_LAYOUT, panda_layout_at
from cram_vrb_lab.specs import (
    ORIGIN,
    PropsSpec,
    RobotSpec,
    SceneSpec,
    Setup,
    SpawnPose,
    Viewport,
)

ROBOTS: Dict[str, RobotSpec] = {
    robot.name: robot for robot in (STRETCH, PANDA, GARMI)
}

SCENES: Dict[str, SceneSpec] = {
    scene.name: scene for scene in (APARTMENT, GARMI_APARTMENT, EMPTY)
}

DEFAULT_ROBOT = STRETCH.name
DEFAULT_SCENE = APARTMENT.name


def _in_front_of(spawn_pose: SpawnPose, ahead: float, side: float = 0.0,
                 up: float = 0.0) -> Tuple[float, float, float]:
    """A point ``ahead`` m in front of the robot, ``side`` m to its left and ``up``
    m above its base, in ``map``. Follows the robot if the demo mounts it
    elsewhere."""
    x, y, z = spawn_pose.position
    cos_yaw, sin_yaw = math.cos(spawn_pose.yaw), math.sin(spawn_pose.yaw)
    return (
        x + cos_yaw * ahead - sin_yaw * side,
        y + sin_yaw * ahead + cos_yaw * side,
        z + up,
    )


def _arm_workspace_viewport(spawn_pose: SpawnPose) -> Viewport:
    """A close-up on a bolted-down arm's workspace, wherever it is bolted down.

    Framed on the props rather than on the room: they are 5 cm objects, and the
    scene's own wide shot leaves them a few pixels.
    """
    x, y, z = spawn_pose.position
    return Viewport(
        eye=(x - 1.4, y - 1.4, z + 1.0),
        target=panda_layout_at(spawn_pose.position, spawn_pose.yaw)
        .cube_start_position,
    )


def _arm_and_what_it_faces_viewport(spawn_pose: SpawnPose) -> Viewport:
    """The arm and the furniture it works on, seen from the open side of the room.

    For an arm that faces something close by (a cabinet run half a metre ahead),
    where the props-framed view above would put the camera inside that furniture.
    """
    return Viewport(
        eye=_in_front_of(spawn_pose, ahead=-1.5, side=-1.2, up=1.4),
        target=_in_front_of(spawn_pose, ahead=0.55, up=0.8),
    )


SETUPS: Dict[Tuple[str, str], Setup] = {
    (setup.robot.name, setup.scene.name): setup
    for setup in (
        Setup(
            robot=STRETCH,
            scene=APARTMENT,
            # Fixed positions in the room: the robot drives to the props, so
            # where it happens to start does not move them.
            props=PropsSpec(layout=lambda spawn_pose: APARTMENT_LAYOUT),
        ),
        Setup(
            robot=STRETCH,
            scene=GARMI_APARTMENT,
            # No props: a layout for this flat has never been measured, and the
            # perception demo this scene exists for uses the YCB objects the
            # scene loader puts on the worktop instead.
        ),
        Setup(
            robot=PANDA,
            scene=APARTMENT,
            # In front of the arm by construction, so the demo cannot put the
            # cube out of reach by mounting the arm somewhere else.
            props=PropsSpec(
                layout=lambda spawn_pose: panda_layout_at(
                    spawn_pose.position, spawn_pose.yaw
                ),
                by_default=True,
            ),
            # viewport=_arm_workspace_viewport,
        ),
        Setup(
            robot=PANDA,
            scene=GARMI_APARTMENT,
            # No props here either: this pairing exists to work on the *scene's*
            # own articulation -- the kitchen cabinet's doors and drawers, which
            # the MJCF twin models and Isaac renders from the same file.
            # viewport=_arm_and_what_it_faces_viewport,
        ),
        Setup(
            robot=GARMI,
            scene=GARMI_APARTMENT,
            # The flat GARMI was built for, and what the robot spec exists for:
            # the same kitchen articulation the Panda works on, reached from a
            # standing robot's shoulder height instead of off the floor.
        ),
    )
}
"""Every runnable robot x scene combination.

A pair that is missing is a pair nobody has aligned, not a pair that is
forbidden: the Panda in the empty stage, say, needs a prop layout measured
against wherever the arm stands there.
"""


def get_setup(robot: str, scene: str) -> Setup:
    """Look up a combination, or explain which ones exist.

    :raises SystemExit: for an unknown combination -- these come from a command
        line, so the message is the whole error handling.
    """
    try:
        return SETUPS[(robot, scene)]
    except KeyError:
        supported = "\n  ".join(
            f"--robot {name} --scene {scene_name}" for name, scene_name in SETUPS
        )
        raise SystemExit(
            f"no setup for robot {robot!r} in scene {scene!r}. Supported:\n"
            f"  {supported}\n"
            "Add one to cram_vrb_lab.setups.SETUPS."
        )


def add_setup_arguments(parser) -> None:
    """Add the flags both entry points share: which combination, and where the
    robot starts."""
    parser.add_argument(
        "--robot",
        choices=sorted(ROBOTS),
        default=DEFAULT_ROBOT,
        help=f"which robot to run (default: {DEFAULT_ROBOT}).",
    )
    parser.add_argument(
        "--scene",
        choices=sorted(SCENES),
        default=DEFAULT_SCENE,
        help=f"which scene to run it in (default: {DEFAULT_SCENE}).",
    )
    parser.add_argument(
        "--spawn-position",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=list(ORIGIN.position),
        help="where the robot starts, in the map frame [m] (default: the "
        "origin). Pass the same value to the sim and to the giskard server.",
    )
    parser.add_argument(
        "--spawn-yaw",
        type=float,
        metavar="RAD",
        default=ORIGIN.yaw,
        help="the robot's starting heading about z [rad] (default: 0).",
    )


def spawn_pose_from_args(args) -> SpawnPose:
    """The spawn pose the flags above describe."""
    return SpawnPose(position=tuple(args.spawn_position), yaw=args.spawn_yaw)
