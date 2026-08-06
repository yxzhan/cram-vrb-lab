"""Which robot in which scene: the registry both demo entry points look up.

``demos/sim.py`` and ``demos/giskard_server.py`` are the same two programs for
every demo; ``--robot`` and ``--scene`` pick a row out of this module. Adding a
combination means adding an entry to :data:`SETUPS` -- no new entry scripts, and
usually no new code at all.

Three kinds of fact:

- :data:`ROBOTS` / :data:`SCENES` -- what a robot or a scene *is*, collected from
  ``robots/<robot>/spec.py`` and ``scenes/<scene>/spec.py``, so everything about
  one of them still lives in its own directory.
- :data:`SETUPS` -- what is true only of a *pair*: where the robot stands in that
  scene, whether the scene has props to pick up, how the viewport frames it.
- :func:`get_setup` and :func:`add_setup_arguments` -- the lookup and the two
  command-line flags, shared by both entry points.

.. note::
   Imported by both the Isaac python and the CRAM venv, so nothing here (or in
   the spec modules it imports) may pull in isaacsim, giskardpy or ROS at module
   scope -- see :mod:`cram_vrb_lab.specs`. The constants below are plain python.
"""

from typing import Dict, Tuple

from cram_vrb_lab.robots.panda.spec import PANDA
from cram_vrb_lab.robots.stretch.spec import STRETCH
from cram_vrb_lab.scenes.apartment.constants import PANDA_BASE_POSITION_IN_MAP
from cram_vrb_lab.scenes.apartment.spec import APARTMENT
from cram_vrb_lab.scenes.empty.spec import EMPTY
from cram_vrb_lab.scenes.garmi_apartment.constants import STRETCH_SPAWN_POSITION
from cram_vrb_lab.scenes.garmi_apartment.spec import GARMI_APARTMENT
from cram_vrb_lab.scenes.props.constants import (
    APARTMENT_LAYOUT,
    PANDA_APARTMENT_LAYOUT,
)
from cram_vrb_lab.specs import PropsSpec, RobotSpec, SceneSpec, Setup, Viewport

ROBOTS: Dict[str, RobotSpec] = {robot.name: robot for robot in (STRETCH, PANDA)}

SCENES: Dict[str, SceneSpec] = {
    scene.name: scene for scene in (APARTMENT, GARMI_APARTMENT, EMPTY)
}

DEFAULT_ROBOT = STRETCH.name
DEFAULT_SCENE = APARTMENT.name

SETUPS: Dict[Tuple[str, str], Setup] = {
    (setup.robot.name, setup.scene.name): setup
    for setup in (
        Setup(
            robot=STRETCH,
            scene=APARTMENT,
            props=PropsSpec(layout=APARTMENT_LAYOUT),
        ),
        Setup(
            robot=STRETCH,
            scene=GARMI_APARTMENT,
            # The map origin of this flat lies outside the rooms, so the default
            # spawn would put the robot in the void next to the building.
            spawn_position=STRETCH_SPAWN_POSITION,
            # No props: their layout is measured against the *other* apartment's
            # geometry, and the perception demo this scene exists for uses the
            # YCB objects the scene loader puts on the worktop instead.
        ),
        Setup(
            robot=PANDA,
            scene=APARTMENT,
            # The arm's own default placement is this apartment's mounting pose
            # (cram_vrb_lab.scenes.apartment.constants.PANDA_BASE_POSITION_IN_MAP),
            # which the giskard world config reads too.
            props=PropsSpec(layout=PANDA_APARTMENT_LAYOUT, by_default=True),
            # Framed on the arm's workspace rather than on the apartment as a
            # whole: the props are 5 cm objects and the default wide shot leaves
            # them a few pixels.
            viewport=Viewport(
                eye=(
                    PANDA_BASE_POSITION_IN_MAP[0] - 1.4,
                    PANDA_BASE_POSITION_IN_MAP[1] - 1.4,
                    PANDA_BASE_POSITION_IN_MAP[2] + 1.0,
                ),
                target=PANDA_APARTMENT_LAYOUT.cube_start_position,
            ),
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
    """Add ``--robot`` / ``--scene`` to an :mod:`argparse` parser."""
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
