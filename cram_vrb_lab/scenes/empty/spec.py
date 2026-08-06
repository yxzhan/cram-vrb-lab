"""The empty stage as the demo entry points see it -- see :mod:`cram_vrb_lab.specs`.

``environment=None``: there is nothing here for giskard to avoid, so its world
contains the robot alone.
"""

from cram_vrb_lab.specs import SceneSpec


def _load(world, render, camera_eye=None, camera_target=None):
    # ``camera_eye`` is ignored: this stage is a workspace close-up with a fixed
    # eye, and only what it looks at is worth choosing.
    from .isaac_scene import load_empty_scene

    if camera_target is None:
        load_empty_scene(world, render)
    else:
        load_empty_scene(world, render, camera_target=camera_target)


EMPTY = SceneSpec(name="empty", load=_load)
