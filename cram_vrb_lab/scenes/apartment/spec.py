"""The apartment as the demo entry points see it -- see :mod:`cram_vrb_lab.specs`."""

from cram_vrb_lab.specs import SceneSpec


def _load(world, render, camera_eye=None, camera_target=None):
    from .isaac_scene import load_apartment_scene

    load_apartment_scene(
        world, render, camera_eye=camera_eye, camera_target=camera_target
    )


def _environment():
    from .giskard_world import apartment_environment

    return apartment_environment()


APARTMENT = SceneSpec(name="apartment", load=_load, environment=_environment)
