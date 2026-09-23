"""The garmi-apartment as the demo entry points see it -- see
:mod:`cram_vrb_lab.specs`."""

from cram_vrb_lab.specs import SceneSpec


def _load(world, render, camera_eye=None, camera_target=None):
    from .isaac_scene import load_garmi_apartment_scene

    load_garmi_apartment_scene(
        world, render, camera_eye=camera_eye, camera_target=camera_target
    )


def _fixed_cameras():
    from .isaac_scene import FIXED_CAMERA_SENSORS

    return dict(FIXED_CAMERA_SENSORS)


def _joints():
    from .constants import SCENE_JOINTS

    return SCENE_JOINTS


def _environment():
    from .giskard_world import garmi_apartment_environment

    return garmi_apartment_environment()


GARMI_APARTMENT = SceneSpec(
    name="garmi_apartment",
    load=_load,
    environment=_environment,
    fixed_cameras=_fixed_cameras,
    joints=_joints,
    prim_root="/World/GarmiApartment",
)
