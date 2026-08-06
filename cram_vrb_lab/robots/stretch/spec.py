"""The Stretch as the demo entry points see it -- see :mod:`cram_vrb_lab.specs`.

Every function here imports inside its body: this module is imported by both the
Isaac python (which has no giskardpy) and the CRAM venv (which has no isaacsim).
"""

from cram_vrb_lab.specs import RobotSpec


def _giskard_world(environment):
    """The stock diff-drive Stretch world, with ``environment`` merged in after it.

    The stock config is unmodified: the robot is built by giskardpy's own
    ``WorldWithStretchConfigDiffDrive`` and the environment is merged onto ``map``
    afterwards. That order is load-bearing -- see
    :func:`cram_vrb_lab.control.giskard_world.build_world_config`.
    """
    from giskardpy.middleware.ros2.scripts.iai_robots.stretch.configs import (
        WorldWithStretchConfigDiffDrive,
    )

    from cram_vrb_lab.control.giskard_world import build_world_config

    from .joints import load_patched_urdf

    return build_world_config(
        WorldWithStretchConfigDiffDrive, environment, urdf=load_patched_urdf()
    )


def _giskard_interface():
    from .giskard_config import StretchRealStyleInterface

    return StretchRealStyleInterface()


def _spawn(world, render, position=None):
    from .isaac_node import spawn_stretch

    if position is None:
        return spawn_stretch(world, render)
    return spawn_stretch(world, render, position=position)


def _ros_node(world, render, robot, args):
    """The ROS bridge plus, unless ``--camera none``, the head camera it publishes."""
    from .isaac_node import StretchROS, create_head_camera

    head_cam = (
        create_head_camera(world, render, want_depth=args.want_depth)
        if args.camera != "none"
        else None
    )
    return StretchROS(
        robot,
        head_cam=head_cam,
        publish_rgb=args.want_rgb,
        publish_depth=args.want_depth,
    )


STRETCH = RobotSpec(
    name="stretch",
    giskard_world=_giskard_world,
    giskard_interface=_giskard_interface,
    spawn=_spawn,
    ros_node=_ros_node,
)
