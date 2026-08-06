"""The Stretch as the demo entry points see it -- see :mod:`cram_vrb_lab.specs`.

Every function here imports inside its body: this module is imported by both the
Isaac python (which has no giskardpy) and the CRAM venv (which has no isaacsim).
"""

from cram_vrb_lab.specs import RobotSpec


def _giskard_world(environment, spawn_pose):
    """The stock diff-drive Stretch world, with ``environment`` merged in after it.

    The stock config is unmodified: the robot is built by giskardpy's own
    ``WorldWithStretchConfigDiffDrive`` and the environment is merged onto ``map``
    afterwards. That order is load-bearing -- see
    :func:`cram_vrb_lab.control.giskard_world.build_world_config`.

    ``spawn_pose`` is deliberately unused. The robot hangs off ``map -> odom`` and
    giskard reads that transform from tf and the base pose from ``/odom``, so where
    the robot actually stands arrives over the wire (the sim publishes odometry
    from wherever it spawned it) exactly as it would from a real robot's
    localization. Building the world at the spawn pose instead would double the
    offset.
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


def _spawn(world, render, spawn_pose):
    from .isaac_node import spawn_stretch

    return spawn_stretch(
        world, render, position=spawn_pose.position, yaw=spawn_pose.yaw
    )


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
