"""GARMI as the demo entry points see it -- see :mod:`cram_vrb_lab.specs`.

Every function here imports inside its body: this module is imported by both the
Isaac python (which has no giskardpy) and the CRAM venv (which has no isaacsim).
"""

from cram_vrb_lab.specs import RobotSpec

from .joints import BASE_LINK_HEIGHT


def _giskard_world(environment, spawn_pose):
    """GARMI on its omni drive, with ``environment`` merged in after it.

    That order is load-bearing -- see
    :func:`cram_vrb_lab.control.giskard_world.build_world_config`.

    ``spawn_pose`` is deliberately unused. The robot hangs off ``map -> odom`` and
    giskard reads that transform from tf and the base pose from ``/odom``, so
    where the robot actually stands arrives over the wire (the sim publishes
    odometry from wherever it spawned it) exactly as it would from a real robot's
    localization. Building the world at the spawn pose instead would double the
    offset.
    """
    from cram_vrb_lab.control.giskard_world import build_world_config

    from .giskard_config import WorldWithGarmiConfig
    from .joints import load_patched_urdf

    return build_world_config(
        WorldWithGarmiConfig, environment, urdf=load_patched_urdf()
    )


def _giskard_interface():
    from .giskard_config import GarmiSimInterface

    return GarmiSimInterface()


def _spawn(world, render, spawn_pose):
    from .isaac_node import spawn_garmi

    return spawn_garmi(
        world,
        render,
        position=spawn_pose.position,
        orientation=spawn_pose.quaternion_wxyz,
    )


def _ros_node(world, render, robot, args):
    """The ROS bridge plus, unless ``--camera none``, the head camera it publishes."""
    from .isaac_node import GarmiROS, create_head_camera

    head_cam = (
        create_head_camera(world, render, want_depth=args.want_depth)
        if args.camera != "none"
        else None
    )
    return GarmiROS(
        robot,
        head_cam=head_cam,
        publish_rgb=args.want_rgb,
        publish_depth=args.want_depth,
    )


def _park(robot, world, render):
    from .isaac_node import move_to_park

    move_to_park(robot, world, render)


GARMI = RobotSpec(
    name="garmi",
    giskard_world=_giskard_world,
    giskard_interface=_giskard_interface,
    spawn=_spawn,
    ros_node=_ros_node,
    # Last: spawning a body resets the world, which throws away the drive gains
    # and the park pose. See move_to_park's warning.
    park=_park,
    # base_link rides above the wheels, and the twin's OmniDrive cannot represent
    # that -- so map -> odom carries it. See BASE_LINK_HEIGHT.
    base_link_height=BASE_LINK_HEIGHT,
)
