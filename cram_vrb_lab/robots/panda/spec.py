"""The Panda as the demo entry points see it -- see :mod:`cram_vrb_lab.specs`.

Every function here imports inside its body: this module is imported by both the
Isaac python (which has no giskardpy) and the CRAM venv (which has no isaacsim).
"""

from cram_vrb_lab.specs import RobotSpec


def _giskard_world(environment):
    """The Panda bolted to ``map`` at its mounting pose, plus ``environment``.

    No odometry and no localization -- the arm does not move -- so the only thing
    the scene contributes is the scenery to avoid.
    """
    from cram_vrb_lab.control.giskard_world import build_world_config

    from .giskard_config import WorldWithPandaConfig
    from .joints import load_patched_urdf

    return build_world_config(
        WorldWithPandaConfig, environment, urdf=load_patched_urdf()
    )


def _giskard_interface():
    from .giskard_config import PandaSimInterface

    return PandaSimInterface()


def _spawn(world, render, position=None):
    from .isaac_node import spawn_panda

    if position is None:
        return spawn_panda(world, render)
    return spawn_panda(world, render, position=position)


def _ros_node(world, render, robot, args):
    """No cameras: the Stretch demos cover perception, and every camera is one
    more way for a grasp to fail that has nothing to do with grasping."""
    from .isaac_node import PandaROS

    return PandaROS(robot)


def _park(robot, world, render):
    from .isaac_node import move_to_park

    move_to_park(robot, world, render)


PANDA = RobotSpec(
    name="panda",
    giskard_world=_giskard_world,
    giskard_interface=_giskard_interface,
    spawn=_spawn,
    ros_node=_ros_node,
    # Last, after the props: spawning a body resets the world, which throws away
    # the drive gains and the park pose. See move_to_park's warning.
    park=_park,
)
