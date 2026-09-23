"""Generic helpers for running a giskard motion-control server against the sim."""

import subprocess


def start_localization_stand_in(base_link_height: float = 0.0) -> subprocess.Popen:
    """Publish a static ``map -> odom`` as a localization stand-in.

    On a real robot AMCL/SLAM owns ``map -> odom``; the sim runs none, so the
    robot boots at the map origin with no odometry drift and the transform is a
    pure translation in z. Giskard's :meth:`sync_6dof_joint_with_tf_frame` blocks
    until this transform is available, so it must run alongside the server.

    :param base_link_height: the robot's
        :attr:`~cram_vrb_lab.specs.RobotSpec.base_link_height`, i.e. how far its
        root link rides above the floor. This transform is the only place that
        height can live: a wheeled base's odometry is planar, so the drive
        connection between ``odom`` and the root link has no z degree of freedom
        ("we can't measure its z-axis position, so z=0" -- ``OmniDrive``'s own
        docstring). Raising ``odom`` by it puts the root link at the right height
        in the twin, and the sim publishes ``odom -> base_link`` planar to match.
        Zero -- the default -- for a robot whose root link is on the floor and for
        one bolted to ``map``.
    """
    return subprocess.Popen(
        [
            "ros2", "run", "tf2_ros", "static_transform_publisher",
            "--x", "0", "--y", "0", "--z", str(base_link_height),
            "--roll", "0", "--pitch", "0", "--yaw", "0",
            "--frame-id", "map", "--child-frame-id", "odom",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def with_scene_joint_states(interface_config, topic: str):
    """Make ``interface_config`` also sync the scene's own joints from ``topic``.

    What ``sync_joint_state_topic`` does for the robot, minus the check that keeps
    it from doing so for anything else: for a topic that is not the robot's it only
    reads at the start of a goal, so a drawer being pulled would still be integrated
    in giskard's own model for the whole motion -- exactly the belief this is here to
    replace with a measurement. So both synchronizers are added here: one for the idle
    loop, which keeps the twin current between goals, and one for the control loop,
    which overwrites whatever the last cycle integrated with what the sim measured.

    Wrapped around the instance rather than mixed into each robot's config class,
    because it depends on the scene, not on the robot.

    :param topic: see :data:`cram_vrb_lab.sim.scene_joints.SCENE_JOINT_STATES_TOPIC`.
    :return: ``interface_config`` itself.
    """
    from giskardpy.middleware.ros2.input_synchronization import (
        LatestJointStateSynchronizer,
        PendingJointStateSynchronizer,
    )

    robot_setup = interface_config.setup

    def setup():
        robot_setup()
        interface_config.motion_server.inputs.synchronizers.append(
            PendingJointStateSynchronizer(world=interface_config.world, topic_name=topic)
        )
        if interface_config.server_config.is_closed_loop:
            interface_config.control_loop.inputs.synchronizers.append(
                LatestJointStateSynchronizer(
                    world=interface_config.world, topic_name=topic
                )
            )

    interface_config.setup = setup
    return interface_config
