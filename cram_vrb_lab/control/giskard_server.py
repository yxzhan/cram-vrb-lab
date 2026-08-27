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
