"""Generic helpers for running a giskard motion-control server against the sim."""

import subprocess


def start_localization_stand_in() -> subprocess.Popen:
    """Publish a static identity ``map -> odom`` as a localization stand-in.

    On a real robot AMCL/SLAM owns ``map -> odom``; the sim runs none, so the
    robot boots at the map origin with no odometry drift and the transform is
    identity. Giskard's :meth:`sync_6dof_joint_with_tf_frame` blocks until this
    transform is available, so it must run alongside the server.
    """
    return subprocess.Popen(
        [
            "ros2", "run", "tf2_ros", "static_transform_publisher",
            "--x", "0", "--y", "0", "--z", "0",
            "--roll", "0", "--pitch", "0", "--yaw", "0",
            "--frame-id", "map", "--child-frame-id", "odom",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
