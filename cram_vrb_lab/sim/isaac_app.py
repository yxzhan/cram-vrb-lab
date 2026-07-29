"""Isaac Sim application bootstrap shared by all sim entry scripts.

.. warning::
   Import order matters. Any module that imports ``isaacsim.core.*``, ``omni``
   or ``pxr`` at module scope (``cram_vrb_lab.scenes.*.isaac_scene``,
   ``cram_vrb_lab.robots.*.isaac_node``) may only be imported AFTER
   :func:`create_simulation_app` has run -- the entry scripts in ``demos/``
   enforce this through their import order.
"""

import argparse
import os
import shutil
import sys


def setup_ros_env():
    """ROS 2 environment for the Isaac ROS2 bridge (must be set before the
    bridge extension loads)."""
    os.environ.setdefault("ROS_DOMAIN_ID", "0")
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
    os.environ.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")


def copy_kit_cache():
    """Copy the precompiled kit cache if it is not present yet (binder image
    mounts it at /mnt/isaacsim-cache; drastically shortens the first startup)."""
    target_dir = "/isaac-sim/kit/cache"
    source_dir = "/mnt/isaacsim-cache/cache"
    if os.path.isdir(source_dir) and not os.path.isdir(target_dir):
        shutil.copytree(source_dir, target_dir)


def parse_scene_args():
    """Parse the scene flags (and strip them from ``sys.argv`` before
    SimulationApp, which parses argv too).

    :return: a namespace with ``camera`` (``rgb`` / ``depth`` / ``both`` /
        ``none``), the derived ``want_rgb`` / ``want_depth`` booleans, and
        ``props``.
    """
    parser = argparse.ArgumentParser(description="Isaac Sim scene")
    parser.add_argument(
        "--camera",
        choices=["rgb", "depth", "both", "none"],
        default="both" if os.environ.get("ISAAC_NO_CAMERA", "0") == "1" else "rgb",
        help="head-camera mode: publish the rgb image, the depth image, both, or "
        "run no camera at all (default: rgb, or none when ISAAC_NO_CAMERA=1).",
    )
    parser.add_argument(
        "--props",
        action="store_true",
        help="spawn the pick-and-place props (a graspable cube on a pedestal, "
        "plus a second pedestal to carry it to). Off by default: the pedestals "
        "stand in floor the other demos navigate through.",
    )
    args, unknown_args = parser.parse_known_args()
    sys.argv = sys.argv[:1] + unknown_args  # hide the scene flags from SimulationApp
    args.want_rgb = args.camera in ("rgb", "both")
    args.want_depth = args.camera in ("depth", "both")
    return args


def create_simulation_app():
    """Set up the environment and start the SimulationApp (the expensive step)."""
    setup_ros_env()
    copy_kit_cache()

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({
        # Headless is opt-in via ISAAC_HEADLESS=1 (e.g. on a machine with no
        # usable X display); the interactive viewer is the default.
        "headless": os.environ.get("ISAAC_HEADLESS", "0") == "1",
        "hide_ui": False,
        "width": 1280,
        "height": 960,
        "renderer": "RaytracedLighting",
        "display_options": 3286,  # show the default grid
    })
    print("SimulationApp Ready!")
    return simulation_app


def render_enabled():
    """Rendering can be disabled with ISAAC_RENDER=0 to run headless physics
    only, e.g. on a machine whose GPU/display cannot do RTX rendering. The
    control path (joint states, odometry, TF from the physics view) needs no
    rendering; a head camera does, so ISAAC_RENDER=0 implies no camera image."""
    return os.environ.get("ISAAC_RENDER", "1") != "0"
