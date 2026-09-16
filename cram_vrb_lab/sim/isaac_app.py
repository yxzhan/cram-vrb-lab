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

from cram_vrb_lab.setups import add_setup_arguments

READY_MARKER = "Isaac Sim scene ready."
"""Printed by :func:`cram_vrb_lab.sim.runner.run` once the scene is up and the ROS
topics exist, and polled for in the log file by ``cram_vrb_lab.control.launcher``.

One string for every combination, so the launcher needs no per-demo knowledge. It
lives here rather than next to the print because the launcher must be able to
import it without pulling in the sim loop (and with it rclpy).
"""


def setup_ros_env():
    """ROS 2 environment for the Isaac ROS2 bridge (must be set before the
    bridge extension loads)."""
    os.environ.setdefault("ROS_DOMAIN_ID", "0")
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
    os.environ.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")


KIT_CACHE_SOURCE = (
    f"cache-{os.environ['ISAACSIM_VERSION']}"
    if os.environ.get("ISAACSIM_VERSION")
    else "cache"
)
"""Which prebuilt kit cache on the shared volume belongs to this Isaac Sim.

A shader and extension cache is only valid for the version that built it, and the
volume now holds one per version, so the version the image exports picks the
directory. Plain ``cache`` is the pre-versioning layout, kept for an image that
sets no ``ISAACSIM_VERSION``; a name that is not on the volume simply means no
cache, and :func:`copy_kit_cache` skips it.
"""

# Shared volume layout (/mnt/isaacsim-cache) -> local destination:
#   cache-<version>/        -> /isaac-sim/kit/cache
#   semantic_digital_twin/  -> ~/.cache/semantic_digital_twin
PREBUILT_CACHES = {
    KIT_CACHE_SOURCE: "/isaac-sim/kit/cache",
    "semantic_digital_twin": os.path.expanduser("~/.cache/semantic_digital_twin"),
}


def copy_kit_cache():
    """Copy the prebuilt caches that are not present yet (the binder image
    mounts them at /mnt/isaacsim-cache; drastically shortens the first startup
    -- the kit cache skips the shader/extension rebuild, the semantic digital
    twin cache the mesh parsing of the kitchen scene)."""
    for name, target_dir in PREBUILT_CACHES.items():
        source_dir = os.path.join("/mnt/isaacsim-cache", name)
        if os.path.isdir(source_dir) and not os.path.isdir(target_dir):
            shutil.copytree(source_dir, target_dir)


def parse_scene_args():
    """Parse the scene flags (and strip them from ``sys.argv`` before
    SimulationApp, which parses argv too).

    :return: a namespace with ``robot`` and ``scene`` (the combination to run,
        see :mod:`cram_vrb_lab.setups`), ``camera`` (``rgb`` / ``depth`` /
        ``both`` / ``none``), the derived ``want_rgb`` / ``want_depth``
        booleans, and ``props``.
    """
    parser = argparse.ArgumentParser(description="Isaac Sim scene")
    add_setup_arguments(parser)
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
        help="spawn the graspable cube for the pick-and-place task. Off by "
        "default, except where the setup spawns it anyway (the Panda demo is "
        "about nothing else); an error for a setup with no prop layout.",
    )
    args, unknown_args = parser.parse_known_args()
    sys.argv = sys.argv[:1] + unknown_args  # hide the scene flags from SimulationApp
    args.want_rgb = args.camera in ("rgb", "both")
    args.want_depth = args.camera in ("depth", "both")
    return args


def window_size():
    """``(width, height)`` for the app window, from ``ISAAC_WINDOW=WxH``.

    A knob because the frame is charged to the *control* loop: the sim steps
    physics, serves giskard and draws in one thread (see
    :mod:`cram_vrb_lab.sim.runner`), so an expensive frame slows the controller.

    What it is worth depends on which half of the frame cost dominates, and the
    ``cost probe`` line says which:

    - the *presentation* half (getting the finished frame onto a remote desktop:
      a CPU copy and a re-encode per frame) is flat in the pixel count. Measured
      on the VNC desktop with an RTX 3080, 1280x960 -> 640x480 recovered 5 ms of
      the 27 ms a native window costs. Little to gain; watch the livestream
      instead.
    - the *rendering* half is not flat, and on a GPU that cannot draw this scene
      quickly it is the whole problem -- an RTX 2070 spends 50 ms on a frame that
      a 3080 renders in single digits.

    Not verified here for the second case: an idle livestream (no client
    attached) does not actually render, so measuring the pixel scaling needs a
    machine where the frame is expensive in the first place.
    """
    width, _, height = os.environ.get("ISAAC_WINDOW", "640x360").partition("x")
    return int(width), int(height)


def create_simulation_app():
    """Set up the environment and start the SimulationApp (the expensive step)."""
    setup_ros_env()
    copy_kit_cache()

    from isaacsim import SimulationApp

    width, height = window_size()
    simulation_app = SimulationApp({
        # Headless is opt-in via ISAAC_HEADLESS=1 (e.g. on a machine with no
        # usable X display); the interactive viewer is the default.
        "headless": os.environ.get("ISAAC_HEADLESS", "0") == "1",
        # "hide_ui": True,
        "width": width,
        "height": height,
        "renderer": "RaytracedLighting",
        "display_options": 3286,  # show the default grid
    })
    print("SimulationApp Ready!")
    return simulation_app


def ensure_urdf_importer():
    """Load the URDF importer extension the SimulationApp's experience leaves out.

    The GUI experience (``isaacsim.exp.base.kit``) lists
    ``isaacsim.asset.importer.urdf``; the one a SimulationApp loads
    (``isaacsim.exp.base.python.kit``) does not, so neither the extension's python
    package nor -- before 6.1 dropped them -- its kit commands are there to be
    had until it is asked for by name.

    Called from :func:`~cram_vrb_lab.sim.urdf_import.import_urdf_robot` rather
    than once at startup, so importing a robot never depends on which entry script
    opened the app; enabling an already-enabled extension is a no-op.
    """
    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("isaacsim.asset.importer.urdf")


def physics_engine():
    """Which physics backend to run, from ``ISAAC_PHYSICS`` (default ``physx``).

    6.1 ships Newton beside PhysX and lets one process register both, so the
    choice is a runtime flag rather than a different build. ``newton`` is
    experimental here: :func:`~cram_vrb_lab.sim.urdf_import.import_urdf_robot`
    picks the matching physics variant out of the imported asset, and the robot
    bridges have to find their link poses elsewhere -- Newton's articulation view
    has no ``get_link_transforms``.
    """
    return os.environ.get("ISAAC_PHYSICS", "physx").lower()


def use_newton():
    """Whether this run is on Newton rather than PhysX."""
    return physics_engine() == "newton"


def render_enabled():
    """Rendering can be disabled with ISAAC_RENDER=0 to run headless physics
    only, e.g. on a machine whose GPU/display cannot do RTX rendering. The
    control path (joint states, odometry, TF from the physics view) needs no
    rendering; a head camera does, so ISAAC_RENDER=0 implies no camera image."""
    return os.environ.get("ISAAC_RENDER", "1") != "0"


def livestream_enabled():
    """Livestreaming is opt-in via ISAAC_LIVESTREAM=1: instead of a local
    viewport, the app streams its viewport over WebRTC, so the scene can be
    watched (and driven with the mouse) from a browser on another machine.

    Goes with ISAAC_HEADLESS=1 -- there is no point paying for both a local
    window and the stream -- and needs ISAAC_RENDER=1, since a stream is
    rendered frames and nothing else."""
    return os.environ.get("ISAAC_LIVESTREAM", "0") == "1"
