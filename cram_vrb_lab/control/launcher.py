"""Shared launcher for the Stretch Isaac Sim demo notebooks.

Starts the Isaac Sim scene (``apartment.py``) and the giskard control server and
waits until each prints its ready marker. Two run modes:

- ``terminal=False`` (default): a detached background subprocess writing to a log
  file, which is polled for the marker.
- ``terminal=True``: a visible ``gnome-terminal`` window running the same command
  (tee'd to the log so readiness is still detected). Handy for watching output.

Used by both ``giskard_demo.ipynb`` and ``cram_demo.ipynb``.
"""

import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

# giskard_stretch/ -> repo root (cram_isaacsim)
REPO = Path(__file__).resolve().parent.parent

# ROS 2 discovery, matched to apartment.py so the notebook kernel sees the topics.
os.environ.setdefault("ROS_DOMAIN_ID", "0")
os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
os.environ.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")

ISAAC_SIM_LOG = "/tmp/apartment_sim.log"
GISKARD_SERVER_LOG = "/tmp/giskard_server.log"

# Sourced before the command in terminal mode: a fresh gnome-terminal shell does
# not inherit the kernel's sourced ROS environment (the background mode does).
_SOURCE = (
    f"source /opt/ros/jazzy/setup.bash && "
    f"source {REPO}/ros2_ws/install/setup.bash && "
)


def _tail(log_path, lines=10):
    return "\n".join(Path(log_path).read_text(errors="ignore").splitlines()[-lines:])


def start(name, args, log_path, marker, timeout, kill_stale=None, terminal=True):
    """Start ``args`` and block until ``marker`` appears in ``log_path``.

    :param terminal: run in a visible gnome-terminal window instead of a detached
        background subprocess.
    :return: the Popen handle (the gnome-terminal launcher when terminal=True).
    """
    if (
        kill_stale
        and subprocess.run(
            ["pkill", "-f", kill_stale], stdout=subprocess.DEVNULL
        ).returncode
        == 0
    ):
        print(f"killed a stale instance of {name}")
        time.sleep(3)

    Path(log_path).write_text("")  # so a stale log never satisfies the marker early
    print(f"starting {name}{' in a terminal' if terminal else ''}, logging to {log_path}")
    if terminal:
        inner = f"{_SOURCE}{shlex.join(args)} 2>&1 | tee {shlex.quote(log_path)}; exec bash"
        proc = subprocess.Popen(["gnome-terminal", "--", "bash", "-c", inner])
    else:
        proc = subprocess.Popen(
            args, stdout=open(log_path, "w"), stderr=subprocess.STDOUT
        )

    start_time = time.time()
    while time.time() - start_time < timeout:
        if marker in Path(log_path).read_text(errors="ignore"):
            print(f"{name} ready after {time.time() - start_time:.0f}s")
            return proc
        # A background subprocess can crash; a gnome-terminal launcher exits at
        # once (the real process lives in the window), so only check in bg mode.
        if not terminal and proc.poll() is not None:
            raise RuntimeError(
                f"{name} exited early (code {proc.returncode}) -- last log lines:\n"
                f"{_tail(log_path)}"
            )
        time.sleep(2)
    raise TimeoutError(
        f"{name}: {marker!r} not found within {timeout}s -- last log lines:\n"
        f"{_tail(log_path)}"
    )


def start_isaac_sim(terminal=False, timeout=900, camera=None):
    """Launch the Isaac Sim apartment scene; wait for the ROS bridge node.

    First startup can take a few minutes (shader compilation).

    :param camera: head-camera mode passed to apartment.py's ``--camera``
        (``"rgb"`` / ``"depth"`` / ``"both"`` / ``"none"``); None uses its default.
    """
    args = [
        f"{REPO}/binder/isaacsim_python_wrapper.sh",
        f"{REPO}/giskard_stretch/apartment.py",
    ]
    if camera is not None:
        args += ["--camera", camera]
    return start(
        "isaac sim",
        args,
        ISAAC_SIM_LOG,
        "StretchROS node ready.",
        timeout=timeout,
        kill_stale="giskard_stretch/apartment.py",
        terminal=terminal,
    )


def start_giskard_server(terminal=False, timeout=300):
    """Launch the giskard control server; wait until it is ready."""
    return start(
        "giskard server",
        [sys.executable, f"{REPO}/giskard_stretch/giskard_stretch_isaac.py"],
        GISKARD_SERVER_LOG,
        "giskard is ready",
        timeout=timeout,
        kill_stale="giskard_stretch_isaac.py",
        terminal=terminal,
    )


def stop():
    """Stop the Isaac Sim scene and the giskard server, however they were started."""
    for pattern in ("giskard_stretch/apartment.py", "giskard_stretch_isaac.py"):
        subprocess.run(["pkill", "-f", pattern], stdout=subprocess.DEVNULL)
    print("stopped isaac sim + giskard server")
