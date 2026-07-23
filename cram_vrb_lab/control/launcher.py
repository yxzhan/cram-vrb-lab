"""Shared launcher for the demo notebooks.

Starts an Isaac Sim scene script and a giskard control server and waits until
each prints its ready marker. Two run modes:

- ``terminal=False`` (default): a detached background subprocess writing to a log
  file, which is polled for the marker.
- ``terminal=True``: a visible ``gnome-terminal`` window running the same command
  (tee'd to the log so readiness is still detected). Handy for watching output.

The defaults launch the Stretch + apartment demo; other robot/scene combinations
pass their own ``sim_script`` / ``server_script`` / ``marker``.
"""

import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from cram_vrb_lab.paths import REPO_DIR as REPO

# ROS 2 discovery, matched to the sim scripts so the notebook kernel sees the topics.
os.environ.setdefault("ROS_DOMAIN_ID", "0")
os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
os.environ.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")

ISAAC_SIM_LOG = "/tmp/isaac_sim.log"
GISKARD_SERVER_LOG = "/tmp/giskard_server.log"

DEFAULT_SIM_SCRIPT = REPO / "demos" / "stretch_apartment_sim.py"
DEFAULT_SERVER_SCRIPT = REPO / "demos" / "stretch_apartment_giskard_server.py"

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
    :param kill_stale: pkill -f pattern for stale instances; note a basename
        pattern matches any process whose command line contains it.
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


def start_isaac_sim(sim_script=None, marker="StretchROS node ready.",
                    terminal=False, timeout=900, camera=None):
    """Launch an Isaac Sim scene script; wait for its ready marker.

    First startup can take a few minutes (shader compilation).

    :param sim_script: scene script run under the Isaac Sim python (default:
        the Stretch apartment demo).
    :param camera: head-camera mode passed as ``--camera``
        (``"rgb"`` / ``"depth"`` / ``"both"`` / ``"none"``); None uses the
        script's default.
    """
    sim_script = Path(sim_script) if sim_script else DEFAULT_SIM_SCRIPT
    args = [
        f"{REPO}/binder/isaacsim_python_wrapper.sh",
        str(sim_script),
    ]
    if camera is not None:
        args += ["--camera", camera]
    return start(
        "isaac sim",
        args,
        ISAAC_SIM_LOG,
        marker,
        timeout=timeout,
        kill_stale=sim_script.name,
        terminal=terminal,
    )


def start_giskard_server(server_script=None, marker="giskard is ready",
                         terminal=False, timeout=300):
    """Launch a giskard control server script; wait until it is ready."""
    server_script = Path(server_script) if server_script else DEFAULT_SERVER_SCRIPT
    return start(
        "giskard server",
        [sys.executable, str(server_script)],
        GISKARD_SERVER_LOG,
        marker,
        timeout=timeout,
        kill_stale=server_script.name,
        terminal=terminal,
    )


def stop(patterns=None):
    """Stop the sim and the giskard server, however they were started."""
    if patterns is None:
        patterns = (DEFAULT_SIM_SCRIPT.name, DEFAULT_SERVER_SCRIPT.name)
    for pattern in patterns:
        subprocess.run(["pkill", "-f", pattern], stdout=subprocess.DEVNULL)
    print("stopped isaac sim + giskard server")
