"""Shared launcher for the demo notebooks.

Starts an Isaac Sim scene script and a giskard control server and waits until
each prints its ready marker. Two run modes:

- ``terminal=False`` (default): a detached background subprocess writing to a log
  file, which is polled for the marker.
- ``terminal=True``: a visible ``gnome-terminal`` window running the same command
  (tee'd to the log so readiness is still detected). Handy for watching output.

There is one sim script and one server script for every demo; which robot in
which scene they run is a ``robot=`` / ``scene=`` argument, defaulting to the
Stretch in the apartment (see :mod:`cram_vrb_lab.setups` for the combinations).
"""

import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:
    from sidecar import Sidecar
except ImportError:
    pass
    # print("Sidecar not available!")

from cram_vrb_lab.paths import REPO_DIR as REPO
from cram_vrb_lab.setups import DEFAULT_ROBOT, DEFAULT_SCENE
from cram_vrb_lab.sim.isaac_app import READY_MARKER as SIM_READY_MARKER

# ROS 2 discovery, matched to the sim scripts so the notebook kernel sees the topics.
os.environ.setdefault("ROS_DOMAIN_ID", "0")
os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
os.environ.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")

ISAAC_SIM_LOG = "/tmp/isaac_sim.log"
GISKARD_SERVER_LOG = "/tmp/giskard_server.log"
RVIZ_LOG = "/tmp/rviz.log"

SIM_SCRIPT = REPO / "demos" / "sim.py"
SERVER_SCRIPT = REPO / "demos" / "giskard_server.py"
DEFAULT_RVIZ_CONFIG = REPO / "demos" / "rviz" / "aicor.rviz"

# Sourced before the command in terminal mode: a fresh gnome-terminal shell does
# not inherit the kernel's sourced ROS environment (the background mode does).
_SOURCE = (
    f"source /opt/ros/jazzy/setup.bash && "
    f"source {REPO}/ros2_ws/install/setup.bash && "
)


def _tail(log_path, lines=10):
    return "\n".join(Path(log_path).read_text(errors="ignore").splitlines()[-lines:])


def _launch(name, args, log_path, kill_stale=None, terminal=True):
    """Kill stale instances, then start ``args`` logging to ``log_path``.

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
        return subprocess.Popen(["gnome-terminal", "--", "bash", "-c", inner])
    return subprocess.Popen(args, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)


def start(name, args, log_path, marker, timeout, kill_stale=None, terminal=True):
    """Start ``args`` and block until ``marker`` appears in ``log_path``.

    Parameters and return value as for :func:`_launch`.
    """
    proc = _launch(name, args, log_path, kill_stale=kill_stale, terminal=terminal)

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


def _spawn_args(spawn_position=None, spawn_yaw=None):
    """``--spawn-position`` / ``--spawn-yaw`` flags, omitted when not given.

    Omitted rather than defaulted here so the map origin is written down in one
    place only -- the scripts' own default (:class:`cram_vrb_lab.specs.SpawnPose`).
    """
    args = []
    if spawn_position is not None:
        args += ["--spawn-position", *[str(float(v)) for v in spawn_position]]
    if spawn_yaw is not None:
        args += ["--spawn-yaw", str(float(spawn_yaw))]
    return args


def start_isaac_sim(robot=DEFAULT_ROBOT, scene=DEFAULT_SCENE,
                    spawn_position=None, spawn_yaw=None,
                    marker=SIM_READY_MARKER, terminal=False, timeout=900,
                    camera=None, props=False):
    """Launch the Isaac Sim scene for ``robot`` in ``scene``; wait until ready.

    First startup can take a few minutes (shader compilation).

    :param robot: which robot, e.g. ``"stretch"`` or ``"panda"``.
    :param scene: which scene, e.g. ``"apartment"`` or ``"garmi_apartment"``.
        The pair must be in :data:`cram_vrb_lab.setups.SETUPS`.
    :param spawn_position: ``(x, y, z)`` in the ``map`` frame [m] the robot
        starts at; None spawns it at the origin.
    :param spawn_yaw: its starting heading about z [rad]; None means 0.
        Pass both to :func:`start_giskard_server` as well -- a bolted-down arm
        has no other way to tell giskard where it is.
    :param camera: head-camera mode passed as ``--camera``
        (``"rgb"`` / ``"depth"`` / ``"both"`` / ``"none"``); None uses the
        script's default.
    :param props: pass ``--props`` to spawn the pick-and-place cube. Off by
        default; only the pick-and-place demos use it, and the Panda setup
        spawns it either way.
    """
    args = [
        f"{REPO}/binder/isaacsim_python_wrapper.sh",
        str(SIM_SCRIPT),
        "--robot", robot,
        "--scene", scene,
        *_spawn_args(spawn_position, spawn_yaw),
    ]
    if camera is not None:
        args += ["--camera", camera]
    if props:
        args += ["--props"]
    return start(
        "isaac sim",
        args,
        ISAAC_SIM_LOG,
        marker,
        timeout=timeout,
        # The full path, not the basename: one script now runs every setup, and
        # a stale sim of *any* combination has to go before a new one starts.
        kill_stale=str(SIM_SCRIPT),
        terminal=terminal,
    )


def start_giskard_server(robot=DEFAULT_ROBOT, scene=DEFAULT_SCENE,
                         spawn_position=None, spawn_yaw=None,
                         marker="giskard is ready", terminal=False, timeout=300):
    """Launch the giskard control server for ``robot`` in ``scene``; wait until
    it is ready.

    Same combination *and the same spawn pose* as :func:`start_isaac_sim`: for a
    robot fixed to ``map`` this is where giskard believes it stands, so a value
    that disagrees with the sim's makes giskard plan for an arm that is not the
    one being rendered.
    """
    return start(
        "giskard server",
        [sys.executable, str(SERVER_SCRIPT), "--robot", robot, "--scene", scene,
         *_spawn_args(spawn_position, spawn_yaw)],
        GISKARD_SERVER_LOG,
        marker,
        timeout=timeout,
        kill_stale=str(SERVER_SCRIPT),
        terminal=terminal,
    )


def _rviz_command(rviz_config):
    """``rviz2 -d <config>``, wrapped in ``vglrun`` only where that exists.

    VirtualGL is how an OpenGL application reaches the GPU on the VRB server,
    whose desktop is a VNC session with no direct 3D context; it is part of that
    image rather than of ROS. Off the server -- a plain workstation, or the
    container run with ``-v /tmp/.X11-unix`` as the README describes -- there is
    no ``vglrun`` and none is wanted: the X server already is the real one, and
    an unconditional ``vglrun`` would just fail to start rviz at all.

    Probed with :func:`shutil.which` in this process. That is the right test even
    for ``terminal=True``, which re-sources ROS in a fresh shell first: vglrun is
    a system binary on ``PATH``, not something a ROS overlay contributes.
    """
    command = ["rviz2", "-d", str(rviz_config)]
    return command if shutil.which("vglrun") is None else ["vglrun", *command]


def start_rviz(rviz_config=None, terminal=False):
    """Launch rviz2 with a config file; returns immediately (no ready marker).

    :param rviz_config: rviz config file (default: the Stretch demo config).
    """
    rviz_config = Path(rviz_config) if rviz_config else DEFAULT_RVIZ_CONFIG
    return _launch(
        "rviz2",
        _rviz_command(rviz_config),
        RVIZ_LOG,
        kill_stale="rviz2",
        terminal=terminal,
    )


def stop(patterns=None):
    """Stop the sim, the giskard server and rviz, however they were started and
    whichever robot/scene they were running."""
    if patterns is None:
        patterns = (str(SIM_SCRIPT), str(SERVER_SCRIPT), "rviz2")
    for pattern in patterns:
        subprocess.run(["pkill", "-f", pattern], stdout=subprocess.DEVNULL)
    print("stopped isaac sim + giskard server + rviz")


def display_desktop(anchor="split-right"):
    """
    Display the remote desktop in a JupyterLab Sidecar tab.
    
    Args:
        anchor (str): Where the Sidecar tab will be placed. Options:
                    'split-right', 'split-left', 'split-top', 'split-bottom',
                    'tab-before', 'tab-after'
    """
    try:
        jupyterhub_user = os.environ["JUPYTERHUB_USER"]
        domain_name = os.environ["BINDER_LAUNCH_HOST"]
        domain_name = domain_name.replace("binder", "jupyter")
    except KeyError:
        jupyterhub_user = None
        domain_name = "http://localhost:8888"
    url_prefix = f"{domain_name}/user/{jupyterhub_user}" if jupyterhub_user is not None else ''

    remote_desktop_url = f"{url_prefix}/desktop"

    display(widgets.HTML(
        value=f'<a href="{remote_desktop_url}"  class="jupyter-button" style="color: #fff;background-color: #1976d2;" target="_blank">Open Desktop in new Tab</a>',
    ))
    
    sc = Sidecar(title='Desktop', anchor=anchor)
    with sc:
        # The inserted custom HTML and CSS snippets are to make the tab resizable
        display(HTML(f"""
            <style>
            body.p-mod-override-cursor div.iframe-widget {{
                position: relative;
                pointer-events: none;
            }}

            body.p-mod-override-cursor div.iframe-widget:before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: transparent;
            }}
            </style>
            <div class="iframe-widget" style="width: calc(100% + 10px);height:100%;">
                <iframe src="{remote_desktop_url}" width="100%" height="100%"></iframe>
            </div>
        """))

