#!/usr/bin/env bash
#
# Auto-launch demos/garmi_demo.py as soon as the graphical session (and $DISPLAY)
# is up.
#
# Wired via XDG autostart (~/.config/autostart/GARMI-Apartment-autostart.desktop).
# In this image the X server is NOT running at container start: the VNC server is
# spawned on demand by jupyter-remote-desktop-proxy the first time the user opens
# the "Desktop" tab, and its xstartup is `exec dbus-launch xfce4-session` -- so
# this script fires exactly then, which is when a display first exists.
#
# The demo needs one: it starts rviz2, and demos/launcher.py hands Isaac Sim a
# DISPLAY unless ISAAC_HEADLESS=1. There is nothing useful to do before the
# desktop session exists.
#
# Opens a terminal on the desktop so the demo has a real tty and its log output
# stays visible. Written defensively so it survives across terminal emulators.
set -u

# Only a fallback. garmi_demo.py sets DISPLAY itself near the top of the file, and
# whatever the desktop session exports is more likely to be right than a guess
# here, so never overwrite an existing value.
: "${DISPLAY:=:1}"
export DISPLAY

REPO_DIR="${REPO_DIR:-$HOME/cram-vrb-lab}"
WRAPPER="${REPO_DIR}/binder/cram_python_wrapper.sh"
TARGET="demos/garmi_demo.py"

# The session manager already started us after X came up, but give the display
# server / window manager a moment to actually accept clients before we spawn
# rviz and the sim. No hard dependency on xdpyinfo -- fall back to the X socket.
for _ in $(seq 1 60); do
    if command -v xdpyinfo >/dev/null 2>&1; then
        xdpyinfo >/dev/null 2>&1 && break
    else
        n="${DISPLAY#*:}"; n="${n%%.*}"
        [ -S "/tmp/.X11-unix/X${n}" ] && break
    fi
    sleep 0.5
done

# No ROS setup here, unlike the isaacsim-template autostart: cram_python_wrapper.sh
# sources /opt/ros/jazzy and the ros2_ws overlay itself before exec'ing the venv
# interpreter, which is the whole reason the demos go through it.

# cd is load-bearing, not tidiness: garmi_demo.py opens with
# `REPO = Path.cwd().resolve()` and puts that on sys.path, so it imports
# cram_vrb_lab and demos/launcher.py out of whatever directory it was started in.
RUN_CMD="cd '${REPO_DIR}' && '${WRAPPER}' '${TARGET}'"

# Pick whatever terminal emulator this image ships. Real terminals are tried
# before the x-terminal-emulator alternative: in this image that alternative
# resolves to Debian's gnome-terminal.wrapper, an arg-translation shim that does
# NOT understand `-- <argv>` and simply hangs when given it. The wrapper only
# speaks the legacy `-e <string>` form, same as the older emulators.
for t in gnome-terminal xfce4-terminal mate-terminal lxterminal konsole xterm x-terminal-emulator; do
    if command -v "$t" >/dev/null 2>&1; then
        TERM_EMU="$t"
        break
    fi
done

# `exec bash` keeps the window open after the demo returns. That matters more here
# than in the isaacsim-template: garmi_demo.py starts Isaac Sim, giskard and rviz
# as child processes and never calls stop(), so the shell is where you go to read
# the logs and shut them down.
case "${TERM_EMU:-}" in
    "")
        # No terminal emulator found: run without a tty. The demo still comes up;
        # only the console log has nowhere to go.
        exec bash -c "${RUN_CMD}"
        ;;
    gnome-terminal|xfce4-terminal|mate-terminal)
        exec "$TERM_EMU" -- bash -c "${RUN_CMD}; exec bash"
        ;;
    *)
        exec "$TERM_EMU" -e "bash -c \"${RUN_CMD}; exec bash\""
        ;;
esac
