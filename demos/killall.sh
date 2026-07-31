#!/bin/bash
# Kill everything this lab starts: Isaac Sim, giskard servers, rviz, the
# localization stand-in, and every process on the CRAM venv python -- which
# includes your Jupyter kernels. Save your notebooks first.
#
# Not `pkill -f`: the shell running pkill has the pattern in its own command
# line, so pkill kills itself (exit 144) and leaves the target running.

pkill -9 -f ipykernel_launcher

for pattern in \
    "cognitive_robot_abstract_machine/.venv/bin/python" \
    isaacsim_python_wrapper \
    static_transform_publisher \
    rviz2
do
    for pid in $(pgrep -f -- "$pattern" 2>/dev/null); do
        [ "$pid" = "$$" ] || [ "$pid" = "$PPID" ] || kill -9 "$pid" 2>/dev/null
    done
done

echo "killed everything (kernels included)"
