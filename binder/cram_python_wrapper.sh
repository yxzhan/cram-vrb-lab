#!/bin/bash

# CRAM venv python wrapper: sources the ROS 2 environment and the ros2_ws
# workspace overlay before exec'ing the venv interpreter, so a jupyter kernel
# (or any script) using it gets the full ROS environment regardless of how the
# parent process was started.

# Deliberately ignore an inherited REPO_DIR: container images export it and a
# stale image would silently point this wrapper at an older checkout.
REPO_DIR="$HOME/cram-vrb-lab"

source "${ROS_PATH:-/opt/ros/jazzy}/setup.bash"
if [ -f "${REPO_DIR}/ros2_ws/install/setup.bash" ]; then
    source "${REPO_DIR}/ros2_ws/install/setup.bash"
fi

# Match the simulation's ROS middleware settings
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-LOCALHOST}"

exec "${REPO_DIR}/cognitive_robot_abstract_machine/.venv/bin/python" "$@"
