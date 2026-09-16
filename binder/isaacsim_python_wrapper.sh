#!/bin/bash

source "${ROS_PATH:-/opt/ros/jazzy}/setup.bash"

# Match the simulation's ROS middleware settings
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-LOCALHOST}"

# Isaac Sim ROS2 Bridge

$ISAACSIM_PYTHON_EXE "$@"