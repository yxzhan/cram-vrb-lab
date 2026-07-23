#!/bin/bash

# Clear default ROS ENV
unset PYTHONPATH

# Match the simulation's ROS middleware settings
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-LOCALHOST}"

# Isaac Sim ROS2 Bridge
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:$ISAACSIM_PATH/exts/isaacsim.ros2.bridge/$ROS_DISTRO/lib

$ISAACSIM_PYTHON_EXE "$@"