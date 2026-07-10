#!/bin/bash

# Isaac-sim python exe wrapper

# unset virtualGL ENV (which breaks ROS2 bridge)
unset LD_PRELOAD
# Clear default ROS ENV
unset PYTHONPATH

# Isaac Sim ROS2 Bridge
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:$ISAACSIM_PATH/exts/isaacsim.ros2.bridge/$ROS_DISTRO/lib

$ISAACSIM_PYTHON_EXE "$@"