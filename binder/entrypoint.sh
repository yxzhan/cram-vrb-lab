#!/bin/bash

# Setup ROS2 environment
source ${ROS_PATH}/setup.bash

# json_msgs overlay (giskard's action interface)
if [ -f "${HOME}/cram_isaacsim/ros2_ws/install/setup.bash" ]; then
    source ${HOME}/cram_isaacsim/ros2_ws/install/setup.bash
fi

# The following line will allow the binderhub start Jupyterlab, should be at the end of the entrypoint.
exec "$@"
