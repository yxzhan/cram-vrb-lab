#!/bin/bash

# Setup ROS2 environment
source ${ROS_PATH}/setup.bash

# ros2_ws overlay (robot descriptions + json_msgs, giskard's action interface)
if [ -f "${HOME}/cram-vrb-lab/ros2_ws/install/setup.bash" ]; then
    source ${HOME}/cram-vrb-lab/ros2_ws/install/setup.bash
fi

# The following line will allow the binderhub start Jupyterlab, should be at the end of the entrypoint.
exec "$@"
