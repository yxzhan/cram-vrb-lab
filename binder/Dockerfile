# Dockerfile of the base image: https://github.com/IntEL4CoRo/jupyter-ros2
FROM intel4coro/jupyter-ros2:jazzy-py3.12

USER ${NB_USER}
#============= Install IsaacSim =============#
ENV ISAACSIM_VERSION="5.1"
ENV ISAACSIM_PATH="/isaac-sim"
ENV ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"

# Download IsaacSim
RUN cd /tmp && \
    wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-${ISAACSIM_VERSION}.0-linux-x86_64.zip && \
    sudo unzip /tmp/isaac-sim-standalone-${ISAACSIM_VERSION}.0-linux-x86_64.zip -d ${ISAACSIM_PATH} && \
    sudo chown ${NB_USER}:users -R ${ISAACSIM_PATH} && \
    rm /tmp/isaac-sim-standalone-${ISAACSIM_VERSION}.0-linux-x86_64.zip

# Extra packages for the isaacsim python env (jupyter/ipywidgets for the
# Isaac Sim jupyter kernel, tqdm for scripts)
RUN $ISAACSIM_PYTHON_EXE -m pip install \
        tqdm \
        jupyter \
        ipywidgets

# Install jupyter kernel of Isaac-sim python
COPY --chown=${NB_USER}:users binder/isaacsim_python_wrapper.sh /home/jovyan/.local/bin/isaacsim_python_wrapper.sh
COPY --chown=${NB_USER}:users binder/kernel.json /opt/conda/share/jupyter/kernels/isaac-sim/kernel.json

# Install UV (builds the giskard venv below)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
#===========================================================#

# VSCode extension to view the simulation desktop from code-server
RUN curl -LO https://github.com/yxzhan/vscode_remote_desktop/releases/download/v0.1.0/vscode-remote-desktop-0.1.0.vsix && \
    code-server --install-extension vscode-remote-desktop-0.1.0.vsix && \
    rm vscode-remote-desktop-0.1.0.vsix

#============= Fast-DDS serialization stack upgrade =============#
# The base image pins the ROS core at an older snapshot; the Isaac Sim ROS2
# bridge needs a consistent (newer) Fast-CDR/Fast-DDS stack, otherwise node
# creation aborts inside Fast-DDS's XTypes type-cache with
#   eprosima::fastcdr::exception::BadParamException
# Upgrade the whole serialization stack together to keep fastcdr and fastrtps
# consistent. Also installs the ROS packages needed by the giskard demo
# (rclpy_message_converter, py_trees_ros).
RUN sudo apt-get update && \
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ros-jazzy-fastcdr \
        ros-jazzy-fastrtps \
        ros-jazzy-rmw-fastrtps-cpp \
        ros-jazzy-rmw-fastrtps-shared-cpp \
        ros-jazzy-rosidl-dynamic-typesupport \
        ros-jazzy-rosidl-dynamic-typesupport-fastrtps \
        ros-jazzy-rosidl-typesupport-fastrtps-cpp \
        ros-jazzy-rosidl-typesupport-fastrtps-c \
        ros-jazzy-rclpy-message-converter \
        ros-jazzy-py-trees \
        ros-jazzy-py-trees-ros && \
    sudo rm -rf /var/lib/apt/lists/*
#================================================================#

# Python packages for the jupyter env
USER ${NB_USER}
COPY --chown=${NB_USER}:users binder/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Copy repo content to the image.
# NOTE: requires the repo to be cloned with --recurse-submodules so the
# cognitive_robot_abstract_machine submodule is present in the build context.
USER ${NB_USER}
ENV REPO_DIR=${HOME}/isaacsim-giskard
RUN mkdir -p ${REPO_DIR}
COPY --chown=${NB_USER}:users . ${REPO_DIR}/
RUN git config --global --add safe.directory ${REPO_DIR}

# Build the giskard python environment (uv workspace -> .venv inside the
# submodule) and add the two packages giskardpy's ROS2 middleware needs that
# are not in its workspace lockfile.
RUN cd ${REPO_DIR}/cognitive_robot_abstract_machine && \
    ~/.local/bin/uv sync && \
    .venv/bin/pip install py_trees argcomplete

# Build the json_msgs ROS workspace (JsonAction used by giskard's action server
# and python client).
RUN cd ${REPO_DIR}/ros2_ws && \
    bash -c "source /opt/ros/jazzy/setup.bash && colcon build --packages-select json_msgs"

# Register the giskard venv as a jupyter kernel for the demo notebook.
RUN ${REPO_DIR}/cognitive_robot_abstract_machine/.venv/bin/python \
        -m ipykernel install --user --name giskard --display-name "Giskard Python (venv)"

# Set Jupyterlab default directory
WORKDIR ${REPO_DIR}
# Set VSCODE default directory
ENV CODE_WORKING_DIRECTORY=${REPO_DIR}

# The entrypoint of the docker image
COPY --chown=${NB_USER}:users binder/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
