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
COPY --chown=${NB_USER}:users binder/isaac-kernel.json /opt/conda/share/jupyter/kernels/isaac-sim/kernel.json

# Install UV (builds the giskard venv below)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
#===========================================================#

# VSCode extension to view the simulation desktop from code-server
RUN curl -LO https://github.com/yxzhan/vscode_remote_desktop/releases/download/v0.1.0/vscode-remote-desktop-0.1.0.vsix && \
    code-server --install-extension vscode-remote-desktop-0.1.0.vsix && \
    rm vscode-remote-desktop-0.1.0.vsix

# Python packages for the jupyter env (rarely changes).
USER ${NB_USER}
COPY --chown=${NB_USER}:users binder/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

ENV REPO_DIR=${HOME}/cram-vrb-lab
RUN mkdir -p ${REPO_DIR}

#============= Pinned CRAM stack (deps + ROS workspace) =============#
# The two builds below are expensive but depend only on the version-pinned
# cognitive_robot_abstract_machine submodule, so they stay cached when only the
# top-level demo scripts change (those are copied last, further down). The
# submodule is copied ALONE here; the rest of the repo is copied afterwards.
# NOTE: requires the repo to be cloned with --recurse-submodules so the
# submodule is present in the build context.
COPY --chown=${NB_USER}:users cognitive_robot_abstract_machine ${REPO_DIR}/cognitive_robot_abstract_machine
RUN git config --global --add safe.directory ${REPO_DIR}/cognitive_robot_abstract_machine

# CRAM python environment (uv workspace -> .venv inside the submodule) plus the
# two packages giskardpy's ROS2 middleware needs that are not in its lockfile.
RUN cd ${REPO_DIR}/cognitive_robot_abstract_machine && \
    ~/.local/bin/uv sync && \
    .venv/bin/pip install nest_asyncio argcomplete usd-core

# Full CRAM ROS 2 workspace (overlay at ${REPO_DIR}/ros2_ws). The setup script
# lives in the submodule and clones the code-iai / cram2 description and message
# repos, colcon-building them with --merge-install. Notably it brings in:
#   - iai_maps -> iai_apartment / iai_kitchen, whose meshes apartment.urdf needs
#     (without them the apartment collision geometry cannot be resolved),
#   - cram_ros2_packages, which provides json_msgs (JsonAction) used by giskard's
#     action server and python client.
# git-lfs is initialized before the clones so the mesh LFS objects are smudged in
# (otherwise the mesh files are left as LFS pointer stubs). The host's ros2_ws is
# excluded via .dockerignore and rebuilt fresh here, so there is no hand-made
# json_msgs to collide with the one in cram_ros2_packages.
RUN sudo apt-get update && \
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git-lfs && \
    git lfs install && \
    OVERLAY_WS=${REPO_DIR}/ros2_ws \
        bash ${REPO_DIR}/cognitive_robot_abstract_machine/.github/docker/setup_ros_workspace.sh && \
    sudo rm -rf /var/lib/apt/lists/*
#===================================================================#

# Install jupyter kernel of the CRAM venv python. The wrapper sources the ROS 2
# environment and the ros2_ws overlay, so notebooks on this kernel can import
# rclpy/json_msgs without the jupyter server having sourced anything.
COPY --chown=${NB_USER}:users binder/cram_python_wrapper.sh /home/jovyan/.local/bin/cram_python_wrapper.sh
COPY --chown=${NB_USER}:users binder/cram-kernel.json /opt/conda/share/jupyter/kernels/cram/kernel.json

# Pinned assets (Isaac USD scenes, stretch_urdf submodule) - rarely change.
COPY --chown=${NB_USER}:users assets ${REPO_DIR}/assets

# Frequently-changing top-level content (package, demos, wrappers).
# Copied LAST so editing them reuses every cached layer above.
COPY --chown=${NB_USER}:users cram_vrb_lab ${REPO_DIR}/cram_vrb_lab
COPY --chown=${NB_USER}:users demos ${REPO_DIR}/demos
COPY --chown=${NB_USER}:users binder ${REPO_DIR}/binder
COPY --chown=${NB_USER}:users README.md pyproject.toml ${REPO_DIR}/
RUN git config --global --add safe.directory ${REPO_DIR}

# Set Jupyterlab default directory
WORKDIR ${REPO_DIR}
# Set VSCODE default directory
ENV CODE_WORKING_DIRECTORY=${REPO_DIR}

# The entrypoint of the docker image
COPY --chown=${NB_USER}:users binder/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
