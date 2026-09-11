#!/bin/bash

# cramera viewer, reached in the browser through jupyter-server-proxy at
# <hub>/user/<name>/proxy/8711/. Started here so a fresh pod has it without anyone
# running a command, and kept in a restart loop: a viewer that died once -- a broken
# scene bundle, the OOM killer -- would otherwise stay dead for the life of the pod
# with nothing watching it. The port is fixed because the viewer's live panel derives
# the demo bridge's URL from the /proxy/<port>/ prefix it was itself served under.
#
# Backgrounded as a whole, so nothing here can delay or block the jupyter server this
# entrypoint execs into, and every failure is a line in the log rather than a pod that
# does not come up. The wrapper sources the ROS 2 environment, which the live panel's
# marker overlay needs.
CRAMERA_PYTHON="${PWD}/binder/cram_python_wrapper.sh"
CRAMERA_LOG="${HOME}/.cramera/viewer.log"
CRAMERA_PORT=8711
CRAMERA_RESTART_DELAY=5

if [ -x "${CRAMERA_PYTHON}" ]; then
    mkdir -p "$(dirname "${CRAMERA_LOG}")"
    (
        while true; do
            "${CRAMERA_PYTHON}" -m cramera.server "${CRAMERA_PORT}" --no-browser
            echo "cramera viewer exited ($?) -- restarting in ${CRAMERA_RESTART_DELAY}s"
            sleep "${CRAMERA_RESTART_DELAY}"
        done
    ) > "${CRAMERA_LOG}" 2>&1 &
fi

# Which commit this pod runs, for the link in the workspace page's header. The page is
# static and the image carries no .git (see .dockerignore), so the ready-made url
# BinderHub puts in the environment is written next to the page for it to read.
if [ -n "${BINDER_REF_URL}" ]; then
    printf '{"url": "%s"}\n' "${BINDER_REF_URL}" \
        > "${HOME}/cram-vrb-lab/demos/web_ui/version.json" 2>/dev/null
fi

# The following line will allow the binderhub start Jupyterlab, should be at the end of the entrypoint.
exec "$@"
