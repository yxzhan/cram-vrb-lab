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

# LeRobot dataset visualizer (port 3000) and the dataset server it reads through
# (port 8011), both reached in the browser through jupyter-server-proxy. Started
# here for the same reason as cramera above: a fresh pod has them without anyone
# running a command, and each is kept in a restart loop.
#
# The two use DIFFERENT proxy prefixes, and that is not a detail:
#
#   viewer  -> /user/<name>/proxy/absolute/3000/   the path arrives unchanged
#   data    -> /user/<name>/proxy/8011/            the prefix is stripped
#
# Next.js can only emit correct links if it knows its public prefix -- that is
# what basePath is -- and basePath is used both to match what arrives and to build
# what is emitted, so the two have to be the same string. Only `proxy/absolute`
# delivers the path unchanged; under plain `/proxy/<port>/` the first page renders
# and then every navigation and every RSC request goes to the domain root. The
# dataset server wants the opposite: it is written against Hub-shaped paths that
# begin at /datasets/..., which is exactly what arrives once the prefix is gone.
#
# Both then come from the same origin as the notebook, so the browser sends the
# JupyterHub session cookie with each and there is no cross-origin question.
VIZ_SRC=/opt/lerobot-visualizer
VIZ_DIR="${HOME}/.lerobot-visualizer"
VIZ_PORT=3000
DATASET_PORT=8011
VIZ_RESTART_DELAY=5

if [ -d "${VIZ_SRC}" ] && [ -n "${JUPYTERHUB_SERVICE_URL}" ]; then
    # /user/<name>, from the spawner rather than from a hard-coded username.
    USER_PATH="/${JUPYTERHUB_SERVICE_URL#*://*/}"
    USER_PATH="${USER_PATH%/}"
    # VSCODE_PROXY_URI carries the public scheme and host; without it the browser
    # would be sent to a URL only the pod can resolve.
    ORIGIN="${VSCODE_PROXY_URI%%/user/*}"

    VIZ_BASE_PATH="${USER_PATH}/proxy/absolute/${VIZ_PORT}"
    VIZ_DATASET_URL="${ORIGIN}${USER_PATH}/proxy/${DATASET_PORT}/datasets"

    # basePath and DATASET_URL are compiled into the client bundle, so the image
    # was built with the placeholders below (see binder/Dockerfile) and they are
    # substituted here. Done on a fresh copy of the pristine tree every start, so
    # a restart -- or the same image run by another user -- never inherits paths
    # rewritten for somebody else.
    rm -rf "${VIZ_DIR}"
    cp -r "${VIZ_SRC}" "${VIZ_DIR}"
    grep -rl 'CRAMVRBLAB_' "${VIZ_DIR}" 2>/dev/null | while read -r f; do
        sed -i \
            -e "s|/CRAMVRBLAB_BASEPATH|${VIZ_BASE_PATH}|g" \
            -e "s|https://CRAMVRBLAB_DATASET/datasets|${VIZ_DATASET_URL}|g" \
            "$f"
    done

    (
        while true; do
            cd "${HOME}/cram-vrb-lab" && \
                python3 -m cram_vrb_lab.datasets.serve --port "${DATASET_PORT}"
            echo "dataset server exited ($?) -- restarting in ${VIZ_RESTART_DELAY}s"
            sleep "${VIZ_RESTART_DELAY}"
        done
    ) > "${HOME}/.lerobot-visualizer-data.log" 2>&1 &

    (
        while true; do
            cd "${VIZ_DIR}" && PORT="${VIZ_PORT}" HOSTNAME=0.0.0.0 node server.js
            echo "dataset visualizer exited ($?) -- restarting in ${VIZ_RESTART_DELAY}s"
            sleep "${VIZ_RESTART_DELAY}"
        done
    ) > "${HOME}/.lerobot-visualizer.log" 2>&1 &
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
