#!/bin/bash

# Launcher page: static http server, reachable via jupyter-server-proxy at
# /user/<name>/proxy/8899/. Wrapped in a loop so it is respawned if it ever dies.
nohup bash -c 'while true; do
    python3 -m http.server 8899 --bind 127.0.0.1 --directory $PWD/demo/web_ui
    echo "[ebim-launcher] http.server exited, restarting in 2s..."
    sleep 2
done' >/tmp/ebim-launcher.log 2>&1 &

# The following line will allow the binderhub start Jupyterlab, should be at the end of the entrypoint.
exec "$@"
