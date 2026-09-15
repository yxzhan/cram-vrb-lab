#!/usr/bin/env bash
#
# cram-vrb-lab launcher.
#
#   bash <(curl -fsSL https://raw.githubusercontent.com/yxzhan/cram-vrb-lab/dev/install.sh)
#
# Resolves the newest commit on the repo's dev branch that has a pre-built
# image on Docker Hub, pulls it, and drops you into a shell inside it, with GPU
# and X11 access, at the repo root -- the demos are started from there by hand.
#
set -euo pipefail

GITHUB_REPO="${GITHUB_REPO:-yxzhan/cram-vrb-lab}"
IMAGE_REPO="${IMAGE_REPO:-intel4coro/yxzhan-2dcram-2dvrb-2dlab-eb909a}"
REF="${REF:-dev}"
IMAGE_TAG="${IMAGE_TAG:-}"
PORT="${PORT:-8888}"
CRAMERA_PORT="${CRAMERA_PORT:-8711}"
BRIDGE_PORT="${BRIDGE_PORT:-8765}"
CACHE_DIR="${CACHE_DIR:-$HOME/isaac_cache}"
PULL=1

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[1;33m warn:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m error:\033[0m %s\n' "$*" >&2; exit 1; }

usage() {
  cat >&2 <<USAGE
Usage: install.sh [options]

  --ref <branch>   branch to track (default: $REF)
  --tag <sha>      pin an exact image tag, skipping resolution
  --port <port>    host port for JupyterLab, for when you start one inside
                   the container (default: $PORT)
  --cramera-port <port>
                   host port for the cramera viewer (default: $CRAMERA_PORT)
  --bridge-port <port>
                   host port for the live demo bridge (default: $BRIDGE_PORT)
  --no-pull        skip 'docker pull' (use an already-downloaded image)
  -h, --help       show this help

Every option also has an env var: REF, IMAGE_TAG, PORT, CRAMERA_PORT,
BRIDGE_PORT, CACHE_DIR, IMAGE_REPO.
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --ref)     REF="${2:?--ref needs a value}"; shift 2 ;;
    --tag)     IMAGE_TAG="${2:?--tag needs a value}"; shift 2 ;;
    --port)    PORT="${2:?--port needs a value}"; shift 2 ;;
    --cramera-port) CRAMERA_PORT="${2:?--cramera-port needs a value}"; shift 2 ;;
    --bridge-port)  BRIDGE_PORT="${2:?--bridge-port needs a value}"; shift 2 ;;
    --no-pull) PULL=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *)         usage; die "unknown option: $1" ;;
  esac
done

# --- resolve the image tag ---------------------------------------------------

# Commit shas on $REF, newest first. The atom feed needs no token and no JSON
# parser; the API is the fallback when it is unavailable.
list_commits() {
  curl -fsSL "https://github.com/${GITHUB_REPO}/commits/${REF}.atom" 2>/dev/null \
    | grep -o 'Commit/[0-9a-f]\{40\}' | sed 's|Commit/||' && return 0
  curl -fsSL -H 'Accept: application/vnd.github+json' \
    "https://api.github.com/repos/${GITHUB_REPO}/commits/${REF}" 2>/dev/null \
    | grep -o '"sha": *"[0-9a-f]\{40\}"' | head -1 | grep -o '[0-9a-f]\{40\}'
}

# The images are pushed by BinderHub, so the newest commit is not always built
# yet; walk back until one is.
tag_exists() {
  curl -o /dev/null -fsS \
    "https://hub.docker.com/v2/repositories/${IMAGE_REPO}/tags/$1" 2>/dev/null
}

resolve_tag() {
  local commits head sha
  log "Resolving the latest built image on '${REF}'..."
  commits="$(list_commits)" || true
  [ -n "$commits" ] || die "could not read commits of '${REF}' from github.com/${GITHUB_REPO}"
  head="$(printf '%s\n' "$commits" | head -1)"

  for sha in $commits; do
    if tag_exists "$sha"; then
      if [ "$sha" != "$head" ]; then
        warn "${REF} is at ${head:0:7}, but its image is not published yet."
        warn "Falling back to ${sha:0:7}, the newest commit that has one."
      fi
      printf '%s\n' "$sha"
      return 0
    fi
  done
  die "no image found on Docker Hub for the last $(printf '%s\n' "$commits" | wc -l) commits of '${REF}'"
}

# --- ports -------------------------------------------------------------------

# 8888 JupyterLab, 8711 the cramera viewer the image's entrypoint starts, 8765 the
# bridge a live demo opens. All three are published, so a port already taken on the
# host would only surface as an obscure 'docker run' failure -- say which one it is,
# and how to move it, before pulling 50 GB.
port_in_use() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${port}\$"
  elif command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1
  elif command -v netstat >/dev/null 2>&1; then
    netstat -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${port}\$"
  else
    # No listener tool available: a successful connect still proves it is taken.
    (exec 3<>"/dev/tcp/127.0.0.1/${port}") >/dev/null 2>&1
  fi
}

port_holder() {
  local port="$1" holder=""
  if command -v lsof >/dev/null 2>&1; then
    holder="$(lsof -nP -iTCP:"${port}" -sTCP:LISTEN -F c 2>/dev/null \
      | sed -n 's/^c//p' | sort -u | paste -sd, -)"
  elif command -v ss >/dev/null 2>&1; then
    holder="$(ss -ltnp 2>/dev/null | grep -E "[:.]${port} " \
      | grep -o '"[^"]*"' | tr -d '"' | sort -u | paste -sd, -)"
  fi
  printf '%s' "$holder"
}

check_port() {
  local port="$1" what="$2" flag="$3" holder
  case "$port" in
    ''|*[!0-9]*) die "invalid ${what} port: '${port}'" ;;
  esac
  port_in_use "$port" || return 0
  holder="$(port_holder "$port")"
  die "port ${port} (${what}) is already in use${holder:+ by ${holder}} -- \
free it, or pick another one with '${flag} <port>'."
}

[ "$PORT" != "$CRAMERA_PORT" ] && [ "$PORT" != "$BRIDGE_PORT" ] \
  && [ "$CRAMERA_PORT" != "$BRIDGE_PORT" ] \
  || die "the three host ports must differ: JupyterLab ${PORT}, cramera ${CRAMERA_PORT}, bridge ${BRIDGE_PORT}"

check_port "$PORT" "JupyterLab" --port
check_port "$CRAMERA_PORT" "cramera viewer" --cramera-port
check_port "$BRIDGE_PORT" "live demo bridge" --bridge-port

# --- preflight ---------------------------------------------------------------

command -v curl >/dev/null || die "curl is required"
command -v docker >/dev/null || \
  die "docker is required: https://docs.docker.com/engine/install/"

DOCKER=(docker)
if ! docker info >/dev/null 2>&1; then
  command -v sudo >/dev/null || die "cannot talk to the docker daemon, and sudo is not available"
  DOCKER=(sudo docker)
  "${DOCKER[@]}" info >/dev/null 2>&1 || die "cannot talk to the docker daemon, even with sudo"
fi

command -v nvidia-smi >/dev/null || \
  warn "nvidia-smi not found -- Isaac Sim needs an NVIDIA RTX GPU with drivers installed."
command -v nvidia-ctk >/dev/null || \
  warn "NVIDIA Container Toolkit not found -- '--gpus all' will fail without it: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"

free_gb="$(df -PBG "$(dirname "$CACHE_DIR")" 2>/dev/null | awk 'NR==2 {gsub(/G/,"",$4); print $4}')"
if [ -n "${free_gb:-}" ] && [ "$free_gb" -lt 50 ] 2>/dev/null; then
  warn "only ${free_gb} GB free on $(dirname "$CACHE_DIR") -- the image needs about 50 GB."
fi

[ -n "$IMAGE_TAG" ] || IMAGE_TAG="$(resolve_tag)"
IMAGE="${IMAGE_REPO}:${IMAGE_TAG}"
log "Image: ${IMAGE}"

mkdir -p "$CACHE_DIR"

if [ -n "${DISPLAY:-}" ]; then
  if command -v xhost >/dev/null; then
    xhost +local:root >/dev/null || warn "xhost failed -- the GUI windows may not open."
  else
    warn "xhost not found -- the GUI windows may not open."
  fi
else
  warn "DISPLAY is not set -- running without a GUI; use the JupyterLab desktop instead."
fi

if [ "$PULL" -eq 1 ]; then
  log "Pulling the image (about 50 GB, this takes a while)..."
  "${DOCKER[@]}" pull "$IMAGE"
fi

# --- run ---------------------------------------------------------------------

log "Web UI: http://localhost:${CRAMERA_PORT}/  (live bridge on :${BRIDGE_PORT})"
log "Starting a shell in the container. Run a demo with:"
log "  binder/cram_python_wrapper.sh demos/garmi_demo.py"
log "Ctrl-D to leave (the container is removed)."

# demos/garmi_demo.py only defaults these to 1. With an X server there is a real
# Isaac window to render into, so turn both off; without one, leave them alone
# and let the demo fall back to headless + WebRTC.
isaac_env=()
if [ -n "${DISPLAY:-}" ]; then
  isaac_env=(--env ISAAC_HEADLESS=0 --env ISAAC_LIVESTREAM=0)
fi

run_args=(
  run --rm --gpus all
  --user root
  --env NVIDIA_DRIVER_CAPABILITIES=all
  --env ACCEPT_EULA=YES
  --env PRIVACY_CONSENT=YES
  --env OMNI_KIT_ACCEPT_EULA=YES
  --env OMNI_KIT_ALLOW_ROOT=1
  --env "DISPLAY=${DISPLAY:-}"
  "${isaac_env[@]}"
  -v /tmp/.X11-unix:/tmp/.X11-unix
  -v /usr/share/vulkan/icd.d/:/etc/vulkan/icd.d
  -v "${CACHE_DIR}:/isaac-sim/kit/cache"
  -p "${PORT}:8888"
  -p "${CRAMERA_PORT}:8711"
  -p "${BRIDGE_PORT}:8765"
)

# An interactive shell, not a server: the image's WORKDIR is the repo root and its
# entrypoint ends in `exec "$@"`, so this is a normal bash with the image's full
# environment -- and the entrypoint has already started the cramera viewer behind it.
#
# When this script is piped into bash, stdin is the pipe rather than the terminal;
# hand docker the real tty instead. Without one there is nothing to attach the shell
# to, so say that rather than starting a container that exits immediately.
[ -e /dev/tty ] && [ -t 1 ] || \
  die "no terminal to attach the shell to -- run install.sh from an interactive terminal:
bash <(curl -fsSL https://raw.githubusercontent.com/${GITHUB_REPO}/${REF}/install.sh)"

run_args+=(-it "$IMAGE" bash)
exec "${DOCKER[@]}" "${run_args[@]}" </dev/tty
