#!/usr/bin/env bash
# ==============================================================================
# Advantech YOLO Vision (JetPack 7 / Jetson Thor) — Build & Launch Script
#
# Adapted from CV_demo/Object-counting/build.sh. The one structural difference:
# that demo PULLS a prebuilt image; this one BUILDS from Dockerfile.jp7, because
# the JP7 image for this repo does not exist in a registry yet.
#
# Usage:
#   ./build.jp7.sh                  # detect demo (default)
#   ./build.jp7.sh --task segment   # segment demo
#   ./build.jp7.sh --task classify  # classify demo
#   ./build.jp7.sh --task all       # build all three images, start none
#   ./build.jp7.sh --run    # Start only (skip build, use existing image)
#   ./build.jp7.sh --shell  # Attach to an already-running container
#   ./build.jp7.sh --stop   # Stop and remove the container
#   ./build.jp7.sh --clean  # Stop and also purge the cached BYOL install
#   ./build.jp7.sh --help
# ==============================================================================
set -euo pipefail
IFS=$'\n\t'

# ── Path Anchoring ────────────────────────────────────────────────────────────
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "${SCRIPT_DIR}"

# ── Logging ───────────────────────────────────────────────────────────────────
log_info()    { echo -e "\e[1;36m[INFO]\e[0m  $1"; }
log_success() { echo -e "\e[1;32m[  OK]\e[0m  $1"; }
log_warn()    { echo -e "\e[1;33m[WARN]\e[0m  $1"; }
log_error()   { echo -e "\e[1;31m[FAIL]\e[0m  $1" >&2; exit 1; }

# ── Parse Flags ───────────────────────────────────────────────────────────────
MODE="build"
TASK="detect"
while [[ $# -gt 0 ]]; do
    arg="$1"
    case "$arg" in
        --task)
            TASK="${2:-}"
            [[ -n "${TASK}" ]] || log_error "--task needs a value: detect|segment|classify|all"
            shift 2; continue ;;
        --task=*) TASK="${arg#*=}"; shift; continue ;;
        --run)    MODE="run" ;;
        --shell)  MODE="shell" ;;
        --stop)   MODE="stop" ;;
        --clean)  MODE="clean" ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "  (no flag)   Build the image, start the container, open a shell"
            echo "  --run       Start the container (skip build, use existing image)"
            echo "  --shell     Attach a shell to the already-running container"
            echo "  --stop      Stop and remove the container"
            echo "  --clean     Stop, remove, and purge the cached BYOL install"
            echo "              (forces a full re-download on the next launch)"
            echo "  --task T    detect (default) | segment | classify | all"
            echo "              'all' builds all three images and starts nothing"
            echo "  --help      Show this help message"
            exit 0 ;;
        *) log_error "Unknown option: ${arg} (see --help)" ;;
    esac
    shift
done

case "${TASK}" in
    detect|segment|classify|all) ;;
    *) log_error "Invalid --task '${TASK}': expected detect|segment|classify|all" ;;
esac

# ── Static Config ─────────────────────────────────────────────────────────────
# There is no .env file. docker-compose.jp7.yml is self-contained so that a
# single file can be shipped to an edge device; these values mirror its
# ${VAR:-default} defaults and exist only for this script's own bookkeeping
# (container lookups, volume purge, hydration polling). Change a default in the
# compose file and mirror it here.
readonly DOCKERFILE="${SCRIPT_DIR}/Dockerfile.jp7"
readonly APP_DIR="/wise-edge/advantech-yolo"
readonly IMAGE_BASE="harbor.edgesync.cloud/weda-ai/advantech-yolo-vision"

# Each task is its own image, carrying only that task's weights. Keep these in
# step with the per-task compose files' ${VAR:-default} values.
weights_for() {
    case "$1" in
        detect)   echo "yolo11n.pt" ;;
        segment)  echo "yolo11n-seg.pt" ;;
        classify) echo "yolo11n-cls.pt" ;;
    esac
}

if [[ "${TASK}" == "detect" ]]; then
    readonly COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.jp7.yml"
    readonly CONTAINER_NAME="${CONTAINER_NAME:-advantech-yolo-jp7}"
    readonly BYOL_VOLUME="${BYOL_VOLUME:-advantech-yolo-jp7-byol}"
else
    readonly COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.jp7-${TASK}.yml"
    readonly CONTAINER_NAME="${CONTAINER_NAME:-advantech-yolo-jp7-${TASK}}"
    readonly BYOL_VOLUME="${BYOL_VOLUME:-advantech-yolo-jp7-${TASK}-byol}"
fi
readonly YOLO_JP7_IMAGE="${YOLO_JP7_IMAGE:-${IMAGE_BASE}:jp7-${TASK}}"
readonly SERVICE_NAME="${SERVICE_NAME:-advantech-yolo}"
readonly HYDRATION_TIMEOUT="${HYDRATION_TIMEOUT:-300}"

[[ -f "${COMPOSE_FILE}" ]] || log_error "docker-compose.jp7.yml not found at ${COMPOSE_FILE}"

# ── Handle --stop / --clean ───────────────────────────────────────────────────
if [[ "${MODE}" == "stop" || "${MODE}" == "clean" ]]; then
    echo -e "\033[1;34m[1/1] Stopping container...\033[0m"
    if docker ps -a -q -f name="${CONTAINER_NAME}" | grep -q .; then
        # Compose still interpolates these even for `down`; give it defaults so
        # it does not abort on an unset variable.
        export X11_SOCKET=${X11_SOCKET:-/tmp/.X11-unix}
        export XAUTHORITY=${XAUTHORITY:-/tmp/.Xauthority}
        export DISPLAY=${DISPLAY:-:0}
        docker compose -f "${COMPOSE_FILE}" down --remove-orphans
        log_success "Container stopped and removed."
    else
        log_warn "Container '${CONTAINER_NAME}' is not running."
    fi

    if [[ "${MODE}" == "clean" ]]; then
        if docker volume inspect "${BYOL_VOLUME}" &> /dev/null; then
            docker volume rm "${BYOL_VOLUME}" > /dev/null \
                && log_success "BYOL cache purged: ${BYOL_VOLUME}" \
                || log_warn "Could not remove volume '${BYOL_VOLUME}' — still in use?"
            log_info "The next launch will re-download Ultralytics."
        else
            log_warn "BYOL cache '${BYOL_VOLUME}' does not exist — nothing to purge."
        fi
    fi
    exit 0
fi

# ── Handle --shell ────────────────────────────────────────────────────────────
if [[ "${MODE}" == "shell" ]]; then
    echo -e "\033[1;34m[1/1] Attaching to running container...\033[0m"
    if docker ps -q -f name="${CONTAINER_NAME}" | grep -q .; then
        log_success "Entering container '${CONTAINER_NAME}'..."
        docker exec -it "${CONTAINER_NAME}" /bin/bash || true
    else
        log_error "Container '${CONTAINER_NAME}' is not running. Start it with: ./build.jp7.sh"
    fi
    exit 0
fi

# ══════════════════════════════════════════════════════════════════════════════
# Full pipeline: preflight → X11 → EULA → build → launch → hydrate → shell
# ══════════════════════════════════════════════════════════════════════════════
TOTAL_STEPS=5
[[ "${MODE}" == "run" ]] && TOTAL_STEPS=4

# ── [1/N] Pre-flight Checks ───────────────────────────────────────────────────
echo -e "\033[1;34m[1/${TOTAL_STEPS}] Pre-flight checks...\033[0m"

command -v docker &> /dev/null || log_error "docker not found."
docker compose version &> /dev/null || log_error "docker compose not found."
docker info &> /dev/null || log_error "Docker daemon not running."
log_success "Docker and Compose ready."

# Thor sanity: the JP6 iGPU nodes do not exist here. Warn early rather than let
# the container start and silently run on CPU.
if [[ ! -e /dev/nvidia0 && ! -e /dev/nvidiactl ]]; then
    log_warn "Neither /dev/nvidia0 nor /dev/nvidiactl is present on this host."
    log_warn "This does not look like a JetPack 7 / Thor device — GPU inference will fail."
fi
if [[ -e /etc/cdi/nvidia.yaml ]] && ! grep -q gstreamer /etc/cdi/nvidia.yaml 2>/dev/null; then
    log_info "CDI mode detected with no GStreamer entries — the compose file's"
    log_info "/opt/nvgst + /opt/nvhostlibs mounts cover this (TROUBLESHOOTING §9.11)."
fi

mkdir -p "${SCRIPT_DIR}/results"
log_success "Output directory ready: ${SCRIPT_DIR}/results"

# ── [2/N] X11 Display — Auto-Detection ───────────────────────────────────────
echo -e "\n\033[1;34m[2/${TOTAL_STEPS}] Detecting X11 Display...\033[0m"
export X11_SOCKET=/tmp/.X11-unix

if [[ -n "${DISPLAY:-}" ]]; then
    log_info "DISPLAY already set in shell: ${DISPLAY} — skipping auto-detect"
else
    [[ -d "${X11_SOCKET}" ]] || log_error "X11 socket directory not found: ${X11_SOCKET}"

    mapfile -t X11_SOCKETS < <(find "${X11_SOCKET}" -name "X[0-9]*" -type s 2>/dev/null | sort)
    [[ ${#X11_SOCKETS[@]} -eq 0 ]] && log_error "No X11 sockets found in ${X11_SOCKET}"

    DETECTED_NUM="$(basename "${X11_SOCKETS[0]}")"
    export DISPLAY=":${DETECTED_NUM#X}"
    log_info "Selected ${X11_SOCKETS[0]} → DISPLAY=${DISPLAY}"

    if [[ ${#X11_SOCKETS[@]} -gt 1 ]]; then
        log_warn "Multiple X11 sockets present; picked the lowest."
        log_warn "To override: export DISPLAY=:N before running build.jp7.sh"
    fi
fi

X11_SOCKET_PATH="${X11_SOCKET}/X${DISPLAY##*:}"
[[ -S "${X11_SOCKET_PATH}" ]] || log_error "X11 socket does not exist: ${X11_SOCKET_PATH}
       DISPLAY=${DISPLAY} but no matching socket found.
       Available: $(ls ${X11_SOCKET}/ 2>/dev/null || echo 'none')"

# ── XAUTHORITY detection ──────────────────────────────────────────────────────
if [[ -n "${XAUTHORITY:-}" && -f "${XAUTHORITY}" ]]; then
    log_info "XAUTHORITY already set in shell: ${XAUTHORITY} — skipping auto-detect"
else
    DETECTED_XAUTH=""
    for candidate in \
        "/run/user/$(id -u)/gdm/Xauthority" \
        "/run/user/$(id -u)/.mutter-Xwaylandauth."* \
        "${HOME}/.Xauthority" \
        "/var/run/lightdm/root/${DISPLAY}"
    do
        if [[ -f "${candidate}" ]]; then
            DETECTED_XAUTH="${candidate}"
            log_info "Found XAUTHORITY at: ${candidate}"
            break
        fi
    done

    [[ -z "${DETECTED_XAUTH}" ]] && log_error "Could not detect XAUTHORITY.
       Export it manually: export XAUTHORITY=/path/to/.Xauthority"

    export XAUTHORITY="${DETECTED_XAUTH}"
fi

[[ -s "${XAUTHORITY}" ]] || log_error "XAUTHORITY file is empty: ${XAUTHORITY}"

command -v xhost &> /dev/null && xhost +local:docker > /dev/null 2>&1 || true
log_success "X11 ready — DISPLAY=${DISPLAY} | XAUTHORITY=${XAUTHORITY}"

# ── [3/N] EULA / BYOL License Gate ───────────────────────────────────────────
echo -e "\n\033[1;34m[3/${TOTAL_STEPS}] BYOL License Acceptance...\033[0m"

if [[ "${ACCEPT_ULTRALYTICS_EULA:-false}" == "true" ]]; then
    log_success "ACCEPT_ULTRALYTICS_EULA=true (set in the environment) — skipping prompt."
else
    echo -e "\033[1;36m╔══════════════════════════════════════════════════════════════════════╗\033[0m"
    echo -e "\033[1;36m║             Ultralytics Component License (BYOL)                     ║\033[0m"
    echo -e "\033[1;36m╚══════════════════════════════════════════════════════════════════════╝\033[0m"
    echo ""
    echo "  Ultralytics is not redistributed in this image. It is installed at"
    echo "  runtime into a local volume once you accept its licence terms."
    echo ""
    printf "  Accept EULA and BYOL terms? [yes/no]: "
    # Under `set -e` a bare `read` at EOF aborts with no diagnostic — which is
    # exactly what a piped or unattended invocation always hits.
    if ! read -r USER_ACCEPT; then
        echo ""
        log_error "No input available — this shell is not interactive.
       Export ACCEPT_ULTRALYTICS_EULA=true to accept unattended."
    fi
    USER_ACCEPT_LOWER=$(echo "${USER_ACCEPT}" | tr '[:upper:]' '[:lower:]')
    if [[ "${USER_ACCEPT_LOWER}" == "yes" || "${USER_ACCEPT_LOWER}" == "y" ]]; then
        export ACCEPT_ULTRALYTICS_EULA=true
        log_success "EULA accepted."
    else
        export ACCEPT_ULTRALYTICS_EULA=false
        log_warn "EULA declined — Ultralytics will not be installed and the demo will fail."
    fi
fi

# ── [4/N] Build ───────────────────────────────────────────────────────────────
STEP=4
if [[ "${MODE}" == "build" ]]; then
    echo -e "\n\033[1;34m[${STEP}/${TOTAL_STEPS}] Building image from Dockerfile.jp7...\033[0m"
    [[ -f "${DOCKERFILE}" ]] || log_error "Dockerfile.jp7 not found at ${DOCKERFILE}"

    build_task() {
        local t="$1"
        log_info "Building ${IMAGE_BASE}:jp7-${t} (weights: $(weights_for "$t"))"
        docker build -f "${DOCKERFILE}" \
            --build-arg "YOLO_TASK=${t}" \
            --build-arg "YOLO_WEIGHTS=$(weights_for "$t")" \
            -t "${IMAGE_BASE}:jp7-${t}" "${SCRIPT_DIR}"
    }

    if [[ "${TASK}" == "all" ]]; then
        for t in detect segment classify; do
            build_task "$t" || log_error "Build failed for task '${t}'."
        done
        log_success "All three images built. Nothing started (--task all builds only)."
        exit 0
    fi

    if build_task "${TASK}"; then
        log_success "Image built: ${YOLO_JP7_IMAGE}"
    elif docker image inspect "${YOLO_JP7_IMAGE}" &> /dev/null; then
        log_warn "Build failed — falling back to the existing local image."
    else
        log_error "Build failed and no local copy of '${YOLO_JP7_IMAGE}' exists.
       The base image harbor.edgesync.cloud/weda-ai/weda-cv:1.0.0-thor must be
       reachable — check network access and 'docker login' credentials."
    fi
    STEP=$((STEP + 1))
fi

# ── [N/N] Launch ──────────────────────────────────────────────────────────────
echo -e "\n\033[1;34m[${STEP}/${TOTAL_STEPS}] Launching container...\033[0m"

# Let Compose decide whether to reuse or recreate. Force-removing the container
# on every run discards the hydrated layer, forcing a full apt + BYOL re-install
# each launch.
docker compose -f "${COMPOSE_FILE}" \
    up -d --remove-orphans "${SERVICE_NAME}"

log_success "Container '${CONTAINER_NAME}' is live."

# ── Wait for Hydration ────────────────────────────────────────────────────────
# The .ready sentinel is cleared by entrypoint.jp7.sh before it hydrates.
# Clearing it from the host here would race with the entrypoint.
echo "  Waiting for environment hydration (first run downloads dependencies)..."

TIMEOUT="${HYDRATION_TIMEOUT:-300}"
READY=false
DEAD=false
FAILED=false
elapsed=0

while [[ ${elapsed} -lt ${TIMEOUT} ]]; do
    if docker exec "${CONTAINER_NAME}" ls "${APP_DIR}/.ready" &> /dev/null; then
        READY=true
        printf "\r  [\033[1;32m%s\033[0m] 100%%\n" "$(printf "%*s" 50 | tr ' ' '#')"
        break
    fi

    # The entrypoint reports an unrecoverable failure explicitly — surface it now
    # rather than making the operator sit through the rest of the budget.
    if docker exec "${CONTAINER_NAME}" ls "${APP_DIR}/.hydration-failed" &> /dev/null; then
        FAILED=true
        printf "\r%*s\r" 70 ""
        break
    fi

    # Fail fast instead of burning the full budget on a container that has died.
    if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null)" != "true" ]]; then
        DEAD=true
        printf "\r%*s\r" 70 ""
        break
    fi

    pct=$(( elapsed * 100 / TIMEOUT ))
    hashes=$(( (pct * 50) / 100 ))
    bar="$(printf "%*s" ${hashes} "" | tr ' ' '#')"
    empty="$(printf "%*s" $(( 50 - hashes )) "")"
    printf "\r  [\033[1;36m%s\033[0m%s] %3d%% " "${bar}" "${empty}" "${pct}"
    sleep 2
    elapsed=$(( elapsed + 2 ))
done

# ── Hydration Failure — Report, Do Not Hand Over a Broken Shell ───────────────
if [[ "${READY}" != "true" ]]; then
    if [[ "${FAILED}" == "true" ]]; then
        echo -e "  [\033[1;31mFAILED\033[0m]"
        log_warn "Hydration reported an unrecoverable failure after ${elapsed}s."
    elif [[ "${DEAD}" == "true" ]]; then
        echo -e "  [\033[1;31mEXITED\033[0m]"
        log_warn "The container stopped during hydration."
    else
        echo -e "\n  [\033[1;31mTIMEOUT\033[0m]"
        log_warn "Hydration did not complete within ${TIMEOUT}s."
        log_warn "Raise HYDRATION_TIMEOUT in the environment if this device is on a slow link."
    fi

    echo ""
    log_info "Last 30 lines of container log:"
    docker logs --tail 30 "${CONTAINER_NAME}" 2>&1 | sed 's/^/           /' || true
    echo ""
    log_warn "The demo is NOT ready — running it now would fail on missing dependencies."
    log_warn "Full log:  docker logs ${CONTAINER_NAME}"
    log_warn "Inspect:   ./build.jp7.sh --shell"
    log_warn "Reset:     ./build.jp7.sh --clean && ./build.jp7.sh"
    exit 1
fi

log_success "Environment hydrated."

# ── Shell ─────────────────────────────────────────────────────────────────────
echo -e "\n\033[1;33m╔══════════════════════════════════════════════════════════╗\033[0m"
echo -e "\033[1;33m║   Advantech YOLO Vision (JP7 / Thor) — Shell Ready        ║\033[0m"
echo -e "\033[1;33m╚══════════════════════════════════════════════════════════╝\033[0m"
echo -e "\n💡 \033[1mRun demo:\033[0m  \033[1;32mpython3 src/advantech-yolo.py \\"
echo -e "              --task ${YOLO_TASK} --model ${YOLO_MODEL} \\"
echo -e "              --input ${YOLO_INPUT} --conf ${YOLO_CONF} \\"
echo -e "              --device ${YOLO_DEVICE} --show --save --loop\033[0m"
echo -e "🖥  \033[1mInteractive:\033[0m \033[1;36mpython3 src/advantech-yolo.py\033[0m"
echo -e "📁 \033[1mResults:\033[0m   ${SCRIPT_DIR}/results\n"

docker exec -it "${CONTAINER_NAME}" /bin/bash || true
