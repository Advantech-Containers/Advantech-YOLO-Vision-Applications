#!/bin/bash
set -uo pipefail

# ==============================================================================
# Advantech YOLO Vision (JetPack 7 / Thor) — Self-Hydrating Entrypoint
#
# Adapted from CV_demo/Object-counting/entrypoint.sh.
#
# Hydration is a best-effort, fail-safe operation: a failure here must never
# kill the container, or the operator loses the shell needed to diagnose it.
# Failures are recorded and reported; the .ready sentinel is created only when
# the environment can genuinely run the demo.
# ==============================================================================

readonly APP_DIR="/wise-edge/advantech-yolo"
readonly READY_FILE="${APP_DIR}/.ready"
# Written when hydration definitively cannot succeed, so build.jp7.sh can report
# the failure at once instead of polling out its whole timeout budget.
readonly FAILED_FILE="${APP_DIR}/.hydration-failed"

# BYOL packages install here rather than into the image venv. The path is backed
# by a named volume (see docker-compose.jp7.yml), so a licensed install survives
# container recreation and hydration runs once per device, not once per launch.
readonly BYOL_DIR="${WEDA_CV_BYOL_DIR:-/opt/weda-cv/byol}"
readonly ULTRALYTICS_VERSION="${ULTRALYTICS_VERSION:-8.4.24}"
readonly ULTRALYTICS_THOP_VERSION="${ULTRALYTICS_THOP_VERSION:-2.0.18}"

# Clear any sentinel inherited from a previous run of this container. Doing this
# here — before hydration — is race-free; doing it from the host after the
# container starts can delete a sentinel this script has already written.
rm -f "${READY_FILE}" "${FAILED_FILE}"

HYDRATION_OK=true

# ── [1] Display Validation ────────────────────────────────────────────────────
echo "[INFO] DISPLAY=${DISPLAY:-<not set>}"
echo "[INFO] XAUTHORITY=${XAUTHORITY:-<not set>}"

if [[ -z "${DISPLAY:-}" ]]; then
    echo "[WARN] DISPLAY is not set — GUI output will be unavailable"
    echo "[WARN] Run the demo without --show, or let build.jp7.sh detect X11"
else
    DISPLAY_NUM="${DISPLAY##*:}"
    X11_SOCKET_PATH="/tmp/.X11-unix/X${DISPLAY_NUM}"

    if [[ ! -S "${X11_SOCKET_PATH}" ]]; then
        echo "[WARN] X11 socket not found: ${X11_SOCKET_PATH}"
        echo "[WARN] Available sockets: $(ls /tmp/.X11-unix/ 2>/dev/null || echo 'none')"
        echo "[WARN] GUI output will be unavailable — drop --show from the command"
    else
        echo "[INFO] X11 socket confirmed: ${X11_SOCKET_PATH}"
    fi
fi

if [[ -z "${XAUTHORITY:-}" ]]; then
    echo "[WARN] XAUTHORITY is not set — display auth will fail"
elif [[ ! -f "${XAUTHORITY}" ]]; then
    echo "[WARN] XAUTHORITY file not found: ${XAUTHORITY}"
    echo "[WARN] Volume mount may have failed — check build.jp7.sh detection output"
else
    echo "[INFO] Xauthority confirmed: ${XAUTHORITY}"
fi

# ── [2] GPU Sanity ────────────────────────────────────────────────────────────
# On Thor the driver is exposed dGPU-style (/dev/nvidia0, no /dev/nvgpu). If the
# nvidia runtime did not engage, torch sees no CUDA and the demo silently falls
# back to CPU at ~2 fps — report it here rather than let it look like a slow GPU.
if python3 -c "import torch" &> /dev/null; then
    CUDA_OK=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
    if [[ "${CUDA_OK}" == "True" ]]; then
        echo "[INFO] CUDA available: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null)"
    else
        echo "[WARN] torch.cuda.is_available() is False — inference will run on CPU."
        echo "[WARN] Check that the stack was started with runtime: nvidia and that"
        echo "[WARN] /dev/nvidia0 and /dev/nvidiactl are visible inside the container."
    fi
else
    echo "[WARN] torch not importable in the base image — cannot verify GPU."
fi

# ── [3] System Dependencies ───────────────────────────────────────────────────
if ! command -v gst-inspect-1.0 &> /dev/null || \
   ! gst-inspect-1.0 v4l2src &> /dev/null 2>&1; then
    echo "[INFO] System dependencies missing — hydrating (this may take several minutes)..."
    export DEBIAN_FRONTEND=noninteractive

    APT_LOG="/tmp/advantech-yolo-apt.log"
    if apt-get update -qq > "${APT_LOG}" 2>&1 && \
       apt-get install -y -q \
            libx11-6 libxext6 x11-utils x11-apps v4l-utils kmod \
            libcanberra-gtk-module libcanberra-gtk3-module at-spi2-core \
            gstreamer1.0-plugins-good gstreamer1.0-tools gstreamer1.0-plugins-base \
            >> "${APT_LOG}" 2>&1; then
        echo "[INFO] System dependencies ready."
    else
        HYDRATION_OK=false
        echo "[FAIL] System dependency installation failed. Last 20 lines:"
        tail -20 "${APT_LOG}" 2>/dev/null | sed 's/^/         /'
        echo "[FAIL] Full log inside the container: ${APT_LOG}"
    fi
else
    echo "[INFO] System dependencies already present."
fi

# ── [4] Ultralytics EULA Gate ─────────────────────────────────────────────────
# Resolve BYOL packages through the same interpreter that runs the demo.
export PYTHONPATH="${BYOL_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

if ! python3 -c "import ultralytics" &> /dev/null; then
    echo "[INFO] Ultralytics not installed — checking EULA acceptance..."
    echo "[INFO] ACCEPT_ULTRALYTICS_EULA=${ACCEPT_ULTRALYTICS_EULA:-<not set>}"

    if [[ "${ACCEPT_ULTRALYTICS_EULA:-}" != "true" ]]; then
        if [[ -t 0 ]]; then
            echo "----------------------------------------------------------------------"
            echo "  Ultralytics components required (BYOL)."
            echo "  To skip this prompt: set ACCEPT_ULTRALYTICS_EULA=true in .env.jp7"
            echo "----------------------------------------------------------------------"
            printf "  Accept EULA and BYOL terms? [yes/no]: "
            read -r USER_ACCEPT
            USER_ACCEPT_LOWER=$(echo "$USER_ACCEPT" | tr '[:upper:]' '[:lower:]')
            if [[ "$USER_ACCEPT_LOWER" == "yes" || "$USER_ACCEPT_LOWER" == "y" ]]; then
                ACCEPT_ULTRALYTICS_EULA=true
            else
                echo "[WARN] EULA declined — Ultralytics not installed. Demo will fail."
                ACCEPT_ULTRALYTICS_EULA=false
            fi
        else
            echo "[WARN] Non-interactive shell and ACCEPT_ULTRALYTICS_EULA != true."
            echo "[WARN] Set ACCEPT_ULTRALYTICS_EULA=true in .env.jp7 to enable auto-install."
            ACCEPT_ULTRALYTICS_EULA=false
        fi
    fi

    if [[ "${ACCEPT_ULTRALYTICS_EULA}" == "true" ]]; then
        echo "[INFO] Installing Ultralytics ${ULTRALYTICS_VERSION} into ${BYOL_DIR}..."
        mkdir -p "${BYOL_DIR}"

        # python3 -m pip (not bare pip) guarantees the install lands in the same
        # interpreter the import check above and the demo below actually use.
        # --no-deps keeps pip from pulling an x86-flavoured torch/opencv over the
        # Thor-built ones already in the base image.
        PIP_LOG="/tmp/advantech-yolo-pip.log"
        if python3 -m pip install \
                --no-cache-dir \
                --retries 5 \
                --timeout 60 \
                --no-deps \
                --target "${BYOL_DIR}" \
                --upgrade \
                "ultralytics==${ULTRALYTICS_VERSION}" \
                "ultralytics-thop==${ULTRALYTICS_THOP_VERSION}" \
                > "${PIP_LOG}" 2>&1; then
            echo "[INFO] Ultralytics install completed."
        else
            echo "[FAIL] Ultralytics install failed. Last 20 lines:"
            tail -20 "${PIP_LOG}" 2>/dev/null | sed 's/^/         /'
            echo "[FAIL] Full log inside the container: ${PIP_LOG}"
        fi

        # Trust the import, not the installer's exit code — a partially unpacked
        # target directory can leave pip happy and the demo broken.
        if python3 -c "import ultralytics" &> /dev/null; then
            echo "[INFO] Ultralytics verified: $(python3 -c 'import ultralytics; print(ultralytics.__version__)' 2>/dev/null)"
        else
            HYDRATION_OK=false
            echo "[FAIL] Ultralytics still not importable after install."
        fi
    else
        HYDRATION_OK=false
        echo "[FAIL] Ultralytics unavailable — the demo cannot run until the EULA is accepted."
    fi
else
    echo "[INFO] Ultralytics already installed: $(python3 -c 'import ultralytics; print(ultralytics.__version__)' 2>/dev/null)"
fi

# ── [5] Environment Protection Patch ─────────────────────────────────────────
# Ultralytics' check_requirements() will pip-install over the base image's
# Thor-built torch/opencv if it dislikes a version. Neutralise it.
if python3 -c "import ultralytics" &> /dev/null; then
    echo "[INFO] Applying environment protection patch..."
    python3 -c "
try:
    import ultralytics.utils.checks as c
    import inspect
    path = inspect.getfile(c)
    with open(path, 'r') as f:
        content = f.read()
    if 'WEDA-CV Protection' not in content:
        with open(path, 'a') as f:
            f.write('\n\n# WEDA-CV Protection: Disable auto-update of dependencies\n')
            f.write('check_requirements = lambda *args, **kwargs: True\n')
        print('[INFO] Protection patch applied.')
    else:
        print('[INFO] Protection patch already present — skipping.')
except Exception as e:
    print(f'[WARN] Protection patch failed: {e}')
    print('[WARN] Ultralytics may attempt to overwrite system packages.')
"
fi

# ── [6] Signal Ready ──────────────────────────────────────────────────────────
if [[ "${HYDRATION_OK}" == "true" ]]; then
    touch "${READY_FILE}"
    echo "[INFO] Environment ready. Launching..."
else
    touch "${FAILED_FILE}"
    echo "[FAIL] Hydration incomplete — not signalling ready."
    echo "[FAIL] The container stays up so the environment can be inspected."
fi
echo "----------------------------------------------------------------------"

exec "$@"
