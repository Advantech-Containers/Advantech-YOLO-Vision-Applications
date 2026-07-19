#!/bin/bash
# ==========================================================================
# Entrypoint for the Advantech YOLO11 looping object-detection demo
# ==========================================================================
# Waits for the device's X server to accept connections before starting the
# demo.  On a device reboot the container can be started by Docker before gdm
# has brought up the display; without this wait the demo would exit on a
# missing DISPLAY and, under `restart: unless-stopped`, enter a crash loop.
#
# Copyright (c) 2026 Advantech Corporation. All rights reserved.
# ==========================================================================

set -euo pipefail

X_WAIT_TIMEOUT_SEC="${X_WAIT_TIMEOUT_SEC:-120}"
DISPLAY="${DISPLAY:-:0}"
export DISPLAY

echo "[entrypoint] waiting up to ${X_WAIT_TIMEOUT_SEC}s for X display ${DISPLAY}"

deadline=$(( SECONDS + X_WAIT_TIMEOUT_SEC ))
until python3 -c "
import sys, cv2, numpy as np
try:
    cv2.namedWindow('_probe', cv2.WINDOW_NORMAL)
    cv2.imshow('_probe', np.zeros((1, 1, 3), np.uint8))
    cv2.waitKey(1)
    cv2.destroyWindow('_probe')
except Exception:
    sys.exit(1)
" >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
        echo "[entrypoint] ERROR: X display ${DISPLAY} unreachable after ${X_WAIT_TIMEOUT_SEC}s."
        echo "[entrypoint] Check that /tmp/.X11-unix and the Xauthority file are mounted,"
        echo "[entrypoint] and that the device's desktop session is running."
        exit 1
    fi
    sleep 3
done

echo "[entrypoint] display ready, starting demo"
exec python3 /advantech/src/demo-od-loop.py
