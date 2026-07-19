#!/usr/bin/env python3
"""
YOLO11 Looping Object-Detection Demo for Advantech Edge AI Devices
==================================================================
Version:      1.0.0
Created:      July 19, 2026
Description:  Unattended, headless-launched object-detection demo that plays a
              video file on repeat and renders annotated detections to the
              device's local screen (X11).

Designed for deployment via the WEDA container-management API, where no TTY is
attached and no interactive input is possible.  Every setting is supplied by
environment variable; the process never prompts.

Differs from advantech-yolo.py in two ways that matter for unattended use:
  * the model is loaded once and stays resident -- the capture is rewound at
    EOF rather than the pipeline being rebuilt, so looping costs no stall;
  * detections are summarised on an interval, never per frame, so container
    logs stay bounded on disk-constrained edge devices.

Copyright (c) 2026 Advantech Corporation. All rights reserved.
"""

import os
import signal
import sys
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

__version__ = "1.0.0"

# --- Configuration (environment-driven; no interactive input) --------------
VIDEO_PATH = os.getenv("VIDEO_PATH", "/advantech/data/OD_Jar.mp4")
MODEL_PATH = os.getenv("MODEL_PATH", "/advantech/models/yolo11n.pt")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.45"))
INFER_DEVICE = os.getenv("INFER_DEVICE", "0")
WINDOW_NAME = os.getenv("WINDOW_NAME", "Advantech YOLO11 - Object Detection")
FULLSCREEN = os.getenv("FULLSCREEN", "true").lower() == "true"
STATS_INTERVAL_SEC = float(os.getenv("STATS_INTERVAL_SEC", "60"))

_shutdown = False


def _handle_signal(signum, _frame):
    """Translate SIGTERM/SIGINT into a clean exit so `docker stop` is graceful."""
    global _shutdown
    print(f"[demo] signal {signum} received, shutting down", flush=True)
    _shutdown = True


def _log(message):
    print(f"[demo] {message}", flush=True)


def validate_environment():
    """Fail fast, and loudly, if a precondition for the demo is absent.

    A silent failure here would leave the container 'running' from WEDA's
    perspective while the screen stays blank, so every check reports the
    specific remedy rather than just raising.
    """
    problems = []

    if not Path(VIDEO_PATH).is_file():
        problems.append(f"video not found: {VIDEO_PATH}")
    if not Path(MODEL_PATH).is_file():
        problems.append(f"model not found: {MODEL_PATH}")

    # The base image ships opencv-python-headless, which shadows JetPack's
    # GTK-enabled build and makes imshow raise at the first frame.  Detect it
    # here instead of 30 seconds into model loading.
    if "GTK" not in cv2.getBuildInformation():
        problems.append(
            f"OpenCV {cv2.__version__} at {cv2.__file__} has no GUI support. "
            "Set PYTHONPATH=/usr/lib/python3.10/dist-packages so JetPack's "
            "OpenCV takes precedence over opencv-python-headless."
        )

    if not os.getenv("DISPLAY"):
        problems.append("DISPLAY is unset -- the demo cannot reach the screen")

    if problems:
        _log("PRECONDITION CHECK FAILED:")
        for problem in problems:
            _log(f"  - {problem}")
        sys.exit(1)

    _log(f"OpenCV {cv2.__version__} (GUI enabled)")
    if torch.cuda.is_available():
        _log(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        _log("WARNING: CUDA unavailable -- inference will fall back to CPU")


def open_capture(path):
    """Open the demo clip, failing loudly rather than yielding empty frames."""
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        _log(f"ERROR: cannot open video: {path}")
        sys.exit(1)
    return capture


def create_window():
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    if FULLSCREEN:
        cv2.setWindowProperty(
            WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )


def main():
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    _log(f"Advantech YOLO11 looping detection demo v{__version__}")
    validate_environment()

    _log(f"loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    capture = open_capture(VIDEO_PATH)
    create_window()
    _log(f"streaming {VIDEO_PATH} on repeat -- detections summarised every "
         f"{STATS_INTERVAL_SEC:.0f}s")

    frames = 0
    loops = 0
    detections = 0
    window_start = time.monotonic()

    while not _shutdown:
        ok, frame = capture.read()
        if not ok:
            # End of clip: rewind in place.  Reopening the file or reloading
            # the model here would stall the demo for seconds on every lap.
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            loops += 1
            ok, frame = capture.read()
            if not ok:
                _log("ERROR: rewind failed, reopening capture")
                capture.release()
                capture = open_capture(VIDEO_PATH)
                continue

        result = model.predict(
            frame,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            device=INFER_DEVICE,
            verbose=False,
        )[0]

        cv2.imshow(WINDOW_NAME, result.plot())

        frames += 1
        if result.boxes is not None:
            detections += len(result.boxes)

        # Interval summary keeps the log bounded; per-frame output would fill
        # the device disk over a multi-day demo run.
        elapsed = time.monotonic() - window_start
        if elapsed >= STATS_INTERVAL_SEC:
            _log(f"{frames / elapsed:.1f} FPS | "
                 f"{detections / max(frames, 1):.1f} objects/frame | "
                 f"laps={loops}")
            frames = 0
            detections = 0
            window_start = time.monotonic()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            _log("quit requested from window")
            break

    capture.release()
    cv2.destroyAllWindows()
    _log("stopped cleanly")


if __name__ == "__main__":
    main()
