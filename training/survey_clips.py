#!/usr/bin/env python3
"""
Survey candidate demo clips with a stock model
==============================================
Version:      1.0.0
Created:      July 19, 2026

Answers the question that actually decides a demo clip: not "what is in this
video" but "what does an out-of-the-box detector confidently draw on it".

A clip whose subject is a class COCO knows (bottle, person, car) demos well
with zero training.  A clip whose subject COCO does not know produces
confidently WRONG labels, which reads worse to a viewer than no detection at
all -- jars labelled `surfboard`, sushi labelled `pizza`.  Those clips need
the make_dataset.py -> train.py pipeline, or should be avoided.

Usage:
    python3 survey_clips.py --dir data/cv_demo_clips --conf 0.40

Copyright (c) 2026 Advantech Corporation. All rights reserved.
"""

import argparse
import collections
import glob
import os

import cv2
from ultralytics import YOLO

__version__ = "1.0.0"


def parse_args():
    p = argparse.ArgumentParser(
        description="Survey clips for out-of-the-box detectability.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dir", required=True, help="directory of video files")
    p.add_argument("--glob", default="*.mp4")
    p.add_argument("--model", default="yolo11l.pt")
    p.add_argument("--conf", type=float, default=0.40)
    p.add_argument("--samples", type=int, default=50,
                   help="frames sampled evenly ACROSS the clip, not from the start")
    p.add_argument("--device", default="0")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    paths = sorted(glob.glob(os.path.join(args.dir, args.glob)))
    if not paths:
        raise SystemExit(f"no clips matching {args.glob} in {args.dir}")

    print(f"model {args.model} @ conf {args.conf}, {args.samples} frames "
          f"sampled evenly across each clip\n")
    print(f"{'clip':22}{'resolution':>12}{'secs':>7}{'coverage':>10}"
          f"{'boxes/frm':>11}  top classes")

    for path in paths:
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if total <= 0:
            cap.release()
            continue
        step = max(1, total // args.samples)

        cls = collections.Counter()
        n = fw = boxes = 0
        for idx in range(0, total, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, f = cap.read()
            if not ok:
                break
            r = model.predict(f, conf=args.conf, device=args.device, verbose=False)[0]
            n += 1
            k = len(r.boxes)
            boxes += k
            if k:
                fw += 1
            for b in r.boxes:
                cls[r.names[int(b.cls)]] += 1
        cap.release()
        if not n:
            continue

        top = ", ".join(f"{k}:{v}" for k, v in cls.most_common(4)) or "-- nothing --"
        print(f"{os.path.basename(path)[:22]:22}{f'{w}x{h}':>12}{total / fps:>7.1f}"
              f"{100.0 * fw / n:>9.0f}%{boxes / n:>11.2f}  {top}")

    print("\nHigh coverage AND correct labels -> use as-is with stock weights.")
    print("High coverage, WRONG labels        -> fine-tune (make_dataset.py).")
    print("Low coverage                       -> poor demo clip; prefer another.")


if __name__ == "__main__":
    main()
