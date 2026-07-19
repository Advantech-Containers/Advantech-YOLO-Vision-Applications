#!/usr/bin/env python3
"""
Evaluate detection models on a clip the way a demo is actually judged
=====================================================================
Version:      1.0.0
Created:      July 19, 2026

Reports, per model and confidence threshold:

  coverage      share of frames with at least one box -- decides whether the
                demo looks alive, and the single most useful number here
  boxes/frame   overlay density
  FPS           sequential throughput on this device
  class split   what is actually being detected, with confidence stats

Two mistakes this tool exists to prevent:

  1. MEASURE THE WHOLE CLIP.  Measuring the first 150 of 495 frames of the
     sushi clip reported 100% coverage where the true figure was 73% -- an
     early window can be far denser than the rest.  --max-frames 0 (default)
     reads everything.

  2. DO NOT TRUST mAP ALONE.  A 2-epoch checkpoint reported val mAP50 0.47
     while its maximum inference confidence was 0.026 -- zero usable
     detections.  --conf 0.001 exposes that immediately.

Usage:
    python3 evaluate.py --video clip.mp4 --models yolo11n.pt yolo26n.pt
    python3 evaluate.py --video clip.mp4 --models best.pt --conf 0.001 --classes-report

Copyright (c) 2026 Advantech Corporation. All rights reserved.
"""

import argparse
import collections
import time

import cv2
from ultralytics import YOLO

__version__ = "1.0.0"


def parse_args():
    p = argparse.ArgumentParser(
        description="Measure coverage, density, throughput and class split on a clip.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video", required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--conf", type=float, nargs="+", default=[0.25, 0.40, 0.50])
    p.add_argument("--imgsz", type=int, nargs="+", default=[640])
    p.add_argument("--classes", type=int, nargs="*", default=None,
                   help="restrict to these class ids; omit for all")
    p.add_argument("--max-frames", type=int, default=0,
                   help="0 = whole clip (recommended); a partial window can mislead badly")
    p.add_argument("--device", default="0")
    p.add_argument("--classes-report", action="store_true",
                   help="also print per-class counts and confidence stats")
    return p.parse_args()


def run_one(model, args, imgsz, conf):
    cap = cv2.VideoCapture(args.video)
    kw = dict(imgsz=imgsz, conf=conf, device=args.device, verbose=False)
    if args.classes:
        kw["classes"] = args.classes

    for _ in range(5):                      # warm up, excluded from timing
        ok, f = cap.read()
        if ok:
            model.predict(f, **kw)
    cap.release()

    cap = cv2.VideoCapture(args.video)
    n = fw = boxes = 0
    per_frame = []
    cls = collections.Counter()
    confs = collections.defaultdict(list)
    t0 = time.time()
    while args.max_frames == 0 or n < args.max_frames:
        ok, f = cap.read()
        if not ok:
            break
        r = model.predict(f, **kw)[0]
        n += 1
        k = len(r.boxes)
        boxes += k
        per_frame.append(k)
        if k:
            fw += 1
        if args.classes_report and r.boxes is not None:
            for b in r.boxes:
                nm = r.names[int(b.cls)]
                cls[nm] += 1
                confs[nm].append(float(b.conf))
    fps = n / max(time.time() - t0, 1e-6)
    cap.release()
    return n, fw, boxes, per_frame, fps, cls, confs


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"clip: {args.video}  {w}x{h}  {total} frames")
    if args.max_frames and args.max_frames < total:
        print(f"WARNING: measuring only {args.max_frames} of {total} frames -- "
              f"a partial window can overstate coverage substantially")
    print()

    print(f"{'model':22}{'imgsz':>7}{'conf':>6}{'coverage':>10}"
          f"{'boxes/frm':>11}{'min':>5}{'max':>5}{'FPS':>8}")
    reports = []
    for name in args.models:
        model = YOLO(name)
        short = name.split("/")[-1][:20]
        for imgsz in args.imgsz:
            for conf in args.conf:
                n, fw, boxes, pf, fps, cls, confs = run_one(model, args, imgsz, conf)
                if n == 0:
                    continue
                print(f"{short:22}{imgsz:>7}{conf:>6.2f}{100.0 * fw / n:>9.0f}%"
                      f"{boxes / n:>11.2f}{min(pf):>5}{max(pf):>5}{fps:>8.1f}")
                if args.classes_report:
                    reports.append((short, imgsz, conf, n, cls, confs))

    for short, imgsz, conf, n, cls, confs in reports:
        print(f"\n{short} @ imgsz={imgsz} conf={conf} -- classes over {n} frames")
        if not cls:
            print("  (no detections at all -- if mAP looked fine, the model is "
                  "undertrained and its confidences are near zero)")
            continue
        print(f"  {'class':16}{'count':>8}{'/frame':>9}{'max':>7}{'mean':>7}{'min':>7}")
        for k, v in cls.most_common():
            c = confs[k]
            print(f"  {k:16}{v:>8}{v / n:>9.2f}{max(c):>7.2f}"
                  f"{sum(c) / len(c):>7.2f}{min(c):>7.2f}")


if __name__ == "__main__":
    main()
