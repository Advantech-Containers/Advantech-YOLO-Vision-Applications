#!/usr/bin/env python3
"""
Build a single-class YOLO dataset from a video via tracked pseudo-labels
=======================================================================
Version:      1.0.0
Created:      July 19, 2026

Turns an unlabelled clip into a trainable dataset with no human annotation.

Why this works when the teacher alone does not:
  A COCO teacher has poor per-frame recall on objects it was never trained on
  (transparent jars, sushi, ...).  Associating its detections into tracks and
  interpolating the gaps recovers the frames it misses -- an object on a
  conveyor moves predictably, so a box interpolated between two confident
  detections is usually closer to truth than the teacher's own miss.  That
  temporal completion is what lets the student exceed the teacher.

Measured on OD_Jar.mp4: the teacher managed 0.13 bottle-detections/frame at
640px; this pipeline produced 68% frame coverage at 2.19 boxes/frame, and the
model trained on it reached mAP50 0.972.

CRITICAL -- teacher resolution:
  The default 640px shrinks a 2560x1440 source past detectability.  Raising
  --teacher-imgsz to 1536 took the same teacher from 0.43 to 2.56 boxes/frame.
  This matters more than teacher model size.

ALWAYS run preview_labels.py afterwards and LOOK at the output before
training.  A duplicate-box defect here was invisible in the summary statistics
and obvious in a six-frame contact sheet.

Usage:
    python3 make_dataset.py --video OD_Jar.mp4 --out dataset --class-name jar

Copyright (c) 2026 Advantech Corporation. All rights reserved.
"""

import argparse
import collections
import os
import sys

import cv2
from ultralytics import YOLO

__version__ = "1.0.0"

# COCO container-ish classes, a reasonable teacher filter for vessels.
# NOTE: vase is 75, NOT 46 (46 is banana).
DEFAULT_TEACHER_CLASSES = [39, 40, 41, 45, 75]  # bottle, wine glass, cup, bowl, vase


def parse_args():
    p = argparse.ArgumentParser(
        description="Build a single-class YOLO dataset from a video via tracked pseudo-labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video", required=True, help="source clip")
    p.add_argument("--out", required=True, help="output dataset directory")
    p.add_argument("--class-name", default="object", help="name for the single output class")
    p.add_argument("--teacher", default="yolo11l.pt", help="teacher weights")
    p.add_argument("--teacher-imgsz", type=int, default=1536,
                   help="teacher inference size -- raise for small objects in 4K sources")
    p.add_argument("--teacher-conf", type=float, default=0.10,
                   help="teacher confidence floor; low on purpose, tracks filter the noise")
    p.add_argument("--teacher-classes", type=int, nargs="*", default=DEFAULT_TEACHER_CLASSES,
                   help="COCO class ids the teacher may emit; empty means all")
    p.add_argument("--iou-match", type=float, default=0.25, help="IoU to associate across frames")
    p.add_argument("--max-miss", type=int, default=12, help="frames a track survives unmatched")
    p.add_argument("--min-track-len", type=int, default=4, help="drop tracks shorter than this")
    p.add_argument("--max-gap", type=int, default=25, help="max frame gap to interpolate across")
    p.add_argument("--nms-iou", type=float, default=0.55, help="per-frame duplicate merge threshold")
    p.add_argument("--val-every", type=int, default=6, help="every Nth frame goes to validation")
    p.add_argument("--device", default="0", help="inference device")
    p.add_argument("--jpeg-quality", type=int, default=92)
    return p.parse_args()


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def detect_tracks(args):
    """Run the teacher and associate detections into tracks by greedy IoU.

    Ultralytics' BoT-SORT needs the `lap` extension, which is absent from the
    JetPack image and needs a source build on aarch64.  A small matcher avoids
    the dependency and keeps the gap logic explicit.
    """
    model = YOLO(args.teacher)
    tracks = collections.defaultdict(dict)   # tid -> {frame: (x1,y1,x2,y2)}
    active = {}                              # tid -> [last_box, frames_missed]
    next_id = 0
    raw_hits = 0
    nframes = 0

    predict_kw = dict(imgsz=args.teacher_imgsz, conf=args.teacher_conf,
                      stream=True, device=args.device, verbose=False)
    if args.teacher_classes:
        predict_kw["classes"] = args.teacher_classes

    for i, r in enumerate(model.predict(source=args.video, **predict_kw)):
        nframes += 1
        dets = [tuple(b) for b in r.boxes.xyxy.cpu().numpy()] if r.boxes is not None else []
        raw_hits += len(dets)

        used = set()
        for d in dets:
            best, best_iou = None, args.iou_match
            for tid, (last, _) in active.items():
                if tid in used:
                    continue
                v = iou(d, last)
                if v > best_iou:
                    best, best_iou = tid, v
            if best is None:
                best = next_id
                next_id += 1
            used.add(best)
            tracks[best][i] = d
            active[best] = [d, 0]

        for tid in list(active):
            if tid not in used:
                active[tid][1] += 1
                if active[tid][1] > args.max_miss:
                    del active[tid]

    return tracks, raw_hits, nframes


def build_frames(tracks, args):
    """Filter short tracks, interpolate interior gaps, merge duplicates."""
    per_frame = collections.defaultdict(list)
    kept = interpolated = 0

    for seq in tracks.values():
        if len(seq) < args.min_track_len:
            continue
        kept += 1
        idxs = sorted(seq)
        for a, b in zip(idxs, idxs[1:]):
            per_frame[a].append(seq[a])
            gap = b - a
            if 1 < gap <= args.max_gap:
                for k in range(1, gap):
                    t = k / gap
                    per_frame[a + k].append(tuple(
                        seq[a][j] + t * (seq[b][j] - seq[a][j]) for j in range(4)))
                    interpolated += 1
        per_frame[idxs[-1]].append(seq[idxs[-1]])

    # The teacher often labels one object under two COCO classes (e.g. wine
    # glass AND cup), producing two tracks over the same thing.  Left in, the
    # student learns to emit stacked duplicates.
    merged = 0
    for f in list(per_frame):
        boxes = sorted(per_frame[f], key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        keep = []
        for b in boxes:
            if any(iou(b, k) > args.nms_iou for k in keep):
                merged += 1
                continue
            keep.append(b)
        per_frame[f] = keep

    return per_frame, kept, interpolated, merged


def write_dataset(per_frame, args):
    for split in ("train", "val"):
        os.makedirs(f"{args.out}/images/{split}", exist_ok=True)
        os.makedirs(f"{args.out}/labels/{split}", exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    written = labelled = total_boxes = 0
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        split = "val" if i % args.val_every == 0 else "train"
        cv2.imwrite(f"{args.out}/images/{split}/f{i:05d}.jpg", frame,
                    [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
        with open(f"{args.out}/labels/{split}/f{i:05d}.txt", "w") as fh:
            for (x1, y1, x2, y2) in per_frame.get(i, []):
                cx, cy = (x1 + x2) / 2.0 / W, (y1 + y2) / 2.0 / H
                bw, bh = (x2 - x1) / W, (y2 - y1) / H
                if bw <= 0 or bh <= 0:
                    continue
                fh.write(f"0 {cx:.6f} {cy:.6f} {min(bw, 1):.6f} {min(bh, 1):.6f}\n")
                total_boxes += 1
        if per_frame.get(i):
            labelled += 1
        written += 1
        i += 1
    cap.release()

    with open(f"{args.out}/data.yaml", "w") as fh:
        fh.write(f"path: {os.path.abspath(args.out)}\n"
                 f"train: images/train\nval: images/val\n"
                 f"names:\n  0: {args.class_name}\n")

    return written, labelled, total_boxes


def main():
    args = parse_args()
    if not os.path.isfile(args.video):
        sys.exit(f"ERROR: video not found: {args.video}")

    print(f"teacher: {args.teacher} @ imgsz={args.teacher_imgsz} conf={args.teacher_conf}")
    tracks, raw_hits, nframes = detect_tracks(args)
    per_frame, kept, interpolated, merged = build_frames(tracks, args)
    written, labelled, total_boxes = write_dataset(per_frame, args)

    print(f"frames processed   : {nframes}")
    print(f"raw teacher hits   : {raw_hits}  ({raw_hits / max(nframes, 1):.2f}/frame)")
    print(f"tracks kept        : {kept} (of {len(tracks)})")
    print(f"interpolated boxes : {interpolated}")
    print(f"duplicates merged  : {merged}")
    print(f"total boxes written: {total_boxes}  ({total_boxes / max(written, 1):.2f}/frame)")
    print(f"frames written     : {written}")
    print(f"frames with labels : {labelled}  ({100.0 * labelled / max(written, 1):.0f}% coverage)")
    print()
    print(f"NEXT: python3 preview_labels.py --dataset {args.out}   <- LOOK at it before training")


if __name__ == "__main__":
    main()
