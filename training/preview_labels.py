#!/usr/bin/env python3
"""
Render a contact sheet of pseudo-labels for visual inspection
=============================================================
Version:      1.0.0
Created:      July 19, 2026

Do not skip this step.  Summary statistics from make_dataset.py cannot tell you
whether the boxes are on the right objects.  Rendering six labelled frames
immediately exposed a defect -- one object labelled under two COCO classes,
producing stacked duplicate boxes -- that the counts showed as a perfectly
healthy 68% coverage.

Training on bad labels is worse than not training.

Usage:
    python3 preview_labels.py --dataset dataset --out preview.jpg

Copyright (c) 2026 Advantech Corporation. All rights reserved.
"""

import argparse
import glob
import os
import sys

import cv2

__version__ = "1.0.0"


def parse_args():
    p = argparse.ArgumentParser(
        description="Render labelled frames from a YOLO dataset as a contact sheet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", required=True, help="dataset directory from make_dataset.py")
    p.add_argument("--out", default="preview.jpg", help="output image")
    p.add_argument("--tiles", type=int, default=6, help="number of frames to show")
    p.add_argument("--cols", type=int, default=3)
    p.add_argument("--tile-width", type=int, default=640)
    return p.parse_args()


def main():
    args = parse_args()

    pairs = []
    for split in ("train", "val"):
        for img in sorted(glob.glob(f"{args.dataset}/images/{split}/*.jpg")):
            lab = img.replace("/images/", "/labels/").replace(".jpg", ".txt")
            if os.path.exists(lab) and os.path.getsize(lab) > 0:
                pairs.append((img, lab))
    if not pairs:
        sys.exit(f"ERROR: no labelled frames found under {args.dataset}")

    # Spread the sample across the clip rather than taking the first N, which
    # would only show the opening seconds.
    step = max(1, len(pairs) // args.tiles)
    picks = pairs[::step][:args.tiles]

    tiles = []
    for img_path, lab_path in picks:
        im = cv2.imread(img_path)
        if im is None:
            continue
        h, w = im.shape[:2]
        n = 0
        for line in open(lab_path):
            parts = line.split()
            if len(parts) != 5:
                continue
            _, cx, cy, bw, bh = (float(x) for x in parts)
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), max(2, w // 400))
            n += 1
        cv2.putText(im, f"{os.path.basename(img_path)}  |  {n} boxes",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, w / 1200.0, (0, 255, 255), 3)
        tw = args.tile_width
        tiles.append(cv2.resize(im, (tw, int(tw * h / w))))

    while len(tiles) % args.cols:
        tiles.append(tiles[-1] * 0)

    rows = [cv2.hconcat(tiles[i:i + args.cols])
            for i in range(0, len(tiles), args.cols)]
    cv2.imwrite(args.out, cv2.vconcat(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"wrote {args.out} from {len(picks)} labelled frames "
          f"({len(pairs)} labelled of the full set)")
    print("Open it. Check the boxes are on the intended object, are not "
          "duplicated, and are not systematically offset.")


if __name__ == "__main__":
    main()
