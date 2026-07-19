#!/usr/bin/env python3
"""
Fine-tune a YOLO detector on a pseudo-labelled dataset
=====================================================
Version:      1.0.0
Created:      July 19, 2026

Run this on a GPU workstation, NOT on the edge device.

  acn.air-520 (2x RTX 6000 Ada)   120 epochs @ 960px  ->  15.6 minutes
  Jetson Orin (7.6 GB shared)     120 epochs @ 960px  ->  ~15 min PER EPOCH

Training on the Orin alongside the running demo drove load average to 74 and
left sshd unable to fork; the device was unreachable for several minutes.

On a shared GPU box, check what else is resident first.  Both air-520 cards are
usually ~93% occupied by an unrelated vLLM server (~2.9 GB free each).  A nano
model at batch=4 fits in 1.6 GB and coexists -- do not evict other people's
services to make room.

WARNING -- a partially-trained checkpoint looks trained and detects nothing:

    2 epochs  : val mAP50 0.47, max inference confidence 0.026  (unusable)
    120 epochs: val mAP50 0.972, max inference confidence 0.95  (good)

mAP is rank-based, so box placement scores well long before the classification
head is calibrated.  Never judge a run by mAP alone -- check the confidence
distribution with evaluate.py.

Usage:
    python3 train.py --data dataset/data.yaml --model yolo26n.pt --epochs 120

Copyright (c) 2026 Advantech Corporation. All rights reserved.
"""

import argparse
import sys

__version__ = "1.0.0"


def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune a YOLO detector on a pseudo-labelled dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", required=True, help="path to data.yaml")
    p.add_argument("--model", default="yolo26n.pt", help="base weights")
    p.add_argument("--epochs", type=int, default=120,
                   help="~100 needed for confidence calibration; fewer detects nothing")
    p.add_argument("--imgsz", type=int, default=960,
                   help="raise for small objects in high-resolution sources")
    p.add_argument("--batch", type=int, default=4,
                   help="keep small to coexist with other GPU tenants")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--project", default="runs")
    p.add_argument("--name", default="finetune")
    p.add_argument("--patience", type=int, default=40)
    return p.parse_args()


def main():
    args = parse_args()

    import torch
    import ultralytics
    from ultralytics import YOLO

    print(f"ultralytics {ultralytics.__version__} | torch {torch.__version__} "
          f"| cuda {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("WARNING: no CUDA device -- CPU training a detector takes days, not hours")
    else:
        free, total = torch.cuda.mem_get_info(0)
        print(f"GPU 0: {free / 2**30:.1f} GB free of {total / 2**30:.1f} GB")
        if free / 2**30 < 2.0:
            print("WARNING: under 2 GB free -- reduce --batch or wait for the GPU to clear")

    # YOLO26 checkpoints need ultralytics >= 8.4; under 8.3.x they are silently
    # coerced into the legacy head and train/predict incorrectly.
    if "26" in args.model and not ultralytics.__version__.startswith("8.4"):
        sys.exit(f"ERROR: {args.model} needs ultralytics >= 8.4, "
                 f"found {ultralytics.__version__}")

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        patience=args.patience,
        val=True,
        plots=True,
        verbose=True,
    )
    print("TRAINING COMPLETE")
    print(f"weights: {args.project}/{args.name}/weights/best.pt")
    print("NEXT: evaluate.py --models <best.pt> --video <clip>   "
          "<- verify confidence, not just mAP")


if __name__ == "__main__":
    main()
