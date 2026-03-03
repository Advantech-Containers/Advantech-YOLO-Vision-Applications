#!/usr/bin/env python3
"""
YOLO11 Model Export Utility for Advantech Edge AI Devices
==========================================================
Version:      2.0.0
Author:       Samir Singh <samir.singh@advantech.com>
Created:      October 9, 2025
Updated:      March 1, 2026
Description:  Export YOLO11 models to optimized formats (TensorRT, ONNX, etc.)

This utility exports YOLO11 models to various formats for optimized inference
on Advantech edge AI devices with NVIDIA Jetson hardware.  Run without
arguments to launch the interactive menu (Task → Size → Format).

Copyright (c) 2025 Advantech Corporation. All rights reserved.
"""

import argparse
import os
import sys
from pathlib import Path
from ultralytics import YOLO
import torch

__version__ = "2.0.0"
__author__ = "Advantech Co. Ltd"
__build_date__ = "2025-12"
__copyright__ = "Copyright (c) 2025 Advantech Corporation. All Rights Reserved."


class Colors:
    GREEN  = '\033[92m'
    YELLOW = '\033[93m'
    RED    = '\033[91m'
    CYAN   = '\033[96m'
    ENDC   = '\033[0m'
    BOLD   = '\033[1m'


def print_banner():
    banner = f"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║     █████╗ ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗███████╗ ██████╗██╗  ██╗  ║
║    ██╔══██╗██╔══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝██╔════╝██╔════╝██║  ██║  ║
║    ███████║██║  ██║╚██╗ ██╔╝███████║██╔██╗ ██║   ██║   █████╗  ██║     ███████║  ║
║    ██╔══██║██║  ██║ ╚████╔╝ ██╔══██║██║╚██╗██║   ██║   ██╔══╝  ██║     ██╔══██║  ║
║    ██║  ██║██████╔╝  ╚██╔╝  ██║  ██║██║ ╚████║   ██║   ███████╗╚██████╗██║  ██║  ║
║    ╚═╝  ╚═╝╚═════╝    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝ ╚═════╝╚═╝  ╚═╝  ║
║                        YOLO11 Model Export v{__version__}                                ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  Author: {__author__:<18}  Build: {__build_date__:<14}                               ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  {__copyright__}             ║
╚══════════════════════════════════════════════════════════════════════════════════╝"""
    print(banner)


def _get_choice(prompt, valid, default="1"):
    try:
        choice = input(f"{prompt} [{default}]: ").strip() or default
        return choice if choice in valid else default
    except (EOFError, KeyboardInterrupt):
        print("\n\nOperation cancelled.")
        sys.exit(0)


def _get_input(prompt, default=""):
    try:
        value = input(f"{prompt} [{default}]: ").strip()
        return value if value else default
    except (EOFError, KeyboardInterrupt):
        print("\n\nOperation cancelled.")
        sys.exit(0)


def interactive_export(gpu_available):
    """Interactive three-step export wizard: Task → Size → Format."""
    os.system('clear' if os.name != 'nt' else 'cls')
    print_banner()

    # --- Step 1: Task ---
    print(f"\n{Colors.BOLD}Step 1 — Select Task{Colors.ENDC}")
    print("  [1] Object Detection      (input size: 640×640)")
    print("  [2] Instance Segmentation (input size: 640×640)")
    print("  [3] Classification        (input size: 224×224)")
    task_choice = _get_choice("Select task", ["1", "2", "3"], "1")
    task_map = {
        "1": ("detect",   "",     640),
        "2": ("segment",  "-seg", 640),
        "3": ("classify", "-cls", 224),
    }
    task_name, task_suffix, imgsz = task_map[task_choice]

    # --- Step 2: Size ---
    print(f"\n{Colors.BOLD}Step 2 — Select Model Size{Colors.ENDC}")
    print("  [1] Nano   (n) — Fastest inference, best for real-time")
    print("  [2] Small  (s) — Good balance of speed and accuracy")
    print("  [3] Medium (m) — Higher accuracy, moderate speed")
    print("  [4] Large  (l) — High accuracy, slower inference")
    print("  [5] XLarge (x) — Maximum accuracy, slowest inference")
    size_choice = _get_choice("Select size", ["1", "2", "3", "4", "5"], "1")
    size_map = {"1": "n", "2": "s", "3": "m", "4": "l", "5": "x"}
    size_letter = size_map[size_choice]

    model_name = f"yolo11{size_letter}{task_suffix}.pt"

    # --- Step 3: Format ---
    print(f"\n{Colors.BOLD}Step 3 — Select Export Format{Colors.ENDC}")
    print("  [1] ONNX (CPU mode)          — Development and testing")
    print("  [2] ONNX (GPU mode, FP16)    — When TensorRT unavailable")
    print(f"  {Colors.GREEN}[3] TensorRT Engine (FP16)   — Recommended for production{Colors.ENDC}")
    print("  [4] PyTorch                  — Ultralytics native")
    fmt_choice = _get_choice("Select format", ["1", "2", "3", "4"], "3")
    fmt_map = {
        "1": ("onnx",         False, False),
        "2": ("onnx",         True,  False),
        "3": ("engine",       True,  False),
        "4": ("torchscript",  False, False),
    }
    fmt_str, half, int8 = fmt_map[fmt_choice]

    if fmt_str == "engine" and not gpu_available:
        print(f"\n{Colors.RED}✗ TensorRT export requires a GPU. "
              f"Please run on a CUDA-capable device.{Colors.ENDC}")
        return

    print(f"\n{Colors.CYAN}ℹ Configuration{Colors.ENDC}")
    print(f"  Model:   {model_name}")
    print(f"  Format:  {fmt_str}")
    print(f"  FP16:    {half}")
    print(f"  imgsz:   {imgsz}")

    export_model(model_name, fmt_str, half, int8, "0" if gpu_available else "cpu", imgsz)

def check_gpu():
    """Check if GPU is available"""
    if torch.cuda.is_available():
        print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA Version: {torch.version.cuda}")
        return True
    else:
        print("✗ No GPU detected - TensorRT export requires GPU")
        return False

def export_model(model_path, format_type, half, int8, device, imgsz=640):
    """Export model to specified format"""
    print(f"\nExporting {model_path} to {format_type.upper()} format...")

    try:
        model = YOLO(model_path)
        exported_model = model.export(
            format=format_type,
            half=half,
            int8=int8,
            device=device,
            imgsz=imgsz,
        )
        print(f"✓ Model exported successfully")
        print(f"  Exported to: {exported_model}")
        return exported_model

    except Exception as e:
        print(f"✗ Export failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Export YOLO11 models for Advantech Edge AI Devices',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Export Formats:
  onnx        - ONNX format (portable, good compatibility)
  engine      - TensorRT engine (best performance on Jetson, RECOMMENDED)
  torchscript - TorchScript format
  openvino    - OpenVINO format

Optimization Options:
  --half    - FP16 quantization (faster, minimal accuracy loss, RECOMMENDED)
  --int8    - INT8 quantization (fastest, some accuracy loss)

Examples:
  python3 advantech-coe-model-export.py                             # Interactive mode
  python3 advantech-coe-model-export.py --model yolo11n.pt --format onnx --half
  python3 advantech-coe-model-export.py --model yolo11n.pt --format engine --half
  python3 advantech-coe-model-export.py --model yolo11n-seg.pt --format engine --half
        """
    )
    parser.add_argument('--model', type=str, default=None,
                        help='Path to YOLO11 model (e.g., yolo11n.pt); '
                             'omit to use interactive mode')
    parser.add_argument('--format', type=str, default='engine',
                        choices=['onnx', 'engine', 'torchscript', 'openvino'],
                        help='Export format (default: engine/TensorRT)')
    parser.add_argument('--half', action='store_true',
                        help='Enable FP16 quantization (recommended)')
    parser.add_argument('--int8', action='store_true',
                        help='Enable INT8 quantization')
    parser.add_argument('--device', type=str, default='0',
                        help='Device to use for export (0 for GPU, cpu for CPU)')
    args = parser.parse_args()

    # Check GPU availability
    gpu_available = check_gpu()

    # No --model provided → interactive mode
    if args.model is None:
        interactive_export(gpu_available)
        return

    # CLI mode
    if args.format == 'engine' and not gpu_available:
        print("\n✗ ERROR: TensorRT export requires GPU")
        print("  Please run this script on a device with CUDA support")
        return

    print("\n" + "=" * 70)
    print("YOLO11 Model Export Utility - Advantech Edge AI")
    print("=" * 70)
    print(f"Model:  {args.model}")
    print(f"Format: {args.format}")
    print(f"FP16:   {args.half}")
    print(f"INT8:   {args.int8}")
    print(f"Device: {args.device}")
    print("=" * 70 + "\n")

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"✗ Error: Model file not found: {args.model}")
        print("\nTip: Run 'python3 src/advantech-coe-model-load.py' to download models first")
        return

    exported_path = export_model(args.model, args.format, args.half, args.int8, args.device)

    if exported_path:
        print("\n" + "=" * 70)
        print("Export Summary")
        print("=" * 70)
        print(f"✓ Export completed successfully")
        print(f"✓ Exported model: {exported_path}")
        print("\nNext steps:")
        print(f"  Run inference: python3 src/advantech-yolo.py --model {exported_path} --input 0 --task detect")
        print("=" * 70)
    else:
        print("\n✗ Export failed")


if __name__ == '__main__':
    main()

