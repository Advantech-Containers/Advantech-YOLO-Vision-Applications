#!/usr/bin/env python3
"""
YOLO11 Vision Application for Advantech Edge AI Devices
========================================================
Version:      2.0.0
Author:       Samir Singh <samir.singh@advantech.com>
Created:      October 9, 2025
Updated:      March 1, 2026
Description:  Complete YOLO11 application supporting detection, segmentation, and classification

This script demonstrates how to use YOLO11 for multiple vision tasks with hardware
acceleration on Advantech edge AI devices.  Run without arguments to launch the
interactive menu (Task → Model → Source → Display options).

Copyright (c) 2025 Advantech Corporation. All rights reserved.
"""

import argparse
import os
import sys
from pathlib import Path
from ultralytics import YOLO
import cv2
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
║                     YOLO11 Inference Pipeline v{__version__}                             ║
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


def interactive_mode():
    """Run interactive configuration and inference."""
    os.system('clear' if os.name != 'nt' else 'cls')
    print_banner()

    # --- Task ---
    print(f"\n{Colors.BOLD}[1] Detection  [2] Segmentation  [3] Classification{Colors.ENDC}")
    task_choice = _get_choice("Select task", ["1", "2", "3"], "1")
    task_map = {"1": "detect", "2": "segment", "3": "classify"}
    task = task_map[task_choice]

    model_hints = {
        "detect":   "yolo11n.pt / yolo11n.engine",
        "segment":  "yolo11n-seg.pt / yolo11n-seg.engine",
        "classify": "yolo11n-cls.pt",
    }
    default_models = {
        "detect":   "yolo11n.pt",
        "segment":  "yolo11n-seg.pt",
        "classify": "yolo11n-cls.pt",
    }

    # --- Model ---
    model_path = _get_input(
        f"Model path ({model_hints[task]})",
        default_models[task]
    )

    # --- Source ---
    print(f"\n{Colors.BOLD}[1] Webcam  [2] RTSP  [3] Video File{Colors.ENDC}")
    src_choice = _get_choice("Select source", ["1", "2", "3"], "1")
    if src_choice == "1":
        device_idx = _get_input("Camera device index", "0")
        source = int(device_idx) if device_idx.isdigit() else device_idx
    elif src_choice == "2":
        source = _get_input("RTSP URL", "rtsp://your-camera-ip:port/")
    else:
        source = _get_input("Video file path", "data/test.mp4")

    # --- Display / Save ---
    show_choice = _get_choice("Show display window? (y/n)", ["y", "n", "Y", "N"], "y")
    show = show_choice.lower() == "y"

    save_choice = _get_choice("Save results to disk? (y/n)", ["y", "n", "Y", "N"], "n")
    save = save_choice.lower() == "y"

    save_dir = '/advantech/results'
    if save:
        save_dir = _get_input("Output directory", save_dir)

    conf_str = _get_input("Confidence threshold", "0.25")
    try:
        conf = float(conf_str)
    except ValueError:
        conf = 0.25

    # --- Run ---
    print(f"\n{Colors.CYAN}ℹ Configuration{Colors.ENDC}")
    print(f"  Task:       {task.upper()}")
    print(f"  Model:      {model_path}")
    print(f"  Source:     {source}")
    print(f"  Confidence: {conf}")
    print(f"  Show:       {show}")
    print(f"  Save:       {save}")
    if save:
        print(f"  Save dir:   {save_dir}")

    # Build a namespace that run_inference() expects
    class _Args:
        pass
    args = _Args()
    args.model    = model_path
    args.input    = str(source)
    args.task     = task
    args.conf     = conf
    args.iou      = 0.45
    args.device   = '0'
    args.show     = show
    args.save     = save
    args.save_dir = save_dir
    args.loop     = False

    run_inference(args)

def check_gpu():
    """Check if GPU is available"""
    if torch.cuda.is_available():
        print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA Version: {torch.version.cuda}")
        print(f"✓ PyTorch Version: {torch.__version__}")
        return True
    else:
        print("✗ No GPU detected, using CPU")
        return False

def validate_task_model(task, model_path):
    """Validate that model matches the task"""
    model_name = Path(model_path).stem.lower()

    if task == 'detect':
        # Detection models shouldn't have -seg or -cls suffix
        if '-seg' in model_name or '-cls' in model_name or '-pose' in model_name:
            print(f"⚠ Warning: Model '{model_path}' may not be suitable for detection task")
            print("  Detection models: yolo11n.pt, yolo11s.pt, yolo11m.pt, etc.")
            return False
    elif task == 'segment':
        # Segmentation models should have -seg suffix
        if '-seg' not in model_name:
            print(f"⚠ Warning: Model '{model_path}' may not be suitable for segmentation task")
            print("  Segmentation models: yolo11n-seg.pt, yolo11s-seg.pt, etc.")
            return False
    elif task == 'classify':
        # Classification models should have -cls suffix
        if '-cls' not in model_name:
            print(f"⚠ Warning: Model '{model_path}' may not be suitable for classification task")
            print("  Classification models: yolo11n-cls.pt, yolo11s-cls.pt, etc.")
            return False

    return True

def run_inference(args):
    """Execute YOLO11 inference with the given args namespace."""
    # Check GPU availability
    gpu_available = check_gpu()

    # Set device
    if args.device == '0' and not gpu_available:
        print("⚠ Warning: GPU not available, falling back to CPU")
        args.device = 'cpu'

    # Validate task and model compatibility
    validate_task_model(args.task, args.model)

    # Print configuration
    print("\n" + "=" * 70)
    print("YOLO11 Vision Application - Advantech Edge AI")
    print("=" * 70)
    print(f"Task:       {args.task.upper()}")
    print(f"Model:      {args.model}")
    print(f"Input:      {args.input}")
    print(f"Device:     {args.device}")
    print(f"Confidence: {args.conf}")
    if args.task != 'classify':
        print(f"IoU:        {args.iou}")
    print(f"Show:       {args.show}")
    print(f"Save:       {args.save}")
    print(f"Loop:       {args.loop}")
    print("=" * 70 + "\n")

    # Load model
    print(f"Loading YOLO11 model: {args.model}")
    try:
        model = YOLO(args.model)
        print("✓ Model loaded successfully")

        model_task = model.task
        if model_task != args.task:
            print(f"⚠ Warning: Model task is '{model_task}' but you requested '{args.task}'")
            print("  This may lead to errors. Please use the correct model for the task.")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("\nTip: Download models using one of these methods:")
        print("  1. python3 src/advantech-coe-model-load.py")
        print("  2. yolo task=detect mode=predict model=yolo11n.pt (auto-downloads)")
        return

    # Convert input to int if it's a digit (for webcam)
    source = int(args.input) if str(args.input).isdigit() else args.input

    predict_params = {
        'source': source,
        'conf': args.conf,
        'device': args.device,
        'show': args.show,
        'save': args.save,
        'project': args.save_dir,
        'stream': True,
    }
    if args.task in ['detect', 'segment']:
        predict_params['iou'] = args.iou

    # Loop mode: force a fixed output folder name + overwrite so the
    # results directory doesn't accumulate predict, predict2, predict3, …
    # on every replay iteration.
    if args.loop and args.save:
        predict_params['name'] = 'predict_loop'
        predict_params['exist_ok'] = True

    print(f"\nRunning {args.task} inference on: {source}")
    if args.loop:
        print("Loop mode: video will replay continuously (press 'q' to quit, "
              "or stop the container)")
    print("Press 'q' to quit\n")

    frame_count = 0
    iteration = 0
    user_quit = False

    try:
        while True:
            iteration += 1
            if args.loop and iteration > 1:
                print(f"\n[loop] replay #{iteration} starting...\n")

            results = model.predict(**predict_params)
            for result in results:
                frame_count += 1

                if args.task == 'detect':
                    if result.boxes is not None and len(result.boxes) > 0:
                        num_detections = len(result.boxes)
                        print(f"Frame {frame_count}: {num_detections} objects detected")
                        classes = result.boxes.cls.cpu().numpy()
                        names = result.names
                        unique_classes = set([names[int(c)] for c in classes])
                        print(f"  Classes: {', '.join(sorted(unique_classes))}")

                elif args.task == 'segment':
                    if result.masks is not None and len(result.masks) > 0:
                        num_segments = len(result.masks)
                        print(f"Frame {frame_count}: {num_segments} instances segmented")
                        if result.boxes is not None:
                            classes = result.boxes.cls.cpu().numpy()
                            names = result.names
                            unique_classes = set([names[int(c)] for c in classes])
                            print(f"  Classes: {', '.join(sorted(unique_classes))}")

                elif args.task == 'classify':
                    if result.probs is not None:
                        top1_idx = result.probs.top1
                        top1_conf = result.probs.top1conf.item()
                        class_name = result.names[top1_idx]
                        print(f"Image {frame_count}: {class_name} ({top1_conf:.2%} confidence)")
                        if hasattr(result.probs, 'top5'):
                            print("  Top 5 predictions:")
                            for idx in result.probs.top5:
                                conf = result.probs.data[idx].item()
                                name = result.names[idx]
                                print(f"    {name}: {conf:.2%}")

                if args.show and cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nStopping inference...")
                    user_quit = True
                    break

            if user_quit or not args.loop:
                break
            # else: video ended naturally and --loop is on → replay

    except KeyboardInterrupt:
        print("\n\nInference stopped by user")
    except Exception as e:
        print(f"\n✗ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return

    suffix = f" across {iteration} iteration{'s' if iteration > 1 else ''}" if args.loop else ""
    print(f"\n✓ Inference completed ({frame_count} frames processed{suffix})")
    if args.save:
        print(f"✓ Results saved to: {args.save_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='YOLO11 Vision Application for Advantech Edge AI Devices',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 advantech-yolo.py                                                 # Interactive mode
  python3 advantech-yolo.py --input 0 --task detect --model yolo11n.pt --show
  python3 advantech-yolo.py --input data/test.mp4 --task segment --model yolo11n-seg.pt --show --save
  python3 advantech-yolo.py --input 0 --task classify --model yolo11n-cls.pt --show
        """
    )
    parser.add_argument('--input', type=str, default=None,
                        help='Input source: 0 for webcam, path to video/image, or RTSP URL')
    parser.add_argument('--model', type=str, default=None,
                        help='YOLO11 model path (e.g., yolo11n.pt, yolo11n-seg.pt)')
    parser.add_argument('--task', type=str, default='detect',
                        choices=['detect', 'segment', 'classify'],
                        help='Task type: detect, segment, or classify')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS (default: 0.45)')
    parser.add_argument('--device', type=str, default='0',
                        help='Device to run on: 0 for GPU, cpu for CPU')
    parser.add_argument('--show', action='store_true',
                        help='Display results in window')
    parser.add_argument('--save', action='store_true',
                        help='Save results to output directory')
    parser.add_argument('--save-dir', type=str, default='/advantech/results',
                        help='Directory to save results (default: /advantech/results)')
    parser.add_argument('--loop', action='store_true',
                        help='Replay the input video (or repeat the image) '
                             'continuously until "q" is pressed or the '
                             'container is stopped')

    args = parser.parse_args()

    # No --model or --input provided → interactive mode
    if args.model is None and args.input is None:
        interactive_mode()
        return

    # Apply defaults for CLI mode if only one was omitted
    if args.model is None:
        args.model = 'yolo11n.pt'
    if args.input is None:
        args.input = '0'

    run_inference(args)


if __name__ == '__main__':
    main()

