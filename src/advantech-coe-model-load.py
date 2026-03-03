#!/usr/bin/env python3
"""
YOLO11 Model Download Utility for Advantech Edge AI Devices
============================================================
Version:      2.0.0
Author:       Samir Singh <samir.singh@advantech.com>
Created:      October 9, 2025
Updated:      March 1, 2026
Description:  Download and verify YOLO11 models for edge deployment

This utility detects Advantech device capabilities and provides
optimized YOLO11 model recommendations for detection, segmentation,
and classification tasks. It shows an interactive menu when run
without arguments.

Copyright (c) 2025 Advantech Corporation. All rights reserved.
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from ultralytics import YOLO
import torch

__version__ = "2.0.0"
__author__ = "Advantech Co. Ltd"
__build_date__ = "2025-12"
__copyright__ = "Copyright (c) 2025 Advantech Corporation. All Rights Reserved."


class Colors:
    CYAN   = '\033[96m'
    GREEN  = '\033[92m'
    YELLOW = '\033[93m'
    RED    = '\033[91m'
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
║                        YOLO11 Model Loader v{__version__}                               ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  Author: {__author__:<18}  Build: {__build_date__:<14}                               ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  {__copyright__}             ║
╚══════════════════════════════════════════════════════════════════════════════════╝"""
    print(banner)


def detect_device():
    """Detect Jetson/Advantech device information."""
    device_info = {
        "model": "Unknown Device",
        "product": "Not Specified",
        "vendor": "Unknown",
        "os": "Unknown",
        "architecture": "Unknown",
        "compute_capability": "Unknown",
        "cuda_cores": 0,
        "memory_gb": 0,
        "jetpack_version": "Unknown",
        "l4t_version": "Unknown",
        "is_jetson": False,
        "is_advantech": False,
        "base_platform": "Unknown",
    }
    try:
        if os.path.exists('/etc/os-release'):
            with open('/etc/os-release', 'r') as f:
                for line in f:
                    if line.startswith('PRETTY_NAME='):
                        device_info["os"] = line.split('=')[1].strip().strip('"')
                        break
    except Exception:
        pass

    dt_model = None
    try:
        if os.path.exists('/proc/device-tree/model'):
            with open('/proc/device-tree/model', 'r') as f:
                dt_model = f.read().strip().replace('\x00', '')
                device_info["product"] = dt_model
    except Exception:
        pass

    l4t_major = None
    try:
        if os.path.exists('/etc/nv_tegra_release'):
            with open('/etc/nv_tegra_release', 'r') as f:
                release_info = f.read().strip()
                match = re.search(r'R(\d+)\s*\(release\),\s*REVISION:\s*([\d.]+)', release_info)
                if match:
                    l4t_major = int(match.group(1))
                    l4t_revision = match.group(2)
                    device_info["l4t_version"] = f"R{l4t_major}.{l4t_revision}"
    except Exception:
        pass

    if l4t_major is not None:
        if l4t_major >= 36:
            device_info["jetpack_version"] = "6.x"
        elif l4t_major >= 35:
            device_info["jetpack_version"] = "5.1.x"
        elif l4t_major >= 32:
            device_info["jetpack_version"] = "4.x"

    if dt_model and "Orin" in dt_model:
        device_info.update({"is_jetson": True, "architecture": "Ampere",
                             "compute_capability": "8.7", "base_platform": "Orin"})
        if "AGX" in dt_model:
            device_info.update({"model": "NVIDIA Jetson AGX Orin", "cuda_cores": 2048})
        elif "NX" in dt_model:
            device_info.update({"model": "NVIDIA Jetson Orin NX", "cuda_cores": 1024})
        else:
            device_info.update({"model": "NVIDIA Jetson Orin Nano", "cuda_cores": 1024})
    elif dt_model and "Xavier" in dt_model:
        device_info.update({"is_jetson": True, "architecture": "Volta",
                             "compute_capability": "7.2", "base_platform": "Xavier"})
        if "NX" in dt_model:
            device_info.update({"model": "NVIDIA Jetson Xavier NX", "cuda_cores": 384})
        else:
            device_info.update({"model": "NVIDIA Jetson AGX Xavier", "cuda_cores": 512})

    try:
        if os.path.exists('/sys/class/dmi/id/board_vendor'):
            with open('/sys/class/dmi/id/board_vendor', 'r') as f:
                vendor = f.read().strip()
                device_info["vendor"] = vendor
                if "Advantech" in vendor:
                    device_info["is_advantech"] = True
                    base = device_info["base_platform"]
                    device_info["model"] = f"Advantech {base}-based AIE"
    except Exception:
        pass

    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            device_info["memory_gb"] = round(props.total_memory / (1024 ** 3))
    except Exception:
        pass

    if device_info["memory_gb"] == 0:
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                mem_mb = int(result.stdout.strip().split('\n')[0])
                device_info["memory_gb"] = round(mem_mb / 1024)
        except Exception:
            pass

    return device_info


def detect_libraries():
    """Detect installed libraries and their versions."""
    libraries = {}
    checks = [
        ("ultralytics", "ultralytics"),
        ("torch",        "torch"),
        ("torchvision",  "torchvision"),
        ("onnxruntime",  "onnxruntime"),
        ("tensorrt",     "tensorrt"),
        ("cv2",          "cv2"),
    ]
    for name, module in checks:
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'installed')
            notes = ""
            if name == "torch":
                import torch as _t
                notes = (f"{Colors.GREEN}CUDA Available{Colors.ENDC}"
                         if _t.cuda.is_available()
                         else f"{Colors.YELLOW}CPU Only{Colors.ENDC}")
            elif name == "onnxruntime":
                import onnxruntime as _ort
                providers = _ort.get_available_providers()
                notes = (f"{Colors.GREEN}GPU Available{Colors.ENDC}"
                         if 'CUDAExecutionProvider' in providers
                         else f"{Colors.YELLOW}CPU Only{Colors.ENDC}")
            libraries[name] = {"installed": True, "version": str(version), "notes": notes}
        except ImportError:
            libraries[name] = {"installed": False, "version": "N/A", "notes": ""}
    return libraries


def display_device_info(device_info):
    """Display detected device information."""
    header = "Detected Advantech Device" if device_info["is_advantech"] else "Detected Device"
    print(f"\n{Colors.BOLD}{header}{Colors.ENDC}")
    print(f"  Model:              {device_info['model']}")
    print(f"  Product:            {device_info['product']}")
    print(f"  Vendor:             {device_info['vendor']}")
    print(f"  OS:                 {device_info['os']}")
    print(f"  JetPack Version:    {device_info['jetpack_version']}")
    print(f"  L4T Version:        {device_info['l4t_version']}")
    print(f"  GPU Architecture:   {device_info['architecture']}")
    print(f"  Compute Capability: {device_info['compute_capability']}")
    print(f"  CUDA Cores:         {device_info['cuda_cores']}")
    print(f"  Memory:             {device_info['memory_gb']} GB")


def display_libraries(libraries):
    """Display detected libraries in a table."""
    print(f"\n{Colors.BOLD}Detected Libraries{Colors.ENDC}")
    print(f"  {'Library':<14} {'Status':<14} {'Version':<20} Notes")
    print("  " + "-" * 72)
    for lib, info in libraries.items():
        status_text = "Installed" if info["installed"] else "Not Installed"
        status_colored = (f"{Colors.GREEN}{status_text}{Colors.ENDC}"
                          if info["installed"]
                          else f"{Colors.RED}{status_text}{Colors.ENDC}")
        pad_status  = 14 - len(status_text)
        pad_version = 20 - len(info["version"])
        print(f"  {lib:<14} {status_colored}{' ' * pad_status} "
              f"{info['version']}{' ' * pad_version} {info['notes']}")


def interactive_mode(save_dir):
    """Run interactive model selection and download."""
    os.system('clear' if os.name != 'nt' else 'cls')
    print_banner()

    print(f"\n{Colors.CYAN}ℹ Detecting device and libraries...{Colors.ENDC}")
    device_info = detect_device()
    libraries   = detect_libraries()

    display_device_info(device_info)
    display_libraries(libraries)

    if not libraries.get("ultralytics", {}).get("installed", False):
        print(f"\n{Colors.RED}✗ ultralytics is required. "
              f"Install with: pip3 install ultralytics{Colors.ENDC}")
        return

    MODEL_OPTIONS = [
        (1, "yolo11n.pt",     "Detection",      True,  "Recommended for real-time applications"),
        (2, "yolo11n-seg.pt", "Segmentation",   True,  "Recommended for real-time applications"),
        (3, "yolo11n-cls.pt", "Classification", True,  "Recommended for real-time applications"),
        (4, "yolo11s.pt",     "Detection",      False, "Higher accuracy, moderate speed"),
        (5, "yolo11s-seg.pt", "Segmentation",   False, "Higher accuracy, moderate speed"),
        (6, "yolo11s-cls.pt", "Classification", False, "Higher accuracy, moderate speed"),
    ]

    print(f"\n{Colors.BOLD}YOLO11 Models for Your Device{Colors.ENDC}")
    print("-" * 70)
    for opt_id, model, task, recommended, desc in MODEL_OPTIONS:
        size_label = "Nano" if "11n" in model else "Small"
        label = f"YOLO11 {size_label} {task}"
        if recommended:
            print(f"{Colors.GREEN}[{opt_id}] {label} (RECOMMENDED){Colors.ENDC}")
        else:
            print(f"[{opt_id}] {label}")
        print(f"    Model: {model}")
        print(f"    {desc}")

    max_id = len(MODEL_OPTIONS)
    try:
        print(f"\nEnter option number (1-{max_id}): ", end="", flush=True)
        choice = input().strip()
        if not choice.isdigit() or not (1 <= int(choice) <= max_id):
            print(f"{Colors.RED}✗ Invalid option. "
                  f"Please enter a number between 1 and {max_id}.{Colors.ENDC}")
            return
        selected_model = MODEL_OPTIONS[int(choice) - 1][1]
        download_model(selected_model, save_dir)
    except KeyboardInterrupt:
        print("\n\nOperation cancelled.")
    except EOFError:
        pass


def check_gpu():
    """Check if GPU is available"""
    if torch.cuda.is_available():
        print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("✗ No GPU detected")
        return False


def download_model(model_name, save_dir):
    """Download a YOLO11 model"""
    print(f"\nDownloading {model_name}...")
    try:
        model = YOLO(model_name)
        print(f"✓ {model_name} downloaded successfully")
        print(f"  Model type: {model.task}")
        print(f"  Model path: {model.ckpt_path}")
        return True
    except Exception as e:
        print(f"✗ Error downloading {model_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download YOLO11 models for Advantech Edge AI Devices',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 advantech-coe-model-load.py                    # Interactive mode
  python3 advantech-coe-model-load.py --model yolo11n    # Download specific model
  python3 advantech-coe-model-load.py --all-detect       # Download n and s detection models
  python3 advantech-coe-model-load.py --all              # Download all recommended models

Available models:
  Detection:      yolo11n, yolo11s, yolo11m, yolo11l, yolo11x
  Segmentation:   yolo11n-seg, yolo11s-seg, yolo11m-seg, yolo11l-seg, yolo11x-seg
  Classification: yolo11n-cls, yolo11s-cls, yolo11m-cls, yolo11l-cls, yolo11x-cls

Recommended for edge devices: yolo11n, yolo11s (nano and small variants)
        """
    )
    parser.add_argument('--model', type=str, default=None,
                        help='Specific model to download (e.g., yolo11n.pt, yolo11s-seg.pt)')
    parser.add_argument('--all-detect', action='store_true',
                        help='Download all recommended detection models (n and s)')
    parser.add_argument('--all-seg', action='store_true',
                        help='Download all recommended segmentation models (n and s)')
    parser.add_argument('--all-cls', action='store_true',
                        help='Download all recommended classification models (n and s)')
    parser.add_argument('--all', action='store_true',
                        help='Download all recommended models for edge deployment')
    parser.add_argument('--save-dir', type=str, default='/advantech/models',
                        help='Directory to save models (default: /advantech/models)')
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # No model-specific args → interactive mode
    if not any([args.model, args.all_detect, args.all_seg, args.all_cls, args.all]):
        interactive_mode(save_dir)
        return

    # CLI mode
    check_gpu()
    print("\n" + "=" * 70)
    print("YOLO11 Model Download Utility - Advantech Edge AI")
    print("=" * 70)
    print(f"Save directory: {save_dir}")
    print("=" * 70 + "\n")

    models_to_download = []
    if args.model:
        model_name = args.model if args.model.endswith('.pt') else f"{args.model}.pt"
        models_to_download.append(model_name)
    elif args.all_detect:
        models_to_download = ['yolo11n.pt', 'yolo11s.pt']
    elif args.all_seg:
        models_to_download = ['yolo11n-seg.pt', 'yolo11s-seg.pt']
    elif args.all_cls:
        models_to_download = ['yolo11n-cls.pt', 'yolo11s-cls.pt']
    elif args.all:
        models_to_download = [
            'yolo11n.pt',     'yolo11s.pt',
            'yolo11n-seg.pt', 'yolo11s-seg.pt',
            'yolo11n-cls.pt', 'yolo11s-cls.pt',
        ]

    print(f"Downloading {len(models_to_download)} model(s)...\n")
    success_count = sum(download_model(m, save_dir) for m in models_to_download)

    print("\n" + "=" * 70)
    print(f"Download Summary: {success_count}/{len(models_to_download)} successful")
    print("=" * 70)

    if success_count > 0:
        print("\n✓ Models are ready to use!")
        print("\nNext steps:")
        print("  1. Export to TensorRT: python3 src/advantech-coe-model-export.py")
        print("  2. Run inference:      python3 src/advantech-yolo.py")


if __name__ == '__main__':
    main()
