# Advantech YOLO11 Vision Applications Container

**Version 1.0.0 | December 2025 | Advantech Corporation**

A containerized toolkit for deploying YOLO11 computer vision applications on Advantech edge AI devices with hardware acceleration.

---

## Overview

This repository provides a ready-to-use Docker environment for running YOLO11 inference on NVIDIA Jetson-based Advantech hardware. The container includes pre-configured software for object detection, instance segmentation, and image classification with full GPU acceleration.

The toolkit automatically detects device capabilities and configures optimal settings for real-time performance.

**Supported Applications:**

| Application | Description |
|:------------|:------------|
| Object Detection | Real-time detection with bounding boxes for 80 COCO classes |
| Instance Segmentation | Pixel-level masks for precise object boundaries |
| Image Classification | Categorization across 1,000 ImageNet classes |

### Supported Hardware

| Specification | Details |
|:--------------|:--------|
| Platform | NVIDIA Jetson (Orin-nano, Orin-nx, AGX Orin) |
| GPU Architecture | Ampere |
| Memory | 8GB, 16GB, 32GB, or 64GB shared |
| JetPack | 6.x |

For troubleshooting, see the [Troubleshooting Guide](TROUBLESHOOTING_YOLO11.md).

---

## Table of Contents

- [System Requirements](#system-requirements)
- [Before You Start](#before-you-start)
- [Quick Start](#quick-start)
- [Model Management](#model-management)
- [Running Inference](#running-inference)
- [Export Formats](#export-formats)
- [Performance Guidelines](#performance-guidelines)
- [Directory Structure](#directory-structure)
- [Limitations](#limitations)
- [License](#license)

---

## System Requirements

### Host System Requirements

| Component | Version/Requirement |
|-----------|---------|
| **JetPack** | 6.x |
| **CUDA** | 12.6.68 |
| **cuDNN** | 9.3.0.75 |
| **TensorRT** | 10.3.0.30 |
| **OpenCV** | 4.8.0 |

* CUDA , CuDNN , TensorRT , OpenCV versions Depends on JetPack version 6.x

### General Required Packages on Host System

Install these components on your Advantech device before using this toolkit.

| Component | Version |
|:----------|:--------|
| JetPack | 6.0+ |
| CUDA | 12.2+ |
| cuDNN | 8.9.4+ |
| TensorRT | 8.6.2+ |
| OpenCV | 4.8+ |
| Docker | 28.1.1 or later |
| Docker Compose | 2.39.1 or later |
| NVIDIA Container Toolkit | 1.11.0 or later |

Component versions depend on your **JetPack Version**. See [NVIDIA JetPack Documentation](https://developer.nvidia.com/embedded/jetpack) for package version and for installation please refer to [SDK Manager](https://developer.nvidia.com/sdk-manager).

### Container Environment

The Docker container includes the following pre-configured components.

| Component | Version | Description |
|:----------|:--------|:------------|
| CUDA | 12.2+ | GPU computing platform |
| cuDNN | 8.9.4+ | Deep Neural Network library |
| TensorRT | 8.6.2+ | Inference optimizer and runtime |
| PyTorch | 2.5.0 | Deep learning framework (user-installed) |
| ONNX Runtime | 1.23.0 | Cross-platform inference engine (user-installed) |
| TorchVision | 0.20.0 | Computer vision library (user-installed) |
| OpenCV | 4.8+ | Computer vision library with CUDA |
| GStreamer | 1.20+ | Multimedia framework |

## Before You Start

Ensure the following prerequisites are met:

- **Docker**: Version `28.1.1` or later
- **Docker Compose**: Version `2.39.1` or later
- **NVIDIA Container Toolkit**: Version `1.11.0` or later

For installation instructions, refer to the [Installation Guide](https://github.com/yqlbu/jetson-packages-family/blob/main/README.md).

Before proceeding, ensure that your system meets the required [general-required-packages-on-host-system](#general-required-packages-on-host-system). If you encounter any issues or inconsistencies in your environment, please consult our [Troubleshooting Guide](TROUBLESHOOTING_YOLO11.md) for solutions and to verify that all prerequisites are properly satisfied.

- Ensure the following components are installed on your device along with other packages mentioned in the [general-required-packages-on-host-system](#general-required-packages-on-host-system):
  - **Docker**
  - **Docker Compose**
  - **NVIDIA Container Toolkit**
  - **NVIDIA Runtime**

---

## Quick Start

### Step 1: Clone the Repository

Download the toolkit to your device.

```bash
git clone https://github.com/Advantech-EdgeSync-Containers/Advantech-YOLO-Vision-Applications.git
cd Advantech-YOLO-Vision-Applications
```

### Step 2: Set Permissions

Grant execute permissions to the setup scripts.

```bash
sudo chmod +x *.sh
```

### Step 3: Start the Container

Launch the Docker environment. This script creates project directories, configures GPU access, and opens an interactive terminal inside the container.

```bash
./build.sh
```

### Step 4: Install Dependencies

Inside the container, install PyTorch, TorchVision, ONNX Runtime, and the YOLO11 framework. These specific versions are validated against the container's software stack for JetPack 6.

```bash
# Install PyTorch 2.5.0 (ARM64 optimized wheel for JetPack 6)
pip3 install --no-cache-dir https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl

# Install TorchVision 0.20.0 (ARM64 optimized wheel)
pip3 install --no-cache-dir https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl

# Install ONNX Runtime GPU 1.23.0 (ARM64 wheel for JetPack 6)
pip3 install --no-cache-dir https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl

# Install Ultralytics YOLO11
pip3 install ultralytics==8.3.0 --no-deps
```

### Step 5: Verify Installation (Optional)

Run a basic test to confirm hardware acceleration is working.

```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

Expected output:

```
PyTorch: 2.5.0a0+872d972e41.nv24.08
CUDA Available: True
```

### Step 6: Verify AI Accelerator (Optional)

Verify that the AI accelerator is properly configured and accessible. Use the Wise-Bench tool to run benchmarks on your device.

```bash
# Ensure you're inside the container
chmod +x wise-bench.sh
./wise-bench.sh
```


The script runs comprehensive tests to validate GPU functionality and report performance metrics. Expected output confirms CUDA availability, memory allocation success, and benchmark completion.

![Wise-Bench Output](data/wisebench.png)

---

## Model Management

### Downloading Models

The model loader detects your hardware and recommends appropriate models.

```bash
python3 src/advantech-coe-model-load.py
```

![Model Load Menu Demo](data/load.gif)

The utility presents an interactive menu. Select a model by entering its number.

| Option | Model | Task | Description |
|:-------|:------|:-----|:------------|
| 1 | YOLO11n | Detection | Recommended for real-time applications |
| 2 | YOLO11n-seg | Segmentation | Recommended for real-time applications |
| 3 | YOLO11n-cls | Classification | Recommended for real-time applications |
| 4 | YOLO11s | Detection | Higher accuracy, moderate speed |
| 5 | YOLO11s-seg | Segmentation | Higher accuracy, moderate speed |
| 6 | YOLO11s-cls | Classification | Higher accuracy, moderate speed |

Nano (n) models provide the fastest inference and are recommended for real-time applications. Small (s) models offer higher accuracy at reduced speed.

### Exporting Models

Convert models to optimized formats for deployment.

```bash
python3 src/advantech-coe-model-export.py
```

![Model Export Menu Demo](data/export.gif)

The export utility guides you through three selections (Task, Size, Format).

**Task Selection:**

| Option | Task | Input Size |
|:-------|:-----|:-----------|
| 1 | Object Detection | 640×640 |
| 2 | Instance Segmentation | 640×640 |
| 3 | Classification | 224×224 |

**Model Size Selection:**

| Option | Size | Characteristics |
|:-------|:-----|:----------------|
| 1 | Nano | Fastest inference, best for real-time |
| 2 | Small | Good balance of speed and accuracy |
| 3 | Medium | Higher accuracy, moderate speed |
| 4 | Large | High accuracy, slower inference |
| 5 | XLarge | Maximum accuracy, slowest inference |

**Format Selection:**

| Option | Format | Recommendation |
|:-------|:-------|:---------------|
| 1 | ONNX (CPU mode) | Development and testing |
| 2 | ONNX (GPU mode, FP16) | When TensorRT unavailable |
| 3 | TensorRT Engine (FP16) | **Recommended for production** |
| 4 | PyTorch | Ultralytics native |

---

## Running Inference

### Interactive Mode (User Mode)

Launch the main application with menu-driven configuration.

```bash
python3 src/advantech-yolo.py
```

The application prompts for task type, model path, input source, confidence threshold, and display/save options. Press `q` in the display window or `Ctrl+C` to stop.

---

## Interactive Workflow Results

### Object Detection

![Detection](data/Detection.gif)

### Instance Segmentation

![Segmentation](data/segmentation.gif)

### Classification

![Classification](data/classification.gif)

---

### Command Line Mode (Developer Mode)

For more details related to CLI parameters:

```bash
python3 src/advantech-yolo.py -h
```

Expected output:

```
usage: advantech-yolo.py [-h] [--input INPUT] [--model MODEL]
                         [--task {detect,segment,classify}]
                         [--conf CONF] [--iou IOU] [--device DEVICE]
                         [--show] [--save] [--save-dir SAVE_DIR]

YOLO11 Vision Application for Advantech Edge AI Devices

options:
  -h, --help            show this help message and exit
  --input INPUT         Input source: 0 for webcam, path to video/image, or RTSP URL
  --model MODEL         YOLO11 model path (e.g., yolo11n.pt, yolo11n-seg.pt)
  --task {detect,segment,classify}
                        Task type: detect, segment, or classify
  --conf CONF           Confidence threshold (default: 0.25)
  --iou IOU             IoU threshold for NMS (default: 0.45)
  --device DEVICE       Device to run on: 0 for GPU, cpu for CPU
  --show                Display results in window
  --save                Save results to output directory
  --save-dir SAVE_DIR   Directory to save results (default: /advantech/results)

Examples:
  python3 advantech-yolo.py
  python3 advantech-yolo.py --input 0 --task detect --model yolo11n.pt --show
  python3 advantech-yolo.py --input data/test.mp4 --task segment --model yolo11n-seg.pt --show --save
  python3 advantech-yolo.py --input 0 --task classify --model yolo11n-cls.pt --show
```

For scripted or automated use, specify options directly.

```bash
python3 src/advantech-yolo.py --model yolo11n.engine --input 0 --task detect
```

```bash
python3 src/advantech-yolo.py --model yolo11n-seg.engine --input /path/to/your/video_file --task segment
```

```bash
python3 src/advantech-yolo.py --model yolo11n-cls.pt --input rtsp://your-camera-ip:port/ --task classify
```

**Available Options:**

| Option | Description | Default |
|:-------|:------------|:--------|
| `--model` | Model file path; omit (with `--input`) to launch interactive mode | `yolo11n.pt` |
| `--input` | Input source (device number, URL, or file path); omit (with `--model`) to launch interactive mode | `0` |
| `--task` | Task type: `detect`, `segment`, or `classify` | `detect` |
| `--conf` | Confidence threshold | `0.25` |
| `--iou` | IoU threshold for NMS | `0.45` |
| `--device` | Device to run on: `0` for GPU, `cpu` for CPU | `0` |
| `--show` | Display results in window | `False` |
| `--save` | Save results to output directory | `False` |
| `--save-dir` | Directory to save results | `/advantech/results` |

---

## Export Formats

Choosing the correct export format significantly affects inference speed.

### TensorRT Engine (Recommended)

TensorRT is NVIDIA's inference optimizer and produces the fastest results on Jetson hardware. It fuses operations, optimizes memory layout, and calibrates precision automatically. FP16 precision reduces memory usage with minimal accuracy loss. Note that engine files are device-specific and not portable between different GPU architectures.

### ONNX (GPU Mode)

ONNX with CUDA execution provides good performance when TensorRT is unavailable. The FP16 variant uses half-precision for improved throughput. This format offers cross-platform compatibility but moderate performance compared to TensorRT.

### ONNX (CPU Mode)

ONNX with CPU execution runs on any system without GPU requirements. This format provides universal compatibility but the slowest performance. It is suitable for development and testing only.

### PyTorch

PyTorch's serialization format maintains ecosystem compatibility as it's Ultralytics native and is useful for PyTorch-specific deployment pipelines. Performance is Fast.

**Performance Comparison:**

| Format | Relative Speed | Primary Use Case |
|:-------|:---------------|:-----------------|
| TensorRT FP16 | Fastest | Production deployment |
| ONNX GPU FP16 | Fast | TensorRT unavailable |
| ONNX CPU | Slow | Development and testing |
| PyTorch | Fast | Ultralytics native |

---

## Performance Guidelines

### Recommended Configurations

| Task | Model | Format | Notes |
|:-----|:------|:-------|:------|
| Object Detection | YOLO11n or YOLO11s | TensorRT/ONNX | Best real-time performance |
| Instance Segmentation | YOLO11n-seg or YOLO11s-seg | TensorRT | Includes mask output |
| Classification | YOLO11n-cls or YOLO11s-cls | PyTorch | 224×224 input size |

### Optimization Notes

Use Nano models for applications that require the maximum frame rate. Use Small models when accuracy is prioritized over speed. TensorRT engines must be rebuilt when moving between different Jetson models. Input resolution affects both speed and accuracy; 640×480 is the default for detection and segmentation. FP16 precision is enabled by default for TensorRT exports.

## Directory Structure

```
Advantech-YOLO-Vision-Applications/
├── src/
│   ├── advantech-coe-model-load.py    # Model download utility
│   ├── advantech-coe-model-export.py  # Model export utility
│   ├── advantech-yolo.py              # Main inference application
│   ├── advantech_core.py              # Inference engine implementations
│   └── advantech_classes.py           # Class label definitions
├── data/                               # Sample data and outputs
├── models/                             # Model storage (created at runtime)
├── docker-compose.yml                  # Container configuration
├── build.sh                            # Container launch script
├── LICENSE                             # GPL-3.0 license
└── README_YOLO11.md                    # This file
```

---

## Limitations

| Limitation | Description |
|:-----------|:------------|
| Model Size | Only Nano and Small variants are optimized for real-time inference on edge devices |
| Pre-trained Only | Custom training requires external tools; import trained models for inference |
| Resolution | Performance degrades above 1080p; use 640×480 or 1280×720 for best results |
| Network | Model download requires internet connectivity; inference runs offline |
| Display | Visualization requires X11 forwarding (run `xhost +local:docker` on host) |
| Classes | Detection uses COCO classes; classification uses ImageNet classes |
| Streams | Single input stream per container instance |

---

## Use Cases

This toolkit is ideal for:

* **Industrial Quality Inspection:** Detect defects and inspect parts with instance segmentation
* **Smart Retail:** Product recognition, customer behavior analysis
* **Smart Cities:** Traffic monitoring, crowd analysis, object tracking
* **Security & Surveillance:** Perimeter monitoring, intrusion detection
* **Agriculture:** Crop monitoring, livestock tracking
* **Healthcare:** Medical image analysis, equipment tracking
* **Robotics:** Environmental perception, object manipulation guidance

---

## License

GNU General Public License v3.0

Copyright © 2025 Advantech Corporation. All rights reserved.
This software is provided by Advantech Corporation "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose, are disclaimed.

For complete license details, see [LICENSE](LICENSE) for complete terms.

For troubleshooting and FAQ, visit the [Wiki](https://github.com/yqlbu/jetson-packages-family/wiki).

---

## Acknowledgments

This toolkit builds upon [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) for the core detection, segmentation, and classification framework, and [NVIDIA](https://developer.nvidia.com/) for CUDA, TensorRT, and the Jetson platform.

Required framework installation:

```bash
# Install PyTorch 2.5.0 (ARM64 optimized wheel for JetPack 6)
pip3 install --no-cache-dir https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl

# Install TorchVision 0.20.0 (ARM64 optimized wheel)
pip3 install --no-cache-dir https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl

# Install ONNX Runtime GPU 1.23.0 (ARM64 wheel for JetPack 6)
pip3 install --no-cache-dir https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl

# Install Ultralytics YOLO11
pip3 install ultralytics==8.3.0 --no-deps
```

---

## Support

For documentation and troubleshooting, visit the [Troubleshooting Guide](TROUBLESHOOTING_YOLO11.md).

For issues, submit to [GitHub Issues](https://github.com/Advantech-EdgeSync-Containers/Advantech-YOLO-Vision-Applications/issues).

---

Advantech Corporation — Center of Excellence
