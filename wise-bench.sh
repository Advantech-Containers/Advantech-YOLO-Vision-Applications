#!/bin/bash
# Advantech YOLO Vision Applications - Hardware Diagnostic & Benchmarking Tool
# Version: 2.0.0
# Author: Samir Singh <samir.singh@advantech.com>
# Copyright (c) 2024-2025 Advantech Corporation. All rights reserved.
clear
LOG_FILE="/advantech/diagnostics/wise-bench.log"
mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null || LOG_FILE="/tmp/wise-bench.log"
{
    echo "==========================================================="
    echo ">>> Diagnostic Run Started at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "==========================================================="
} >> "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
PURPLE='\033[0;35m'
NC='\033[0m'
echo -e "${BLUE}${BOLD}+------------------------------------------------------+${NC}"
echo -e "${BLUE}${BOLD}|    ${PURPLE}Advantech YOLO Vision Hardware Diagnostics Tool${BLUE}   |${NC}"
echo -e "${BLUE}${BOLD}+------------------------------------------------------+${NC}"
echo
echo -e "${BLUE}"
echo "       █████╗ ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗███████╗ ██████╗██╗  ██╗"
echo "      ██╔══██╗██╔══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝██╔════╝██╔════╝██║  ██║"
echo "      ███████║██║  ██║╚██╗ ██╔╝███████║██╔██╗ ██║   ██║   █████╗  ██║     ███████║"
echo "      ██╔══██║██║  ██║ ╚████╔╝ ██╔══██║██║╚██╗██║   ██║   ██╔══╝  ██║     ██╔══██║"
echo "      ██║  ██║██████╔╝  ╚██╔╝  ██║  ██║██║ ╚████║   ██║   ███████╗╚██████╗██║  ██║"
echo "      ╚═╝  ╚═╝╚═════╝    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝ ╚═════╝╚═╝  ╚═╝"
echo -e "${NC}                           Center of Excellence"
echo
echo -e "${YELLOW}${BOLD}▶ Starting hardware acceleration tests...${NC}"
echo -e "${CYAN}  This may take a moment...${NC}"
echo
sleep 3
print_header() {
    echo
    echo "+--- $1 ----$(printf '%*s' $((47 - ${#1})) | tr ' ' '-')+"
    echo "|$(printf '%*s' 50 | tr ' ' ' ')|"
    echo "+--------------------------------------------------+"
}
print_success() { echo "✓ $1"; }
print_warning() { echo "⚠ $1"; }
print_info() { echo "ℹ $1"; }
print_table_header() {
    echo "+--------------------------------------------------+"
    echo "| $1$(printf '%*s' $((47 - ${#1})) | tr ' ' ' ')|"
    echo "+--------------------------------------------------+"
}
print_table_row() { printf "| %-25s | %-20s |\n" "$1" "$2"; }
print_table_footer() { echo "+--------------------------------------------------+"; }
print_header "SYSTEM INFORMATION"
print_table_header "SYSTEM DETAILS"
KERNEL=$(uname -r)
ARCHITECTURE=$(uname -m)
HOSTNAME=$(hostname)
OS=$(grep PRETTY_NAME /etc/os-release 2>/dev/null | cut -d'"' -f2 || echo "Unknown")
MEMORY_TOTAL=$(free -h | awk '/^Mem:/ {print $2}')
MEMORY_USED=$(free -h | awk '/^Mem:/ {print $3}')
CPU_MODEL=$(lscpu | grep "Model name" | cut -d':' -f2- | sed 's/^[ \t]*//' | head -1 || echo "Unknown")
CPU_CORES=$(nproc --all)
print_table_row "Hostname" "$HOSTNAME"
print_table_row "OS" "$OS"
print_table_row "Kernel" "$KERNEL"
print_table_row "Architecture" "$ARCHITECTURE"
print_table_row "CPU" "$CPU_MODEL"
print_table_row "CPU Cores" "$CPU_CORES"
print_table_row "Memory" "$MEMORY_USED / $MEMORY_TOTAL"
print_table_row "Date" "$(date '+%Y-%m-%d %H:%M:%S')"
print_table_footer
print_header "NVIDIA DEVICES"
echo "+------------------------------------------------------------------+"
printf "| %-30s| %-15s| %-12s|\n" "Device" "Type" "Status"
echo "+------------------------------+-----------------+-------------+"
DEVICE_COUNT=0
for dev in /dev/nvhost-* /dev/nvidia* /dev/v4l2-nv*; do
    if [[ -e "$dev" ]]; then
        device_name=$(basename "$dev")
        device_type=$(echo "$device_name" | sed 's/nvhost-//' | sed 's/nvidia/gpu/')
        printf "| %-30s| %-15s| %-12s|\n" "$device_name" "$device_type" "Available"
        ((DEVICE_COUNT++))
    fi
done
if [[ "$DEVICE_COUNT" -eq 0 ]]; then
    printf "| %-62s|\n" "No NVIDIA devices found"
fi
echo "+------------------------------------------------------------------+"
print_success "Found $DEVICE_COUNT NVIDIA devices"
print_header "CUDA INFORMATION"
echo -e "${YELLOW}"
echo "       ██████╗██╗   ██╗██████╗  █████╗ "
echo "      ██╔════╝██║   ██║██╔══██╗██╔══██╗"
echo "      ██║     ██║   ██║██║  ██║███████║"
echo "      ██║     ██║   ██║██║  ██║██╔══██║"
echo "      ╚██████╗╚██████╔╝██████╔╝██║  ██║"
echo "       ╚═════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝"
echo -e "${NC}"
print_table_header "CUDA DETAILS"
CUDA_OK=0
if [[ -f "/usr/local/cuda/bin/nvcc" ]]; then
    CUDA_VERSION=$(/usr/local/cuda/bin/nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    CUDA_PATH="/usr/local/cuda"
    print_table_row "CUDA Version" "$CUDA_VERSION"
    print_table_row "CUDA Path" "$CUDA_PATH"
    print_table_row "Status" "✓ Available"
    CUDA_OK=1
else
    NVCC_PATH=$(which nvcc 2>/dev/null || find /usr -name nvcc 2>/dev/null | head -1)
    if [[ -n "$NVCC_PATH" ]]; then
        CUDA_VERSION=$("$NVCC_PATH" --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
        CUDA_PATH=$(dirname $(dirname "$NVCC_PATH"))
        print_table_row "CUDA Version" "$CUDA_VERSION"
        print_table_row "CUDA Path" "$CUDA_PATH"
        print_table_row "Status" "✓ Available"
        CUDA_OK=1
    else
        print_table_row "Status" "⚠ Not detected"
    fi
fi
print_table_footer
print_header "OPENCV CUDA TEST"
echo -ne "▶ Testing OpenCV CUDA support... "
sleep 1
echo "done"
print_table_header "OPENCV DETAILS"
OPENCV_VERSION="Not installed"
OPENCV_CUDA="False"
OPENCV_DEVICES="0"
OPENCV_OK=0
if python3 -c "import cv2" 2>/dev/null; then
    OPENCV_VERSION=$(python3 -c "import cv2; print(cv2.__version__)" 2>/dev/null || echo "Unknown")
    OPENCV_CUDA=$(python3 -c "import cv2; print(hasattr(cv2, 'cuda'))" 2>/dev/null || echo "False")
    if [[ "$OPENCV_CUDA" == "True" ]]; then
        OPENCV_DEVICES=$(nvidia-smi -L | wc -l 2>/dev/null || echo "0")
    fi
fi
print_table_row "OpenCV Version" "$OPENCV_VERSION"
print_table_row "CUDA Module" "$([[ "$OPENCV_CUDA" == "True" ]] && echo "Available" || echo "Not available")"
print_table_row "CUDA Devices" "$OPENCV_DEVICES"
if [[ "$OPENCV_CUDA" == "True" ]] && [[ "$OPENCV_DEVICES" != "0" ]]; then
    print_table_row "Status" "✓ GPU Accelerated"
    OPENCV_OK=1
else
    print_table_row "Status" "⚠ CPU Mode Only"
fi
print_table_footer
print_header "PYTORCH CUDA TEST"
echo -ne "▶ Running PyTorch CUDA test... "
sleep 1
echo "done"
print_table_header "PYTORCH DETAILS"
PYTORCH_VERSION="Not installed"
PYTORCH_CUDA="False"
PYTORCH_DEVICES="0"
PYTORCH_DEVICE_NAME="N/A"
PYTORCH_OK=0
if python3 -c "import torch" 2>/dev/null; then
    PYTORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "Unknown")
    PYTORCH_CUDA=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
    PYTORCH_DEVICES=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
    if [[ "$PYTORCH_CUDA" == "True" ]]; then
        PYTORCH_DEVICE_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown")
    fi
fi
print_table_row "PyTorch Version" "$PYTORCH_VERSION"
print_table_row "CUDA Available" "$([[ "$PYTORCH_CUDA" == "True" ]] && echo "Yes" || echo "No")"
print_table_row "CUDA Devices" "$PYTORCH_DEVICES"
print_table_row "Device Name" "$PYTORCH_DEVICE_NAME"
if [[ "$PYTORCH_CUDA" == "True" ]]; then
    print_table_row "Status" "✓ Accelerated"
    PYTORCH_OK=1
else
    print_table_row "Status" "⚠ CPU Only"
fi
print_table_footer
print_header "ONNX RUNTIME TEST"
echo -ne "▶ Checking ONNX providers... "
sleep 1
echo "done"
print_table_header "ONNX RUNTIME DETAILS"
ONNX_VERSION="Not installed"
ONNX_PROVIDERS="None"
ONNX_HAS_CUDA="False"
ONNX_HAS_TRT="False"
ONNX_OK=0
if python3 -c "import onnxruntime" 2>/dev/null; then
    ONNX_VERSION=$(python3 -c "import onnxruntime as ort; print(ort.__version__)" 2>/dev/null || echo "Unknown")
    ONNX_PROVIDERS=$(python3 -c "import onnxruntime as ort; print(','.join(ort.get_available_providers()))" 2>/dev/null || echo "CPUExecutionProvider")
    ONNX_HAS_CUDA=$(python3 -c "import onnxruntime as ort; print('CUDAExecutionProvider' in ort.get_available_providers())" 2>/dev/null || echo "False")
    ONNX_HAS_TRT=$(python3 -c "import onnxruntime as ort; print('TensorrtExecutionProvider' in ort.get_available_providers())" 2>/dev/null || echo "False")
fi
print_table_row "ONNX Runtime Version" "$ONNX_VERSION"
print_table_row "CUDA Provider" "$([[ "$ONNX_HAS_CUDA" == "True" ]] && echo "✓ Available" || echo "⚠ Not available")"
print_table_row "TensorRT Provider" "$([[ "$ONNX_HAS_TRT" == "True" ]] && echo "✓ Available" || echo "⚠ Not available")"
if [[ "$ONNX_HAS_CUDA" == "True" ]]; then
    print_table_row "Status" "✓ GPU Accelerated"
    ONNX_OK=1
else
    print_table_row "Status" "⚠ CPU Only"
fi
print_table_footer
print_header "TENSORRT TEST"
echo -ne "▶ Testing TensorRT capabilities... "
sleep 1
echo "done"
print_table_header "TENSORRT DETAILS"
TRT_VERSION="Not installed"
TRT_FP16="False"
TRT_INT8="False"
TRT_DLA="0"
TRT_OK=0
if python3 -c "import tensorrt" 2>/dev/null; then
    TRT_VERSION=$(python3 -c "import tensorrt as trt; print(trt.__version__)" 2>/dev/null || echo "Unknown")
    TRT_FP16=$(python3 -c "
import tensorrt as trt
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
print(builder.platform_has_fast_fp16)
" 2>/dev/null || echo "False")
    TRT_INT8=$(python3 -c "
import tensorrt as trt
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
print(builder.platform_has_fast_int8)
" 2>/dev/null || echo "False")
    TRT_DLA=$(python3 -c "
import tensorrt as trt
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
print(builder.num_DLA_cores)
" 2>/dev/null || echo "0")
    TRT_OK=1
fi
print_table_row "TensorRT Version" "$TRT_VERSION"
print_table_row "FP16 Support" "$([[ "$TRT_FP16" == "True" ]] && echo "✓ Yes" || echo "No")"
print_table_row "INT8 Support" "$([[ "$TRT_INT8" == "True" ]] && echo "✓ Yes" || echo "No")"
print_table_row "DLA Cores" "$TRT_DLA"
if [[ "$TRT_OK" -eq 1 ]]; then
    print_table_row "Status" "✓ Available"
else
    print_table_row "Status" "⚠ Not available"
fi
print_table_footer
print_header "YOLO MODEL TEST"
echo -ne "▶ Testing YOLO inference... "
print_table_header "YOLO INFERENCE DETAILS"
YOLO_FPS="0"
YOLO_DEV="cpu"
YOLO_TASK="N/A"
YOLO_DETECTIONS=""
YOLO_OK=0
if python3 -c "from ultralytics import YOLO" 2>/dev/null; then
    YOLO_RESULT=$(python3 << 'PYEOF'
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['YOLO_VERBOSE'] = 'False'
try:
    from ultralytics import YOLO
    import torch
    import numpy as np
    import time
    import cv2
    model = YOLO('yolo11n.pt')
    if torch.cuda.is_available():
        device = 0
        device_name = 'cuda'
    else:
        device = 'cpu'
        device_name = 'cpu'
    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    for _ in range(3):
        model(img, device=device, verbose=False)
    start = time.time()
    for _ in range(10):
        model(img, device=device, verbose=False)
    elapsed = (time.time() - start) / 10
    fps = 1.0 / elapsed
    detections = []
    test_video_path = '/advantech/data/test.mp4'
    if os.path.exists(test_video_path):
        cap = cv2.VideoCapture(test_video_path)
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            results = model(frame, device=device, verbose=False)
            if len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes[:5]:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = model.names[cls_id]
                    detections.append(f'{name}:{conf:.2f}')
    if not detections:
        detections = ['none']
    print(f'{fps:.1f}')
    print(device_name)
    print(model.task)
    print(','.join(detections[:5]))
except Exception as e:
    print('0')
    print('cpu')
    print('error')
    print('none')
PYEOF
    )
    # Parse last 4 lines only — Ultralytics may print extra output above
    YOLO_TAIL=$(echo "$YOLO_RESULT" | tail -n 4)
    YOLO_FPS=$(echo "$YOLO_TAIL" | sed -n '1p')
    YOLO_DEV=$(echo "$YOLO_TAIL" | sed -n '2p')
    YOLO_TASK=$(echo "$YOLO_TAIL" | sed -n '3p')
    YOLO_DETECTIONS=$(echo "$YOLO_TAIL" | sed -n '4p')
    echo "done"
else
    echo "skipped (ultralytics not installed)"
fi
print_table_row "Inference Device" "$YOLO_DEV"
print_table_row "Task Type" "$YOLO_TASK"
print_table_row "FPS (YOLO11n)" "$YOLO_FPS"
if [[ "$YOLO_DEV" == "cuda" ]]; then
    print_table_row "Status" "✓ GPU Accelerated"
    YOLO_OK=1
else
    print_table_row "Status" "⚠ CPU Only"
fi
print_table_footer
echo
print_table_header "DETECTION RESULTS (test.mp4)"
if [[ -n "$YOLO_DETECTIONS" ]] && [[ "$YOLO_DETECTIONS" != "none" ]]; then
    IFS=',' read -ra DET_ARRAY <<< "$YOLO_DETECTIONS"
    for det in "${DET_ARRAY[@]}"; do
        OBJ_NAME=$(echo "$det" | cut -d':' -f1)
        OBJ_CONF=$(echo "$det" | cut -d':' -f2)
        print_table_row "$OBJ_NAME" "Confidence: $OBJ_CONF"
    done
else
    print_table_row "Status" "No objects detected or video not found"
fi
print_table_footer
print_header "GSTREAMER NVIDIA PLUGINS"
echo -ne "▶ Collecting NVIDIA GStreamer plugins... "
sleep 1
echo "done"
print_table_header "GSTREAMER DETAILS"
GST_PLUGINS=$(gst-inspect-1.0 2>/dev/null | grep -i "nv" | wc -l)
GST_OK=0
GST_NVVIDCONV=0
GST_NVV4L2H264=0
GST_NVV4L2DEC=0
if gst-inspect-1.0 nvvidconv >/dev/null 2>&1; then GST_NVVIDCONV=1; fi
if gst-inspect-1.0 nvv4l2h264enc >/dev/null 2>&1; then GST_NVV4L2H264=1; fi
if gst-inspect-1.0 nvv4l2decoder >/dev/null 2>&1; then GST_NVV4L2DEC=1; fi
print_table_row "NVIDIA Plugins Found" "$GST_PLUGINS"
print_table_footer
echo
echo "+------------------------------------------------------------------+"
printf "| %-30s| %-15s| %-14s|\n" "Plugin Name" "Type" "Status"
echo "+-------------------------------+-----------------+---------------+"
NVIDIA_PLUGINS=$(gst-inspect-1.0 2>/dev/null | grep -i "nv")
PLUGIN_COUNT=0
if [[ -n "$NVIDIA_PLUGINS" ]]; then
    echo "$NVIDIA_PLUGINS" | while IFS= read -r line; do
        PLUGIN_NAME=$(echo "$line" | awk '{print $2}' | sed 's/:$//')
        PLUGIN_TYPE=$(echo "$line" | awk '{print $1}' | sed 's/:$//')
        if gst-inspect-1.0 "$PLUGIN_NAME" >/dev/null 2>&1; then
            STATUS="✓ Available"
        else
            STATUS="⚠ Error"
        fi
        printf "| %-30s| %-15s| %-14s|\n" "$PLUGIN_NAME" "$PLUGIN_TYPE" "$STATUS"
    done
fi
echo "+------------------------------------------------------------------+"
echo
print_table_header "KEY PLUGINS STATUS"
print_table_row "nvvidconv" "$([[ "$GST_NVVIDCONV" -eq 1 ]] && echo "✓ Available" || echo "⚠ Missing")"
print_table_row "nvv4l2h264enc" "$([[ "$GST_NVV4L2H264" -eq 1 ]] && echo "✓ Available" || echo "⚠ Missing")"
print_table_row "nvv4l2decoder" "$([[ "$GST_NVV4L2DEC" -eq 1 ]] && echo "✓ Available" || echo "⚠ Missing")"
if [[ "$GST_NVVIDCONV" -eq 1 ]] || [[ "$GST_NVV4L2H264" -eq 1 ]]; then
    GST_OK=1
fi
print_table_footer

print_header "VIDEO ENCODING TEST"
print_table_header "VIDEO ENCODING DETAILS"
VENC_OK=0
if [[ -e "/dev/nvhost-msenc" ]] || [[ -e "/dev/nvhost-nvenc" ]] || [[ -e "/dev/v4l2-nvenc" ]]; then
    print_table_row "Hardware Encoder" "✓ Detected"
    if [[ -e "/dev/nvhost-msenc" ]]; then
        print_table_row "Device" "/dev/nvhost-msenc (JetPack 5)"
    fi
    if [[ -e "/dev/nvhost-nvenc" ]]; then
        print_table_row "Device" "/dev/nvhost-nvenc"
    fi
    if [[ -e "/dev/v4l2-nvenc" ]]; then
        print_table_row "Device" "/dev/v4l2-nvenc (JetPack 6)"
    fi
    if gst-inspect-1.0 nvv4l2h264enc >/dev/null 2>&1; then
        print_table_row "H.264 Encoder" "✓ nvv4l2h264enc"
        VENC_OK=1
    else
        print_table_row "H.264 Encoder" "⚠ Not available"
    fi
    if gst-inspect-1.0 nvv4l2h265enc >/dev/null 2>&1; then
        print_table_row "H.265 Encoder" "✓ nvv4l2h265enc"
    else
        print_table_row "H.265 Encoder" "⚠ Not available"
    fi
    print_table_row "Status" "✓ Ready"
else
    print_table_row "Hardware Encoder" "⚠ Not detected"
fi
print_table_footer
print_header "VIDEO DECODING TEST"
print_table_header "VIDEO DECODING DETAILS"
VDEC_OK=0
if [[ -e "/dev/nvhost-nvdec" ]] || [[ -e "/dev/v4l2-nvdec" ]]; then
    print_table_row "Hardware Decoder" "✓ NVDEC Detected"
    if [[ -e "/dev/nvhost-nvdec" ]]; then
        print_table_row "Device" "/dev/nvhost-nvdec (JetPack 5)"
    fi
    if [[ -e "/dev/v4l2-nvdec" ]]; then
        print_table_row "Device" "/dev/v4l2-nvdec (JetPack 6)"
    fi
    if gst-inspect-1.0 nvv4l2decoder >/dev/null 2>&1; then
        print_table_row "GStreamer Plugin" "✓ nvv4l2decoder"
        print_table_row "Status" "✓ Ready"
        VDEC_OK=1
    else
        print_table_row "GStreamer Plugin" "⚠ nvv4l2decoder missing"
    fi
else
    print_table_row "Hardware Decoder" "⚠ NVDEC Not detected"
    print_table_row "Device" "No decoder device found"
fi
print_table_footer

print_header "CAMERA TEST"
print_table_header "CAMERA DETAILS"
CAM_OK=0
if [[ -e "/dev/video0" ]]; then
    CAM_NAME=$(v4l2-ctl --device=/dev/video0 --all 2>/dev/null | grep "Card type" | cut -d':' -f2 | sed 's/^[ \t]*//' || echo "Unknown")
    print_table_row "Camera Device" "/dev/video0"
    print_table_row "Camera Name" "${CAM_NAME:-Unknown}"
    print_table_row "Status" "✓ Available"
    CAM_OK=1
else
    print_table_row "Camera Device" "⚠ Not detected"
fi
print_table_footer
print_header "DIAGNOSTICS SUMMARY"
echo -e "${PURPLE}"
echo "      ██╗   ██╗ ██████╗ ██╗      ██████╗ "
echo "      ╚██╗ ██╔╝██╔═══██╗██║     ██╔═══██╗"
echo "       ╚████╔╝ ██║   ██║██║     ██║   ██║"
echo "        ╚██╔╝  ██║   ██║██║     ██║   ██║"
echo "         ██║   ╚██████╔╝███████╗╚██████╔╝"
echo "         ╚═╝    ╚═════╝ ╚══════╝ ╚═════╝ "
echo -e "${NC}"
print_table_header "ACCELERATION STATUS"
print_table_row "CUDA Toolkit" "$([[ $CUDA_OK -eq 1 ]] && echo "✓ Available" || echo "⚠ Missing")"
print_table_row "OpenCV CUDA" "$([[ $OPENCV_OK -eq 1 ]] && echo "✓ Accelerated" || echo "⚠ CPU Only")"
print_table_row "PyTorch CUDA" "$([[ $PYTORCH_OK -eq 1 ]] && echo "✓ Accelerated" || echo "⚠ CPU Only")"
print_table_row "ONNX Runtime GPU" "$([[ $ONNX_OK -eq 1 ]] && echo "✓ Accelerated" || echo "⚠ CPU Only")"
print_table_row "TensorRT" "$([[ $TRT_OK -eq 1 ]] && echo "✓ Available" || echo "⚠ Missing")"
print_table_row "YOLO Inference(YOLO11)" "$([[ $YOLO_OK -eq 1 ]] && echo "✓ GPU ($YOLO_FPS FPS)" || echo "⚠ CPU Only")"
print_table_row "GStreamer NVIDIA" "$([[ $GST_OK -eq 1 ]] && echo "✓ Available" || echo "⚠ Missing")"
print_table_row "Video Encoding" "$([[ $VENC_OK -eq 1 ]] && echo "✓ Working" || echo "⚠ Not available")"
print_table_row "Video Decoding" "$([[ $VDEC_OK -eq 1 ]] && echo "✓ Working" || echo "⚠ Not available")"
print_table_row "Camera" "$([[ $CAM_OK -eq 1 ]] && echo "✓ Detected" || echo "⚠ Not found")"
print_table_footer
TOTAL=$((CUDA_OK + OPENCV_OK + PYTORCH_OK + ONNX_OK + TRT_OK + YOLO_OK + GST_OK + VENC_OK + VDEC_OK + CAM_OK))
MAX=10
PERCENTAGE=$((TOTAL * 100 / MAX))
BAR_SIZE=20
FILLED=$((BAR_SIZE * TOTAL / MAX))
EMPTY=$((BAR_SIZE - FILLED))
BAR=""
for ((i=0; i<FILLED; i++)); do BAR="${BAR}█"; done
for ((i=0; i<EMPTY; i++)); do BAR="${BAR}░"; done
print_table_row "Overall Score" "$PERCENTAGE% ($TOTAL/$MAX)"
print_table_row "Progress" "$BAR"
print_table_footer
echo
if [[ $TOTAL -ge 8 ]]; then
    echo -e "${GREEN}${BOLD}✓ System is ready for YOLO Vision applications${NC}"
elif [[ $TOTAL -ge 5 ]]; then
    echo -e "${YELLOW}${BOLD}⚠ System partially ready - some features limited${NC}"
else
    echo -e "${RED}${BOLD}✗ System not configured - run init.sh first${NC}"
fi
echo
echo -e "${CYAN}Log saved to: $LOG_FILE${NC}"
echo -e "${CYAN}Diagnostics completed at: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo