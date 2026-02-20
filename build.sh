#!/bin/bash
# ==========================================================================
# Advantech-YOLO11-Vision-Application Docker Compose Build Script
# ==========================================================================
# Version:      1.6.0
# Author:       Samir Singh <samir.singh@advantech.com>
# Created:      March 10, 2025
# Last Updated: October 9, 2025
#
# Description:
#   This script sets up and launches the Advantech-YOLO11-Vision-Application
#   container environment with proper X11 forwarding for GUI applications.
#   It automatically creates required directories and configures Docker
#   environment for hardware acceleration on Advantech edge AI devices with
#   JetPack 6.0 support.
#
# Terms and Conditions:
#   1. This software is provided by Advantech Corporation "as is" and any
#      express or implied warranties, including, but not limited to, the implied
#      warranties of merchantability and fitness for a particular purpose are
#      disclaimed.
#   2. In no event shall Advantech Corporation be liable for any direct, indirect,
#      incidental, special, exemplary, or consequential damages arising in any way
#      out of the use of this software.
#   3. Redistribution and use in source and binary forms, with or without
#      modification, are permitted provided that the above copyright notice and
#      this permission notice appear in all copies.
#
# Copyright (c) 2025 Advantech Corporation. All rights reserved.
# ==========================================================================

clear

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${BLUE}"
echo "       █████╗ ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗███████╗ ██████╗██╗  ██╗     ██████╗ ██████╗ ███████╗"
echo "      ██╔══██╗██╔══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝██╔════╝██╔════╝██║  ██║    ██╔════╝██╔═══██╗██╔════╝"
echo "      ███████║██║  ██║██║   ██║███████║██╔██╗ ██║   ██║   █████╗  ██║     ███████║    ██║     ██║   ██║█████╗  "
echo "      ██╔══██║██║  ██║╚██╗ ██╔╝██╔══██║██║╚██╗██║   ██║   ██╔══╝  ██║     ██╔══██║    ██║     ██║   ██║██╔══╝  "
echo "      ██║  ██║██████╔╝ ╚████╔╝ ██║  ██║██║ ╚████║   ██║   ███████╗╚██████╗██║  ██║    ╚██████╗╚██████╔╝███████╗"
echo "      ╚═╝  ╚═╝╚═════╝   ╚═══╝  ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝ ╚═════╝╚═╝  ╚═╝     ╚═════╝ ╚═════╝ ╚══════╝"
echo -e "${WHITE}                                  Center of Excellence${NC}"
echo
echo -e "${CYAN}  This may take a moment...${NC}"
echo

sleep 7

mkdir -p src models data results

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

echo "Starting Docker containers..."
if command_exists docker-compose; then
    echo "Using docker-compose command..."
    docker-compose up -d
elif command_exists docker && command_exists compose; then
    echo "Using docker compose command..."
    docker compose up -d
else
    echo "Error: Neither docker-compose nor docker compose commands are available."
    exit 1
fi

echo "Connecting to container..."
docker exec -it advantech-yolo-vision bash