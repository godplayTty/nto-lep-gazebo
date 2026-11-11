#!/usr/bin/env bash
set -euo pipefail

. /etc/os-release
if [[ "${ID:-}" != ubuntu || "${VERSION_ID:-}" != 24.04 ]]; then
    echo "Требуется Ubuntu 24.04 или приложенный Dev Container." >&2
    exit 2
fi

if (( EUID == 0 )); then
    sudo_cmd=()
else
    sudo_cmd=(sudo)
fi

if [[ ! -f /opt/ros/jazzy/setup.bash ]] && ! dpkg -s ros2-apt-source >/dev/null 2>&1; then
    "${sudo_cmd[@]}" apt-get update
    "${sudo_cmd[@]}" apt-get install -y ca-certificates curl
    pkg="$(mktemp -d)"
    trap 'rm -rf "$pkg"' EXIT
    release="$(curl -fsSL https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest |
        python3 -c 'import json,sys; print(json.load(sys.stdin)["tag_name"])')"
    curl -fsSL "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${release}/ros2-apt-source_${release}.noble_all.deb" \
        -o "$pkg/ros2-apt-source.deb"
    "${sudo_cmd[@]}" dpkg -i "$pkg/ros2-apt-source.deb"
fi

"${sudo_cmd[@]}" apt-get update
"${sudo_cmd[@]}" apt-get install -y --no-install-recommends ros-jazzy-ros-base ros-jazzy-ros-gz \
    ros-jazzy-tf2-ros python3-opencv python3-numpy libgl1-mesa-dri libegl-mesa0
python3 -c 'import cv2; assert hasattr(cv2, "aruco"); print("OpenCV", cv2.__version__)'
