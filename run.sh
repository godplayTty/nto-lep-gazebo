#!/usr/bin/env bash
set -eo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if [[ ! -f /opt/ros/jazzy/setup.bash ]]; then
    echo "В Ubuntu 24.04 сначала выполните ./setup.sh." >&2
    exit 2
fi
source /opt/ros/jazzy/setup.bash
set -u
bridge_prefix="$(ros2 pkg prefix ros_gz_bridge)"

mode=gui
if [[ "${1:-}" == --headless ]]; then
    mode=headless
    shift
fi
python3 generate_world.py "$@" --out generated
export GZ_SIM_RESOURCE_PATH="$PWD/generated${GZ_SIM_RESOURCE_PATH:+:${GZ_SIM_RESOURCE_PATH}}"
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export GZ_PARTITION="${GZ_PARTITION:-lep_inspection}"
pids=()
cleanup() {
    trap - EXIT INT TERM
    for pid in "${pids[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    wait || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ "$mode" == headless ]]; then
    gz sim -r -s --headless-rendering "$PWD/generated/lep.sdf" &
else
    gz sim -r "$PWD/generated/lep.sdf" &
fi
pids+=("$!")
"$bridge_prefix/lib/ros_gz_bridge/parameter_bridge" --ros-args -p "config_file:=$PWD/bridge.yaml" &
pids+=("$!")
python3 dashboard.py --generated "$PWD/generated" --port "${LEP_WEB_PORT:-8080}" &
pids+=("$!")
wait -n "${pids[@]}"
