#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${ISAAC_CONTAINER_NAME:-isaac_ros_dev-aarch64-container}"
PID_FILE="/tmp/vio_px4_new_pipeline_cuvslam.pid"

docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null | grep -qx true || exit 0

# Only stop processes carrying this workspace's inherited ownership marker.
# This remains reliable if Docker reparents ROS children after docker-exec
# exits, and never matches a cuVSLAM pipeline started manually.
docker exec "$CONTAINER_NAME" bash -lc '
  pids=""
  for env_file in /proc/[0-9]*/environ; do
    if { tr "\0" "\n" < "$env_file" | grep -qx "VIO_PIPELINE_OWNER=vio_px4_new_pipeline"; } 2>/dev/null; then
      pid=${env_file#/proc/}; pid=${pid%/environ}; pids="$pids $pid"
    fi
  done
  test -n "$pids" || { rm -f /tmp/vio_px4_new_pipeline_cuvslam.pid; exit 0; }
  kill -TERM $pids 2>/dev/null || true
  for unused in 1 2 3 4 5 6 7 8 9 10; do
    alive=""
    for pid in $pids; do kill -0 "$pid" 2>/dev/null && alive="$alive $pid"; done
    test -z "$alive" && { rm -f /tmp/vio_px4_new_pipeline_cuvslam.pid; exit 0; }
    sleep 0.2
  done
  kill -KILL $pids 2>/dev/null || true
  rm -f /tmp/vio_px4_new_pipeline_cuvslam.pid
'
