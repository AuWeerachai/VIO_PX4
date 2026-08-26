#!/usr/bin/env bash
set -euo pipefail

: "${ISAAC_ROS_WS:?ISAAC_ROS_WS must point to the Isaac ROS workspace}"
cd "${ISAAC_ROS_WS}/src/isaac_ros_common"

rviz2 -d "$(ros2 pkg prefix isaac_ros_visual_slam --share)/rviz/default.cfg.rviz"
