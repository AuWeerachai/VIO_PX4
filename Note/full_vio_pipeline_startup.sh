#!/bin/bash
# =============================================================================
# Tabs:
#   T0: Docker container (foreground, keeps container alive)
#   T1: VIO output (start_vio.sh)
#   T2: RViz (start_rviz.sh)
#   T3: ROS params (ros2 param set)
#   T4: TF camera_link -> drone_link
#   T5: TF map -> ground
#   T6: TF odom -> odom_tilt
#   T7: Rotated odometry relay
#   T8: MAVROS
#   T9: GPS origin
# =============================================================================



# ---------------------------------------------------------------------------
# *** ALL TUNABLE PARAMETERS  ***

# tilt angle starting from camera to drone in degrees (negative = tilt up)
CAMERA_TILT_DEG=-30.0

# GPS origin
GPS_LAT="40.41367045298802"
GPS_LON="-79.94633160928028"
GPS_ALT="300.0"

# ROS params (T3)
PARAM_JITTER=60.0
PARAM_IMU_FUSION=True

# Delays (seconds)
DELAY_T1=8             # T1: wait for container before launching VIO
DELAY_T2=8             # T2: wait for container before launching RViz
DELAY_T3_CONTAINER=8   # T3: wait for container to be up (same as T1/T2)
DELAY_T3_NODE=25       # T3: additional wait for visual_slam_node to initialize
DELAY_T7_T8=3          # T8: wait after T7 starts
DELAY_T8_T9=15         # T9: wait after T8 starts (MAVROS needs time to connect)

# ---------------------------------------------------------------------------


# Workspace and container
WS="${ISAAC_ROS_WS:-$HOME/workspaces/isaac_ros-dev}"
CONTAINER_NAME="isaac_ros_dev-aarch64-container"
CONTAINER_WS="/workspaces/isaac_ros-dev"
FCU_URL="serial:///dev/ttyUSB0:921600"


# ---------------------------------------------------------------------------
# Derived values (do not edit)
# ---------------------------------------------------------------------------
TILT_RAD_NEG=$(python3 -c "import math; print(math.radians($CAMERA_TILT_DEG))")
SCRIPTS="$WS/src/isaac_ros_common/scripts"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PANE_DIR="$SCRIPT_DIR/vio_panes"

echo "=================================================="
echo " Isaac ROS VIO Launcher"
echo " Camera tilt : ${CAMERA_TILT_DEG} deg  (${TILT_RAD_NEG} rad)"
echo " Workspace   : ${WS}"
echo " Container   : ${CONTAINER_NAME}"
echo " T1/T2 delay : ${DELAY_T1}s"
echo " T3 delay    : ${DELAY_T3_CONTAINER}s (container) + ${DELAY_T3_NODE}s (node)"
echo " T8 delay    : ${DELAY_T7_T8}s after T7"
echo " T9 delay    : ${DELAY_T8_T9}s after T8"
echo "=================================================="

# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------
for cmd in gnome-terminal python3 docker; do
    if ! command -v $cmd &>/dev/null; then
        echo "ERROR: '$cmd' is not installed."
        exit 1
    fi
done

mkdir -p "$PANE_DIR"


# ---------------------------------------------------------------------------
# Generate per-pane scripts
# ---------------------------------------------------------------------------

# -- T0: Start container (foreground, keeps TTY alive) -----------------------
cat > "$PANE_DIR/t0_container.sh" << PANE_EOF
#!/bin/bash
echo "=== T0: Starting Docker container ==="
cd "$SCRIPTS"
./run_dev.sh
PANE_EOF

# -- T1: VIO -----------------------------------------------------------------
cat > "$PANE_DIR/t1_vio.sh" << PANE_EOF
#!/bin/bash
echo "=== T1: Waiting ${DELAY_T1}s for container to start ==="
sleep $DELAY_T1
echo "=== T1: Launching start_vio.sh ==="
docker exec -it -u admin --workdir "$CONTAINER_WS" "$CONTAINER_NAME" \
    bash -c "
        source /opt/ros/humble/setup.bash
        cd $CONTAINER_WS
        ./src/isaac_ros_common/scripts/start_vio.sh
        EXIT_CODE=\$?
        echo '=== T1: VIO shutdown complete (exit code: '\$EXIT_CODE') ==='
        echo '=== T1: Press Enter to close this tab ==='
        read
    "
exec bash
PANE_EOF

# -- T2: RViz ----------------------------------------------------------------
cat > "$PANE_DIR/t2_rviz.sh" << PANE_EOF
#!/bin/bash
echo "=== T2: Waiting ${DELAY_T2}s for container to start ==="
sleep $DELAY_T2
echo "=== T2: Running start_rviz.sh ==="
docker exec -it -u admin --workdir "$CONTAINER_WS" "$CONTAINER_NAME" \
    bash -c "source /opt/ros/humble/setup.bash && cd $CONTAINER_WS && ./src/isaac_ros_common/scripts/start_rviz.sh"
PANE_EOF

# -- T3: ROS params ----------------------------------------------------------
cat > "$PANE_DIR/t3_params.sh" << PANE_EOF
#!/bin/bash
echo "=== T3: Waiting ${DELAY_T3_CONTAINER}s for container to start ==="
sleep $DELAY_T3_CONTAINER
echo "=== T3: Waiting ${DELAY_T3_NODE}s for visual_slam_node to initialize ==="
sleep $DELAY_T3_NODE
echo "=== T3: Setting ROS parameters ==="
docker exec -it -u admin --workdir "$CONTAINER_WS" "$CONTAINER_NAME" \
    bash -c "source /opt/ros/humble/setup.bash && \
             ros2 param set /visual_slam_node image_jitter_threshold_ms $PARAM_JITTER && \
             ros2 param set /visual_slam_node enable_imu_fusion $PARAM_IMU_FUSION && \
             echo '=== T3: Parameters set OK ==='"
docker exec -it -u admin --workdir "$CONTAINER_WS" "$CONTAINER_NAME" bash
PANE_EOF

# -- T4: TF camera_link -> drone_link ----------------------------------------
cat > "$PANE_DIR/t4_tf_cam.sh" << PANE_EOF
#!/bin/bash
echo "=== T4: TF camera_link -> drone_link ==="
ros2 run tf2_ros static_transform_publisher 0 0 0 0 $TILT_RAD_NEG 0 camera_link drone_link
exec bash
PANE_EOF

# -- T5: TF map -> ground ----------------------------------------------------
cat > "$PANE_DIR/t5_tf_map.sh" << PANE_EOF
#!/bin/bash
echo "=== T5: TF map -> ground ==="
ros2 run tf2_ros static_transform_publisher 0 0 0 0 $TILT_RAD_NEG 0 map ground
exec bash
PANE_EOF

# -- T6: TF odom -> odom_tilt ------------------------------------------------
cat > "$PANE_DIR/t6_tf_odom.sh" << PANE_EOF
#!/bin/bash
echo "=== T6: TF odom -> odom_tilt ==="
ros2 run tf2_ros static_transform_publisher 0 0 0 0 $TILT_RAD_NEG 0 odom odom_tilt
exec bash
PANE_EOF

# -- T7: Rotated odometry relay ----------------------------------------------
cat > "$PANE_DIR/t7_odom_relay.sh" << PANE_EOF
#!/bin/bash
echo "=== T7: Rotated odometry relay ==="
ros2 run vio_utils rotated_odometry_relay_vio_px4 \
    --ros-args -p pitch_deg:=$CAMERA_TILT_DEG
exec bash
PANE_EOF

# -- T8: MAVROS --------------------------------------------------------------
cat > "$PANE_DIR/t8_mavros.sh" << PANE_EOF
#!/bin/bash
echo "=== T8: Waiting ${DELAY_T7_T8}s for T7 to start ==="
sleep $DELAY_T7_T8
echo "=== T8: Launching MAVROS ==="
ros2 launch mavros px4.launch fcu_url:=$FCU_URL
exec bash
PANE_EOF

# -- T9: GPS origin ----------------------------------------------------------
cat > "$PANE_DIR/t9_gps_origin.sh" << PANE_EOF
#!/bin/bash
echo "=== T9: Waiting ${DELAY_T8_T9}s for MAVROS to start ==="
sleep $DELAY_T8_T9
echo "=== T9: Setting GPS origin ==="
echo "  latitude : $GPS_LAT"
echo "  longitude: $GPS_LON"
echo "  altitude : $GPS_ALT"
ros2 topic pub --once /mavros/global_position/set_gp_origin \
  geographic_msgs/msg/GeoPointStamped "
header:
  frame_id: ''
position:
  latitude: $GPS_LAT
  longitude: $GPS_LON
  altitude: $GPS_ALT"
echo "=== T9: GPS origin set ==="
echo "  latitude : $GPS_LAT"
echo "  longitude: $GPS_LON"
echo "  altitude : $GPS_ALT"
exec bash
PANE_EOF

chmod +x "$PANE_DIR"/*.sh

# ---------------------------------------------------------------------------
# Launch gnome-terminal with 10 tabs
# ---------------------------------------------------------------------------
echo "Launching gnome-terminal with 10 tabs..."
gnome-terminal \
    --tab --title "T0 - Container"   --command "bash $PANE_DIR/t0_container.sh" \
    --tab --title "T1 - VIO"         --command "bash $PANE_DIR/t1_vio.sh" \
    --tab --title "T2 - RViz"        --command "bash $PANE_DIR/t2_rviz.sh" \
    --tab --title "T3 - Params"      --command "bash $PANE_DIR/t3_params.sh" \
    --tab --title "T4 - TF Cam"      --command "bash $PANE_DIR/t4_tf_cam.sh" \
    --tab --title "T5 - TF Map"      --command "bash $PANE_DIR/t5_tf_map.sh" \
    --tab --title "T6 - TF Odom"     --command "bash $PANE_DIR/t6_tf_odom.sh" \
    --tab --title "T7 - Odom Relay"  --command "bash $PANE_DIR/t7_odom_relay.sh" \
    --tab --title "T8 - MAVROS"      --command "bash $PANE_DIR/t8_mavros.sh" \
    --tab --title "T9 - GPS Origin"  --command "bash $PANE_DIR/t9_gps_origin.sh" &

echo "=================================================="
echo " All 10 tabs launched!"
echo " Switch tabs: Ctrl+PgUp / Ctrl+PgDn"
echo "=================================================="
