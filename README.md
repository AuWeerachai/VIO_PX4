# VIO_PX4 Bridge (ROS 2 Humble)

This package publishes external vision data to PX4 using `px4_msgs/VehicleOdometry` on:
- `/fmu/in/vehicle_visual_odometry`

It is intended for Isaac ROS Visual SLAM output (`nav_msgs/Odometry`) and converts:
- ROS ENU world -> PX4 NED world
- ROS FLU body -> PX4 FRD body

## Why this path (instead of MAVROS)
For your setup (ROS 2 Humble + PX4 uXRCE-DDS), direct `px4_msgs` is simpler, lower-latency, and avoids MAVROS/MAVLink translation.

## Build
Prerequisites:
- ROS 2 Humble installed.
- `px4_msgs` available in your environment (recommended via your PX4 ROS workspace at `~/ws_px4_dev`).
- SciPy installed from apt.

From `~/Desktop/VIO_PX4`:

```bash
source /opt/ros/humble/setup.bash

# Required by visual_odometry_bridge.py
sudo apt install -y python3-scipy

# Required so Python can import px4_msgs at runtime
# (skip if you installed px4_msgs another way)
source ~/ws_px4_dev/install/setup.bash

cd ~/Desktop/VIO_PX4
colcon build --packages-select vio_px4_bridge
source install/setup.bash
```

If `~/ws_px4_dev/install/setup.bash` does not exist yet:

```bash
source /opt/ros/humble/setup.bash
cd ~/ws_px4_dev
colcon build --packages-select px4_msgs px4_ros_com
source install/setup.bash
```

## Run
```bash
source /opt/ros/humble/setup.bash
source ~/ws_px4_dev/install/setup.bash
source ~/Desktop/VIO_PX4/install/setup.bash

ros2 run vio_px4_bridge visual_odometry_bridge --ros-args \
  -p odom_topic:=/visual_slam/tracking/odometry \
  -p px4_topic:=/fmu/in/vehicle_visual_odometry
```

## PX4 EKF2 parameters to set
Use QGroundControl or MAVLink shell to set:

- `EKF2_EV_CTRL = 15`
- `EKF2_EV_DELAY = 0`
- `EKF2_HGT_REF = Vision`

Notes:
- `15` enables EV position + EV velocity + EV yaw fusion.
- If yaw is unstable, start with position+velocity only and add yaw later.

## Verify flow
```bash
ros2 topic hz /visual_slam/tracking/odometry
ros2 topic echo /fmu/in/vehicle_visual_odometry
```

On PX4 side, check EKF status flags for EV fusion.
