# VIO_PX4

Jetson Orin Nano cuVSLAM bridge for PX4, supporting VIO-as-GPS and MAVROS
external-vision paths.

- Run the installed system with `./vio-launch`.
- Default camera mode: **VIO (stereo + RealSense IMU)**. The menu shows the selected mode.
- VO fallback: `~/vio-imu-mode vo`, then fully stop and restart the camera pipeline.
- ROS source is under `src/`.
- Operator launcher code is under `scripts/vio-launch/`.
- Runtime architecture documentation is under `docs/`.
