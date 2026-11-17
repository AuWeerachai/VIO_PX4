# VIO_PX4

Jetson Orin Nano cuVSLAM bridge for PX4 using Path A: home GPS spoof followed
by live VIO-as-GPS over MAVLink `HIL_GPS`.

- Run the installed system with `./vio-launch`.
- Camera mode: **VO (stereo only)**. RealSense IMU fusion is disabled; current extrinsics and bridge safety checks are retained.
- VO fallback: `~/vio-imu-mode vo`, then fully stop and restart the camera pipeline.
- ROS source is under `src/`.
- Operator launcher code is under `scripts/vio-launch/`.
- Runtime architecture documentation is under `docs/`.
