# VIO_PX4

Jetson Orin Nano cuVSLAM bridge for PX4, supporting VO-as-GPS
with optional VO heading over direct MAVLink.

- Run the installed system with `./vio-launch`.
- Camera mode: **VO (stereo only)**. RealSense IMU fusion is disabled; current extrinsics and bridge safety checks are retained.
- VO fallback: `~/vio-imu-mode vo`, then fully stop and restart the camera pipeline.
- ROS source is under `src/`.
- Operator launcher code is under `scripts/vio-launch/`.
- Runtime architecture documentation is under `docs/`.
- Recovery / replacement-Jetson setup: [bootstrap checklist](bootstrap/docs/JETSON_RECOVERY.md).
  Git restores the project layer, not the saved vehicle configuration, PX4 firmware,
  RTK workspace or recordings. Preserve the NVMe and config before reinstalling.
