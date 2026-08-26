# Staged RealSense IMU implementation

This is a future implementation plan. The default pipeline remains the proven
visual-only configuration until this sequence passes bench testing.

## Confirmed cuVSLAM behavior

`enable_imu_fusion` is consumed when `visual_slam_node` is constructed. The IMU
subscription is not created when it starts false, so fusion cannot be enabled
later with a ROS parameter change. When it starts true, tracker initialization
waits for camera calibration and at least one IMU message.

## Planned safe startup

```text
Start RealSense infrared stereo with accel/gyro disabled
  -> launch visual_slam_node with enable_imu_fusion=True
  -> wait for /realsense2_camera
  -> set unite_imu_method=2
  -> set supported gyro and accelerometer rates
  -> enable gyro and accelerometer
  -> verify the combined IMU topic and TF to drone_link
  -> first IMU sample allows cuVSLAM tracker initialization
  -> verify body-frame odometry before starting either PX4 path
```

Do not use `initial_reset: true` by default. The previous eager-IMU experiment
produced RealSense USB control-transfer warnings and motion-module force pauses.

## Required bench checks

- cuVSLAM prints `Enable IMU Fusion: true`.
- The combined IMU topic is continuous and correctly remapped to
  `visual_slam/imu`.
- The IMU frame has a valid transform to `drone_link`.
- The tracker initializes reliably over repeated cold starts.
- No persistent `Motion Module force pause` occurs.
- Stereo remains 640x360 at 90 Hz.
- Stationary pose/attitude is at least as stable as the visual-only baseline.
- Hand motion has correct axis signs.
- The VIO pose-jump continuity gate still keeps global output continuous.

Retain a visual-only configuration switch as rollback until flight validation is
complete.
