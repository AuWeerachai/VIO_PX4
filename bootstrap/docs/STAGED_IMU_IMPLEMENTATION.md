# Staged RealSense IMU implementation

This is an experimental implementation on the `exp-turn-imu-on` branch. The
`vio-as-gps` branch remains the proven visual-only rollback configuration until
this sequence passes bench testing.

## Confirmed cuVSLAM behavior

`enable_imu_fusion` is consumed when `visual_slam_node` is constructed. The IMU
subscription is not created when it starts false, so fusion cannot be enabled
later with a ROS parameter change. When it starts true, tracker initialization
waits for camera calibration and at least one IMU message.

## Experimental startup

```text
Construct RealSense with infrared stereo, accel, gyro, and unite_imu_method=2
  -> construct visual_slam_node with enable_imu_fusion=True
  -> route the combined IMU topic to visual_slam/imu
  -> first synchronized IMU sample allows cuVSLAM tracker initialization
  -> verify body-frame odometry before starting either PX4 path
```

Do not enable the gyro and accelerometer through sequential runtime parameter
changes. Bench testing on the D456 showed that each change restarts sensors and
causes a sustained flood of USB control-transfer errors. Configure both motion
streams atomically when the RealSense node is constructed. Do not use
`initial_reset: true` by default.

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
