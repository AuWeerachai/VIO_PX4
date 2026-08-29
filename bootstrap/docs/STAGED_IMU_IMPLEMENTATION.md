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
- Stereo remains 640x360 at 30 Hz while IMU fusion is enabled; higher image
  rates must be treated as a separate USB-bandwidth experiment.
- Stationary pose/attitude is at least as stable as the visual-only baseline.
- Hand motion has correct axis signs.
- The VIO pose-jump continuity gate still keeps global output continuous.

Retain a visual-only configuration switch as rollback until flight validation is
complete.

## 2026-08-29 D456 bench result

The experimental launch correctly constructed cuVSLAM with IMU fusion enabled,
enabled both D456 motion streams at 200 Hz, selected 640x360x30 stereo, and
created the remapped `/visual_slam/imu` topic. The camera nevertheless published
neither IMU nor image data and continuously reported USB control-transfer
failures. A scoped USB unbind/rebind did not change the result.

The deployed versions are not the vendor-recommended pair:

- RealSense ROS 4.51.1 / librealsense 2.55.1
- D456 firmware 5.17.0.10

The librealsense 2.55.1 release recommends D400 firmware 5.16.0.1. Do not resume
the VIO test until either the camera firmware is changed to 5.16.0.1 or the
Isaac ROS RealSense stack is upgraded and validated against firmware 5.17.0.10.
Firmware changes require a separate explicit bench procedure and rollback plan.
