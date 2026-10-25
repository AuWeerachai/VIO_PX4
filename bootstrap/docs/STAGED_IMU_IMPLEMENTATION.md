# RealSense IMU / default VIO startup

Implemented and stationary-bench tested on 2026-10-21. VIO is the default;
VO remains an explicit fallback. This does not certify flight performance.

## Reproducible sequence

1. The CLI selects VIO unless the saved camera mode explicitly selects VO.
2. The camera launcher builds/reuses the pinned SDK 2.56.5 + NVIDIA camera wrapper
   in the mounted `.vio-imu/` workspace folder. No system driver or firmware is replaced.
3. Launch stereo and cuVSLAM with `enable_imu_fusion=True` from construction.
4. After stereo CameraInfo arrives, enable gyro/accel at 200 Hz and combine
   them using interpolation (`unite_imu_method=2`).
5. After the parameter changes complete, require a new continuous two-second
   IMU window containing at least 300 samples. Verify finite values, increasing
   timestamps, sensor and delivery gaps no greater than 0.1 s, plausible
   stationary gravity, the expected message frame, and fresh stereo CameraInfo.
   A startup gap discards the preceding window and starts validation again;
   it does not extend the 25-second deadline. Invalid data, wrong frames, or
   duplicate/regressing timestamps still fail immediately. Both sensor time
   and actual receipt time must span two seconds, so queued bursts cannot pass.
6. Before either PX4 path starts/reuses the camera, check its actual fusion
   parameter, incoming IMU, and fresh `drone_link` odometry inside the container.

Changing a ROS parameter after starting cuVSLAM without fusion does not create
its IMU subscription. Always restart when changing VO/VIO mode.

## Source fixes

`apply_vio_imu_fixes.py` applies/validates two narrowly scoped changes:

- Remap `visual_slam/imu` to `/imu` in the core launch fragment.
- Pass each IMU message's `imu_ts` to `CUVSLAM_RegisterImuMeasurement`, not the
  latest image timestamp for every sample.

The NVIDIA wrapper labels combined data `camera_imu_optical_frame`; values
preserve the motion optical-axis convention. cuVSLAM's configured
`camera_gyro_optical_frame` transform supplies its camera/IMU extrinsic.
The mount remains the single `drone_link -> camera_link` transform.

## Evidence and remaining validation

Old SDK: zero motion samples in standalone tests. SDK 2.56.5: video and motion
stream together; combined ROS IMU about 198 Hz, ordered timestamps, gravity about
9.80 m/s², and successful stationary cuVSLAM tracking.

Default-launcher verification on the Jetson (2026-10-21): 41 regression tests
passed. After a fresh camera start using the canonical cached overlay, a
25-second observation recorded 4,949 IMU samples (197.90 Hz), no timestamp
regressions, a maximum 15.03 ms IMU gap, and 2,004 body-odometry messages.
The runtime fusion parameter was true. The root `~/vio-launch` menu displayed
`VIO (stereo + IMU)`, and `~/vio-launch vio` passed the live readiness check.
A deliberately mismatched VO readiness check failed as intended. GPS output
was kept stopped. This verified the cached deployment, not a complete rebuild
on a freshly flashed Jetson.

### Restart failure traced on 2026-10-21

A subsequent fresh-container start stalled before camera components loaded.
The staged helper correctly refused startup after no stereo CameraInfo arrived.
GDB showed the component manager blocked in Fast DDS
`SharedMemTransport::CreateInputChannelResource`, with an orphaned
`/dev/shm/fastrtps_port7419` mapped. Its matching lock file was missing, so the
standard `fastdds shm clean` tool did not remove that port. After stopping the
hung process and checking that neither file had users, the port and its mutex
were moved into a backup; VIO then initialized successfully. This was a ROS
runtime shutdown/restart problem, not evidence that camera IMU fusion failed.

The launcher now interrupts the camera and waits for its processes to exit
before closing the owning terminal/container. Startup failures save the last
200 terminal lines to `~/.local/state/vio-launch/logs/camera-startup-failure.log`
before cleanup and show the last 30 lines in the error. Do not indiscriminately
delete shared-memory files: other ROS applications may own active ports.

The driver still warns that IMU calibration is unavailable and defaults are
used. Walking comparisons, axis checks, repeated cold starts, calibration review,
and bridge recovery tests remain necessary before flight use.

## Fallback

Startup-gap correction (2026-10-25): the previous checker permanently included
the first samples, so one 295 ms interruption during initialization stopped the
whole launch. The new checker waits for a fresh healthy window, without relaxing
the gap threshold. Persistent interruptions still stop startup. This is a
startup check, not an in-flight IMU watchdog or a fix for the separate driver
warning about missing IMU calibration.

Run `~/vio-imu-mode vo`, fully stop the camera session, then restart `~/vio-launch`.
VO uses the original base-image driver without the VIO overlay. Reselect VIO
using `~/vio-imu-mode vio` and restart. The menu shows the selected mode;
startup verifies that the running camera actually matches it.
