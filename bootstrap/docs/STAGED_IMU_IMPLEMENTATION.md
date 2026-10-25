# RealSense IMU experiment (inactive)

Current status (2026-10-25): restored stereo-only VO after reported stationary
cuVSLAM jumps with IMU enabled. The root cause is not proven. The active launcher
does not build/source this overlay or enable IMU fusion. Extrinsics and bridge
quarantine/recovery fixes remain enabled. The following records the historical
IMU implementation and bench checks, not the current launch sequence.

## Reproducible sequence

1. The CLI selects VIO unless the saved camera mode explicitly selects VO.
2. The camera launcher builds/reuses the pinned SDK 2.56.5 + NVIDIA camera wrapper
   in the mounted `.vio-imu/` workspace folder. No system driver or firmware is replaced.
3. Launch stereo and cuVSLAM with `enable_imu_fusion=True` from construction.
4. After stereo CameraInfo arrives, enable gyro/accel at 200 Hz and combine
   them using interpolation (`unite_imu_method=2`).
5. The normal camera launcher uses `stage_realsense_imu.py --configure-only`.
   Parameter configuration errors still stop the launch, but early IMU gaps or
   backward timestamps no longer kill cuVSLAM through the startup checker.
6. Path A confirms the camera node and selected fusion mode, then connects to
   PX4. After RC channel 6 triggers the 15-second stationary spoof, the bridge
   gathers new evidence: two continuous healthy seconds of body odometry and,
   in VIO mode, at least 300 IMU samples spanning two seconds. Sensor and delivery
   gaps must be no greater than 0.1 s for IMU / 0.5 s for odometry. Invalid data,
   wrong frames and timestamp discontinuities discard the affected window;
   later healthy data can recover. Receipt time also must span two seconds.
7. Handoff additionally requires a recently confirmed fusion setting, completed
   heading alignment and no pose quarantine. Pre-trigger evidence is discarded;
   the origin and heading initialization are refreshed while stationary.
8. At 15 seconds, either switch to live GPS, or stop GPS output and latch the
   failure. The CLI then stops the GPS bridge and cuVSLAM/RViz/container session,
   saving `camera-spoof-failure.log`. Start Path A again to retry. The 15-second
   deadline is not extended by bad samples. Keep the vehicle stationary and on
   the ground until live handoff. This is not flight certification.
9. Path B and the standalone `vio` command retain their strict readiness probe
   before reporting success. VO fallback does not require IMU samples, but does
   require the actual camera fusion mode to match and healthy body odometry.

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

Spoof-period validation (2026-10-25): after testing a bounded startup window,
validation was moved to the bridge's spoof phase. Early glitches are diagnostic
evidence, not proof that cuVSLAM itself has failed. The bridge requires recovery
before live output; the launcher shuts the camera down if handoff fails. Existing
in-flight pose-jump and stale-odometry safeguards remain unchanged. This new
readiness check is not a continuous in-flight IMU watchdog and does not fix the
separate driver warning about missing IMU calibration. The standalone staged
helper's strict bench-check mode remains available when `--configure-only` is
omitted, but the normal launcher does not invoke that mode.

Run `~/vio-imu-mode vo`, fully stop the camera session, then restart `~/vio-launch`.
VO uses the original base-image driver without the VIO overlay. Reselect VIO
using `~/vio-imu-mode vio` and restart. The menu shows the selected mode;
startup verifies that the running camera actually matches it.
