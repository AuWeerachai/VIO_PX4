# VIO_PX4

Jetson Orin Nano cuVSLAM bridge for PX4 using Path A: home GPS spoof followed
by live VIO-as-GPS over MAVLink `HIL_GPS`.

Full 3D recovery is enabled on `main`. After a coordinate reset,
recovery aligns the complete VO body orientation to fresh
PX4 roll/pitch/yaw, including the small real tilt of a nearly level aircraft.
The resulting rigid rotation/translation stays fixed until another reset.
Previous positions and the home reference are not rotated or replaced.

Recovery still requires the existing independent velocity/yaw-change agreement
and stable-camera windows. HIL_GPS remains stopped while quarantined. The
rotation-rate setting now checks full 3D pose steps instead of Euler yaw alone,
including roll/pitch-only resets and avoiding false yaw jumps at singularities.
No heading is sent to PX4, no IMU is routed to cuVSLAM, and height output,
position-propagation guards, and uncertainty floors are unchanged.

Local tests cover arbitrary/inverted resets, repeated resets, moving recovery,
stationary recovery followed by a rectangle, and stale/reordered telemetry.
They establish frame-transform correctness, not real-world VO accuracy or
flight readiness. Begin a ground test with a fresh VO/bridge session; this
change does not repair an already corrupted startup frame, recover unknown
lost motion, or remove accumulated global uncertainty.

- Run the installed system with `./vio-launch`.
- Camera mode: **VO (stereo only)**. RealSense IMU fusion is disabled; current extrinsics and bridge safety checks are retained.
- VO fallback: `~/vio-imu-mode vo`, then fully stop and restart the camera pipeline.
- ROS source is under `src/`.
- Operator launcher code is under `scripts/vio-launch/`.
- Runtime architecture documentation is under `docs/`.
