# VIO_PX4

Jetson Orin Nano cuVSLAM bridge for PX4 using Path A: home GPS spoof followed
by live VIO-as-GPS over MAVLink `HIL_GPS`.

## Experimental branch: stabilized PX4 heading

`vio-px4-heading-stabilized` branches from `main` at `ce9eba3`.
It uses time-matched PX4 heading to orient **new VO displacement only**.
It never rotates previous positions, sends VO heading to PX4, or routes IMU
data into cuVSLAM. Stable `main` is unchanged.

A configurable 1 cm horizontal noise band holds small fluctuations without
discarding slow travel. Heading-difference smoothing avoids directly applying
every heading fluctuation to position. Raw pose checks precede filtering;
heading resets, invalid data and stale telemetry stop GPS and require recovery.
This is an experiment, not a guarantee of zero real-world drift or flight readiness.

See [implementation, settings and validation](docs/PX4_HEADING_STABILIZED.md).
The Heading alignment menu enables/disables the experiment and configures it.
Camera settings and PX4 firmware/parameters are not changed by this branch.

## Stable baseline retained

Full 3D recovery is enabled on `main`. After a coordinate reset,
recovery aligns the complete VO body orientation to fresh
PX4 roll/pitch/yaw, including the small real tilt of a nearly level aircraft.
On stable `main` the resulting rigid rotation/translation stays fixed until
another reset. This experimental branch retains that tilt alignment, while
correcting the direction of new horizontal movements during tracking.
Previous positions and the home reference are not rotated or replaced.

Recovery still requires the existing independent velocity/yaw-change agreement
and stable-camera windows. HIL_GPS remains stopped while quarantined. The
rotation-rate setting now checks full 3D pose steps instead of Euler yaw alone,
including roll/pitch-only resets and avoiding false yaw jumps at singularities.
No heading is sent to PX4, no IMU is routed to cuVSLAM, and height output,
position-propagation guards and inherited uncertainty floors are retained.
The experiment also prevents reported horizontal accuracy from being tighter
than its configured position-noise radius.

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
