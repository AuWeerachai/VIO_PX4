# PX4 interface checklist (Cube + Jetson)

Target checked: local `PX4-Autopilot` main at
`v1.17.0-alpha1-708-g6e418096b7`. Re-check this document after changing PX4
firmware because message handling and estimator requirements can change.

## Common preflight gates

- [ ] Cube and Jetson share a working MAVLink/UART link; the selected `/dev/tty*`
  exists and is readable/writable by the user.
- [ ] A PX4 heartbeat is received before either bridge reports success.
- [ ] The configured home latitude, longitude and MSL altitude are intentional.
- [ ] cuVSLAM publishes fresh `nav_msgs/Odometry` at the configured topic.
- [ ] The odometry `frame_id`, `child_frame_id`, twist frame, and camera-to-body
  mounting convention have been verified on the actual Jetson.
- [ ] Only one of Path A (GPS) and Path B (external vision) is active.
- [ ] The real GNSS receiver is disabled or deliberately configured for blending;
  injected GPS must not silently compete for the same GPS instance.
- [ ] PX4 estimator status and innovations are observed before arming.

## Path A: HIL_GPS contract

- [ ] `MAV_USEHILGPS=1` and PX4 has been rebooted if required.
- [ ] Bridge MAVLink source system ID equals PX4 `MAV_SYS_ID`. PX4's receiver
  rejects non-HIL `HIL_GPS` from a different system ID.
- [ ] `EKF2_GPS_CTRL` enables only the intended data. Normally use bits 0, 1 and
  2 (value 7); do **not** enable dual-antenna heading bit 3.
- [ ] Send at 10 Hz without long gaps. PX4 timestamps receipt locally.
- [ ] `fix_type >= 3`, satellites >= `EKF2_REQ_NSATS`, speed accuracy below
  `EKF2_REQ_SACC`, and EPH/EPV below their configured gates.
- [ ] Hold valid, stationary spoof data for at least `EKF2_REQ_GPS_H` (10 s by
  default). The configured 15 s window provides only 5 s margin.
- [ ] Latitude/longitude are WGS84 degrees scaled by 1e7; altitude is MSL metres
  scaled to millimetres; N/E/D velocity is centimetres per second.
- [ ] For this PX4 version, `HIL_GPS.eph/epv` are decoded as centimetres and
  converted to `SensorGps.eph/epv` metres, despite the generic MAVLink field
  description calling them dilution values.
- [ ] Course-over-ground is derived from aligned N/E velocity and is unknown
  (`UINT16_MAX`) below the minimum speed. It is never treated as body heading.
- [ ] `HIL_GPS.yaw=0` (unavailable). This PX4 receiver sets GPS heading to NaN;
  do not enable GPS dual-antenna heading fusion.
- [ ] Static spoof has zero velocity and cannot establish heading.
- [ ] Live handoff is forbidden until VIO is fresh and its world yaw has been
  aligned to true north from an independent reference.
- [ ] Loss/staleness of VIO stops GPS messages rather than freezing a plausible
  moving fix.

## Non-circular heading alignment

- [ ] `MAG_DECLINATION_RESOLVED` reports the configured initial/home LLA, the
  expected east-positive declination, bundled table path, and
  `frozen_for_flight=true` before compass samples are accepted.
- [ ] A table load/validation failure blocks launch. Manual fallback requires
  explicitly selecting `mag_declination_source=manual`; never silently use 0°.
- [ ] Request and receive PX4 `HIGHRES_IMU` magnetometer and `ATTITUDE` streams.
- [ ] Use calibrated body-FRD magnetometer plus FC roll/pitch for tilt
  compensation. Do not use FC/EKF yaw in the alignment calculation.
- [ ] Apply magnetic declination for the configured home position.
- [ ] Apply the measured cuVSLAM child-frame-to-vehicle-body mounting rotation.
- [ ] Collect a stationary, fresh sample window; reject excessive circular
  spread, stale attitude/magnetometer, invalid magnitude, or moving VIO.
- [ ] Freeze the accepted yaw offset. Never continuously update it from PX4
  fused yaw after injected GPS begins.
- [ ] Rotate both VIO displacement and world velocity by the same offset.
- [ ] On cuVSLAM reset/relocalization, stop live GPS and require realignment.
- [ ] A single innovation failure enters quarantine and stops live GPS; it must
  not immediately move the local-to-global origin.
- [ ] Accepted-trajectory prediction uses only the last trusted velocity. An
  incoming/quarantined velocity must not influence that prediction.
- [ ] Verify an isolated bad pose that returns to the old trajectory is dropped
  without incrementing the continuity epoch.
- [ ] Require `continuity_confirmation_samples` internally consistent poses
  before declaring a new cuVSLAM coordinate epoch.
- [ ] Verify the reconstructed first pose of a confirmed new epoch equals the
  last accepted pose of the old epoch. Motion within the confirmation window
  should be retained when output resumes.
- [ ] After re-anchoring, keep GPS stopped for
  `continuity_recovery_samples` additional valid poses.
- [ ] Validate continuity thresholds against the vehicle's maximum real speed
  and acceleration so genuine motion is not misclassified as relocalization.
- [ ] For this fully actuated platform, do not use body tilt as a horizontal
  motion gate; approximately level attitude does not imply zero translation.
- [ ] Confirm the configured 10 m/s hard speed limit matches the flight-control
  envelope and leaves an acceptable allowance for VIO velocity noise.
- [ ] Confirm the configured 5 m/s² acceleration limit against real flight logs;
  noisy differentiated VIO velocity must not cause false quarantine events.
- [ ] Connect cuVSLAM's explicit tracking/reset event as the primary re-anchor
  trigger; residual thresholds are a conservative fallback, not sole evidence.

## Path B: MAVROS ODOMETRY contract

- [ ] MAVROS exclusively owns `/dev/ttyUSB0:921600` and `/mavros/state` reports
  `connected: true` before the EV relay starts.
- [ ] Micro XRCE-DDS and `px4_msgs` are not required or started for Path B.
- [ ] Publish at 30-50 Hz when practical; PX4 documentation warns that low-rate
  external vision may not be fused.
- [ ] `EKF2_EV_CTRL` enables only intended position, height, velocity and yaw
  fields. Do not fuse EV yaw until the VIO frame has been aligned deliberately.
- [ ] cuVSLAM publishes `child_frame_id=drone_link`; the relay rejects
  `camera_link` so camera pose cannot be mistaken for vehicle-body pose.
- [ ] Position, orientation, velocity, covariance and timestamps use consistent
  PX4 frames and time domain.
- [ ] Configure `EKF2_EV_DELAY` and `EKF2_EV_POS_X/Y/Z` for measured latency and
  camera lever arm.

## Acceptance checks before propellers

- [ ] PX4 `listener sensor_gps` (Path A) or MAVLink `ODOMETRY` inspection
  (Path B) shows the intended rate, frame, values and no dropouts.
- [ ] Move the vehicle north/east/up by a known direction and verify signs and
  scale independently of PX4's fused output.
- [ ] Point the vehicle along independently known north/east headings and verify
  aligned VIO displacement/velocity; do not validate PX4 using only data that
  originated from PX4.
- [ ] Confirm EKF GNSS/EV checks pass and inspect position, velocity, height and
  magnetometer innovation test ratios.
- [ ] Disconnect VIO and the serial link separately and confirm the expected PX4
  aiding-loss/failsafe behavior while disarmed.
- [ ] Perform the first test without propellers, then restrained/controlled,
  before any free flight.

Primary references:

- PX4 EKF2 guide: https://docs.px4.io/main/en/advanced_config/tuning_the_ecl_ekf
- PX4 external position guide: https://docs.px4.io/main/en/ros/external_position_estimation
- MAVLink common messages: https://mavlink.io/en/messages/common.html
- Local PX4 receiver: `src/modules/mavlink/mavlink_receiver.cpp`
- Local PX4 GNSS parameters: `src/modules/ekf2/params_gnss.yaml`
