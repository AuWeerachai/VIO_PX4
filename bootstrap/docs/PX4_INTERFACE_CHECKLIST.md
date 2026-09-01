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
- [ ] Identify the physical Here and injected VIO instances from live PX4 data.
  Do not infer instance numbers from connector names or startup order.
- [ ] PX4 estimator status and innovations are observed before arming.

## Path A: HIL_GPS contract

- [ ] `MAV_USEHILGPS=1` and PX4 has been rebooted if required.
- [ ] `SENS_GPS_MASK=0` (no blending), if this firmware exposes the legacy
  selector. Set `SENS_GPS_PRIME` only after identifying the VIO instance on the
  actual hardware. Missing selector parameters require firmware-specific
  inspection before flight.
- [ ] Physical Here GNSS is retained with `GPS_1_CONFIG=201`, while
  `GPS_2_CONFIG=0` because VIO arrives via MAVLink. The currently installed
  Jetson pymavlink uses the base `HIL_GPS` message without the optional `id`
  extension, so bridge `gps_id` alone cannot prove the PX4 instance number.
- [ ] Bridge MAVLink source system ID equals PX4 `MAV_SYS_ID`. PX4's receiver
  rejects non-HIL `HIL_GPS` from a different system ID.
- [ ] `EKF2_GPS_CTRL=1`: fuse longitude/latitude only, matching vns-sdk.
  Altitude remains on the configured height source, velocity is not fused, and
  heading remains on the compass.
- [ ] Send at 10 Hz without long gaps. PX4 timestamps receipt locally.
- [ ] `fix_type=3`, synthetic satellites=40, speed accuracy
  below `EKF2_REQ_SACC`, and EPH/EPV below their configured gates.
- [ ] Set `EKF2_GPS_P_NOISE=0.1 m` to match the bridge's healthy horizontal
  EPH. Do not claim RTK fix types to influence source selection.
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
  moving fix, allowing PX4 to time out that source and fall back to Here GNSS.
- [ ] Confirm `VIO_GPS_OUTPUT_STOPPED` and `VIO_GPS_OUTPUT_RESUMED` transitions
  in the bridge log and corresponding `VIO GPS:` messages in QGroundControl
  during a propeller-off VIO disconnect/recovery test.

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
- [ ] Verify an isolated bad pose that returns to the old trajectory is dropped
  without incrementing the continuity epoch.
- [ ] Require `continuity_confirmation_samples` internally consistent poses
  before declaring a new cuVSLAM coordinate epoch.
- [ ] Verify candidate and recovery VIO samples cannot modify the last trusted
  output anchor.
- [ ] While HIL_GPS is silent, use only the change in PX4 `ODOMETRY` local
  position and ATTITUDE yaw; never copy PX4's absolute/global position.
- [ ] Reject non-autopilot/non-NED odometry, excessive position variance, and
  any estimator reset-counter change.
- [ ] Verify a stationary disturbance resumes exactly at the trusted anchor.
- [ ] Verify real motion during quarantine resumes at the trusted anchor plus
  the guarded PX4 inertial displacement.
- [ ] Verify stale, non-finite, timestamp-regressed, excessive-gap, over-speed,
  over-acceleration, over-yaw-rate, and over-duration PX4 data discards the
  propagated displacement and recovers the new VIO segment from the frozen pose.
- [ ] Require consecutive VIO/PX4 velocity agreement before accepting a
  successful propagated handoff.
- [ ] Require consecutive unique PX4 attitude samples where post-reset VIO yaw
  change agrees with PX4 yaw change. Never compare the arbitrary absolute yaw
  origin of the recovered VIO epoch.
- [ ] Verify a yaw disagreement resets the recovery window without allowing
  rejected samples to contribute to the next agreement window.
- [ ] Resume with EPH no tighter than PX4's horizontal uncertainty, then tighten
  gradually to the configured 0.1 m VIO accuracy.
- [ ] After re-anchoring, keep GPS stopped for
  `continuity_recovery_samples` additional valid poses.
- [ ] Validate continuity thresholds against the vehicle's maximum real speed
  and acceleration so genuine motion is not misclassified as relocalization.
- [ ] Verify position and heading changes are limited using message timestamp
  intervals (`speed × dt`, `yaw_rate × dt`) at both 30 Hz and 90 Hz.
- [ ] For this fully actuated platform, do not use body tilt as a horizontal
  motion gate; approximately level attitude does not imply zero translation.
- [ ] Confirm the configured 10 m/s hard speed limit matches the flight-control
  envelope and leaves an acceptable allowance for VIO velocity noise.
- [ ] Confirm the configured 5 m/s² acceleration limit against real flight logs;
  noisy differentiated VIO velocity must not cause false quarantine events.
- [ ] Connect cuVSLAM's explicit tracking/reset event as the primary re-anchor
  trigger; timestamp-derived physical rate checks remain the fallback.

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
