# VIO_PX4 Bridge (ROS 2 Humble)

Two ways to feed the Cube+/PX4 EKF from Jetson VIO:

1. **External vision (original)** — `/fmu/in/vehicle_visual_odometry`
2. **Internship-style GPS (new)** — boot GPS spoof, then VIO projected as GPS via `HIL_GPS`

The internship ArduPilot stack uses MAVLink `GPS_INPUT`. Stock PX4 does **not** fuse that the same way. PX4’s supported inject path is **`HIL_GPS`** with **`MAV_USEHILGPS=1`** (source sysid must match the vehicle, usually `1`).

## Build

```bash
source /opt/ros/humble/setup.bash
source ~/ws_px4_dev/install/setup.bash   # or wherever px4_msgs is
sudo apt install -y python3-scipy python3-pymavlink

cd ~/Desktop/VIO_PX4
colcon build --packages-select vio_px4_bridge
source install/setup.bash
```

## Automate test_procedure (`vio-test` CLI)

Commander-style helper (inspired by `theseus-packages`) lives in `tools/vio-test/`:

```bash
ln -sf ~/Desktop/VIO_PX4/tools/vio-test/vio-test ~/.local/bin/vio-test
vio-test doctor
vio-test run a    # GPS spoof → live
vio-test run b    # EV path
vio-test stop
```

See `tools/vio-test/README.md` and `~/test_procedure.txt`.

## 1) External vision bridge (unchanged)

```bash
ros2 run vio_px4_bridge vio_px4_bridge --ros-args \
  -p odom_topic:=/visual_slam/tracking/odometry \
  -p px4_topic:=/fmu/in/vehicle_visual_odometry \
  -p input_world_frame:=local_flu
```

PX4 params (EV):
- Start **without EV yaw fusion** until the deployed cuVSLAM frames and mounting
  transform have been verified. Do not blindly use `EKF2_EV_CTRL = 15`.
- `EKF2_HGT_REF = Vision`

NVIDIA defines this odometry as `odom_frame -> base_frame`, relative to startup.
The bridge therefore publishes `POSE_FRAME_FRD` for the default `local_flu`
mode. Use `input_world_frame:=earth_enu` only when an upstream system genuinely
georeferences the world axes to east/north/up.

## 2) Internship-style GPS spoof + live VIO GPS (recommended for GPS path)

Same *pattern* as vns-sdk `GpsSpoof` → `BasaltGpsBridge`:

1. Wait for the RC channel 6 LOW/MID→HIGH trigger.
2. Stream static **home** GPS while VIO and independent heading alignment become ready.
3. Align cuVSLAM yaw using calibrated Cube magnetometer + FC roll/pitch and
   magnetic declination. FC yaw is deliberately not used.
4. Sanitize cuVSLAM through a continuous local-pose layer. Translation/yaw
   resets are re-anchored locally and never passed through as vehicle motion.
5. Send `SET_GPS_GLOBAL_ORIGIN` once.
6. Hand off only when VIO is fresh and heading alignment is stable; the accepted
   offset is frozen before live VIO → LLA GPS starts.

### Run (MAVLink / HIL_GPS — works on stock PX4)

Point `mavlink_url` at the Cube+ MAVLink endpoint reachable from the Jetson
(example UDP; change to your serial/UDP route):

```bash
ros2 run vio_px4_bridge vio_px4_gps_bridge --ros-args \
  -p transport:=mavlink \
  -p mavlink_url:=udpout:127.0.0.1:14540 \
  -p mavlink_sysid:=1 \
  -p odom_topic:=/visual_slam/tracking/odometry \
  -p home_lat_deg:=40.4433 \
  -p home_lon_deg:=-79.9436 \
  -p home_alt_m:=300.0 \
  -p spoof_duration_s:=15.0 \
  -p spoof_until_vio:=true \
  -p heading_source:=compass \
  -p mag_declination_deg:=0.0 \
  -p child_to_body_yaw_deg:=0.0 \
  -p rate_hz:=10.0
```

### PX4 parameters (GPS inject)

| Param | Value | Why |
|-------|-------|-----|
| `MAV_USEHILGPS` | `1` | Accept companion `HIL_GPS` |
| `EKF2_GPS_CTRL` | normally `7` | Position, altitude, velocity; heading bit off |
| `EKF2_EV_CTRL` | `0` (while testing GPS path) | Avoid fighting EV |
| `EKF2_HGT_REF` | GPS or Baro | Match your height source |
| Onboard GNSS | disable or carefully blend | Real Cube+ GPS will fight the inject |

Reboot FC after param changes.

The bridge refuses to start if `MAV_USEHILGPS` is not `1`, position/velocity
fusion is missing, GPS heading fusion is enabled, or the sender system ID does
not match PX4. Set the real magnetic declination and measured cuVSLAM
child-to-body yaw in the CLI's **Heading alignment** menu. Review
`PX4_INTERFACE_CHECKLIST.md` before hardware testing.

### cuVSLAM pose-jump continuity

Path A maintains a separate continuous local control frame before converting to
LLA. During normal tracking it passes VIO motion through. A suspicious sample is
first quarantined and GPS output is paused. If the raw stream returns to the old
trajectory, that isolated sample is discarded. Only a consistent cluster of new
samples confirms a cuVSLAM coordinate reset; then the bridge changes the
raw-VIO-to-continuous transform so that:

```text
continuous_pose_after_jump = continuous_pose_before_jump
```

GPS output pauses for a short stable-sample recovery window. Subsequent motion
is accumulated relative to the new cuVSLAM epoch. The bridge does not read
PX4's fused global position for this, avoiding feedback/self-reference. Default
fallback gates are configurable with `continuity_position_residual_m`,
`continuity_yaw_residual_deg`, `continuity_max_gap_s`,
`continuity_confirmation_samples`, `continuity_recovery_samples`,
`continuity_max_speed_m_s`, `continuity_max_acceleration_m_s2`, and
`continuity_max_yaw_rate_deg_s`. The explicit cuVSLAM tracking/reset signal
should be connected as the primary detector once verified on the Jetson.

Continuity transitions are written to the `gps-bridge` log with the prefix
`VIO_CONTINUITY`. Each gate event includes its reason, measured value, configured
limit, epoch, and outcome (`quarantine_started`, `isolated_outlier_rejected`,
`candidate_window_restarted`, or `reset_confirmed`). This is transition logging,
not per-frame logging, so it remains useful without rapidly growing the log.

The current vehicle profile sets hard limits of 10 m/s speed and 5 m/s²
acceleration. The platform is fully actuated, so the gate deliberately does not
infer horizontal motion or acceleration from roll/pitch: valid forward,
backward, or lateral translation may occur while the body remains approximately
level.

Compass heading defaults to an offline magnetic-declination lookup. The bundled
10-degree IGRF-derived grid is the same table used by vns-sdk (extracted from
ArduPilot `AP_Declination`). At bridge startup, `home_lat_deg/home_lon_deg` are
bilinearly interpolated once, the result is logged as
`MAG_DECLINATION_RESOLVED`, and the value is frozen for that flight. The bridge
does not update it from its own generated GPS output. A missing or invalid table
blocks compass alignment; select `mag_declination_source:=manual` explicitly to
use `mag_declination_deg` instead.

### Optional ROS2 SensorGps transport

```bash
ros2 run vio_px4_bridge vio_px4_gps_bridge --ros-args \
  -p transport:=ros2 \
  -p ros2_gps_topic:=/fmu/in/sensor_gps \
  ...
```

Stock PX4 DDS yaml only has **out** GPS. To use this, add the subscription in
`px4_dds_sensor_gps_in.snippet.yaml` to `uxrce_dds_client/dds_topics.yaml` and rebuild PX4.
Prefer `transport:=mavlink` unless you want that firmware change.

## What was *not* done

Full port of `BasaltGpsBridge` / vns-sdk onto Jetson+PX4 (eCAL, map-match, ArduPilot `GPS_INPUT`) is a product migration, not a drop-in. This package implements the same **boot spoof + GPS position stream** idea on PX4’s real inject path.

## Verify

```bash
# Companion logs should show waiting → spoof/alignment → GPS [live]
# In QGC: vehicle has 3D fix at home before VIO, then position tracks VIO motion
```
