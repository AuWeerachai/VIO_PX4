# VIO_PX4 Bridge (ROS 2 Humble)

Two ways to feed the Cube+/PX4 EKF from Jetson VIO:

1. **External vision** — cuVSLAM body odometry through MAVROS `ODOMETRY`
2. **Internship-style GPS (new)** — boot GPS spoof, then VIO projected as GPS via `HIL_GPS`

The internship ArduPilot stack uses MAVLink `GPS_INPUT`. Stock PX4 does **not** fuse that the same way. PX4’s supported inject path is **`HIL_GPS`** with **`MAV_USEHILGPS=1`** (source sysid must match the vehicle, usually `1`).

## Fresh Jetson setup

The repository contains the VIO bridges, operator launcher, and project-owned
Isaac ROS launch overrides. For a Jetson that does not yet have Isaac ROS,
provision the compatible NVIDIA base and this project together:

```bash
cd ~/workspaces
git clone -b vio-as-gps git@github.com:AuWeerachai/VIO_PX4.git
cd VIO_PX4
./scripts/bootstrap_isaac_ros.sh --install-deps
./vio-launch
```

If the compatible Isaac ROS/cuVSLAM workspace already exists, only install the
project layer:

```bash
cd ~/workspaces
git clone -b vio-as-gps git@github.com:AuWeerachai/VIO_PX4.git
cd VIO_PX4
./scripts/bootstrap_jetson.sh --install-deps
./vio-launch
```

If the Isaac workspace is elsewhere, pass `--isaac-ws PATH`. The bootstrap
creates backups of existing Isaac launch scripts, installs the versioned
overrides, builds `vio_px4_bridge`, and creates both `~/vio-launch` and
`~/.local/bin/vio-launch`.

The deployment keeps NVIDIA Isaac ROS and MAVROS as standalone sibling
workspaces rather than vendoring their generated build trees:

```text
~/workspaces/isaac_ros-dev
~/workspaces/mavros
~/workspaces/VIO_PX4
```

The full bootstrap creates both dependency workspaces at the pinned, proven
revisions. Path A uses `isaac_ros-dev` but owns MAVLink directly; Path B uses
both `isaac_ros-dev` and MAVROS.

See `jetson-config/README.md` for the installed paths. Generated ROS build trees,
logs, bags, credentials, and Docker images are intentionally not stored in Git.

## Operator launcher

Run the repository-root entry point; its implementation is versioned under
`scripts/vio-launch/`:

```bash
./vio-launch
```

See `scripts/vio-launch/README.md`.

## 1) External vision through MAVROS

```bash
vio-launch run b
```

PX4 params (EV):
- Start **without EV yaw fusion** until the deployed cuVSLAM frames and mounting
  transform have been verified. Do not blindly use `EKF2_EV_CTRL = 15`.
- `EKF2_HGT_REF = Vision`

Path B starts MAVROS on `/dev/ttyUSB0:921600`, then relays validated body-frame
cuVSLAM odometry to `/mavros/odometry/out`. Micro XRCE-DDS and `px4_msgs` are not
part of the Jetson deployment. The relay rejects raw `camera_link` poses; use
the installed `jetson-config/isaac_ros_overrides/scripts/start_vio.sh` so cuVSLAM
publishes `child_frame_id=drone_link` directly.

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

Point `mavlink_url` at the Cube+ MAVLink endpoint reachable from the Jetson:

```bash
ros2 run vio_px4_bridge vio_px4_gps_bridge --ros-args \
  -p transport:=mavlink \
  -p mavlink_url:=/dev/ttyUSB0:921600 \
  -p mavlink_sysid:=1 \
  -p odom_topic:=/visual_slam/tracking/odometry \
  -p home_lat_deg:=40.4433 \
  -p home_lon_deg:=-79.9436 \
  -p home_alt_m:=300.0 \
  -p spoof_duration_s:=15.0 \
  -p spoof_until_vio:=false \
  -p boot_accuracy_m:=0.5 \
  -p cruise_eph_m:=0.5 \
  -p horizontal_only_output:=true \
  -p heading_source:=compass \
  -p mag_declination_source:=table \
  -p child_to_body_yaw_deg:=0.0 \
  -p rate_hz:=10.0
```

### PX4 parameters (GPS inject)

| Param | Value | Why |
|-------|-------|-----|
| `MAV_USEHILGPS` | `1` | Accept companion `HIL_GPS` |
| `EKF2_GPS_CTRL` | `1` | Fuse GPS longitude/latitude only, matching vns-sdk |
| `EKF2_EV_CTRL` | `0` (while testing GPS path) | Avoid fighting EV |
| `EKF2_HGT_REF` | Baro | PX4 owns height; VIO GPS contributes horizontal position only |
| `GPS_1_CONFIG` | `201` | Keep physical Here GNSS on GPS1 |
| `GPS_2_CONFIG` | `0` | VIO arrives over MAVLink, not a second GPS serial driver |
| `SENS_GPS_MASK` | `0` | Do not blend physical GNSS and VIO |
| `SENS_GPS_PRIME` | verify on hardware | Instance numbering changes when physical GNSS is absent |

Reboot FC after param changes.

The bridge refuses to start if `MAV_USEHILGPS` is not `1`, horizontal-position
fusion is missing, unsupported GPS fusion bits are enabled, the sender system ID does
not match PX4, or exposed dual-GPS selector values contradict the VIO-first
policy. Firmware without `SENS_GPS_MASK`/`SENS_GPS_PRIME` produces an explicit
unverified-selection warning and requires inspection before flight. Set the real magnetic declination and measured cuVSLAM
child-to-body yaw in the CLI's **Heading alignment** menu. Review
`docs/PX4_INTERFACE_CHECKLIST.md` before hardware testing.

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

Path A owns `/dev/ttyUSB0` directly. MAVROS must not run concurrently; the CLI
checks for an existing serial-port owner and fails rather than competing for it.

`HIL_GPS` requires altitude and velocity fields on the wire. In the deployed
horizontal-only mode, altitude is held at the configured origin and velocity is
zero; `EKF2_GPS_CTRL=1` leaves height and velocity estimation to PX4. Only
latitude and longitude carry live VIO motion. Horizontal accuracy remains
0.5 m instead of loosening after the boot phase.

## What was *not* done

Full port of `BasaltGpsBridge` / vns-sdk onto Jetson+PX4 (eCAL, map-match, ArduPilot `GPS_INPUT`) is a product migration, not a drop-in. This package implements the same **boot spoof + GPS position stream** idea on PX4’s real inject path.

## Verify

```bash
# Companion logs should show waiting → spoof/alignment → GPS [live]
# In QGC: vehicle has 3D fix at home before VIO, then position tracks VIO motion
```
