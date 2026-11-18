# vio-launch

Camera selection is explicit: in **Camera extrinsic**, enter the rear D456's
SDK serial from `rs-enumerate-devices -s`. This is not necessarily the serial
shown in USB/sysfs. The CLI stores it as `camera_serial` in its user config.
The launcher passes both the serial and a D456 model filter to RealSense,
preserving the installed stereo settings. It never falls back to the front
camera. Readiness also checks the actual opened device through `device_info`,
including when reusing an existing session. Restart VIO after changing it.

During deployment, close existing CLI instances and stop the navigation
publishers before replacing camera/bridge files. Reopen the CLI after the update.
An already-open updated launcher detects changes to its own source and refuses
new launches or settings saves; Leave and Stop remain available. Older launchers
without this guard must be closed explicitly during their first upgrade.
The direct `gps` command also verifies camera identity. The CLI does
not automatically restart the camera underneath a running navigation publisher;
stop the pipeline deliberately first. Leftover EV/MAVROS publishers from older
deployments are rejected before the camera/session is touched.

Interactive companion CLI (inspired by `theseus-packages`) for VIO ↔ PX4.

**On Jetson (AnyDesk):** run `vio-launch` with no args → a numbered menu pops up.

```bash
cd ~/workspaces/VIO_PX4/scripts/vio-launch
./vio-launch
```

## Main menu

The header shows **Selected camera mode: VO (stereo only)**.
The camera path is restored to the pre-IMU baseline (`50a159b`): fusion is
explicitly disabled and the base-image RealSense driver is used. No experimental
IMU overlay is built or sourced. Current extrinsics and the newer bridge
quarantine/recovery fixes remain in use.

Path A checks recent body odometry during its 15-second stationary spoof,
along with fusion mode, heading alignment and pose-gate health. Startup glitches
can recover; failed validation stops GPS and then cuVSLAM/RViz. Keep the drone
stationary until live handoff.
Fully stop the old camera session and reopen the CLI before restarting.
Old IMU selection files do not re-enable fusion; a mismatched running session
or a shell with the experimental IMU overlay still sourced is rejected.
The archived IMU tools/cache are not needed for VO and are not deleted.

| Item | What it does |
|------|----------------|
| **Path A — GPS spoof → live** | Internship-style global GPS (`HIL_GPS`) |
| **Status (option 2)** | Live GPS navigation and recovery status |
| **FC link** | Pick direct UART/USB serial, UDP networking, or keep the current setting |
| **Home latitude/longitude** | Horizontal spoof/origin position |
| **Heading alignment** | Use PX4 estimated heading (default) or explicit manual true heading |
| **Camera extrinsic** | Configure the full ROS FLU `drone_link → camera_link` XYZ/RPY transform |
| **Pose-jump gate limits** | Configure physical rate, timing, and recovery limits |
| **RViz auto-launch** | Persistently toggle automatic RViz startup; default OFF |
| **Leave CLI** | Close the menu and keep processes running |
| **Stop all processes** | Stop bridges, Isaac sessions, and the container |

The CLI is hardware-only: it runs on the Jetson and talks to the real Cube.
Path A uses MAVLink directly and exclusively owns `/dev/ttyUSB0`. MAVROS and
Micro XRCE-DDS are not required. The former Path B launcher and relay have been
removed; `run b` and `ev-bridge` are no longer supported.

Settings persist in `~/.config/vio-launch/config.json`.

## Heading-fusion retry (experimental branch)

This branch starts from GPS-only release `23b10a8`, not the failed heading
integration. In **Heading alignment / VO heading**, explicitly switch VO heading
ON. The only new tuning value is heading standard deviation in degrees (1 sigma,
default 1 degree). The existing PX4/manual initial reference choice remains.
Camera mounting is configured only in **Camera extrinsic**, not here.
Old `external_yaw_enabled`, age, rate and tolerance settings are ignored; they
cannot silently enable this experiment. New settings are `send_vo_heading` and
`heading_std_deg`.

The launcher sends HIL_GPS at 30 Hz and heading-only MAVLink ODOMETRY at 30 Hz.
There is no rate prompt. Direct ROS users can still set the GPS `rate_hz`
parameter; heading output is fixed at 30 Hz. Only distinct accepted camera poses
are transmitted as heading observations, with their capture timestamps preserved.
Actual rates can be lower if the camera, scheduler or connection cannot keep up.

The stable position/yaw-jump gates, startup, and recovery algorithm are unchanged.
A real position/yaw jump or an existing tracking timeout stops both aids. The
failed experiment's 200-ms gate, extra heading-agreement startup gate, and
time-matched recovery replacement are absent. Heading timestamp/clock faults
pause heading output only; they cannot start navigation quarantine, change the
position anchor, or increase GPS uncertainty. Heading age uses the existing
`vio_timeout_s` (default 1 s), not a new CLI limit. PX4 may independently reject
delayed EV measurements; sending packets is not evidence that they were fused.

Heading starts after the normal GPS live handoff and MAVLink clock synchronization
warmup. Warmup does not block GPS. During live tracking, heading is the corrected
body VO yaw plus the frozen initial earth-heading offset. It is not continuously
steered toward PX4. After a genuine quarantine, the existing recovery uses current
PX4 heading; the heading transmitter inherits conservative PX4 uncertainty rather
than claiming the operator's nominal std removes dead-reckoning error. Missing
recovery covariance pauses heading only. Initial std is an operator assumption
about a verified starting heading, not measured or guaranteed absolute accuracy.

See [PX4 setup and verification](../../bootstrap/docs/PX4_INTERFACE_CHECKLIST.md#optional-heading-fusion-retry)
before enabling. No PX4 parameters are changed automatically. This branch needs
a disarmed hardware test before flight; local tests cannot establish EKF fusion
or real-world heading accuracy.

The CLI verifies all seven copied camera/runtime files against its
`camera_release.json` before starting or reusing the camera, querying identity,
or starting RViz. Missing or different files block startup instead of mixing
releases. `git pull` and a bridge build alone do not update the copies inside
`isaac_ros-dev`. Stop the pipeline before deployment and synchronize the
project-owned overrides using `bootstrap/bootstrap_jetson.sh`; then run
`python3 bootstrap/verify_camera_deployment.py` and reopen the CLI.
Camera, RViz and readiness probes use the same environment: the customized
cuVSLAM package plus the matched base-image RealSense driver/messages. The
experimental IMU overlays and other workspace driver copies are not loaded.
This camera compatibility setup does not change GPS navigation or send EV yaw.

Old `heading_source=compass` settings automatically migrate to `px4` when loaded
(persisted on the next settings save). The bridge also accepts `compass` as a
deprecated alias for `px4`, with a warning. No bridge declination is added to
PX4 yaw. Finish calibration and verify PX4's true heading before starting Path A.
Fresh, stationary startup samples align the local trajectory to PX4; the offset
then stays fixed in normal tracking. Existing quarantine recovery is unchanged.
This does not certify north alignment or remove magnetic interference.

Camera translation uses ROS FLU axes: +X forward, +Y left, and +Z up.
Positive roll, pitch, and yaw follow the right-hand rule. In particular,
positive pitch points a forward-facing camera downward. Gate changes apply the
next time Path A starts; camera-extrinsic changes apply the next time cuVSLAM
starts.

## Units and pose-jump gate meanings

Camera `X/Y/Z` are entered in **metres**. Camera `roll/pitch/yaw` are entered
in **degrees** and converted to radians for ROS. Gate speed is in `m/s`,
acceleration in `m/s²`, yaw rate in `deg/s`, and maximum tracking gap in
seconds. Confirmation and recovery values are numbers of odometry samples.

For every pair of messages, the bridge derives the allowed position and
heading change from the configured rate and the measured timestamp interval:
`distance = speed × dt` and `angle = yaw_rate × dt`. This keeps the physical
gate consistent when the effective odometry frequency changes.

A coordinate reset means cuVSLAM has started reporting the same physical scene
from a different raw origin or heading. After the first suspicious sample, the
bridge pauses GPS. Even if samples return to the old trajectory, full recovery
checks are required; there is no single-sample shortcut. Mutually consistent
samples confirm a new coordinate segment, followed by the configured stable
sample window and fresh velocity/yaw-change agreement. The default maximum
yaw rate is `360 deg/s`.

Recovery prefers PX4's dead-reckoned horizontal position change from the start
of quarantine, anchored to the last trusted bridge position. It uses current
FCU heading to orient the recovered camera segment. There is **no fixed
two-second cutoff** and no extra integration of PX4 velocity. Old saved
`inertial_max_duration_s` settings are ignored. No PX4 failsafe setting is changed.

Freshness, finite values, frame/source, reported failure quality, position
uncertainty, speed, acceleration, position-step and estimator-reset checks
remain. A failed position check freezes the original anchor for that quarantine;
fresh heading and camera/velocity agreement can still permit recovery there.
No continuous PX4-position-to-GPS feedback runs after the one-time handoff.

After a successful dead-reckoned handoff, retain the highest reported horizontal
uncertainty seen during the outage as an accuracy floor. Stable relative vision
does not remove inherited absolute position error; a new independent reference
is needed before claiming better global accuracy. Reported uncertainty itself
is an estimator estimate, not a guarantee of true error.

The old accuracy-tightening prompt is removed: its timer cannot lower the
inherited recovery uncertainty. Existing `recovery_accuracy_tighten_s` settings
are retained for compatibility, without changing recovery behavior. The camera
timestamp-gap setting checks consecutive capture timestamps, not the separate
missing-message timeout (`vio_timeout_s`, default 1 s). Reset confirmation and
stable-camera counts are necessary but not sufficient: fresh velocity and
yaw-change agreement checks must also pass.

Freezing cannot account for movement during the outage. Recovery therefore
retains a conservative global uncertainty floor (maximum configured speed
times outage duration), rather than automatically claiming the healthy 0.01 m floor.
PX4 may reject the resulting degraded GPS. A verified origin and new bridge
session are needed to clear that uncertainty.

Healthy live GPS reports 0.01 m hAcc (one centimetre on the HIL_GPS wire).
This configured uncertainty is not proof of centimetre accuracy. Static spoof
still reports 0.5 m. The launcher does not change `EKF2_GPS_P_NOISE`; if PX4
keeps that parameter at 0.1 m, its measurement-noise floor remains 0.1 m.

## Selecting an option

Type the number shown next to an option and press `Enter`.

- `0` returns to the previous menu
- `q` also returns or exits

Arrow keys are intentionally not used because some AnyDesk terminals do not
forward their escape sequences correctly.

## First-time on Jetson

1. Clone this repository into `~/workspaces/VIO_PX4`.
2. Run `./bootstrap/bootstrap_jetson.sh --install-deps` from the repository root.
3. Run `~/vio-launch`.
4. **FC link → UART** → pick your serial device / baud.
5. **Path A** for GPS spoof followed by live VIO-as-GPS.

PX4 still needs `MAV_USEHILGPS=1` for Path A.

Path A checks the serial device and waits for a PX4 MAVLink heartbeat, then
waits for RC channel 6 LOW/MID→HIGH before starting a 15-second home spoof.
Live VIO takes over when ready. Path A requires body-frame cuVSLAM odometry
with `child_frame_id=drone_link`; raw `camera_link` poses are rejected.

## Direct launch commands (optional)

Direct commands still work:

```bash
./vio-launch run a
./vio-launch gps
```
