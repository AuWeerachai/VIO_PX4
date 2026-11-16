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
stop the pipeline deliberately first. Conflicting paths are rejected before
the camera/session is touched.

Stop verifies the whole owned process group, not only its shell PID. Surviving
children remain tracked and receive a bounded TERM/KILL shutdown. A failed
ownership check or failed shutdown retains state and blocks replacement;
the camera is not stopped underneath an unverified live GPS publisher.

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
| **Status** | Live output health, values, uncertainty and age (option 2) |
| **FC link** | Pick direct UART/USB serial, UDP networking, or keep the current setting |
| **Home latitude/longitude** | Horizontal spoof/origin position |
| **Heading alignment** | Choose the initial heading reference: PX4 estimate (default) or manual true heading |
| **Camera extrinsic** | Configure the full ROS FLU `drone_link → camera_link` XYZ/RPY transform |
| **Pose-jump gate limits** | Configure physical rate, timing, and recovery limits |
| **RViz auto-launch** | Persistently toggle automatic RViz startup; default OFF |
| **Leave CLI** | Close the menu and keep processes running |
| **Stop all processes** | Stop bridges, Isaac sessions, and the container |

The CLI is hardware-only: it runs on the Jetson and talks to the real Cube.
Path A uses MAVLink directly and exclusively owns `/dev/ttyUSB0`. Path B has
been removed; neither MAVROS nor Micro XRCE-DDS is started. The launcher still
blocks startup if it detects a legacy publisher left by an older version.

Settings persist in `~/.config/vio-launch/config.json`.

**Heading alignment** also offers **Send VO heading to PX4 (on/off)** for Path A.
It defaults to off. When enabled, position and heading share the same stop gate,
while their recovery checks collect evidence independently. See the
[setup and bench checklist](../../bootstrap/docs/PATH_A_EXTERNAL_YAW.md) before
enabling it; the CLI does not change PX4 parameters for you.

Camera capture-age protection (default 0.2 s), timestamp/frame/quaternion checks,
and full quarantine recovery also apply with heading output **off**. The shared
age limit is available in **Pose-jump gate limits** as well as heading settings.
Blackout recovery uses FCU movement from the last trusted camera capture, not
from the later returning frame. GPS-only operation now services TIMESYNC for
this matching too; no optional-yaw PX4 parameters are required for GPS-only use.

**Status (option 2)** refreshes every 0.5 s. Output states and PX4 connection are
on the first line. The next two lines show **HIL_GPS** and **VO heading**, each
with its health, last sent value, reported uncertainty and sample age. There
are no duplicate VIO/pose-gate health lines. GPS-only mode omits the heading
line. Startup checks, independent velocity/yaw
recovery counters, retained position uncertainty and blocking reasons appear
only when relevant. Detailed diagnostics remain in `navigation-status.json`;
the display does not claim PX4 is fusing a measurement or has entered a failsafe.

HIL_GPS shows latitude/longitude and `hAcc` (the sent horizontal accuracy in
metres, **not** a variance). Values include the MAVLink degE7/cm rounding.
Live sample age comes from the accepted camera timestamp, so retransmitting
the same fix does not make its age look fresh. Static spoof samples instead
age from their generation time; unknown camera ages are shown as unavailable.
Unsent candidates/settings and failed writes never replace the displayed fix.

The sent heading is in degrees clockwise from north, and sample age is in
milliseconds. The aligned uncertainty column shows `hAcc` in metres and
`yaw std` in degrees (one sigma). Heading standard deviation is calculated as
`degrees(sqrt(last_sent_variance_rad2))`, not copied from the configured noise
setting. The raw rad² variance remains in `navigation-status.json`; outgoing
MAVLink covariance is unchanged. The underlying variance is
at least the configured heading-noise floor and retains uncertainty inherited
from PX4 at startup/recovery. It is not a live estimate of all subsequent VO
drift. The temporary fixed-one-degree policy was withdrawn before deployment.
Paused output is labelled **last sent** with the blocking health state; before the
first send it says **not sent yet**. These are bridge-reported values, not PX4's
internal EKF uncertainty. The camera-age limit remains 200 ms by default.

Heading means the forward (+X) direction of `drone_link`, not the camera.
Camera mounting is configured only in **Camera extrinsics** and applied by
cuVSLAM before publishing body odometry. There is no child-to-body yaw field
or additional mounting rotation in the bridge. Old saved
`child_to_body_yaw_deg` settings are ignored and removed on the next save;
direct ROS launches that request a nonzero legacy offset are rejected.

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
next time Path A starts; camera-extrinsic changes apply the next time either
path starts cuVSLAM.

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

Freezing cannot account for movement during the outage. Recovery therefore
retains a conservative global uncertainty floor (maximum configured speed
times outage duration), rather than automatically claiming 0.1 m accuracy.
PX4 may reject the resulting degraded GPS. A verified origin and new bridge
session are needed to clear that uncertainty.

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
5. **Path A** for GPS spoof followed by live VO-as-GPS.

PX4 still needs `MAV_USEHILGPS=1` for Path A.

Path A checks the serial device and waits for a PX4 MAVLink heartbeat, then
waits for RC channel 6 LOW/MID→HIGH before starting a 15-second home spoof.
Live VIO takes over when ready. Body-frame cuVSLAM odometry must use
`child_frame_id=drone_link`; raw `camera_link` poses are rejected.

## Direct launch commands (optional)

Direct commands still work:

```bash
./vio-launch run a
```
