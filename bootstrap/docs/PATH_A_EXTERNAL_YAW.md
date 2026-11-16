# Path A: optional VO heading alongside HIL_GPS

Implemented locally; not deployed or flight-validated. Default **OFF** preserves
GPS-only operation. No PX4 firmware patch, parameter write, arming command,
camera change, or second serial connection is part of this change.

## Independent inputs, one stop decision

Normal tracking sends horizontal position via HIL_GPS and heading via MAVLink 2
ODOMETRY on the existing connection. The camera pose has already been converted
to the vehicle body by cuVSLAM. The bridge applies its corrected local epoch and
fixed earth-heading offset, converts FLU/ENU to FRD/NED, and preserves VO tilt in
the quaternion. PX4 fuses **only yaw** from this message, not roll, pitch,
position, velocity or height.

The ODOMETRY position fields contain the real corrected local position, not
NaNs: this PX4 receiver otherwise fails to mark the orientation as LOCAL_NED.
GNSS fusion and EV yaw require that earth-aligned frame. Velocity/rate fields
are unknown/NaN because they are not used by this input. Quality 100 means the
bridge's binary health checks passed, not a measured percentage confidence.

Either a position/yaw jump, invalid body frame/quaternion, stale camera capture,
timestamp regression, or loss of timing/link health blocks **both** live
measurement outputs. Heartbeats/telemetry/TIMESYNC/status are not navigation
measurements and continue. A valid old pose is never re-stamped and sent as a
new heading observation. Staleness is detected at the configured age limit;
PX4 still has buffered measurements and its own source-loss timeout.

Recovery checks are evaluated independently: velocity disagreement does not
prevent the relative-yaw window from collecting evidence, and vice versa.
Both must pass the final decision, along with the full local stability window.
No turning is required: stable stationary samples are valid evidence. There
is no fixed two-second recovery cutoff and no single-good-sample shortcut.

FCU ODOMETRY is matched to camera capture time by interpolation or bounded
constant-rate extrapolation. Interpolation across a reset is forbidden.
Recovered position uses FCU displacement from the last trusted camera sample;
recovered yaw uses FCU heading at the recovery sample. This alignment happens
once per recovery, not on every subsequent live sample. Existing position
reset/uncertainty/physical limits still discard unsafe translation and fall
back to the frozen anchor with enlarged position uncertainty.

The recovered yaw uncertainty is conservatively derived from PX4's body-angle
variances (trace divided by cos(pitch)^2). That uncertainty becomes a persistent
lower bound; neither a timer nor steady relative tracking makes it disappear.
The existing inherited position-uncertainty floor also remains. This prevents
an immediate claim of baseline accuracy, but does not make the recovered
reference independent of the EKF: PX4/VO correlation still needs validation.

Raw cuVSLAM epoch changes are **not** forwarded as ODOMETRY reset-counter
changes. The bridge has already repaired them. In this firmware a changed EV
reset counter can force an estimator yaw reset, so the outgoing counter remains
zero. Large manual in-flight heading corrections are not commanded.

## CLI and PX4 settings

Open `./vio-launch`, then **Heading alignment**, and select **Send VO heading to
PX4: on**. Existing saved configurations default to off. New settings:

| Setting | Default | Meaning |
|---|---:|---|
| Heading noise floor | 5 degrees | One-sigma floor; not a claim of true-north accuracy |
| Heading output rate | 30 Hz | 30–50 Hz; never duplicate a camera sample |
| Maximum camera sample age | 0.2 s | Capture age, not callback-arrival age |
| Startup readback tolerance | 10 degrees | PX4 vs seeded VO heading, before live handoff |

Existing CLI physical limits, recovery message age/gap, yaw agreement limits
and stable-sample counts remain authoritative. The new option requires
`drone_link` body odometry and a zero legacy child-to-body yaw correction.
Extrinsics remain entirely in cuVSLAM; no extra camera rotation is added.

Set parameters manually while on the ground:

| PX4 parameter | Value | Purpose |
|---|---:|---|
| `MAV_USEHILGPS` | `1` | Existing GPS injection |
| `EKF2_GPS_CTRL` | `1` | Horizontal GPS position only |
| `EKF2_EV_CTRL` | `8` | EV yaw only; rejects duplicate position/height fusion |
| `EKF2_EV_NOISE_MD` | `0` | Honour reported covariance, including recovery uncertainty |
| `EKF2_MAG_TYPE` | `6` for PX4 seed; `5` for manual seed | Init-only compass, or no compass fusion |
| `EKF2_HGT_REF` | Keep `0` or `2` | Existing barometer/rangefinder height source |

The bridge reads and verifies these; it does not change them. GPS-only Path A
now verifies `EKF2_EV_CTRL=0` as well. Disabling the CLI feature therefore also
requires restoring this parameter. Path B and Path A remain mutually exclusive.

Keep EKF innovation gates and failsafes at validated aircraft values. Review
`EKF2_EV_DELAY`: capture timestamps are preserved; do not blindly reuse a delay
calibrated for the old MAVROS pipeline. Verify `EKF2_EV_QMIN` permits quality 100.
`EKF2_EVA_NOISE` can further reduce heading weight; the CLI noise is a floor,
not a guarantee of final estimator weight. The audited firmware's yaw control
also takes a maximum with `ev_att_noise` directly, while EKF2's receiver layer
squares that parameter. This implementation does not change that firmware
behavior or silently reduce its noise floor.

### Initial heading

**PX4 seed:** verify true heading before launch. The existing stationary
alignment collects PX4 yaw, then freezes the alignment for normal VO tracking.
PX4 uncertainty is carried into the new heading output.

**Manual seed:** enter the vehicle's true heading clockwise from north. Set
`EKF2_MAG_TYPE=5` and start with PX4 disarmed; normally reboot PX4 first if it
already has a different heading reference. Once camera and timing are healthy,
the bridge can send this EV yaw during stationary spoofing. Live GPS handoff
requires matching fresh FCU heading readbacks. A manual CLI number is **not**
a command that forcibly resets an already-aligned EKF. If PX4 rejects it,
startup fails instead of widening gates or forcing a reset.

### Time synchronization

ODOMETRY uses the camera capture timestamp translated from ROS system time to
the companion monotonic clock. Standard bidirectional MAVLink TIMESYNC relates
that clock to PX4 boot time; do not pre-convert ODOMETRY into PX4 time as well.
ROS simulated time and clock discontinuities are rejected for hardware output.

PX4's audited time synchronizer needs 500 accepted exchanges. The bridge
requests 50 Hz TIMESYNC, responds promptly, and waits for 600 low-delay replies
plus its own low-RTT offset estimate. Status shows clock warming/ready. Trigger
RC only after ready (normally about 15 s); an early HIGH does not begin spoof
and must be followed by LOW/MID→HIGH again. RTT is limited by PX4's 10 ms
protocol constant, not an aircraft motion parameter.

TIMESYNC has no remote convergence acknowledgement. The warmup/RTT checks are
conservative local checks, **not proof** PX4 has accepted every exchange.
Confirm `timesync_status` and EV sample timestamps on hardware before flight.
They must show plausible delay, no boot/Unix epoch mismatch and no repeated
time-jump resets. A busy Jetson/UART may fail these checks; do not bypass them.

## What happens while both inputs are silent?

PX4 propagates attitude and motion with its IMU and any remaining valid aids.
With init-only/no magnetometer, it does not gain a new magnetic heading
reference during quarantine. Heading uncertainty can grow. Barometer or
rangefinder height remains independent. Position validity and the configured
position-loss failsafe eventually apply if no horizontal aid is available.
Do not assume a Mission necessarily falls into Altitude mode; validate the
aircraft's actual fallback configuration.

## Verification checklist

- [ ] No props / ground bench first; changing an estimator input is not flight
  clearance merely because Python tests pass.
- [ ] Confirm the required parameter readbacks and body/frame conventions.
- [ ] Confirm `timesync_status` convergence and `vehicle_visual_odometry`
  sample timestamps; heading message rate should be 30–50 Hz with distinct
  timestamps. GPS remains at the existing 10 Hz.
- [ ] Check `estimator_status_flags`: `cs_ev_yaw=true`, `cs_gnss_pos=true`,
  `cs_ev_pos=false`, `cs_ev_vel=false`, `cs_ev_hgt=false`, `cs_gps_hgt=false`.
  Check the configured range/barometer source remains active.
- [ ] Check EV yaw innovations and yaw covariance in ULog; CLI "ready" reports
  bridge readiness, not proof of estimator fusion.
- [ ] Turn slowly both ways, across +/-180 degrees, then translate with the
  rearward tilted camera. Confirm both direction and heading are correct.
- [ ] Induce translation-only, yaw-only, combined jumps and complete blackout.
  Verify both incoming measurement streams stop after detection. Do not infer
  yaw behavior from QGC's satellite count alone.
- [ ] Repeat a 30–90 s loss, restore a stationary view, then walk/turn. Check
  recovery without a compulsory turn or restart, no duplicate yaw correction,
  and retained uncertainty. Also test missing/stale FCU telemetry and resets.
- [ ] Verify RTL/Mission/position-loss behavior separately on this aircraft.

Local regression tests exercise production callback/gate/recovery methods,
MAVLink 2 encode/decode, full attitude basis conversion, independent agreement
windows, long loss, invalid frames/timestamps, manual seed readback, conflicting
parameter rejection and ROS node construction. They do not simulate all of
PX4's EKF or replace the above hardware checks.

Audited firmware: Flying-Hand PX4 `7cfb72feeaaad43f4b9b2fb17c428e06cb2e7331`
(v1.0.1/manual declination), particularly `mavlink_receiver.cpp`, `Timesync.*`,
`EKF2.cpp`, `ev_control.cpp`, and `ev_yaw_control.cpp`.
Reference protocols: [PX4 external position](https://docs.px4.io/v1.16/en/ros/external_position_estimation)
and [MAVLink time sync](https://mavlink.io/en/services/timesync.html).
