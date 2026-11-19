# Stabilized PX4-heading displacement experiment

Branch: `vio-px4-heading-stabilized`, based on `main` commit
`ce9eba3a09e703541d3a365cb4b89e4844b22dae` (19 November 2026).
Local implementation and tests only; not ground/flight validated or deployed.

## Purpose and scope

Use the flight controller's heading to correct the direction of each new VO
movement, without making the reported position move just because heading
changes. The previous dynamic experiment already used incremental movement;
incremental integration alone does not stop jitter. Small noisy position steps
can fail to cancel when they are rotated using different headings.

This branch adds bounded timestamp matching, a position noise band, and
smoothing of the heading difference. It does not establish the physical cause
of the earlier twitching: simultaneous raw VO, transmitted GPS and PX4-estimate
measurements are still needed to distinguish bridge motion from EKF corrections.

No EV/yaw measurement is sent to PX4. No IMU is added to cuVSLAM. Camera serial,
extrinsics, stereo-only operation, GPS rate (10 Hz), configured home, height
handling, physical gate limits and firmware settings are not changed.
The original full-3D recovery and independent recovery checks are retained.

## Each cycle

1. Request PX4 `ODOMETRY` and `ATTITUDE` at the existing 20 Hz. Use the
   autopilot's NED/FRD quaternion in `ODOMETRY` for normal tracking, with its
   capture timestamp and reset counter. Ignore other system/component IDs.
2. Establish the FCU-boot/Jetson-monotonic clock offset using MAVLink `TIMESYNC`.
   Require five matching replies, reject excessive round trips, and require a
   recent bounded offset estimate. Capture timestamps, not receipt times,
   select the attitude. ROS-clock discontinuities invalidate pairing.
3. Briefly queue camera samples when the upper PX4 timestamp has not arrived.
   Interpolate attitude between valid bracketing quaternions; never extrapolate.
   Drain samples in order, including suspicious ones. The capture-age limit
   bounds waiting; the memory queue is also bounded. Each raw sample goes
   through the existing physical/timing gate before it can affect output.
4. Compare the corrected VO body yaw with PX4 yaw at that camera timestamp.
   Smooth this **difference**, not the aircraft's heading. A turn tracked
   equally by both sensors does not introduce a smoothing lag in that turn.
5. Apply a horizontal position-noise band in the fixed, tilt-corrected VO
   frame. Keep its anchor fixed while the sample lies inside the band. Outside
   the band, move the anchor only by the excess distance. This transition is
   continuous, not a one-centimetre snap.
6. Rotate only that new filtered horizontal displacement by the changing yaw
   correction and add it to the previous output position. Use a midpoint
   correction for the interval. Previous positions never rotate.
7. Convert the continuous position to latitude/longitude using the unchanged
   geographic home and startup basis. Publish through the existing HIL_GPS
   gate/timer. Velocity/altitude remain the existing horizontal-only fillers.

In compact form, within one accepted coordinate segment:

```
requested correction = wrap(matched PX4 yaw - corrected VO body yaw)
correction += (1 - exp(-dt / smoothing_time)) * wrapped correction change
new filtered step = position-noise-filter(raw position) - previous filtered position
new output = previous output + yaw_rotation(midpoint correction) * new filtered step
```

The fixed full-3D segment transform is applied before these horizontal steps.
Smoothing time zero uses the unsmoothed difference. No PX4 position is used in
healthy tracking. Geographic home remains fixed; the continuous output is the
accumulator for new movement.

## Why slow movement is not lost

The filter does **not** discard each frame-to-frame movement below 1 cm.
Instead, displacement accumulates relative to a retained anchor. A vehicle
moving slowly eventually leaves the band and continues being reported, with
up to one band radius of local filtering lag. The filter is timestamp/rate
independent. At reversals it can take up to twice the radius of travel to cross
the band. This is a deliberate small-motion tradeoff, not a claim that the
drone is physically stationary.

Inside the band, changing PX4/VO headings cannot move horizontal output at all.
Outside it, slow VO drift or larger noise can still be indistinguishable from
real travel. The radius bounds local filter lag, **not accumulated global
error**. There is no learned noise average that quarantine samples could taint.

## Quarantine and recovery

Raw jumps still use the original gate. A PX4 reset, implausible attitude step,
timestamp regression, invalid source, lost clock synchronization or stale
matched output also stops HIL_GPS and enters the same recovery path. Small
normal pairing waits do not create separate quarantine episodes. A buffered
jump is never replaced by a later good sample; it is checked before any
subsequent output. Detection includes the bounded matching latency.

During quarantine, the existing guarded PX4 displacement can supply the
recovery position. Its boundary is interpolated at the **last accepted camera
capture time**, not simply at the latest telemetry arrival. If that boundary
cannot be obtained, position propagation is rejected and the original frozen
position/uncertainty policy applies. Position failure does not permanently
block otherwise valid heading recovery.

Recovery still requires stable camera samples, independent velocity/yaw-change
checks, fresh FCU attitude and a valid time-matched sample. The new raw frame is
aligned with full PX4 roll/pitch/yaw at that sample's capture time. Position
and velocity at handoff are interpolated to that same time, so telemetry
latency does not become an extra displacement. Interpolated samples do not
increase the number of independent FCU packets counted by recovery checks.
Reseed the movement filter
at that recovered position; discard the old filter's pending displacement so
it cannot be counted twice. Normal heading-assisted increments then resume.

Reported GPS accuracy retains the existing quarantine uncertainty floors, and
cannot be tighter than the configured noise radius. Agreement with PX4 is not
independent evidence of accuracy: GPS position aiding can itself influence
PX4 attitude. This experiment does not eliminate that estimator correlation,
VO scale drift, wrong initial north, or unknown motion during loss of tracking.

## CLI and status

The Heading alignment menu exposes:

| Setting | Default | Meaning |
|---|---:|---|
| Stabilized PX4 heading | on (this branch) | Correct new movement throughout tracking |
| Horizontal noise radius | 0.01 m | Hold small fluctuations around a retained anchor |
| Heading-difference smoothing | 0.5 s | Time constant for the correction, not vehicle yaw |
| Timestamp-pairing tolerance | 0.03 s | Bounds synchronization uncertainty and nearby attitude support |

The existing Pose-jump gate limits still supply maximum PX4 message age/gap,
timestamp-reordering tolerance and physical rate limits. Matching uses those
same freshness limits; it does not read a hardcoded camera frame rate.
TIMESYNC uses a 10 Hz request cadence, five-reply minimum, 20-reply history and
two-second reply expiry as protocol implementation constants. MAVLink input
is drained at 100 Hz; this does not change the GPS publication rate.

The CLI and bridge reject incompatible settings before opening the transport:
PX4 heading, ENU `drone_link` odometry, zero extra body-yaw offset, MAVLink and
horizontal-only output are required. Camera extrinsics remain in cuVSLAM.
Turning the experiment off restores startup/recovery-only heading alignment.
All changes require a new bridge session; do not switch modes while flying.

Status adds the heading-pairing state, current correction and whether the
noise band is holding or passing movement. The status JSON also includes the
filter's remaining heading residual and candidate latitude/longitude. Candidate
coordinates are not proof of transmitted output; use the existing output and
quarantine fields, or a MAVLink capture, for that distinction.

## Local validation and required hardware check

Initial validation on 19 November 2026: **396 tests passed, none skipped**, with ROS
Humble sourced. An isolated `colcon build --packages-select vio_px4_bridge`
also passed; the existing workspace installation was not replaced.

Tests cover bounded stationary noise, very slow travel at 10/30/90/200 Hz,
turns and yaw wrap, gradual VO yaw drift, tilted/reset coordinate segments,
quarantine recovery, old/duplicate/reordered packets, clock and estimator
resets, invalid inputs, and camera/PX4 rate and latency differences. A real
ROS-node smoke test constructs the node, messages and timers, and checks the
HIL_GPS arguments with the serial/network transport mocked.

Run the regression suite from the repository with ROS Humble sourced:

```bash
PYTHONPATH="src/vio_px4_bridge:$PYTHONPATH" PYTHONDONTWRITEBYTECODE=1 python3 -m pytest src/vio_px4_bridge/test tests -q -p no:cacheprovider
```

Before flight, compare raw body VO, transmitted HIL_GPS and PX4 estimated
position in a stationary test; then test slow translation, turns, return to a
marked position and intentional visual loss/recovery on the ground. Check
that actual capture ages fit the configured bound and that the 1 cm band is
appropriate for the observed noise. No hardware jitter or accuracy claim is
established by the synthetic tests alone. Stable `main` remains the rollback.

## Code map

- `px4_heading_reference.py`: synchronization, attitude history and matching.
- `stabilized_displacement.py`: anchored noise band and incremental movement.
- `local_continuity.py`: original raw gate, recovery and experiment hook.
- `vio_px4_gps_bridge.py`: bounded queue, MAVLink dispatch, output and status.
- `scripts/vio-launch/vio_launch.py`: persistent settings and validation.

Protocol references: [MAVLink TIMESYNC](https://mavlink.io/en/services/timesync.html)
and [ODOMETRY fields](https://mavlink.io/en/messages/common.html#ODOMETRY).
The expected PX4 stream conventions were checked against the local firmware's
`src/modules/mavlink/streams/ODOMETRY.hpp`.

### Ground-test startup correction

The first Jetson start exposed a spoof-deadline race: trying to match the next
camera frame clears the transient `last_match`, even though the previously
accepted time-matched output is still fresh. The deadline now checks the age
and reference generation of the accepted output instead. A normal pending
frame does not invalidate it; stale data, clock loss and estimator/history
resets still prevent handoff. Regression tests reproduce the original failure
and check all three rejection cases. No timing limit was relaxed.

### Stationary-test timing corrections

The subsequent stationary Jetson run failed: fresh, unchanged raw VO was
quarantined by the new heading-pairing path. This is a software timing failure,
not evidence of physical motion. The original tests varied message arrival
latency but used exactly 50 ms PX4 capture intervals; they missed uneven
capture spacing.

- A known 70 ms PX4 bracket has points more than 30 ms from either endpoint.
  Those points cannot satisfy the configured pairing tolerance. Previously
  the queue waited indefinitely for a better bracket, blocking later frames
  until the old frame exceeded its age limit. Now an unusable bracket (or a
  frame older than retained history) is processed through the raw pose gate
  without updating output. The following samples can still be matched. A
  genuinely missing future bracket waits within the existing freshness bound.
- Recovery now replays every cached PX4 motion sample after the last accepted
  camera-capture boundary through the existing displacement, speed,
  acceleration, uncertainty, reset and timestamp guards. Previously it compared
  the next live packet directly with that older boundary, inventing a gap and
  unnecessarily freezing recovery. No intermediate packet is skipped to get
  to the newest one. Real missing data still rejects propagation.
- In experimental-mode status, the counter is labelled **Quarantine episodes
  (pose/timing/reference)**. Timing and FCU-reset events must not be presented
  as confirmed visual pose jumps. JSON keys remain backward compatible.

The added regressions first reproduced the old failures. They cover uneven
20 Hz PX4 capture spacing with 90 Hz stationary/moving VO, a real jump in an
unpairable frame, fresh output age, cached-motion replay, faults in intermediate
cached samples and a genuine PX4 gap. Timing/physical thresholds, publication
rates, firmware parameters and the stable branch are unchanged. These local
tests do not replace the required stationary and movement retest on Jetson.

Local validation after the corrections: **429 tests passed, none skipped**,
including 14 timing regression cases. An isolated colcon package build and a
ROS-node smoke test importing that installed package also passed. The startup
handoff regression additionally verifies that a raw, unmatched startup sample
cannot replace the previously matched output. No active Jetson process was
changed during this validation.

Clock/rate clarification: `HIGHRES_IMU` contains sensor measurements, not the
estimated attitude used here. Startup uses `ATTITUDE`; continuous pairing uses
the `ODOMETRY` quaternion and its capture timestamp/reset counter. Both streams
are **requested** at 20 Hz; the delivered rate can vary. PX4 boot time, Jetson
monotonic time and ROS timestamps are different time bases. `TIMESYNC` estimates
the PX4-to-Jetson offset; the bridge also maps ROS time to Jetson monotonic time.
It does not set either device's clock or assume their timestamps are equal.
