# Experimental dynamic positioning using PX4 heading

Branch: `vio-dynamic-repositioning`, based on stable `main` at
`23b10a88877208a6623a9258a91f1ddd1c11c5a7`. Local implementation/test date:
2026-11-18. Deployed and build/test checked on Jetson on the same date;
live navigation/recovery and flight validation are still pending.

## What changes

During normal tracking, each newly accepted VO displacement is oriented using
time-matched PX4 heading. VO orientation is still needed to interpret its own
coordinate frame; it is **not** the absolute heading authority. Accumulated
positions never rotate when the reference heading changes. Roll/pitch mapping
is established at startup/validated full-rotation recovery, not continuously
replaced with FCU roll/pitch.

A cuVSLAM coordinate reset can change roll and pitch as well as yaw and position.
The old recovery transform corrected yaw and translation only. This branch
corrects all three rotation axes at a verified recovery, using the FCU's
estimated attitude. The vehicle must be nearly level at handoff. Small real
roll/pitch angles are retained; they are not forced to zero.

This is still stereo VO, using the existing camera extrinsics and `drone_link`
output. There is no camera/FCU IMU integration and no VO heading sent to PX4.
HIL_GPS remains the only navigation aid transmitted by the bridge. The healthy
horizontal-accuracy configuration and horizontal-only height policy are not
changed by this feature.

## Incremental position, not whole-map rotation

Let `C_old` map raw VO coordinates to the current continuous local output frame.
For a matched camera/FCU sample, compute a new correction `C_new` by adding a
world-Z rotation so that `C_new * q_vo` has PX4's heading expressed in that same
local output frame. This keeps the previously established tilt correction.

Then:

```text
raw_step     = new_raw_position - last_output_raw_position
step_mapping = midpoint rotation between C_old and C_new
new_output   = last_output_position + step_mapping * raw_step
```

Use the midpoint correction for a changing heading offset over the measurement
interval. Velocity is transformed with the endpoint correction; it is not
integrated to generate normal-tracking position. Update the raw/output anchor
after accepting the step. The fixed startup heading offset only converts this
continuous local frame to earth coordinates; it never rotates the accumulated
path in response to a new heading reading.

All raw camera samples still pass the physical/reset gates. Samples lacking a
time-matched heading do not refresh GPS position or its accepted age. Their
movement is included in the next matched displacement if all intervening raw
samples passed the gates. A corrupt intermediate sample still triggers
quarantine. If matching/reference freshness exceeds the configured limit, stop
GPS and require recovery; there is no fallback to camera yaw as absolute heading.

PX4 estimator resets, significant clock regression, telemetry gaps and attitude
changes exceeding the configured rotation-rate limit stop GPS. A new direction
cannot silently redirect a step across one of these events. Fresh recovery may
adopt the FCU's new reference, without moving old points. The MAVLink reset
counter covers several state resets, not only yaw, so its handling is conservative.

## Recovery sequence

1. Stop HIL_GPS when a physical/timing gate fires, a camera blackout times out,
   an invalid sample arrives, or persistent VO/FCU tilt disagreement is detected.
2. Hold the last accepted bridge position. While GPS is silent, track guarded
   PX4 horizontal displacement, referenced to the last accepted camera sample's
   measurement time. If its position guards fail, discard that displacement
   and keep the frozen reference instead.
3. Require a stable new VO segment and fresh, time-matched FCU pose samples.
   The recovered VO numbers may have any constant tilt or origin; requiring
   those raw numbers to be level would prevent correcting the reset.
4. Form one candidate rotation from raw VO orientation to FCU orientation,
   expressed in the bridge's existing local frame. Keep this candidate fixed
   while subsequent attitude samples test it. Do not refit every sample to
   hide disagreement. Independently check transformed VO/PX4 velocity.
5. Only after all checks and the stable-sample period pass, apply that rotation
   and translate the new segment onto the frozen or PX4-propagated reference.
6. Resume HIL_GPS with retained recovery uncertainty. Stop using PX4 position
   for propagation. Resume incremental VO displacement with current PX4 heading.
   Keep the recovered tilt correction until another validated recovery.

The geographic home does **not** move. Each recovery creates a new local
segment anchor inside the continuous trajectory. Thus global coordinates still
use home plus corrected local displacement, while a reset is corrected around
the last trusted/recovered position, not around home or the raw VO origin.
This does not continuously copy PX4 position back as GPS. Nevertheless, the
position aid now deliberately depends on PX4 attitude throughout tracking; it
is not independent GPS truth. An incorrect FCU heading can still misdirect new
motion. Existing PX4 checks require horizontal-position-only GNSS fusion;
no velocity/yaw/height aid or PX4 parameter change is introduced by this branch.

## Time and coordinate conventions

- Quaternion order in the new helpers is `w,x,y,z`. ROS odometry arrives in
  `x,y,z,w`; both ENU/NED and FLU/FRD basis changes are applied explicitly.
- Camera mounting geometry is already applied by cuVSLAM. It is not reapplied
  here. Full-rotation mode requires `drone_link`, ENU ROS odometry, and zero
  extra bridge child-to-body yaw.
- The FCU's `ODOMETRY` quaternion and position share `time_usec`, sourced from
  PX4's `timestamp_sample`. Its system/component, frame, estimator type,
  covariance, reset counter and quaternion are checked. `ATTITUDE` remains the
  existing startup/current-heading reference, not the new 3D pairing clock.
- Echoed MAVLink TIMESYNC exchanges map FCU boot time to companion monotonic
  time. Camera ROS timestamps are converted using their capture age. Matching
  interpolates only between bracketing FCU samples; it never extrapolates.
- Camera and FCU rates may differ. Waiting briefly for the next bracketing FCU
  sample does not erase recent recovery evidence. It does not count as new
  evidence or authorize a new position update. Stale data, clock changes and estimator resets
  discard the relevant evidence and require fresh checks.
- The MAVLink receive timer runs at 100 Hz to avoid adding the GPS output timer's
  delay to TIMESYNC round trips. GPS output rate is unchanged. Buffers are
  bounded; incremental updates use small quaternion/vector operations, without
  a graph optimizer, raw IMU integration or another middleware process.

## CLI controls

Under **Pose-jump gate limits**, this branch defaults **Dynamic positioning:
continuous PX4 heading + 3D recovery** to **ON**. The Heading option explains this
mode rather than requesting a manual heading or another camera mounting angle.
Legacy manual/child-yaw settings are not passed by the CLI in dynamic mode.
Saved configuration is passed to the ROS bridge. Limits are
operator settings, not claimed PX4 requirements or measured sensor accuracy.

| Setting | Default | Purpose |
| --- | --- | --- |
| Maximum PX4 body tilt when resuming | 10 degrees | Do not establish a level-body recovery while actually tilted substantially |
| Maximum 3D rotation disagreement | 5 degrees | Consistency with the fixed candidate rotation |
| VO/PX4 tilt-disagreement trigger | 10 degrees, 5 unique FCU samples | Detect persistent false VO roll/pitch during normal tracking |
| Maximum pairing/clock uncertainty | 0.03 seconds | Time-reference uncertainty and distance to a bracketing sample |
| Maximum time-sync round trip | 0.05 seconds | Reject slow clock measurements |
| Maximum time-sync age | 2 seconds | Require recent clock evidence |

Existing maximum PX4 message age/gap, velocity agreement, confirmation/stability
counts and physical limits remain in effect. The existing attitude-agreement
sample count is reused for full-rotation agreement. The old yaw-only discrepancy
limit is inactive in 3D mode. The rotation-rate limit applies to total 3D
orientation change, avoiding Euler-yaw singularities near a vertical raw pose.
Status displays the 3D recovery state, progress and FCU body tilt instead of an
unused yaw-only check. Status also shows continuous PX4 heading and its last
paired value. Maximum PX4 message age also bounds waiting for a matched output;
this is a continuous dependency now, not just an outage-recovery setting.

Turning dynamic positioning **OFF** and restarting the bridge selects the
previous yaw-only path. A full code rollback must also restore the previous
working-tree files and rebuild: merely switching branches does not remove
uncommitted deployment changes. Do not switch branches or restart navigation
publishers while flying.

## Local tests and remaining ground validation

Final local result: **342 tests passed**, including real ROS node startup/timer
checks with FCU transport mocked. Python compilation and `git diff --check`
passed. The continuous-heading extension tests slow VO yaw drift, real turns,
missing matched samples, reset-triggered pauses and preservation of already
accepted positions. The duplicate-launch CLI fix is shared with `main`;
experimental positioning and recovery remain exclusive to this branch.

### Jetson deployment check, 2026-11-18

The experimental working tree was deployed on `vio-dynamic-repositioning`.
The host bridge built successfully and all **337 tests passed on Jetson**,
including transport-mocked ROS node startup tests. Installed bridge modules
were hash-checked against source; all seven camera-helper release checks
passed. Saved home, camera extrinsics, gates and other operator settings were
unchanged. No camera/Docker rebuild or PX4 parameter change was performed.

A separate ten-second passive test, with PX4 confirmed disarmed, received
898 camera samples in `drone_link`, 203 PX4 odometry messages and 98 accepted
clock-sync replies. Camera mean sample age was 19.5 ms; 149 camera samples
had matching PX4 interpolation brackets. Other samples were waiting for
initial clock sync or the next PX4 bracket, not marked as tracking failures.
This check sent no HIL_GPS or origin commands. It establishes live timing
availability, not successful navigation or quarantine recovery.

The previous workspace/build/install and operator configuration were backed
up on Jetson at `/home/teamc/vio-rollback-before-dynamic-nceH2dmx`.
The old bridge and CLI were stopped with ground/disarm confirmation; camera
and RViz were left running. Reopen `~/vio-launch` to begin the ground test.

The subsequent startup attempt used the experimental bridge, but live handoff
was rejected: the reused VO session and FCU disagreed about tilt by about
123 degrees while FCU body tilt was below one degree. The existing failure
handler stopped the bridge and camera. The cause of that disagreement has not
yet been established; do not treat this attempt as a successful navigation
test or bypass the tilt check. A separate CLI bug (silently reusing a running
bridge when selecting Path A) was fixed and locally tested afterward. That
fix does not change the tilt gate or camera behavior.

Run dependency-free tests:

```bash
PYTHONPATH=src/vio_px4_bridge python3 -m pytest src/vio_px4_bridge/test tests -q
```

Optional real ROS node startup test, with FCU transport mocked and ROS restricted
to localhost on a separate domain:

```bash
source /opt/ros/humble/setup.bash
PYTHONPATH="$PWD/src/vio_px4_bridge:$PYTHONPATH" \
  python3 -m pytest tests/test_dynamic_ros_smoke.py -q
```

Tests cover arbitrary rigid resets, near-vertical rotations and yaw wrap,
quaternion sign equivalence, ENU/FLU conversions, no-motion and moving recovery,
body-frame velocity, differing camera/FCU rates, camera blackout, a 30-second
disturbance, FCU resets and clock changes, frozen-position fallback, invalid
messages, source filtering, CLI persistence and launch arguments. Geometry
tests include 3,500 transformed displacements across large rotations.
Synthetic drift cases generate VO displacements whose directions drift with
the VO yaw, while PX4 heading stays correct. Correction succeeds under that
model; it is not proof that every real VO position error comes from yaw.

Before deployment is accepted, perform disarmed ground tests on the actual
Jetson: verify TIMESYNC/pairing freshness, establish a normal trajectory,
induce visual loss, verify GPS stops, restore the view with the body nearly
level, and verify forward/sideways movement is correct after recovery. While
stationary away from the origin, verify that changing heading does not orbit
the position around home. Repeat
while moving and after a prolonged loss; inspect the logged uncertainty and
`VIO_3D_RECOVERY_APPLIED` transform. A substantially tilted FCU must block
resumption until level and freshly validated. Ground-test both recovery and
mode-off rollback before relying on the branch in flight.

This correction cannot repair VO scale error, map deformation, poor visual
tracking, incorrect camera calibration, or an incorrect FCU reference. It also
does not level the raw RViz point cloud: only the bridge's corrected trajectory
is transformed. Local synthetic tests are not a flight-safety certification.
