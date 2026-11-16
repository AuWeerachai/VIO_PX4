# Jetson camera startup and Path A heading verification — 2026-11-16

Scope: ground/disarmed checks on the replacement Jetson and rear D456. This
record does not certify flight performance, accuracy, or all load conditions.
The base repository revision was `a7273e6`; the environment fixes described here
were deployed as working-tree changes. No PX4 parameters, camera extrinsics,
home coordinates or gate limits were changed. The initial repair did not alter
saved CLI settings; the subsequent requested heading change is recorded below.

## Camera failure and repair

The normal Isaac workspace overlay selected the base-image RealSense 4.51.1
driver but a workspace `realsense2_camera_msgs` installation with a seven-field
`DeviceInfo` response. The base driver expects six fields. `ldd` confirmed that
the older driver resolved its message typesupport library from the other
installation. This is an incompatible binary/interface combination, not merely
a missing camera-discovery response. The previous `.vio-imu` path check did not
detect this ordinary workspace overlay.

The repeated exit `-11` failures occurred during startup and identity checks.
There was no usable core from these specific failures, so an exact crash stack
was not established. Nevertheless, the incompatible libraries were directly
verified and removed from the runtime selection. This class of failure is
described in [colcon's ABI-override guidance](https://colcon.readthedocs.io/en/released/user/overriding-packages.html#undefined-behavior-when-overridden-package-breaks-abi).

The repair keeps cached files on disk. `vo_environment.sh` removes inherited
Isaac install paths, sources Humble, then sources only
`install/isaac_ros_visual_slam/share/isaac_ros_visual_slam/package.bash`.
Camera, RViz and identity-check shells use this environment. The new guard checks
prefixes, matching driver/message versions, native library resolution and Python
service layout before any identity-service request. It also refuses to silently
use stock cuVSLAM when the customized source overlay should be active.

Verified on the actual image and hardware:

- The deliberately contaminated environment is rejected before hardware access.
- The repaired environment selects base-image 4.51.1 driver and messages, plus
  the customized cuVSLAM overlay.
- `/proc/<camera-pid>/maps` confirms these libraries in the real camera process,
  not just the diagnostic shell.
- One completed cold startup with the finalized helper produced fresh
  `odom` / `drone_link` odometry and started RViz. An earlier camera-only run was
  intentionally stopped while correcting the package-local overlay path; it
  is not counted as a successful finalized startup.
- Three repeated camera identity/data checks succeeded against the same rear
  D456, without a crash or bypass of serial/model verification.
- A subsequent normal `vio-launch vio` invocation reused the running camera
  successfully while the live bridge remained active.

## Serial timing and PX4 heading evidence

The FC FTDI adapter had its default `latency_timer=16` ms. PX4 reported a
14.820 ms time-sync round trip, beyond its 10 ms sample acceptance threshold.
Clock readiness was slow/intermittent. Setting host receive buffering to 1 ms
improved observed PX4 round trips to 2.152–2.996 ms. No time-sync thresholds were
relaxed. An adapter-serial-specific udev rule preserves the setting on reconnect;
`bootstrap/configure_fcu_serial_latency.sh` reproduces this setup. Other serial
adapters are not matched by that rule.

The user activated RC channel 6. The production startup sequence completed the
15-second spoof, verified the 200 m global-origin altitude, and entered live
VO-as-GPS. A temporary ground observer used the bridge's sole serial connection
for fixed, read-only PX4 `listener` commands; no second serial reader or firmware
parameter writes were used.

Readbacks during live operation:

| Evidence | Observed value |
|---|---|
| `vehicle_visual_odometry` | Published; sample about 24–30 ms before receive timestamp |
| `cs_ev_yaw` | `True` |
| `cs_gnss_pos` | `True` |
| `cs_ev_pos`, `cs_ev_vel`, `cs_ev_hgt`, `cs_gps_hgt` | `False` |
| `cs_rng_hgt`, `cs_baro_hgt` | `True` |
| `cs_inertial_dead_reckoning` | `False` during healthy live output |
| EV yaw aid `fused` | `True` |
| EV yaw aid `innovation_rejected` | `False` |
| EV yaw innovation | Approximately 0.00058 rad at that stationary sample |
| EV yaw `observation_variance` | 0.10000 rad², about 18.1° standard deviation |
| Bridge heading output | Approximately 30 messages/s during healthy periods |
| HIL_GPS | Live horizontal output; synthetic height fusion remains disabled |

The 5° CLI noise floor is **not** a claim of 5° actual heading accuracy or the
final EKF observation noise. The bridge inherited approximately 17.2° startup
uncertainty from PX4; PX4's observed variance floor was larger still. No heading
noise/firmware tuning was done in this repair. Also, PX4 reported magnetic-field
disturbance during this test; no continuous magnetic-heading/3-D fusion was
active. Correct true-north alignment still requires its separate validation.

## Test coverage and limitations

- 267 automated tests passed locally and on Jetson with the host Humble setup
  sourced. A first Jetson test invocation omitted ROS setup and had five import
  failures; rerunning in the required environment passed all tests.
- Shell syntax and Git whitespace checks passed.
- The hardware test validates camera startup, identity, live odometry, clock
  behavior, startup handoff, and actual external-yaw fusion. It does not prove
  a flight trajectory or true-north accuracy.
- Heavy on-device test activity coincided with camera samples exceeding the
  configured 0.2 s age limit. Both aids were withheld by the existing quarantine
  logic; recovery completed automatically. Subsequent observations were healthy.
  This is a remaining load-margin limitation, not grounds for widening gates.
  Avoid running builds/regression tests during flight; sustained operational
  load and repeated cold boots still need dedicated acceptance testing.

Raw bridge/PX4 readback is in the Jetson's
`~/.local/state/vio-launch/logs/gps-bridge.log`, starting at the 14:04:03 session.
The temporary observer is `/tmp/vio_heading_bench.py`; it wraps the unchanged
production bridge, retains the normal RC trigger/gates, and does not persist in
the normal CLI launch command after this session ends.

## Rollback

Jetson backups: `~/.local/state/vio-launch/deploy-camera-env-20261116-jHPHFo/`.
They contain the previous project helpers, deployed Isaac scripts, and unchanged
CLI configuration. Restore matching project and deployed scripts together only
with the pipeline stopped. Restoring the old whole-workspace environment would
also restore the known incompatible-library risk.

The adapter rule is `/etc/udev/rules.d/99-vio-fcu-ftdi-FTGNUI4Q.rules`. Removing
that specific rule and reloading udev restores default behavior on reconnection;
the current adapter's timer can also be restored to 16 ms while disarmed.

## Heading-confidence revision: fixed-variance policy withdrawn before deployment

A temporary local change used the configured one-degree uncertainty instead of
inheriting PX4 reference covariance. After discussing relative-turn precision
versus absolute-heading uncertainty, the operator chose to retain inheritance.
The fixed-variance policy was not deployed. Local code again preserves the
larger reference uncertainty at PX4 startup and each recovery, without resetting
it to one degree on resumption. Normal live tracking does not continuously copy
PX4 yaw/covariance. Position uncertainty and all quarantine checks are unchanged.

The configured minimum remains one degree as requested; existing saved choices
are preserved. The Jetson saved setting was separately changed to one degree
with backup `~/.local/state/vio-launch/heading-noise-1deg-jzozUB/`. The observed
running bridge still had its old five-degree setting and inherited 17.2-degree
floor (PID 110559). No live process was restarted for these local revisions.

Status now reports the last successfully transmitted heading and its variance,
including the inherited floor. Unsent candidates cannot replace the displayed
sent sample; paused and stale output are labelled, and no standard-deviation
duplicate line is shown. Tests cover the actual sent covariance, startup
inheritance, retained recovery floors, independent normal tracking, transport
failure, stale status, ROS status-file serialization and the CLI display.

Inheritance preserves the prior conservative policy; it does not add a complete
VO drift covariance model. cuVSLAM covariance semantics and calibration remain
separate work before claiming a full estimated heading-error model.

The PX4 yaw-control floor remains a separate limitation: firmware `7cfb72fe`
compares a variance against unsquared `ev_att_noise`. The observed floor was
0.1 rad² (18.1 degrees standard deviation). Reporting one degree from the bridge
does not bypass this. No firmware code, flash, or PX4 parameter was changed.

## Passive camera-derived odometry timing

An independent host ROS subscriber observed `/visual_slam/tracking/odometry`
for 45 seconds, with the same best-effort/depth-5 QoS as the bridge. There was
no serial access, ROS publication, restart, or on-device build/test load.
This measures header timestamp to subscriber receipt, including the camera,
VO computation and message delivery; it is not a direct image-exposure latency
measurement or the exact production callback scheduling delay.

| Measurement | Value |
|---|---:|
| Valid odometry samples | 4,000 |
| Source rate | 89.75 Hz |
| Mean arrival age | 19.57 ms |
| Population standard deviation | 6.06 ms |
| Mean + 2 standard deviations | 31.69 ms |
| 95th / 99th percentile arrival age | 27.30 / 34.39 ms |
| Largest arrival age | 100.92 ms |
| Samples older than mean + 2 standard deviations | 66 (1.65%) |
| Invalid/duplicate/regressing timestamps | 0 |
| Samples older than existing 200 ms limit | 0 |

Bridge health was sampled at 4 Hz, not per camera callback. All observations
were accompanied by fresh healthy/live status snapshots. Thus this is a short
healthy bench distribution, not a mixture deliberately trained on quarantine.

The stop gate also runs between arrivals. A 100 Hz passive timer saw the latest
sample grow older while waiting for the next frame: mean 24.68 ms, standard
deviation 6.88 ms, largest 108.31 ms. The arrival-derived 31.69 ms threshold was
exceeded in 424 of 4,455 timer observations (9.52%). These are observations, not
424 separate quarantine events. Even fitting that timer distribution's own
mean + 2 standard deviations (38.44 ms) leaves 1.73% of observations above it.

Decision: do **not** replace the live 200 ms cutoff with mean + 2 standard
deviations. It would interrupt this otherwise healthy stream frequently, and
continuously learning from slow/quarantined data could widen the threshold
while the pipeline deteriorates. A statistical latency warning can be useful,
but changing the flight cutoff requires representative load/motion validation
and an explicit bounded policy. No adaptive timing gate was introduced.

Before optional heading output (`0606a1f`), GPS-only operation used a 1-second
accepted-callback timeout. The stricter capture-age check was added with heading
output and gates both aids when that option is enabled. Neither timeout is the
PX4 position-loss failsafe timer.

Raw probe data and script are preserved under `/tmp` on the desktop and Jetson:
`vio-camera-age-20261116-trusted-heading/{summary.json,odometry_age.csv,latest_sample_age.csv}`
and `vio_camera_age_probe.py`. Temporary files may be removed by a reboot.

## High-priority review fixes — local only, not deployed

The later local review reproduced and fixed these failures:

- Recovery could retain an old yaw-agreement pass while faster camera samples
  kept changing yaw against the same FCU sample. Every camera observation now
  rechecks disagreement; only distinct FCU samples earn agreement credit.
- Disabling heading output also disabled capture timestamp and quaternion
  validation. GPS-only and GPS-plus-yaw now share mandatory validation and the
  configured capture-age watchdog (default 200 ms). Bad data or a blackout
  latches full recovery; a single good return does not bypass it.
- GPS-only blackout recovery started its displacement reference when the camera
  returned, discarding motion during the loss. Both modes now use time-matched
  FCU displacement from the last trusted camera capture. Missing matching
  evidence cannot be substituted by a newly invented reference. Unsafe
  translation still falls back to the frozen anchor with enlarged uncertainty.
- Launcher shutdown watched only the parent shell. It now retains and verifies
  the owned process group, including surviving children, through TERM/KILL
  shutdown. Zombies are not considered running publishers. Ownership or
  shutdown failures retain the record and prevent replacement startup.

Path B's active CLI commands, package entry point, relay, MAVROS manifest/patch
and bootstrap provisioning were removed. Only migration checks for already
running legacy publishers remain. Incremental-install artifacts are retired
with backups by bootstrap; local generated Path B artifacts were removed.
Historical Git revisions remain available. Status is now menu option 2.

Preserved: first RC HIGH can trigger GPS-only startup; no respoof feature or
automatic in-flight reset was introduced. Extrinsics, physical limits, origin
and height-fusion policy, and inherited uncertainty policy are unchanged. The
obsolete accuracy-tightening prompt was removed because no such tightening is
performed. Capture-age configuration is also available with heading disabled.

Verification on the desktop: **380 tests passed**, including production
callback tests for both output modes, 2/30-second moving blackouts, clock
resets, malformed/stale inputs, and real disposable process groups with
TERM-ignoring children. The ROS package built successfully; shell syntax and
`git diff --check` passed. No Jetson runtime, PX4 parameters, firmware or
aircraft process was changed for these fixes. These are software regression
checks, not hardware or flight validation of this revision.
