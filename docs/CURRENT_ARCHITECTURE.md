# Current Jetson-to-PX4 architecture

## Common VIO front end

```text
vio-launch
  -> Isaac ROS run_dev.sh
  -> version-controlled start_vio.sh override
  -> RealSense D456 stereo
  -> cuVSLAM(base_frame=drone_link)
  -> /visual_slam/tracking/odometry
     frame_id=odom, child_frame_id=drone_link
```

The physical camera mount is represented once by a static ROS FLU transform:

```text
drone_link -> camera_link
```

The launcher supplies the full translation and rotation through
`CAMERA_{X,Y,Z}_M` and `CAMERA_{ROLL,PITCH,YAW}_RAD`. There is no downstream
correction of completed odometry. Translation uses ROS FLU axes (+X forward,
+Y left, +Z up), and rotation follows the right-hand rule.

## Path A: horizontal VIO as GPS

```text
body-frame cuVSLAM odometry
  -> physical-limit and pose-jump continuity gate
  -> fixed heading alignment using PX4 at startup (or a manual reference)
  -> continuous local displacement anchored at configured latitude/longitude
  -> MAVLink HIL_GPS at 10 Hz
  -> PX4 EKF2 horizontal-position fusion
```

Path A waits for RC channel 6 LOW/MID to HIGH, publishes a fixed 15-second home
spoof, and then hands off to live VIO. Only latitude/longitude convey motion.
The protocol-required altitude is fixed and velocity is zero. PX4 uses
`EKF2_GPS_CTRL=1`, so PX4 owns height and velocity estimation.

When VIO is stale or quarantined, the bridge stops sending HIL_GPS. PX4 handles
source timeout, physical-GNSS selection, dead reckoning, and configured
failsafes. Short relocalizations are absorbed locally without a global jump.
The operator can change all continuity thresholds from `vio-launch`; saved
values are passed explicitly to the bridge at every Path A launch.
The bridge converts speed and yaw-rate limits into allowed pose changes using
the measured timestamp interval between each pair of odometry messages. A
confirmed coordinate reset increments the internal segment counter. While
HIL_GPS is silent, the bridge measures guarded PX4 `ODOMETRY` horizontal
displacement. There is no fixed duration cutoff: telemetry freshness, resets,
uncertainty, speed and acceleration determine whether translation can be used.
Both GPS-only and GPS-plus-heading operation match FCU samples to camera
capture time. The displacement starts at the **last trusted camera capture**,
not at the later moment a blackout is noticed or the camera returns.

Before handoff, the camera must form a stable coordinate segment. Velocity and
relative-yaw agreement windows collect independently, and both must pass.
Duplicate FCU evidence cannot add another good sample, but the newest camera
observation can revoke a previous pass. No physical turn is needed to recover.
The recovered segment is placed at the last trusted output plus the validated
FCU displacement, with yaw aligned to current time-matched FCU heading.

If position propagation becomes unsafe, its movement is discarded. Recovery
can still finish at the frozen position once camera/FCU agreement passes;
reported position uncertainty includes the possible lost movement. Missing
time-matched heading/velocity evidence keeps output stopped, not permanently
latched against later healthy telemetry. Neither candidate samples nor bad
inputs move the trusted output anchor.

Capture timestamps, finite values, the body frame and unit quaternion are
checked regardless of whether optional heading output is enabled. A camera
blackout latches full recovery at the configured capture-age limit (200 ms by
default); one returning good sample cannot bypass recovery. With heading
enabled, both live measurement streams stop together. Telemetry and TIMESYNC
continue while navigation measurements are paused.

After handoff, inherited position and heading uncertainty are retained. Stable
relative VO alone does not justify tightening the global reference back to
0.1 m or one degree. Numeric gate limits are configured in `vio-launch`.

## Reproducibility boundary

This Git repository owns:

- Path A HIL_GPS and its optional yaw-only ODOMETRY output;
- the numbered operator CLI;
- camera/body launch overrides;
- Jetson bootstrap and dependency checks;
- PX4 interface and staged-IMU documentation.

The external Isaac ROS workspace owns NVIDIA packages, Docker support, and
`run_dev.sh`. `bootstrap/bootstrap_jetson.sh` installs this repository's reviewed
overrides into that workspace. Docker images, ROS build products, runtime logs,
flight bags, maps, and credentials are intentionally excluded from Git.
