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
  -> independent compass/declination heading alignment
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
displacement and yaw change for at most two seconds. PX4 position variance and
the estimator reset counter are checked before that displacement is accepted.
The recovered raw VIO segment is aligned to the last trusted pose plus only that
short-term displacement. Before handoff, consecutive VIO/PX4 velocities must
agree and consecutive relative yaw changes must agree. Absolute post-reset VIO
yaw is intentionally ignored because a new coordinate epoch may start at any
angle; its stable relative rotation is mapped onto the last trusted yaw plus
the guarded PX4 yaw change. Duplicate PX4 attitude samples are not counted.
Candidate/recovery VIO samples never move the trusted anchor. A failed
propagation is discarded and recovery continues from the frozen pose. On a
successful handoff, reported GPS
accuracy starts at PX4's accumulated horizontal uncertainty and tightens back
to 0.1 m. All numeric limits are configured in
`vio-launch`.

## Path B: external vision through MAVROS

```text
body-frame cuVSLAM odometry
  -> validated frame relay
  -> /mavros/odometry/out
  -> MAVROS ODOMETRY
  -> PX4 EKF2 external-vision fusion
```

Path A directly owns the FC serial port. Path B gives that port to MAVROS. They
are mutually exclusive. Micro XRCE-DDS, SITL, fake odometry, and
`test_fmu_vio` are not part of this hardware pipeline.

## Reproducibility boundary

This Git repository owns:

- both PX4 bridge paths;
- the numbered operator CLI;
- camera/body launch overrides;
- Jetson bootstrap and dependency checks;
- PX4 interface and staged-IMU documentation.

The external Isaac ROS workspace owns NVIDIA packages, Docker support, and
`run_dev.sh`. `bootstrap/bootstrap_jetson.sh` installs this repository's reviewed
overrides into that workspace. Docker images, ROS build products, runtime logs,
flight bags, maps, and credentials are intentionally excluded from Git.
