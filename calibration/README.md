# Camera extrinsic calibration

Standalone, read-only tool stored alongside VO for deployment. It records cuVSLAM
floor points, live camera extrinsics, VO pose, and a passive copy of PX4 attitude.
It proposes absolute camera roll/pitch in ROS FLU degrees. Mounting yaw is fixed at
180 degrees; XYZ is unchanged. It never applies calibration or sends FCU commands.

## Source layout

- `local/`: desktop transfer receiver, offline analysis plots and workstation workflow.
- `onboard-computer/`: Jetson recorder and receive-only telemetry reader.
- `common/`: CLI, saved-data analysis and geometry shared by both machines.
- `tests/`: offline tests and optional ROS synthetic integration tests.
- `onboard-computer/bridge-integration/`: explicit passive-telemetry patch, not automatically applied.

The installer assembles a flat runtime at `~/extrinsic_calibration` on each machine.
Recorded data and plots are NOT committed. No passwords, IP addresses, user-specific
extrinsics or flight recordings are required in Git.

## Dependencies

Both computers: Ubuntu Python 3, NumPy, SciPy, Matplotlib, OpenSSH client.
Use system Python (`/usr/bin/python3 -s`) to avoid user-site NumPy ABI conflicts.
If absent, install explicitly:

```bash
sudo apt install python3-numpy python3-scipy python3-matplotlib openssh-client
```

Jetson additionally needs the existing working ROS 2 Humble/cuVSLAM pipeline,
`rclpy`, `sensor_msgs`, `nav_msgs`, `tf2_ros`, `tf2_py`, and live TF visible to the
host. This tool does not install/rebuild Isaac ROS or change Docker. Match the
pipeline's ROS_DOMAIN_ID/RMW environment. The wrapper sources `/opt/ros/humble/setup.bash`.

Required live data:

- `/visual_slam/vis/observations_cloud` (`sensor_msgs/PointCloud2`)
- `/visual_slam/tracking/odometry` (`nav_msgs/Odometry`)
- timestamped TF among `map` (or chosen RViz frame), `odom`, `drone_link`, `camera_link`
- passive bridge JSON on **127.0.0.1:14691**, ATTITUDE and LOCAL_NED ODOMETRY.

Only one recorder can bind that port. Never open the Cube serial port from this tool.
If the capture reports PX4 MISSING, see the bridge-integration section below.

## Deploy on a new Jetson

From the cloned VIO_PX4 repository:

```bash
bash calibration/install.sh onboard-computer
~/extrinsic_calibration/extrinsic_calibration --version
```

This installs only calibration files; it does not restart VO or the flight bridge.
Start the existing VO pipeline normally and complete spoof/startup so the bridge is live.

## Deploy on the local computer

From the same repository:

```bash
bash calibration/install.sh local
~/extrinsic_calibration/start_receiver teamc@JETSON_IP
```

Authenticate normally and leave that receiver window open. It uses outgoing SSH;
**no SSH server on the local computer is needed**. Restart the receiver after a
computer/Jetson reboot or loss of its authenticated connection. Closing it does
not stop VO. Files remain on Jetson if transfer fails.

## Record on Jetson

```bash
~/extrinsic_calibration/extrinsic_calibration
```

Enter **height from Cube Orange to the physical ground plane at START (metres)**,
then the RViz Fixed Frame. Type START once and press Enter; wait for **RECORDING STARTED**.
Type STATUS and Enter for sample counts. Interactive status does not print repeatedly
over your input. STOP (or DONE), followed by Enter, finishes.
A second operator should run the terminal during flight.

After STOP:

```text
Transfer recording to your computer and generate plots? [Y/n]:
```

Y requests the connected desktop receiver to copy this capture, analyze locally,
and open the comparison image. Jetson displays the actual result path or failure.
N preserves the recording without transferring. An SSH session without a graphical
display can still save plots; image opening is best-effort.

If no receiver is connected, the fallback asks for the computer's SSH login; that
fallback requires an SSH server and the tool installed at `~/extrinsic_calibration`
(the older `~/Desktop/extrinsic_calibration` location is also supported).

Alternative: launch everything from the PC with
`~/extrinsic_calibration/extrinsic_calibration remote teamc@JETSON_IP`.
In this mode STOP automatically downloads and plots, without a second transfer prompt.

## Outputs

Jetson raw recordings: `~/.local/state/extrinsic_calibration/<run>/`.
Local results: `~/Desktop/Camera_Calibration_Results/<date_time_run>/`:

```text
capture/                         original recording, never deleted from Jetson
plots/calibration_before_after.png
plots/current_floor_view.png
plots/predicted_floor_view.png
plots/orientation_history.png
plots/recommended_cli_settings.txt
plots/report.txt
plots/report.json
```

The before/after figure shows isometric, X and Y views plus
`current CLI angle +/- adjustment = new CLI angle`. The predicted image is not
a new measurement. Three decimals are display precision, not demonstrated accuracy.
If samples disagree, the fit is labeled provisional. A horizontal floor, suitable
features and valid PX4 roll/pitch are assumptions; yaw cannot be calibrated from
a floor. No calibration is automatically applied. Review the result and enter
values into the VO CLI manually, then restart VO for the new extrinsics to take effect.

To analyze existing data on the PC:

```bash
~/extrinsic_calibration/extrinsic_calibration analyze /path/to/capture --output /path/to/NEW/plots --open
```

## Passive bridge integration (explicit, ground-only deployment)

The currently tested bridge addition is commit `3b89769`, based on main `ce9eba3`.
It copies already-received PX4 messages to localhost; it does NOT send VO heading,
route IMU into cuVSLAM, change HIL_GPS or change recovery thresholds.
The patch is included so another agent can reproduce the integration without a
hidden dependency on this Jetson. The installer never applies it automatically.

If the bridge already imports `DiagnosticTelemetry`, do not apply it again.
Otherwise, from the VIO_PX4 repo, while disarmed and not flying:

```bash
git apply --check calibration/onboard-computer/bridge-integration/*.patch
git apply calibration/onboard-computer/bridge-integration/*.patch
source /opt/ros/humble/setup.bash
colcon build --packages-select vio_px4_bridge
```

Review any conflict instead of forcing the patch. Stop/restart Path A on the ground
to load the rebuilt bridge. The existing extrinsics/configuration stay unchanged.
This prerequisite is separate from installing the recorder; main flight code is
not modified by adding this calibration folder.

## Tests

From this repository, using an isolated install:

```bash
calibration_test_dir=$(mktemp -d /tmp/calibration-test.XXXXXX)
bash calibration/install.sh local "$calibration_test_dir"
PYTHONPATH="$calibration_test_dir" /usr/bin/python3 -s -m unittest discover -s calibration/tests -p 'test_*.py'
```

ROS smoke tests are optional and use synthetic DDS data, not the real camera/FCU.
They establish code/data-path behavior, not physical calibration accuracy.
