# vio-launch

Interactive companion CLI (inspired by `theseus-packages`) for VIO ↔ PX4.

**On Jetson (AnyDesk):** run `vio-launch` with no args → a numbered menu pops up.

```bash
cd ~/workspaces/VIO_PX4/scripts/vio-launch
./vio-launch
```

## Main menu

| Item | What it does |
|------|----------------|
| **Path A — GPS spoof → live** | Internship-style global GPS (`HIL_GPS`) |
| **Path B — External vision** | Body VIO → MAVROS → Cube (`ODOMETRY`) |
| **FC link** | Pick direct UART/USB serial, UDP networking, or keep the current setting |
| **Home latitude/longitude** | Horizontal spoof/origin position |
| **Heading alignment** | Configure compass/manual true-heading alignment |
| **Camera extrinsic** | Configure the full ROS FLU `drone_link → camera_link` XYZ/RPY transform |
| **Pose-jump gate limits** | Configure speed, acceleration, residual, timing, and recovery limits |
| **RViz auto-launch** | Persistently toggle automatic RViz startup; default OFF |
| **Leave CLI** | Close the menu and keep processes running |
| **Stop all processes** | Stop bridges, Isaac sessions, and the container |

The CLI is hardware-only: it runs on the Jetson and talks to the real Cube.
Path A uses MAVLink directly and exclusively owns `/dev/ttyUSB0`. Path B starts
the installed MAVROS workspace on that serial link. The paths are mutually
exclusive; Micro XRCE-DDS is not used.

Settings persist in `~/.config/vio-launch/config.json`.

Camera translation uses ROS FLU axes: +X forward, +Y left, and +Z up.
Positive roll, pitch, and yaw follow the right-hand rule. In particular,
positive pitch points a forward-facing camera downward. Gate changes apply the
next time Path A starts; camera-extrinsic changes apply the next time either
path starts cuVSLAM.

## Units and pose-jump gate meanings

Camera `X/Y/Z` are entered in **metres**. Camera `roll/pitch/yaw` are entered
in **degrees** and converted to radians for ROS. Gate speed is in `m/s`,
acceleration in `m/s²`, position residual in metres, yaw rate in `deg/s`, yaw
residual in degrees, and maximum tracking gap in seconds. Confirmation and
recovery values are numbers of odometry samples.

Position residual is the difference between measured displacement and the
displacement predicted from the last trusted velocity. Yaw residual is the
equivalent difference for heading change and yaw rate. A pose **epoch** is one
continuous cuVSLAM coordinate segment. Consistent suspicious samples confirm
that cuVSLAM started a new epoch; the bridge aligns it with the last accepted
pose. It then requires the configured recovery samples to remain stable before
GPS publication resumes. The default maximum yaw rate is `360 deg/s`.

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
5. **Path A** for internship-style GPS, or **Path B** for local EV.

PX4 still needs `MAV_USEHILGPS=1` for Path A.

Path A checks the serial device and waits for a PX4 MAVLink heartbeat, then
waits for RC channel 6 LOW/MID→HIGH before starting a 15-second home spoof.
Live VIO takes over when ready. Both paths require body-frame cuVSLAM odometry
with `child_frame_id=drone_link`; raw `camera_link` poses are rejected. Path B
waits for VIO, starts MAVROS, waits for the Cube connection, then starts its
validated odometry relay.

## Direct launch commands (optional)

Direct commands still work:

```bash
./vio-launch run a
./vio-launch run b
```
