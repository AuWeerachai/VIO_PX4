# vio-launch

Interactive companion CLI (inspired by `theseus-packages`) for VIO ↔ PX4.

**On Jetson (AnyDesk):** run `vio-launch` with no args → a numbered menu pops up.

```bash
cd ~/workspaces/VIO_PX4_NEW_PIPELINE/tools/vio-launch
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
| **Camera tilt** | Configure the `drone_link → camera_link` pitch extrinsic |
| **RViz auto-launch** | Persistently toggle automatic RViz startup; default OFF |
| **Leave CLI** | Close the menu and keep processes running |
| **Stop all processes** | Stop bridges, Isaac sessions, and the container |

The CLI is hardware-only: it runs on the Jetson and talks to the real Cube.
Path A uses MAVLink directly and exclusively owns `/dev/ttyUSB0`. Path B starts
the installed MAVROS workspace on that serial link. The paths are mutually
exclusive; Micro XRCE-DDS is not used.

Settings persist in `~/.config/vio-launch/config.json`.

## Selecting an option

Type the number shown next to an option and press `Enter`.

- `0` returns to the previous menu
- `q` also returns or exits

Arrow keys are intentionally not used because some AnyDesk terminals do not
forward their escape sequences correctly.

## First-time on Jetson

1. Clone this repository into `~/workspaces/VIO_PX4`.
2. Run `./scripts/bootstrap_jetson.sh --install-deps` from the repository root.
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
