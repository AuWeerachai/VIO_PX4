# vio-test

Interactive companion CLI (inspired by `theseus-packages`) for VIO ↔ PX4.

**On Jetson (AnyDesk):** run `vio-test` with no args → a numbered menu pops up.

```bash
cd ~/Desktop/VIO_PX4/tools/vio-test   # or wherever you copied it
./vio-test
```

## Main menu

| Item | What it does |
|------|----------------|
| **Path A — GPS spoof → live** | Internship-style global GPS (`HIL_GPS`) |
| **Path B — External vision** | Local VIO → Cube (`vehicle_visual_odometry`) |
| **FC link** | Pick direct UART/USB serial, UDP networking, or keep the current setting |
| **Home lat/lon/alt** | Spoof / origin |
| Doctor / Status / Logs / Stop | Ops |

The CLI is hardware-only: it runs on the Jetson and talks to the real Cube.
Path A uses MAVLink directly. Path B uses whichever PX4↔ROS 2 middleware you
already configured; the CLI checks for `/fmu/out/*` topics but does not start a
specific middleware agent.

Settings persist in `~/.config/vio-test/config.json`.

## Selecting an option

Type the number shown next to an option and press `Enter`.

- `0` returns to the previous menu
- `q` also returns or exits

Arrow keys are intentionally not used because some AnyDesk terminals do not
forward their escape sequences correctly.

## First-time on Jetson

1. Copy `tools/vio-test/` (at least `vio-test` + `vio_test.py`) to the Orin
2. `chmod +x vio-test`
3. `./vio-test`
4. **FC link → UART** → pick your serial device / baud
5. **Path A** for internship-style GPS, or **Path B** for local EV

PX4 still needs `MAV_USEHILGPS=1` for Path A.

Path A checks the serial device and waits for a PX4 MAVLink heartbeat, then
waits for RC channel 6 LOW/MID→HIGH before starting a 15-second home spoof.
Live VIO takes over when ready. Path B waits for a
live `/fmu/out/vehicle_status` message and a live VIO odometry message before
launching. See `~/test_procedure.txt` for the ordered multi-terminal startup.

## Scripting (optional)

Direct commands still work:

```bash
./vio-test doctor
./vio-test run a
./vio-test stop
```
