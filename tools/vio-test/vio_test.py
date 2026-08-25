#!/usr/bin/env python3
"""vio-test — interactive hardware CLI for Jetson VIO ↔ PX4 Cube.

Default (no args): numbered interactive menu that works reliably over AnyDesk.
Also supports direct subcommands: doctor, run a, gps, stop, ...
"""

from __future__ import annotations

import argparse
import errno
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def expand(p: str) -> Path:
    return Path(os.path.expanduser(p)).resolve()


def first_existing(candidates: list[str]) -> Optional[Path]:
    for c in candidates:
        path = expand(c)
        if path.exists():
            return path
    return None


def user_config_path() -> Path:
    return Path.home() / ".config" / "vio-test" / "config.json"


@dataclass
class Config:
    vio_px4_dir: Path = field(
        default_factory=lambda: Path.home() / "workspaces/VIO_PX4_NEW_PIPELINE"
    )
    ros_setup: Path = field(default_factory=lambda: Path("/opt/ros/humble/setup.bash"))
    mavros_dir: Path = field(default_factory=lambda: Path.home() / "workspaces/mavros")
    px4_msgs_setup: Optional[Path] = None
    home_lat: float = 40.4433
    home_lon: float = -79.9436
    home_alt: float = 300.0
    # Jetson↔Cube is typically UART, but a routed UDP MAVLink link also works.
    mavlink_url: str = "/dev/ttyUSB0:921600"
    mavlink_sysid: int = 1
    odom_topic: str = "/visual_slam/tracking/odometry"
    heading_source: str = "compass"
    manual_heading_deg: float = 0.0
    mag_declination_source: str = "table"
    mag_declination_deg: float = 0.0
    child_to_body_yaw_deg: float = 0.0
    state_dir: Path = field(default_factory=lambda: Path.home() / ".local/state/vio-test")

    @property
    def log_dir(self) -> Path:
        return self.state_dir / "logs"

    @property
    def state_file(self) -> Path:
        return self.state_dir / "state.json"

def default_config() -> Config:
    home = Path.home()
    desktop = home / "Desktop"
    return Config(
        vio_px4_dir=first_existing([
            "~/workspaces/VIO_PX4_NEW_PIPELINE",
            str(desktop / "VIO_PX4"),
            "~/VIO_PX4",
        ])
        or (desktop / "VIO_PX4"),
        ros_setup=first_existing(["/opt/ros/humble/setup.bash", "/opt/ros/jazzy/setup.bash"])
        or Path("/opt/ros/humble/setup.bash"),
        px4_msgs_setup=first_existing(
            [
                str(home / "ws_px4_dev/install/setup.bash"),
                str(desktop / "VIO_PX4/install/setup.bash"),
            ]
        ),
        mavlink_url="/dev/ttyUSB0:921600",
    )


def load_saved_overrides() -> dict:
    path = user_config_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_config(cfg: Config) -> None:
    path = user_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mavlink_url": cfg.mavlink_url,
        "mavlink_sysid": cfg.mavlink_sysid,
        "home_lat": cfg.home_lat,
        "home_lon": cfg.home_lon,
        "home_alt": cfg.home_alt,
        "odom_topic": cfg.odom_topic,
        "heading_source": cfg.heading_source,
        "manual_heading_deg": cfg.manual_heading_deg,
        "mag_declination_source": cfg.mag_declination_source,
        "mag_declination_deg": cfg.mag_declination_deg,
        "child_to_body_yaw_deg": cfg.child_to_body_yaw_deg,
        "vio_px4_dir": str(cfg.vio_px4_dir),
        "ros_setup": str(cfg.ros_setup),
        "mavros_dir": str(cfg.mavros_dir),
    }
    atomic_write_json(path, payload)


def resolved_declination_deg(cfg: Config) -> Optional[float]:
    """Preview the same bundled grid lookup used by the ROS bridge."""
    if cfg.mag_declination_source == "manual":
        return cfg.mag_declination_deg
    table_path = (cfg.vio_px4_dir / "src/vio_px4_bridge/vio_px4_bridge/data/"
                  "declination_table.json")
    try:
        table = json.loads(table_path.read_text())
        grid = table["declination_deg"]
        lon = cfg.home_lon
        while lon > 180.0: lon -= 360.0
        while lon < -180.0: lon += 360.0
        i = min(max((cfg.home_lat-table["lat_min_deg"])/table["lat_step_deg"], 0.0), len(grid)-1.0)
        j = min(max((lon-table["lon_min_deg"])/table["lon_step_deg"], 0.0), len(grid[0])-1.0)
        i0, j0 = min(int(i), len(grid)-2), min(int(j), len(grid[0])-2)
        fi, fj = i-i0, j-j0
        return (grid[i0][j0]*(1-fi)*(1-fj) + grid[i0+1][j0]*fi*(1-fj)
                + grid[i0][j0+1]*(1-fi)*fj + grid[i0+1][j0+1]*fi*fj)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def declination_label(cfg: Config) -> str:
    value = resolved_declination_deg(cfg)
    if cfg.mag_declination_source == "table":
        return (f"table(home)={value:.2f}°" if value is not None
                else "table unavailable (launch blocked)")
    return f"manual={cfg.mag_declination_deg:.2f}°"


def atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON without leaving an empty/corrupt file after an I/O failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def apply_dict(cfg: Config, data: dict) -> Config:
    updates: dict = {}
    for key in (
        "mavlink_url",
        "mavlink_sysid",
        "home_lat",
        "home_lon",
        "home_alt",
        "odom_topic",
        "heading_source",
        "manual_heading_deg",
        "mag_declination_source",
        "mag_declination_deg",
        "child_to_body_yaw_deg",
    ):
        if key in data and data[key] is not None:
            updates[key] = data[key]
    for key in ("vio_px4_dir", "ros_setup", "mavros_dir"):
        if key in data and data[key]:
            updates[key] = expand(str(data[key]))
    return replace(cfg, **updates) if updates else cfg


def load_config(ns: argparse.Namespace | None = None) -> Config:
    cfg = default_config()
    cfg = apply_dict(cfg, load_saved_overrides())
    saved = load_saved_overrides()
    if "mavlink_url" not in saved:
        cfg = replace(cfg, mavlink_url="/dev/ttyUSB0:921600")
    if ns is None:
        return cfg
    updates = {}
    if getattr(ns, "mavlink_url", None):
        updates["mavlink_url"] = ns.mavlink_url
    if getattr(ns, "odom_topic", None):
        updates["odom_topic"] = ns.odom_topic
    if getattr(ns, "home_lat", None) is not None:
        updates["home_lat"] = ns.home_lat
    if getattr(ns, "home_lon", None) is not None:
        updates["home_lon"] = ns.home_lon
    if getattr(ns, "home_alt", None) is not None:
        updates["home_alt"] = ns.home_alt
    return replace(cfg, **updates) if updates else cfg


# ---------------------------------------------------------------------------
# Interactive select menu (theseus-packages SelectMenu analogue)
# ---------------------------------------------------------------------------


@dataclass
class MenuItem:
    label: str
    value: str
    description: str = ""
    disabled: bool = False


def _clear() -> None:
    if sys.stdout.isatty():
        # Clear the visible terminal and move the cursor home so repeated menu
        # screens do not pile up. Standard ANSI sequences work in GNOME
        # Terminal over AnyDesk and avoid spawning an external `clear` command.
        print("\033[2J\033[H", end="", flush=True)


def select_menu(
    title: str,
    items: list[MenuItem],
    subtitle: str = "",
) -> Optional[str]:
    """Numbered menu (works over AnyDesk / serial / plain SSH).

    Type a number and press Enter. Also accepts q / 0 to cancel.
    """
    choices = [i for i in items if not i.disabled]
    if not choices:
        return None

    while True:
        _clear()
        print("=" * 52)
        print(f"  vio-test  v{VERSION}")
        print(f"  {title}")
        if subtitle:
            print(f"  {subtitle}")
        print("=" * 52)
        print()

        # Section headers (disabled items with a label) for readability
        choice_num = 0
        num_map: dict[int, MenuItem] = {}
        for item in items:
            if item.disabled:
                if item.label:
                    print(f"  --- {item.label} ---")
                else:
                    print()
                continue
            choice_num += 1
            num_map[choice_num] = item
            print(f"  {choice_num}) {item.label}")
            if item.description:
                print(f"      {item.description}")

        print()
        print("  0) Cancel / back")
        print()
        try:
            raw = input("Enter number and press Enter: ").strip().lower()
        except EOFError:
            return None

        if raw in {"0", "q", "quit", "exit", "b", "back"}:
            return None
        if raw == "":
            print("  Please type an option number, then press Enter.")
            continue
        try:
            n = int(raw)
        except ValueError:
            print(f"  Invalid choice '{raw}'. Try again.")
            continue
        if n in num_map:
            return num_map[n].value
        print(f"  Choice {n} out of range. Try again.")


def prompt_line(label: str, default: str = "") -> Optional[str]:
    suffix = f" [{default}]" if default else ""
    try:
        raw = input(f"{label}{suffix}: ").strip()
    except EOFError:
        return None
    if raw == "":
        return default
    return raw


def pause(msg: str = "Press Enter to continue...") -> None:
    try:
        input(msg)
    except EOFError:
        pass


# ---------------------------------------------------------------------------
# Process manager
# ---------------------------------------------------------------------------


def ensure_dirs(cfg: Config) -> None:
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)


def read_state(cfg: Config) -> dict:
    ensure_dirs(cfg)
    if not cfg.state_file.exists():
        return {"procs": []}
    try:
        state = json.loads(cfg.state_file.read_text())
        if not isinstance(state, dict) or not isinstance(state.get("procs", []), list):
            return {"procs": []}
        return state
    except (OSError, json.JSONDecodeError, TypeError):
        return {"procs": []}


def write_state(cfg: Config, state: dict) -> None:
    ensure_dirs(cfg)
    atomic_write_json(cfg.state_file, state)


def process_start_ticks(pid: int) -> Optional[int]:
    """Return Linux process start ticks, used to detect PID reuse."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return int(fields[19])
    except (OSError, IndexError, ValueError):
        return None


def managed_alive(proc: dict) -> bool:
    try:
        pid = int(proc["pid"])
    except (KeyError, TypeError, ValueError):
        return False
    current_ticks = process_start_ticks(pid)
    if current_ticks is None:
        return False
    saved_ticks = proc.get("startTicks")
    return saved_ticks is None or current_ticks == saved_ticks


def prune_state(cfg: Config) -> list[dict]:
    state = read_state(cfg)
    alive = [p for p in state.get("procs", []) if managed_alive(p)]
    # Do not rewrite the state file on every menu refresh. Apart from needless
    # disk I/O, that used to crash the menu when the filesystem was full.
    if alive != state.get("procs", []):
        try:
            write_state(cfg, {"procs": alive})
        except OSError as exc:
            if exc.errno != errno.ENOSPC:
                raise
            print("Warning: disk is full; process state could not be updated.", file=sys.stderr)
    return alive


def find_proc(cfg: Config, name: str) -> Optional[dict]:
    return next((p for p in prune_state(cfg) if p["name"] == name), None)


def register_proc(cfg: Config, proc: dict) -> None:
    state = read_state(cfg)
    procs = [
        p
        for p in state.get("procs", [])
        if p.get("name") != proc["name"] and managed_alive(p)
    ]
    procs.append(proc)
    write_state(cfg, {"procs": procs})


def start_managed(
    cfg: Config,
    name: str,
    bash_command: str,
    startup_check_s: float = 0.75,
) -> dict:
    existing = find_proc(cfg, name)
    if existing:
        raise RuntimeError(f"{name} already running (pid {existing['pid']}). Stop it first.")
    ensure_dirs(cfg)
    log_file = cfg.log_dir / f"{name}.log"
    if log_file.exists() and log_file.stat().st_size > 5 * 1024 * 1024:
        # Bound unattended log growth. Keep one previous log for diagnosis.
        os.replace(log_file, log_file.with_suffix(".log.1"))
    with log_file.open("a", encoding="utf-8") as log:
        log.write(f"\n===== {time.strftime('%Y-%m-%dT%H:%M:%S')} start {name} =====\n")
        log.write(f"$ {bash_command}\n")
        log.flush()
        proc = subprocess.Popen(
            ["bash", "-lc", bash_command],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=os.environ.copy(),
        )
    time.sleep(startup_check_s)
    if proc.poll() is not None:
        recent = log_file.read_text(errors="replace").strip().splitlines()[-12:]
        raise RuntimeError(
            f"{name} failed to start. Recent log:\n" + "\n".join(recent)
        )

    managed = {
        "name": name,
        "pid": proc.pid,
        "startTicks": process_start_ticks(proc.pid),
        "logFile": str(log_file),
        "startedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "command": bash_command,
    }
    try:
        register_proc(cfg, managed)
    except Exception:
        # Never leave an untracked publisher running if state persistence fails.
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except OSError:
            pass
        raise
    return managed


def stop_managed(cfg: Config, name: Optional[str] = None) -> list[str]:
    procs = prune_state(cfg)
    targets = [p for p in procs if name is None or p["name"] == name]
    stopped: list[str] = []
    for p in targets:
        pid = int(p["pid"])
        try:
            os.killpg(pid, signal.SIGTERM)
        except OSError:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
        deadline = time.monotonic() + 2.0
        while managed_alive(p) and time.monotonic() < deadline:
            time.sleep(0.05)
        if managed_alive(p):
            try:
                os.killpg(pid, signal.SIGKILL)
            except OSError:
                pass
        stopped.append(p["name"])
    if targets:
        stopped_names = {p["name"] for p in targets}
        remaining = [p for p in procs if p.get("name") not in stopped_names]
        write_state(cfg, {"procs": remaining})
    return stopped


def ros_prefix(cfg: Config, extra_setups: Optional[list[Path]] = None) -> str:
    setups = [cfg.ros_setup]
    if cfg.px4_msgs_setup and cfg.px4_msgs_setup.exists():
        setups.append(cfg.px4_msgs_setup)
    setups.extend(setup for setup in (extra_setups or []) if setup.exists())
    # VIO_PX4 may be both the discovered px4_msgs overlay and the explicitly
    # requested package overlay. Source each setup file only once.
    seen: set[Path] = set()
    parts = []
    for setup in setups:
        resolved = setup.resolve()
        if resolved not in seen:
            seen.add(resolved)
            parts.append(f'source "{resolved}"')
    return " && ".join(parts)


def normalize_path(value: str) -> str:
    v = value.lower()
    if v in {"a", "gps", "hil", "internship"}:
        return "a"
    if v in {"b", "ev", "vision", "external-vision"}:
        return "b"
    raise RuntimeError(f"Unknown path '{value}'. Use a|gps or b|ev.")


def validate_mavlink_url(value: str) -> Optional[str]:
    if value.startswith("/dev/"):
        device, separator, baud = value.rpartition(":")
        if not separator or not device or not baud.isdigit() or int(baud) <= 0:
            return "Serial link must look like /dev/ttyUSB0:921600"
        return None
    if value.startswith(("udpout:", "udpin:")):
        parts = value.split(":")
        if len(parts) != 3 or not parts[1]:
            return "UDP link must look like udpout:HOST:PORT"
        try:
            port = int(parts[2])
        except ValueError:
            return "UDP port must be a number"
        if not 1 <= port <= 65535:
            return "UDP port must be between 1 and 65535"
        return None
    return "Use a serial link (/dev/...:BAUD) or UDP link (udpout:HOST:PORT)"


def serial_users(device: Path) -> list[int]:
    result = subprocess.run(
        ["fuser", str(device)], capture_output=True, text=True, check=False
    )
    return [int(value) for value in result.stdout.split() if value.isdigit()]


def ros_topic_names(cfg: Config) -> set[str]:
    command = (
        f"{ros_prefix(cfg, [cfg.vio_px4_dir / 'install/setup.bash'])} && "
        "ros2 topic list --no-daemon"
    )
    try:
        result = subprocess.run(
            ["bash", "-lc", command],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Timed out while checking ROS 2 topics") from exc
    if result.returncode != 0:
        raise RuntimeError(
            "Could not query ROS 2 topics. Confirm ROS is running and start your "
            "configured PX4↔ROS middleware before using Path B."
        )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def require_px4_ros_link(cfg: Config) -> None:
    topics = ros_topic_names(cfg)
    status_topic = "/fmu/out/vehicle_status"
    if status_topic not in topics:
        raise RuntimeError(
            f"PX4 ROS 2 topic {status_topic} is not available. Start your configured "
            "PX4↔ROS middleware, then retry Path B."
        )
    require_ros_message(cfg, status_topic, "PX4 status", timeout_s=5.0)


def require_ros_message(
    cfg: Config,
    topic: str,
    label: str,
    timeout_s: float,
) -> None:
    command = (
        f"{ros_prefix(cfg, [cfg.vio_px4_dir / 'install/setup.bash'])} && "
        f"ros2 topic echo --no-daemon --once {shlex.quote(topic)}"
    )
    try:
        result = subprocess.run(
            ["bash", "-lc", command],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"{label} is not ready: no message arrived on {topic} "
            f"within {timeout_s:.0f} seconds"
        ) from exc
    if result.returncode != 0:
        raise RuntimeError(
            f"{label} check failed on {topic}. Confirm the publisher and your "
            "ROS 2 middleware are running."
        )


def require_mavros_connected(cfg: Config, timeout_s: float = 10.0) -> None:
    command = (
        f"{ros_prefix(cfg, [cfg.mavros_dir / 'install/setup.bash'])} && "
        "ros2 topic echo --no-daemon --once /mavros/state"
    )
    try:
        result = subprocess.run(
            ["bash", "-lc", command], capture_output=True, text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("MAVROS did not publish state before timeout") from exc
    if result.returncode != 0 or "connected: true" not in result.stdout.lower():
        raise RuntimeError("MAVROS is running but is not connected to the Cube")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_doctor(cfg: Config) -> int:
    url_error = validate_mavlink_url(cfg.mavlink_url)
    checks = [
        ("MAVLink configuration", url_error is None, url_error or cfg.mavlink_url),
        ("ROS setup", cfg.ros_setup.exists(), str(cfg.ros_setup)),
        (
            "VIO_PX4 install",
            (cfg.vio_px4_dir / "install/setup.bash").exists(),
            str(cfg.vio_px4_dir),
        ),
        (
            "MAVROS install (Path B)",
            (cfg.mavros_dir / "install/setup.bash").exists(),
            str(cfg.mavros_dir),
        ),
    ]
    if cfg.heading_source == "compass" and cfg.mag_declination_source == "table":
        resolved_declination = resolved_declination_deg(cfg)
        checks.append((
            "Magnetic declination table",
            resolved_declination is not None,
            (f"home lookup {resolved_declination:.3f} deg"
             if resolved_declination is not None else "missing or invalid"),
        ))
    if cfg.mavlink_url.startswith("/dev/"):
        dev = cfg.mavlink_url.rsplit(":", 1)[0]
        exists = Path(dev).exists()
        checks.append(("Serial device", exists, dev))
        checks.append(
            ("Serial permission", exists and os.access(dev, os.R_OK | os.W_OK), dev)
        )

    print("vio-test doctor\n")
    failed = 0
    for label, ok, detail in checks:
        print(f"{'[OK]' if ok else '[!!]'} {label}: {detail}")
        if not ok:
            failed += 1
    print(f"\nhome LLA: {cfg.home_lat}, {cfg.home_lon}, {cfg.home_alt}")
    print(f"odom:     {cfg.odom_topic}")
    print(f"heading:  {cfg.heading_source}, declination={declination_label(cfg)}, "
          f"child-to-body={cfg.child_to_body_yaw_deg} deg")
    print(f"checklist: {cfg.vio_px4_dir / 'PX4_INTERFACE_CHECKLIST.md'}")
    print(f"config:   {user_config_path()}")
    return 0 if failed == 0 else 1


def cmd_status(cfg: Config) -> None:
    procs = prune_state(cfg)
    if not procs:
        print("No managed processes.")
        return
    print("Running:\n")
    for p in procs:
        phase = "running"
        log_path = Path(p.get("logFile", ""))
        if p.get("name") == "gps-bridge" and log_path.is_file():
            recent = "\n".join(log_path.read_text(errors="replace").splitlines()[-80:])
            states = [
                (recent.rfind("VIO odometry stale"), "VIO stale; GPS updates stopped"),
                (recent.rfind("GPS spoof expired"), "waiting for RC trigger; previous spoof expired"),
                (recent.rfind("GPS [live]"), "live VIO GPS"),
                (recent.rfind("heading alignment completes"), "spoofing home; aligning heading"),
                (recent.rfind("heading alignment locked"), "heading aligned; preparing live GPS"),
                (recent.rfind("GPS [spoof]"), "spoofing home; waiting for VIO"),
                (recent.rfind("Waiting for RC channel"), "waiting for RC channel 6 HIGH"),
                (recent.rfind("PX4 heartbeat received"), "PX4 connected; initializing"),
            ]
            latest_state = max(states, key=lambda item: item[0])
            if latest_state[0] >= 0:
                phase = latest_state[1]
        elif p.get("name") == "ev-bridge":
            phase = "MAVROS external-vision relay running"
        print(f"  {p['name']}  {phase}  pid={p['pid']}  since={p['startedAt']}")
        print(f"    log: {p['logFile']}")


def cmd_ev_bridge(cfg: Config) -> None:
    setup = cfg.vio_px4_dir / "install/setup.bash"
    if not setup.exists():
        raise RuntimeError(f"Missing {setup} — build vio_px4_bridge first")
    print(f"Checking live VIO odometry on {cfg.odom_topic}...", flush=True)
    require_ros_message(cfg, cfg.odom_topic, "VIO odometry", timeout_s=5.0)
    print("VIO odometry ready.")
    if not find_proc(cfg, "mavros"):
        mavros_setup = cfg.mavros_dir / "install/setup.bash"
        if not mavros_setup.exists():
            raise RuntimeError(f"Missing MAVROS setup: {mavros_setup}")
        if not cfg.mavlink_url.startswith("/dev/"):
            raise RuntimeError("Path B currently requires the Jetson serial link")
        serial_device, _, baud = cfg.mavlink_url.rpartition(":")
        holders = serial_users(Path(serial_device))
        if holders:
            raise RuntimeError(
                f"{serial_device} is already open by pid(s) {holders}; stop Path A "
                "or the old MAVROS launcher first"
            )
        mavros_cmd = (
            f"{ros_prefix(cfg, [mavros_setup])} && ros2 launch mavros px4.launch "
            f"fcu_url:=serial://{serial_device}:{baud}"
        )
        mavros_proc = start_managed(cfg, "mavros", mavros_cmd, startup_check_s=3.0)
        print(f"Started MAVROS pid={mavros_proc['pid']}; waiting for FCU connection...")
        require_mavros_connected(cfg, timeout_s=10.0)
    cmd = (
        f"{ros_prefix(cfg, [setup])} && ros2 run vio_px4_bridge mavros_ev_bridge --ros-args"
        f" -p {shlex.quote(f'odom_topic:={cfg.odom_topic}')}"
        " -p expected_child_frame:=drone_link"
        " -p output_parent_frame:=odom"
    )
    proc = start_managed(cfg, "ev-bridge", cmd)
    print(f"Started EV bridge pid={proc['pid']}")
    print(f"log: {proc['logFile']}")


def cmd_vio(cfg: Config) -> None:
    script = cfg.vio_px4_dir / "scripts/start_cuvslam_body.sh"
    if not script.exists():
        raise RuntimeError(f"Missing cuVSLAM launcher: {script}")
    cmd = f"{ros_prefix(cfg)} && bash {shlex.quote(str(script))}"
    proc = start_managed(cfg, "cuvslam", cmd, startup_check_s=3.0)
    print(f"Started body-frame cuVSLAM pid={proc['pid']}; waiting for odometry...")
    try:
        require_ros_message(cfg, cfg.odom_topic, "body-frame cuVSLAM odometry", 30.0)
    except Exception:
        stop_managed(cfg, "cuvslam")
        raise
    print("cuVSLAM body odometry ready.")


def cmd_gps(cfg: Config) -> None:
    setup = cfg.vio_px4_dir / "install/setup.bash"
    if not setup.exists():
        raise RuntimeError(
            f"VIO_PX4 not built. Run: cd {cfg.vio_px4_dir} && "
            "colcon build --packages-select vio_px4_bridge"
        )
    url_error = validate_mavlink_url(cfg.mavlink_url)
    if url_error:
        raise RuntimeError(url_error)
    if cfg.mavlink_url.startswith("/dev/"):
        serial_device = Path(cfg.mavlink_url.rsplit(":", 1)[0])
        if not serial_device.exists():
            raise RuntimeError(
                f"Cube serial device not found: {serial_device}\n"
                "Connect the Cube, then use FC link to select the detected device."
            )
        if not os.access(serial_device, os.R_OK | os.W_OK):
            raise RuntimeError(f"No read/write permission for {serial_device}")
        holders = serial_users(serial_device)
        if holders:
            raise RuntimeError(
                f"{serial_device} is already open by pid(s) {holders}. Path A needs "
                "exclusive serial access; stop MAVROS first."
            )
    params = {
        "transport": "mavlink",
        "mavlink_url": cfg.mavlink_url,
        "mavlink_sysid": cfg.mavlink_sysid,
        "home_lat_deg": cfg.home_lat,
        "home_lon_deg": cfg.home_lon,
        "home_alt_m": cfg.home_alt,
        "spoof_duration_s": 15.0,
        "spoof_until_vio": "true",
        "rc_trigger_enabled": "true",
        "rc_channel": 6,
        "rc_low_pwm_max": 1300,
        "rc_high_pwm_min": 1700,
        "rc_request_rate_hz": 10.0,
        "rate_hz": 10.0,
        "odom_topic": cfg.odom_topic,
        "heading_source": cfg.heading_source,
        "manual_heading_deg": cfg.manual_heading_deg,
        "mag_declination_source": cfg.mag_declination_source,
        "mag_declination_deg": cfg.mag_declination_deg,
        "child_to_body_yaw_deg": cfg.child_to_body_yaw_deg,
        "expected_child_frame": "drone_link",
    }
    param_args = " ".join(
        f"-p {shlex.quote(f'{name}:={value}')}" for name, value in params.items()
    )
    cmd = (
        f"{ros_prefix(cfg, [setup])} && ros2 run vio_px4_bridge vio_px4_gps_bridge --ros-args"
        f" {param_args}"
    )
    proc = start_managed(cfg, "gps-bridge", cmd, startup_check_s=6.0)
    print(f"Path A connected to PX4 on {cfg.mavlink_url} (pid {proc['pid']}).")
    print("Waiting for RC channel 6 LOW/MID -> HIGH to start the 15-second home spoof.")
    print("Live VIO will take over automatically when odometry becomes ready.")
    print(f"Log: {proc['logFile']}")


def cmd_run(
    cfg: Config,
    path_arg: str,
) -> None:
    which = normalize_path(path_arg)
    label = "GPS spoof + VIO GPS" if which == "a" else "external vision"
    print(f"Starting Path {which.upper()} ({label})...\n", flush=True)

    try:
        require_ros_message(cfg, cfg.odom_topic, "body-frame cuVSLAM odometry", 2.0)
    except RuntimeError:
        if not find_proc(cfg, "cuvslam"):
            cmd_vio(cfg)
        else:
            raise RuntimeError("cuVSLAM is running but body-frame odometry is not ready")

    if which == "b":
        if find_proc(cfg, "gps-bridge"):
            raise RuntimeError("Path A is already running. Stop it before starting Path B.")
        if not find_proc(cfg, "ev-bridge"):
            cmd_ev_bridge(cfg)
        print("\nPath B started (local EV → Cube).")
    else:
        if find_proc(cfg, "ev-bridge"):
            raise RuntimeError("Path B is already running. Stop it before starting Path A.")
        if not find_proc(cfg, "gps-bridge"):
            cmd_gps(cfg)
        # cmd_gps reports success only after receiving a PX4 heartbeat.


def cmd_logs(cfg: Config, name: str, lines: int = 40, full: bool = False) -> None:
    proc = find_proc(cfg, name)
    log_file = Path(proc["logFile"]) if proc else (cfg.log_dir / f"{name}.log")
    if not log_file.exists():
        raise RuntimeError(f"No log for '{name}' at {log_file}")
    raw = log_file.read_text(errors="replace").strip()
    print(log_file)
    if full:
        print(raw)
        return

    # Interactive view: show only the latest launch and collapse Python
    # tracebacks to the useful final error. The raw file remains untouched.
    marker = "===== "
    latest = marker + raw.rsplit(marker, 1)[-1] if marker in raw else raw
    session = latest.splitlines()
    errors = [
        line
        for line in session
        if line.startswith(("RuntimeError:", "FileNotFoundError:", "PermissionError:"))
        or "SerialException:" in line
        or line.startswith("[ros2run]:")
    ]
    if "Traceback (most recent call last):" in latest or errors:
        print("Result: FAILED")
        preferred = next((line for line in reversed(errors) if "SerialException:" in line), None)
        if preferred is None:
            preferred = next((line for line in reversed(errors) if not line.startswith("[ros2run]:")), None)
        print(f"Error: {preferred or 'process exited during startup'}")
        return

    print("Result: latest launch output")
    print("\n".join(session[-lines:]))


def cmd_stop(cfg: Config, name: Optional[str] = None) -> None:
    stopped = stop_managed(cfg, name)
    if not stopped:
        print(f"Nothing named '{name}' to stop." if name else "Nothing to stop.")
        return
    print(f"Stopped: {', '.join(stopped)}")


# ---------------------------------------------------------------------------
# Interactive TUI (default)
# ---------------------------------------------------------------------------


def configure_link(cfg: Config) -> Config:
    _clear()
    print("Configure FC link (Jetson ↔ Cube)\n")
    choice = select_menu(
        "MAVLink transport",
        [
            MenuItem("UART / serial (Jetson FTDI → TELEM2)", "uart", "e.g. /dev/ttyUSB0:921600"),
            MenuItem("UDP out", "udp", "e.g. udpout:127.0.0.1:14540"),
            MenuItem("Keep current", "keep", cfg.mavlink_url),
            MenuItem("Back", "back"),
        ],
        subtitle=f"current: {cfg.mavlink_url}",
    )
    if choice in (None, "back", "keep"):
        return cfg
    if choice == "uart":
        devices = (
            sorted(Path("/dev").glob("ttyTHS*"))
            + sorted(Path("/dev").glob("ttyUSB*"))
            + sorted(Path("/dev").glob("ttyACM*"))
        )
        items = [MenuItem(str(d), str(d), "detected") for d in devices] or [
            MenuItem("/dev/ttyTHS1", "/dev/ttyTHS1", "common Orin UART"),
            MenuItem("/dev/ttyTHS0", "/dev/ttyTHS0"),
            MenuItem("/dev/ttyUSB0", "/dev/ttyUSB0"),
        ]
        items.append(MenuItem("Custom path…", "custom"))
        items.append(MenuItem("Back", "back"))
        dev = select_menu("Serial device", items)
        if dev in (None, "back"):
            return cfg
        if dev == "custom":
            dev = prompt_line("Device path", "/dev/ttyUSB0")
            if not dev:
                return cfg
        baud = prompt_line("Baud", "921600") or "921600"
        candidate = f"{dev}:{baud}"
        error = validate_mavlink_url(candidate)
        if error:
            print(f"Invalid link: {error}")
            pause()
            return cfg
        cfg = replace(cfg, mavlink_url=candidate)
    elif choice == "udp":
        url = prompt_line("UDP URL", "udpout:127.0.0.1:14540") or "udpout:127.0.0.1:14540"
        error = validate_mavlink_url(url)
        if error:
            print(f"Invalid link: {error}")
            pause()
            return cfg
        cfg = replace(cfg, mavlink_url=url)
    save_config(cfg)
    print(f"Saved link: {cfg.mavlink_url}")
    pause()
    return cfg


def configure_home(cfg: Config) -> Config:
    _clear()
    print("Configure home / spoof origin (lat lon alt)\n")
    print(f"Current: {cfg.home_lat}, {cfg.home_lon}, {cfg.home_alt}")
    lat = prompt_line("Latitude deg", str(cfg.home_lat))
    lon = prompt_line("Longitude deg", str(cfg.home_lon))
    alt = prompt_line("Altitude MSL m", str(cfg.home_alt))
    if lat is None:
        return cfg
    try:
        lat_value = float(lat)
        lon_value = float(lon or cfg.home_lon)
        alt_value = float(alt or cfg.home_alt)
        if not all(math.isfinite(value) for value in (lat_value, lon_value, alt_value)):
            raise ValueError("latitude, longitude, and altitude must be finite numbers")
        if not -90.0 <= lat_value <= 90.0:
            raise ValueError("latitude must be between -90 and 90")
        if not -180.0 <= lon_value <= 180.0:
            raise ValueError("longitude must be between -180 and 180")
        cfg = replace(
            cfg,
            home_lat=lat_value,
            home_lon=lon_value,
            home_alt=alt_value,
        )
    except ValueError as exc:
        print(f"Invalid home: {exc}")
        pause()
        return cfg
    save_config(cfg)
    print("Saved home.")
    pause()
    return cfg


def configure_heading(cfg: Config) -> Config:
    _clear()
    print("Configure independent heading alignment\n")
    print("Compass mode uses Cube magnetometer + roll/pitch, but never PX4 yaw.")
    print("Angles are degrees; declination is east-positive, west-negative.\n")
    source = prompt_line("Source (compass/manual)", cfg.heading_source)
    if source is None:
        return cfg
    source = (source or cfg.heading_source).strip().lower()
    if source not in ("compass", "manual"):
        print("Invalid source; use compass or manual.")
        pause()
        return cfg
    declination_source = prompt_line(
        "Declination source (table/manual)", cfg.mag_declination_source
    )
    if declination_source is None:
        return cfg
    declination_source = (declination_source or cfg.mag_declination_source).lower()
    if declination_source not in ("table", "manual"):
        print("Invalid declination source; use table or manual.")
        pause()
        return cfg
    declination = str(cfg.mag_declination_deg)
    if declination_source == "manual":
        declination = prompt_line("Magnetic declination", declination)
    mounting = prompt_line(
        "cuVSLAM child-to-body yaw", str(cfg.child_to_body_yaw_deg)
    )
    manual = str(cfg.manual_heading_deg)
    if source == "manual":
        manual = prompt_line("Body true heading at alignment", manual)
    try:
        values = tuple(float(x) for x in (declination, mounting, manual))
        if not all(math.isfinite(x) for x in values):
            raise ValueError("angles must be finite")
        cfg = replace(
            cfg,
            heading_source=source,
            mag_declination_source=declination_source,
            mag_declination_deg=values[0],
            child_to_body_yaw_deg=values[1],
            manual_heading_deg=values[2],
        )
    except (TypeError, ValueError) as exc:
        print(f"Invalid heading configuration: {exc}")
        pause()
        return cfg
    save_config(cfg)
    print("Saved heading alignment configuration.")
    pause()
    return cfg


def pick_log(cfg: Config) -> None:
    ensure_dirs(cfg)
    # Only show logs that exist for components used by the hardware paths.
    names = [
        name
        for name in ("gps-bridge", "ev-bridge")
        if (cfg.log_dir / f"{name}.log").is_file()
    ]
    if not names:
        _clear()
        print("No hardware bridge logs yet.")
        pause()
        return

    items = []
    for name in names:
        log_path = cfg.log_dir / f"{name}.log"
        size_kib = log_path.stat().st_size / 1024.0
        updated = time.strftime("%Y-%m-%d %H:%M", time.localtime(log_path.stat().st_mtime))
        items.append(MenuItem(name, name, f"{size_kib:.1f} KiB · updated {updated}"))
    items.append(MenuItem("Back", "back"))
    choice = select_menu("View logs", items)
    if choice in (None, "back"):
        return
    try:
        cmd_logs(cfg, choice, 60)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}")
    pause()


def interactive_main(cfg: Config) -> int:
    while True:
        running = prune_state(cfg)
        run_note = f"{len(running)} running" if running else "idle"
        choice = select_menu(
            "Main Menu",
            [
                MenuItem("", "h1", disabled=True),
                MenuItem("Flight paths", "hdr1", disabled=True),
                MenuItem(
                    "Path A — GPS spoof → live VIO (global)",
                    "run-a",
                    "Recommended internship-style HIL_GPS path",
                ),
                MenuItem(
                    "Path B — External vision (local)",
                    "run-b",
                    "body-frame VIO → MAVROS → PX4 ODOMETRY",
                ),
                MenuItem("", "h2", disabled=True),
                MenuItem("Setup", "hdr2", disabled=True),
                MenuItem(
                    "FC link (UART / UDP)",
                    "link",
                    cfg.mavlink_url,
                ),
                MenuItem(
                    "Home lat/lon/alt",
                    "home",
                    f"{cfg.home_lat}, {cfg.home_lon}, {cfg.home_alt}",
                ),
                MenuItem(
                    "Heading alignment",
                    "heading",
                    f"{cfg.heading_source}; declination={declination_label(cfg)}, "
                    f"child→body={cfg.child_to_body_yaw_deg}°",
                ),
                MenuItem("", "h3", disabled=True),
                MenuItem("Ops", "hdr3", disabled=True),
                MenuItem(
                    "Doctor (Checks ROS, bridge, FC link, and required files)",
                    "doctor",
                ),
                MenuItem(
                    f"Status (Shows running bridge processes; {run_note})",
                    "status",
                ),
                MenuItem(
                    "Logs (Shows recent output/errors from a process)",
                    "logs",
                ),
                MenuItem("Stop all", "stop"),
                MenuItem("", "h4", disabled=True),
                MenuItem("Exit", "exit"),
            ],
            subtitle=f"Hardware: Jetson + Cube · link={cfg.mavlink_url}",
        )

        if choice in (None, "exit"):
            _clear()
            return 0

        try:
            if choice == "run-a":
                _clear()
                cmd_run(cfg, "a")
                pause()
            elif choice == "run-b":
                _clear()
                cmd_run(cfg, "b")
                pause()
            elif choice == "link":
                cfg = configure_link(cfg)
            elif choice == "home":
                cfg = configure_home(cfg)
            elif choice == "heading":
                cfg = configure_heading(cfg)
            elif choice == "doctor":
                _clear()
                cmd_doctor(cfg)
                pause()
            elif choice == "status":
                _clear()
                cmd_status(cfg)
                pause()
            elif choice == "logs":
                pick_log(cfg)
            elif choice == "stop":
                _clear()
                cmd_stop(cfg)
                pause()
        except KeyboardInterrupt:
            print("\nCancelled.")
            pause()
        except Exception as exc:  # noqa: BLE001
            print(f"\nError: {exc}")
            pause()


# ---------------------------------------------------------------------------
# CLI argparse (still available for scripting)
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vio-test",
        description="VIO/PX4 hardware CLI for Jetson + Cube. Run with no args for menu.",
    )
    p.add_argument("--version", action="version", version=VERSION)
    p.add_argument("--mavlink-url")
    p.add_argument("--odom-topic")
    p.add_argument("--home-lat", type=float)
    p.add_argument("--home-lon", type=float)
    p.add_argument("--home-alt", type=float)
    p.add_argument("--no-tui", action="store_true", help="Do not open menu; print help")

    sub = p.add_subparsers(dest="command")
    sub.add_parser("tui", help="Open interactive menu (default)")
    sub.add_parser("doctor")
    sub.add_parser("status")
    sub.add_parser("ev-bridge")
    sub.add_parser("vio")
    sub.add_parser("gps")

    run_p = sub.add_parser("run")
    run_p.add_argument("path")

    logs_p = sub.add_parser("logs")
    logs_p.add_argument("name")
    logs_p.add_argument("-n", "--lines", type=int, default=40)
    logs_p.add_argument("--full", action="store_true", help="Show the complete raw log")

    stop_p = sub.add_parser("stop")
    stop_p.add_argument("name", nargs="?")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args)

    # Default: numbered interactive menu (reliable over AnyDesk/SSH).
    if not args.command or args.command == "tui":
        if getattr(args, "no_tui", False):
            parser.print_help()
            return 0
        return interactive_main(cfg)

    try:
        if args.command == "doctor":
            return cmd_doctor(cfg)
        if args.command == "status":
            cmd_status(cfg)
        elif args.command == "ev-bridge":
            cmd_ev_bridge(cfg)
        elif args.command == "vio":
            cmd_vio(cfg)
        elif args.command == "gps":
            cmd_gps(cfg)
        elif args.command == "run":
            cmd_run(cfg, args.path)
        elif args.command == "logs":
            cmd_logs(cfg, args.name, args.lines, full=args.full)
        elif args.command == "stop":
            cmd_stop(cfg, args.name)
        else:
            parser.print_help()
            return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
