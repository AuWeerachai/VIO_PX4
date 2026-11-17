#!/usr/bin/env python3
"""vio-launch — interactive hardware CLI for Jetson VIO ↔ PX4 Cube.

Default (no args): numbered interactive menu that works reliably over AnyDesk.
Also supports direct subcommands: doctor, run a, gps, stop, ...
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import math
import os
import select
import shutil
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from functools import wraps
from pathlib import Path
from typing import Optional

VERSION = "0.1.0"
_CLI_FILE = Path(__file__).resolve()
_LOADED_CLI_HASH = hashlib.sha256(_CLI_FILE.read_bytes()).digest()


class StaleLauncherError(RuntimeError):
    pass


def require_current_cli() -> None:
    """Do not launch or save from a Python process predating a deployment."""
    try:
        current = hashlib.sha256(_CLI_FILE.read_bytes()).digest()
    except OSError as exc:
        raise StaleLauncherError('Cannot verify this launcher. Reopen ~/vio-launch.') from exc
    if current != _LOADED_CLI_HASH:
        raise StaleLauncherError(
            'vio-launch was updated while this menu was open. '
            'Leave CLI (keep processes running), then reopen ~/vio-launch. '
            'No launch or settings save was performed.')


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
    return Path.home() / ".config" / "vio-launch" / "config.json"


def legacy_user_config_path() -> Path:
    """Read pre-rename settings once so existing installations retain values."""
    return Path.home() / ".config" / "vio-test" / "config.json"


@dataclass
class Config:
    vio_px4_dir: Path = field(
        default_factory=lambda: Path.home() / "workspaces/VIO_PX4_NEW_PIPELINE"
    )
    ros_setup: Path = field(default_factory=lambda: Path("/opt/ros/humble/setup.bash"))
    isaac_ros_dir: Path = field(
        default_factory=lambda: Path.home() / "workspaces/isaac_ros-dev"
    )
    px4_msgs_setup: Optional[Path] = None
    home_lat: float = 40.4433
    home_lon: float = -79.9436
    home_alt: float = 200.0
    # Jetson↔Cube is typically UART, but a routed UDP MAVLink link also works.
    mavlink_url: str = "/dev/ttyUSB0:921600"
    mavlink_sysid: int = 1
    odom_topic: str = "/visual_slam/tracking/odometry"
    heading_source: str = "px4"
    manual_heading_deg: float = 0.0
    external_yaw_enabled: bool = False
    external_yaw_noise_deg: float = 1.0
    external_yaw_rate_hz: float = 30.0
    external_yaw_max_age_s: float = 0.2
    external_yaw_startup_tolerance_deg: float = 10.0
    mag_declination_source: str = "table"
    mag_declination_deg: float = 0.0
    # SDK serial from rs-enumerate-devices, NOT the USB ASIC identifier.
    camera_serial: str = ""
    # drone_link -> camera_link extrinsic in ROS FLU coordinates.
    camera_x_m: float = 0.0
    camera_y_m: float = 0.0
    camera_z_m: float = 0.0
    camera_roll_deg: float = 0.0
    camera_pitch_deg: float = 0.0
    camera_yaw_deg: float = 0.0
    # Local-continuity safety gate. These defaults match the bridge defaults.
    gate_max_gap_s: float = 1.0
    gate_recovery_samples: int = 10
    gate_confirmation_samples: int = 3
    gate_max_speed_m_s: float = 10.0
    gate_max_acceleration_m_s2: float = 5.0
    gate_max_yaw_rate_deg_s: float = 360.0
    inertial_max_duration_s: float = 2.0  # legacy config, ignored (no time cutoff)
    inertial_max_message_age_s: float = 0.25
    inertial_max_message_gap_s: float = 0.25
    inertial_reorder_tolerance_s: float = 0.05
    inertial_max_position_uncertainty_m: float = 3.0
    recovery_velocity_agreement_m_s: float = 2.0
    recovery_velocity_agreement_samples: int = 3
    recovery_yaw_agreement_deg: float = 20.0
    recovery_yaw_agreement_samples: int = 5
    recovery_reset_rotation_budget_deg: float = 2.0  # legacy config, inactive
    recovery_accuracy_tighten_s: float = 5.0
    rviz_enabled: bool = False
    state_dir: Path = field(default_factory=lambda: Path.home() / ".local/state/vio-launch")

    @property
    def log_dir(self) -> Path:
        return self.state_dir / "logs"

    @property
    def state_file(self) -> Path:
        return self.state_dir / "state.json"

def default_config() -> Config:
    home = Path.home()
    desktop = home / "Desktop"
    repository_root = Path(__file__).resolve().parents[2]
    return Config(
        vio_px4_dir=first_existing([
            str(repository_root),
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
    if not path.exists() and legacy_user_config_path().exists():
        path = legacy_user_config_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_config(cfg: Config) -> None:
    require_current_cli()
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
        "external_yaw_enabled": cfg.external_yaw_enabled,
        "external_yaw_noise_deg": cfg.external_yaw_noise_deg,
        "external_yaw_rate_hz": cfg.external_yaw_rate_hz,
        "external_yaw_max_age_s": cfg.external_yaw_max_age_s,
        "external_yaw_startup_tolerance_deg": cfg.external_yaw_startup_tolerance_deg,
        "mag_declination_source": cfg.mag_declination_source,
        "mag_declination_deg": cfg.mag_declination_deg,
        "camera_serial": cfg.camera_serial,
        "camera_x_m": cfg.camera_x_m,
        "camera_y_m": cfg.camera_y_m,
        "camera_z_m": cfg.camera_z_m,
        "camera_roll_deg": cfg.camera_roll_deg,
        "camera_pitch_deg": cfg.camera_pitch_deg,
        "camera_yaw_deg": cfg.camera_yaw_deg,
        "gate_max_gap_s": cfg.gate_max_gap_s,
        "gate_recovery_samples": cfg.gate_recovery_samples,
        "gate_confirmation_samples": cfg.gate_confirmation_samples,
        "gate_max_speed_m_s": cfg.gate_max_speed_m_s,
        "gate_max_acceleration_m_s2": cfg.gate_max_acceleration_m_s2,
        "gate_max_yaw_rate_deg_s": cfg.gate_max_yaw_rate_deg_s,
        "inertial_max_message_age_s": cfg.inertial_max_message_age_s,
        "inertial_max_message_gap_s": cfg.inertial_max_message_gap_s,
        "inertial_reorder_tolerance_s": cfg.inertial_reorder_tolerance_s,
        "inertial_max_position_uncertainty_m": cfg.inertial_max_position_uncertainty_m,
        "recovery_velocity_agreement_m_s": cfg.recovery_velocity_agreement_m_s,
        "recovery_velocity_agreement_samples": cfg.recovery_velocity_agreement_samples,
        "recovery_yaw_agreement_deg": cfg.recovery_yaw_agreement_deg,
        "recovery_yaw_agreement_samples": cfg.recovery_yaw_agreement_samples,
        "recovery_accuracy_tighten_s": cfg.recovery_accuracy_tighten_s,
        "rviz_enabled": cfg.rviz_enabled,
        "vio_px4_dir": str(cfg.vio_px4_dir),
        "ros_setup": str(cfg.ros_setup),
        "isaac_ros_dir": str(cfg.isaac_ros_dir),
    }
    atomic_write_json(path, payload)


def heading_label(cfg: Config) -> str:
    if cfg.heading_source in ("px4", "compass"):
        return "PX4 estimated heading (declination handled by PX4)"
    return f"manual true heading {cfg.manual_heading_deg:g}°"


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
        "external_yaw_enabled",
        "external_yaw_noise_deg",
        "external_yaw_rate_hz",
        "external_yaw_max_age_s",
        "external_yaw_startup_tolerance_deg",
        "mag_declination_source",
        "mag_declination_deg",
        "camera_serial",
        "camera_x_m",
        "camera_y_m",
        "camera_z_m",
        "camera_roll_deg",
        "camera_pitch_deg",
        "camera_yaw_deg",
        "gate_max_gap_s",
        "gate_recovery_samples",
        "gate_confirmation_samples",
        "gate_max_speed_m_s",
        "gate_max_acceleration_m_s2",
        "gate_max_yaw_rate_deg_s",
        "inertial_max_duration_s",
        "inertial_max_message_age_s",
        "inertial_max_message_gap_s",
        "inertial_reorder_tolerance_s",
        "inertial_max_position_uncertainty_m",
        "recovery_velocity_agreement_m_s",
        "recovery_velocity_agreement_samples",
        "recovery_yaw_agreement_deg",
        "recovery_yaw_agreement_samples",
        "recovery_reset_rotation_budget_deg",
        "recovery_accuracy_tighten_s",
        "rviz_enabled",
    ):
        if key in data and data[key] is not None:
            updates[key] = data[key]
    for key in ("vio_px4_dir", "ros_setup", "isaac_ros_dir"):
        if key in data and data[key]:
            updates[key] = expand(str(data[key]))
    if str(updates.get("heading_source", "")).strip().lower() == "compass":
        updates["heading_source"] = "px4"
    return replace(cfg, **updates) if updates else cfg


def load_config(ns: argparse.Namespace | None = None) -> Config:
    cfg = default_config()
    saved = load_saved_overrides()
    # Legacy child_to_body_yaw_deg is intentionally not loaded: camera mounting
    # belongs entirely to the cuVSLAM extrinsics, and input is drone_link.
    cfg = apply_dict(cfg, saved)
    # Backward compatibility with the former pitch-only menu setting.
    if "camera_pitch_deg" not in saved and "camera_tilt_deg" in saved:
        try:
            cfg = replace(cfg, camera_pitch_deg=float(saved["camera_tilt_deg"]))
        except (TypeError, ValueError):
            pass
    if "mavlink_url" not in saved:
        cfg = replace(cfg, mavlink_url="/dev/ttyUSB0:921600")
    # Migrate configurations created before camera-to-body odometry became a
    # required safety boundary for the tilted D456 installation.
    if saved.get("odom_topic") == "/vio/body/odometry":
        cfg = replace(cfg, odom_topic="/visual_slam/tracking/odometry")
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
        print(f"  vio-launch  v{VERSION}")
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


_PROCESS_LOCKS = threading.local()
_STARTUPS = threading.local()


class StartupCancelled(RuntimeError):
    pass


class ProcessOperationBusy(RuntimeError):
    pass


def startup_generation(cfg: Config):
    path = cfg.state_dir / "startup-generation.json"
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict) or not isinstance(value.get("generation"), str):
            raise ValueError("invalid cancellation generation")
        return value["generation"]
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot verify startup cancellation state: {path}: {exc}") from exc


def check_startup_current(cfg: Config) -> None:
    key = (os.getpid(), cfg.state_dir.resolve())
    active = getattr(_STARTUPS, "active", {})
    if key in active and startup_generation(cfg) != active[key]:
        raise StartupCancelled("Startup cancelled by Stop all. No further launch actions performed.")


@contextmanager
def startup_operation(cfg: Config):
    """One starter at a time, without excluding Stop all during readiness waits."""
    key = (os.getpid(), cfg.state_dir.resolve())
    if not hasattr(_STARTUPS, "active"):
        _STARTUPS.active = {}
    if key in _STARTUPS.active:
        check_startup_current(cfg)
        yield
        return
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    with (cfg.state_dir / "startup.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ProcessOperationBusy(
                "Another vio-launch startup is in progress. Stop all can cancel it.") from exc
        try:
            with process_operation(cfg):
                generation = uuid.uuid4().hex
                atomic_write_json(cfg.state_dir / "startup-generation.json",
                                  {"generation": generation})
                _STARTUPS.active[key] = generation
            yield
        finally:
            _STARTUPS.active.pop(key, None)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def serialized_startup(function):
    @wraps(function)
    def guarded(cfg, *args, **kwargs):
        with startup_operation(cfg):
            return function(cfg, *args, **kwargs)
    return guarded


@contextmanager
def process_operation(cfg: Config):
    """Serialize complete process transactions across CLI instances.

    Reentrant only within the owning thread/process. The stable lock file must
    never be unlinked: replacing it would let another CLI lock a different inode.
    Nonblocking acquisition reports contention instead of hanging a terminal.
    The descriptor is not inherited by executed ROS/Docker processes.
    """
    key = (os.getpid(), cfg.state_dir.resolve())
    check_startup_current(cfg)
    if not hasattr(_PROCESS_LOCKS, "held"):
        _PROCESS_LOCKS.held = set()
    if key in _PROCESS_LOCKS.held:
        yield
        return
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    with (cfg.state_dir / "process-manager.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ProcessOperationBusy(
                "Another vio-launch process operation is in progress. "
                "No changes made by this request; retry when it finishes.") from exc
        _PROCESS_LOCKS.held.add(key)
        try:
            check_startup_current(cfg)
            yield
        finally:
            _PROCESS_LOCKS.held.remove(key)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def serialized_process_operation(function):
    @wraps(function)
    def locked(cfg, *args, **kwargs):
        with process_operation(cfg):
            return function(cfg, *args, **kwargs)
    return locked


def ensure_dirs(cfg: Config) -> None:
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)


@serialized_process_operation
def read_state(cfg: Config) -> dict:
    ensure_dirs(cfg)
    try:
        state = json.loads(cfg.state_file.read_text())
        if not isinstance(state, dict) or not isinstance(state.get("procs"), list):
            raise ValueError("expected an object containing a procs list")
        names, pids = set(), set()
        for proc in state["procs"]:
            if (not isinstance(proc, dict)
                    or not isinstance(proc.get("name"), str) or not proc["name"]
                    or type(proc.get("pid")) is not int or proc["pid"] <= 1
                    or type(proc.get("startTicks")) is not int or proc["startTicks"] < 0
                    or ("bootId" in proc and
                        (not isinstance(proc["bootId"], str) or not proc["bootId"]))):
                raise ValueError("invalid process ownership record")
            if proc["name"] in names or proc["pid"] in pids:
                raise ValueError("duplicate process ownership records")
            names.add(proc["name"])
            pids.add(proc["pid"])
        return state
    except FileNotFoundError:
        # A genuinely new installation has no state yet. Unreadable, empty or
        # malformed existing files are NOT evidence that publishers are stopped.
        return {"procs": []}
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError(
            f"Cannot verify managed process state in {cfg.state_file}: {exc}. "
            "Processes may still be running. Start/stop/restart is blocked; "
            "repair the state file or its permissions before retrying. "
            "Do not delete it to bypass this check.") from exc


@serialized_process_operation
def write_state(cfg: Config, state: dict) -> None:
    ensure_dirs(cfg)
    atomic_write_json(cfg.state_file, state)


def process_identity(pid: int) -> Optional[tuple[str, int, int, int]]:
    """State, process group, session, start ticks. Zombies own no running work."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[0], int(fields[2]), int(fields[3]), int(fields[19])
    except (FileNotFoundError, ProcessLookupError):
        return None
    # Permission/I/O failures must not be mistaken for a stopped publisher.


def process_start_ticks(pid: int) -> Optional[int]:
    identity = process_identity(pid)
    return None if identity is None or identity[0] == "Z" else identity[3]


def managed_group(proc: dict) -> list[int]:
    """Find live members of this launcher's private session, even after bash exits.

    A running child retains the group/session IDs. Linux cannot reuse those IDs
    for a new process until the old group is gone. Start ticks and boot identity
    prevent signalling an unrelated process from a stale saved record.
    """
    pid = int(proc["pid"])
    ticks = proc.get("startTicks")
    if pid <= 1 or pid == os.getpgrp():
        raise RuntimeError("Refusing to manage a non-private process group")
    if proc.get("bootId") and proc["bootId"] != Path(
            "/proc/sys/kernel/random/boot_id").read_text().strip():
        return []
    leader = process_identity(pid)
    if leader is not None and ticks is not None and leader[3] != ticks:
        return []  # PID reused: this is no longer our process.
    if ticks is None:
        raise RuntimeError(f"Cannot verify ownership of {proc.get('name', pid)}; "
                           "missing saved process start time")
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        identity = process_identity(int(entry.name))
        if identity is None or identity[0] == "Z" or identity[1] != pid:
            continue
        if identity[2] != pid or identity[3] < ticks:
            raise RuntimeError(f"Cannot verify ownership of process group {pid}")
        members.append(int(entry.name))
    return members


def managed_alive(proc: dict) -> bool:
    return bool(managed_group(proc))


def managed_leader_alive(proc: dict) -> bool:
    """A surviving orphan is a stop target, not a successful bridge startup."""
    ticks = process_start_ticks(int(proc["pid"]))
    return ticks is not None and ticks == proc.get("startTicks")


def same_managed_process(first: dict, second: dict) -> bool:
    return all(first.get(key) == second.get(key)
               for key in ("name", "pid", "startTicks", "bootId"))


def terminate_managed(proc: dict) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if not managed_alive(proc):
            return
        try:
            os.killpg(int(proc["pid"]), sig)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 2.0
        while managed_alive(proc) and time.monotonic() < deadline:
            time.sleep(0.05)
    if managed_alive(proc):
        raise RuntimeError(f"{proc['name']} process group has not stopped; "
                           "state retained and replacement startup blocked")


@serialized_process_operation
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


@serialized_process_operation
def register_proc(cfg: Config, proc: dict) -> None:
    state = read_state(cfg)
    procs = [p for p in state["procs"] if managed_alive(p)]
    if any(p["name"] == proc["name"] for p in procs):
        raise RuntimeError(f"{proc['name']} already running; refusing to overwrite ownership")
    procs.append(proc)
    write_state(cfg, {"procs": procs})


@serialized_process_operation
def spawn_managed(
    cfg: Config,
    name: str,
    bash_command: str,
) -> tuple[subprocess.Popen, dict]:
    require_current_cli()
    existing = find_proc(cfg, name)
    if existing:
        raise RuntimeError(f"{name} already running (pid {existing['pid']}). Stop it first.")
    ensure_dirs(cfg)
    log_file = cfg.log_dir / f"{name}.log"
    runner = _CLI_FILE.with_name("managed_process.py")
    proc = subprocess.Popen(
        [sys.executable, "-u", str(runner), "--log", str(log_file), "--name", name,
         "--", "bash", "-lc", bash_command],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        # No inherited terminal/SSH pipes: leaving the CLI must return even
        # though this supervisor and its publisher continue in their session.
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        env=os.environ.copy(),
    )
    identity = process_identity(proc.pid)
    managed = {
        "name": name,
        "pid": proc.pid,
        "startTicks": None if identity is None else identity[3],
        "bootId": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "logFile": str(log_file),
        "startedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "command": bash_command,
    }
    try:
        # Save ownership before waiting: a shell can exit leaving children.
        register_proc(cfg, managed)
    except Exception:
        # Never leave an untracked publisher running if state persistence fails.
        terminate_managed(managed)
        raise
    return proc, managed


def start_managed(cfg: Config, name: str, bash_command: str,
                  startup_check_s: float = 0.75) -> dict:
    proc, managed = spawn_managed(cfg, name, bash_command)
    # Ownership is already saved. Stop all may cancel/terminate this process
    # during the check; do not keep the state lock while waiting for readiness.
    deadline = time.monotonic() + startup_check_s
    while proc.poll() is None and time.monotonic() < deadline:
        check_startup_current(cfg)
        time.sleep(min(.05, max(0., deadline-time.monotonic())))
    with process_operation(cfg):
        exit_code = proc.poll()
        if exit_code is not None or not managed_leader_alive(managed):
            stop_managed(cfg, name, expected_proc=managed)
            log_file = Path(managed["logFile"])
            try:
                recent = log_file.read_text(errors="replace").strip().splitlines()[-12:]
            except OSError as exc:
                recent = [f"Log unavailable: {exc}"]
            raise RuntimeError(f"{name} failed to start (exit code {exit_code}). Recent log:\n" + "\n".join(recent))
        return managed


@serialized_process_operation
def stop_managed(cfg: Config, name: Optional[str] = None, *,
                 expected_proc: Optional[dict] = None) -> list[str]:
    procs = prune_state(cfg)
    targets = [p for p in procs if name is None or p["name"] == name]
    if expected_proc is not None and any(
            not same_managed_process(p, expected_proc) for p in targets):
        raise RuntimeError(
            "The original bridge was replaced by another launch. "
            "The replacement process was left running.")
    stopped: list[str] = []
    for p in targets:
        terminate_managed(p)
        stopped.append(p["name"])
    if targets:
        stopped_names = {p["name"] for p in targets}
        remaining = [p for p in procs if p.get("name") not in stopped_names]
        try:
            write_state(cfg, {"procs": remaining})
        except OSError as exc:
            if exc.errno != errno.ENOSPC:
                raise
            # The groups are verified stopped. Retain their old ownership
            # records for a later prune, but do not block camera teardown.
            print("Warning: disk is full; stopped-process records retained until space is available.",
                  file=sys.stderr)
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
    raise RuntimeError(f"Unknown path '{value}'. Only a|gps is supported; Path B was removed.")


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


ISAAC_CONTAINER = "isaac_ros_dev-aarch64-container"


def isaac_container_running() -> bool:
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", ISAAC_CONTAINER],
        capture_output=True, text=True, check=False, timeout=5,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def ensure_isaac_container(cfg: Config, timeout_s: float = 600.0) -> None:
    if isaac_container_running():
        print(f"Isaac container already running: {ISAAC_CONTAINER}")
        return
    run_dev = cfg.isaac_ros_dir / "src/isaac_ros_common/scripts/run_dev.sh"
    if not run_dev.exists():
        raise RuntimeError(f"Missing NVIDIA container launcher: {run_dev}")
    if not shutil.which("gnome-terminal"):
        raise RuntimeError("gnome-terminal is required to host NVIDIA's interactive container")
    terminal_env = os.environ.copy()
    if not terminal_env.get("DISPLAY"):
        # An SSH shell is not attached to the graphical login, but the Jetson's
        # local GNOME session is. Explicitly target that session so the window
        # is visible through AnyDesk.
        uid = os.getuid()
        xauthority = Path(f"/run/user/{uid}/gdm/Xauthority")
        session_bus = Path(f"/run/user/{uid}/bus")
        if not xauthority.exists() or not session_bus.exists():
            raise RuntimeError(
                "No active Jetson graphical session was found. Log into the "
                "Jetson desktop/AnyDesk first, then retry."
            )
        terminal_env["DISPLAY"] = ":0"
        terminal_env["XAUTHORITY"] = str(xauthority)
        terminal_env["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path={session_bus}"
        terminal_env["XDG_RUNTIME_DIR"] = f"/run/user/{uid}"
    # A disconnected AnyDesk client does not make desktop windows disappear.
    # Remove stale windows from earlier failed/retried launches before opening
    # one new, uniquely titled run_dev terminal.
    if shutil.which("xdotool"):
        for stale_title in (
            "Isaac ROS run_dev.sh",
            "vio-launch visibility probe",
            "vio-test visibility probe",
        ):
            found = subprocess.run(
                ["xdotool", "search", "--name", stale_title],
                capture_output=True, text=True, env=terminal_env, check=False,
            )
            for window_id in found.stdout.split():
                subprocess.run(
                    ["xdotool", "windowclose", window_id], env=terminal_env,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    check=False,
                )
    command = (
        f"export ISAAC_ROS_WS={shlex.quote(str(cfg.isaac_ros_dir))}; "
        "printf 'teamc@teamc-desktop:~/workspaces$ cd ${ISAAC_ROS_WS}/src/isaac_ros_common && \\\n   ./scripts/run_dev.sh\\n'; "
        f"cd {shlex.quote(str(run_dev.parent))} && "
        "./run_dev.sh; "
        "rc=$?; echo \"Isaac run_dev.sh exited (code $rc).\"; "
        "echo 'This diagnostic shell remains open; type exit to close it.'; exec bash"
    )
    subprocess.Popen(
        ["gnome-terminal", "--window",
         "--title", "Isaac ROS run_dev.sh", "--",
         "bash", "-lc", command],
        start_new_session=True,
        env=terminal_env,
    )
    print(
        f"Opened 'Isaac ROS run_dev.sh' on Jetson desktop "
        f"{terminal_env['DISPLAY']}; waiting for Docker readiness..."
    )
    deadline = time.monotonic() + timeout_s
    stable_since = None
    next_progress = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if isaac_container_running():
            stable_since = stable_since or time.monotonic()
            if time.monotonic() - stable_since >= 2.0:
                print(f"Isaac container ready: {ISAAC_CONTAINER}")
                print(
                    "run_dev.sh is visible in the separate 'Isaac ROS run_dev.sh' window; "
                    "vio-launch is continuing here.", flush=True
                )
                return
        else:
            stable_since = None
        if time.monotonic() >= next_progress:
            print("Still waiting; follow build/startup output in the run_dev.sh window...", flush=True)
            next_progress = time.monotonic() + 15.0
        time.sleep(1.0)
    raise RuntimeError(
        f"Isaac container did not become ready within {timeout_s:.0f}s. "
        "Inspect the 'Isaac ROS Container' terminal tab."
    )


def require_ros_message(
    cfg: Config,
    topic: str,
    label: str,
    timeout_s: float,
    message_type: Optional[str] = None,
) -> None:
    type_arg = f" {shlex.quote(message_type)}" if message_type else ""
    command = (
        f"{ros_prefix(cfg, [cfg.vio_px4_dir / 'install/setup.bash'])} && "
        f"ros2 topic echo --no-daemon --once {shlex.quote(topic)}{type_arg}"
    )
    deadline = time.monotonic() + timeout_s
    last_error = ""
    # Do not gate this on `ros2 topic list --no-daemon`: short-lived DDS
    # discovery on the Jetson intermittently returns an empty list even while
    # type resolution and samples are available. A real received sample is the
    # only readiness criterion and also rejects stale endpoint names.
    while time.monotonic() < deadline:
        try:
            result = subprocess.run(
                ["bash", "-lc", command], capture_output=True, text=True,
                timeout=max(0.2, min(8.0, deadline - time.monotonic())),
            )
        except subprocess.TimeoutExpired:
            last_error = "topic discovered but no sample received"
            continue
        if result.returncode == 0:
            return
        details = (result.stderr or result.stdout).strip().splitlines()
        last_error = details[-1] if details else "ros2 topic echo failed"
        time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))

    raise RuntimeError(
        f"{label} is not ready: no current message arrived on {topic} within "
        f"{timeout_s:.0f} seconds ({last_error or 'topic not discovered'})"
    )


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
            "Isaac ROS workspace",
            (cfg.isaac_ros_dir / "src/isaac_ros_common/scripts/run_dev.sh").exists(),
            str(cfg.isaac_ros_dir),
        ),
    ]
    if cfg.mavlink_url.startswith("/dev/"):
        dev = cfg.mavlink_url.rsplit(":", 1)[0]
        exists = Path(dev).exists()
        checks.append(("Serial device", exists, dev))
        checks.append(
            ("Serial permission", exists and os.access(dev, os.R_OK | os.W_OK), dev)
        )

    print("vio-launch doctor\n")
    failed = 0
    for label, ok, detail in checks:
        print(f"{'[OK]' if ok else '[!!]'} {label}: {detail}")
        if not ok:
            failed += 1
    print(f"\nhorizontal origin: {cfg.home_lat}, {cfg.home_lon}")
    print(f"odom:     {cfg.odom_topic}")
    print(f"initial heading reference: {heading_label(cfg)}")
    print(f"checklist: {cfg.vio_px4_dir / 'bootstrap/docs/PX4_INTERFACE_CHECKLIST.md'}")
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
        print(f"  {p['name']}  {phase}  pid={p['pid']}  since={p['startedAt']}")
        print(f"    log: {p['logFile']}")


ISAAC_TMUX_VIO = "vio-launch-isaac-vio"
ISAAC_TMUX_RVIZ = "vio-launch-isaac-rviz"
LEGACY_ISAAC_TMUX_SESSIONS = ("vio-test-isaac-vio", "vio-test-isaac-rviz")


def jetson_desktop_env() -> dict[str, str]:
    env = os.environ.copy()
    if env.get("DISPLAY"):
        return env
    uid = os.getuid()
    xauthority = Path(f"/run/user/{uid}/gdm/Xauthority")
    session_bus = Path(f"/run/user/{uid}/bus")
    if not xauthority.exists() or not session_bus.exists():
        raise RuntimeError(
            "No active Jetson graphical session. Log into AnyDesk first, then retry."
        )
    env.update({
        "DISPLAY": ":0",
        "XAUTHORITY": str(xauthority),
        "DBUS_SESSION_BUS_ADDRESS": f"unix:path={session_bus}",
        "XDG_RUNTIME_DIR": f"/run/user/{uid}",
    })
    return env


def tmux_command(session: str, command: str) -> None:
    subprocess.run(
        ["tmux", "send-keys", "-t", session, "-l", command], check=True, timeout=5
    )
    subprocess.run(["tmux", "send-keys", "-t", session, "Enter"], check=True, timeout=5)


def startup_tmux_command(cfg: Config, session: str, command: str) -> None:
    with process_operation(cfg):
        tmux_command(session, command)


def tmux_session_exists(session: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", session],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def tmux_recent_output(session: str, lines: int = 80) -> str:
    result = subprocess.run(
        ["tmux", "capture-pane", "-pt", session, "-S", f"-{lines}"],
        capture_output=True,
        text=True,
    )
    return result.stdout if result.returncode == 0 else ""


def start_visible_isaac_session(cfg: Config, session: str, title: str) -> None:
    with process_operation(cfg):
        replacing_vio_owner = session == ISAAC_TMUX_VIO and tmux_session_exists(session)
        subprocess.run(["tmux", "kill-session", "-t", session], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
    if replacing_vio_owner:
        # run_dev.sh owns an auto-removed Docker container. Killing its terminal
        # begins asynchronous container teardown. Starting another run_dev.sh
        # too soon can see that dying container, try to attach, and then fail
        # with "No such container". Wait for teardown to finish first.
        deadline = time.monotonic() + 45.0
        while isaac_container_running() and time.monotonic() < deadline:
            check_startup_current(cfg)
            time.sleep(0.5)
        if isaac_container_running():
            raise RuntimeError(
                "Previous Isaac container did not stop after its VIO session closed. "
                "Use option 8, wait a few seconds, then retry."
            )
    with process_operation(cfg):
        env = jetson_desktop_env()
        subprocess.run(["tmux", "new-session", "-d", "-s", session], check=True, timeout=5)
        subprocess.Popen(
            ["gnome-terminal", "--window", "--title", title, "--",
             "tmux", "attach-session", "-t", session],
            env=env, start_new_session=True,
        )
        run_dev_dir = cfg.isaac_ros_dir / "src/isaac_ros_common"
        tmux_command(
            session,
            f"export ISAAC_ROS_WS={shlex.quote(str(cfg.isaac_ros_dir))}; "
            f"cd {shlex.quote(str(run_dev_dir))} && ./scripts/run_dev.sh",
        )


def stop_visible_isaac_sessions() -> None:
    # RViz attaches to the existing container; stop it first. Stopping the VIO
    # session then ends the owning docker-run shell and removes the container.
    stop_rviz_session()
    if tmux_session_exists(ISAAC_TMUX_VIO) and isaac_container_running():
        # Let ROS release its shared-memory ports before run_dev's auto-removed
        # container is torn down. Killing the terminal first can strand them.
        subprocess.run(["tmux", "send-keys", "-t", ISAAC_TMUX_VIO, "C-c"],
                       check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + 12.0
        while time.monotonic() < deadline:
            result = subprocess.run(
                ["docker", "exec", ISAAC_CONTAINER, "pgrep", "-f",
                 "[s]cripts/start_vio.sh|[r]clcpp_components/component_container_mt"],
                capture_output=True, timeout=3,
            )
            if result.returncode != 0:
                break
            time.sleep(0.25)
    for session in (ISAAC_TMUX_VIO, *LEGACY_ISAAC_TMUX_SESSIONS):
        subprocess.run(["tmux", "kill-session", "-t", session], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stop_rviz_session() -> None:
    """Stop RViz inside Docker, then remove its terminal/session wrapper."""
    if isaac_container_running():
        subprocess.run(
            ["docker", "exec", ISAAC_CONTAINER, "pkill", "-TERM", "-x", "rviz2"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            alive = subprocess.run(
                ["docker", "exec", ISAAC_CONTAINER, "pgrep", "-x", "rviz2"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode == 0
            if not alive:
                break
            time.sleep(0.25)
        else:
            subprocess.run(
                ["docker", "exec", ISAAC_CONTAINER, "pkill", "-KILL", "-x", "rviz2"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    for session in (ISAAC_TMUX_RVIZ, "vio-test-isaac-rviz"):
        subprocess.run(
            ["tmux", "kill-session", "-t", session], check=False,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )


@serialized_startup
def start_rviz_session(cfg: Config) -> None:
    with process_operation(cfg):
        stop_rviz_session()
    print("- Opening Terminal 2: Isaac ROS RViz (run_dev.sh → start_rviz.sh)...")
    start_visible_isaac_session(cfg, ISAAC_TMUX_RVIZ, "Isaac ROS RViz")
    time.sleep(3.0)
    startup_tmux_command(
        cfg, ISAAC_TMUX_RVIZ,
        "cd ${ISAAC_ROS_WS}/src/isaac_ros_common && ./scripts/start_rviz.sh; exit",
    )


def selected_camera_mode(cfg: Config) -> str:
    # Restored VO baseline: ignore stale IMU opt-in files from the experiment.
    return "vo"


def camera_mode_label(cfg: Config) -> str:
    mode = selected_camera_mode(cfg)
    return "VIO (stereo + IMU)" if mode == "vio" else "VO (stereo only)"


def require_camera_ready(cfg: Config, started_only: bool = False) -> None:
    check_startup_current(cfg)
    mode = selected_camera_mode(cfg)
    command = (
        "source ${ISAAC_ROS_WS}/src/isaac_ros_common/scripts/vo_environment.sh && "
        "python3 ${ISAAC_ROS_WS}/src/isaac_ros_common/scripts/check_camera_ready.py "
        # Each probe creates a new ROS participant. Identity-service discovery
        # needs the same budget even when only checking startup, not data health.
        f"{mode} --timeout 12"
        + f" --serial {shlex.quote(cfg.camera_serial)}"
        + (" --started-only" if started_only else "")
    )
    for attempt in range(2 if started_only else 1):
        result = subprocess.run(
            ["docker", "exec", "-u", "admin", ISAAC_CONTAINER, "bash", "-lc", command],
            capture_output=True, text=True, timeout=20,
        )
        check_startup_current(cfg)
        if not result.returncode:
            print(f"- {result.stdout.strip()}")
            return
        error = result.stdout.strip() or result.stderr.strip() or "Camera readiness check failed"
        # Retry only an unanswered identity request, with a fresh ROS client.
        # An actual wrong serial/model or fusion-mode mismatch must fail closed.
        if (started_only and attempt == 0
                and 'not ready: D456 identity verified=False' in error):
            print("- Camera identity confirmation timed out; retrying once (camera is not restarted).",
                  flush=True)
            continue
        raise RuntimeError(error)


def require_no_publishers_for_camera_restart(cfg: Config) -> None:
    active = [name for name in ("gps-bridge", "ev-bridge") if find_proc(cfg, name)]
    if active:
        raise RuntimeError(
            f"Cannot restart the camera while {', '.join(active)} is running. "
            "Stop all processes deliberately before restarting the camera. "
            "No camera restart was performed.")


@serialized_startup
def cmd_vio(cfg: Config, defer_health: bool = False) -> None:
    require_current_cli()
    print(f"- Selected camera mode: {camera_mode_label(cfg)}")
    if not isinstance(cfg.camera_serial, str) or not cfg.camera_serial.isascii() or not cfg.camera_serial.isdecimal():
        raise RuntimeError('Set the rear D456 SDK serial in Camera extrinsic settings first. Automatic camera selection is disabled.')
    print(f"- Selected camera: rear D456, serial {cfg.camera_serial}")
    if not shutil.which("tmux") or not shutil.which("gnome-terminal"):
        raise RuntimeError("tmux and gnome-terminal are required for visible Isaac sessions")
    raw_topic = "/visual_slam/tracking/odometry"
    if tmux_session_exists(ISAAC_TMUX_VIO) and isaac_container_running():
        try:
            if defer_health:
                require_camera_ready(cfg, started_only=True)
            else:
                require_ros_message(
                    cfg, raw_topic, "existing cuVSLAM body odometry", 6.0,
                    message_type="nav_msgs/msg/Odometry",
                )
        except (StartupCancelled, ProcessOperationBusy):
            raise
        except RuntimeError:
            require_no_publishers_for_camera_restart(cfg)
            print("Existing Isaac VIO session is not healthy; restarting it...", flush=True)
        else:
            check_startup_current(cfg)
            if not defer_health:
                require_camera_ready(cfg)
            print("- Existing cuVSLAM session reused; health will be checked during spoofing."
                  if defer_health else "- Existing cuVSLAM body odometry is healthy; reusing Terminal 1.")
            if cfg.rviz_enabled and not tmux_session_exists(ISAAC_TMUX_RVIZ):
                start_rviz_session(cfg)
            print(
                "- RViz is running." if tmux_session_exists(ISAAC_TMUX_RVIZ)
                else "- RViz auto-launch is OFF; continuing without RViz."
            )
            print("- Camera started; keep the drone stationary through spoofing."
                  if defer_health else "- cuVSLAM body odometry ready.")
            return
    require_no_publishers_for_camera_restart(cfg)
    print("- Opening Terminal 1: Isaac ROS VIO (run_dev.sh → start_vio.sh)...")
    start_visible_isaac_session(cfg, ISAAC_TMUX_VIO, "Isaac ROS VIO")
    deadline = time.monotonic() + 2700.0
    while not isaac_container_running():
        check_startup_current(cfg)
        recent = tmux_recent_output(ISAAC_TMUX_VIO)
        if (
            "Error response from daemon: No such container" in recent
            or "Container isaac_ros_dev-aarch64-container is not running" in recent
        ):
            with process_operation(cfg):
                stop_visible_isaac_sessions()
            raise RuntimeError(
                "Isaac run_dev.sh failed while attaching to a stale container. "
                "The stale session was cleaned up; start the path again."
            )
        if time.monotonic() >= deadline:
            with process_operation(cfg):
                stop_visible_isaac_sessions()
            raise RuntimeError("Isaac container did not start within 45 minutes")
        time.sleep(1.0)
    # The keystrokes remain queued until run_dev.sh presents the admin shell.
    time.sleep(2.0)
    startup_tmux_command(
        cfg, ISAAC_TMUX_VIO,
        "cd ${ISAAC_ROS_WS} && "
        "source src/isaac_ros_common/scripts/vo_environment.sh && "
        "if [[ -f src/isaac_ros_common/isaac_ros_visual_slam/isaac_ros_visual_slam/package.xml ]]; then "
        "echo '=== Incremental build: isaac_ros_visual_slam ==='; "
        "colcon build --packages-select isaac_ros_visual_slam || exit $?; "
        "else echo '=== Using installed isaac_ros_visual_slam package ==='; fi; "
        "source src/isaac_ros_common/scripts/vo_environment.sh && "
        "echo \"=== Using: $(ros2 pkg prefix isaac_ros_visual_slam) ===\" && "
        "cd ${ISAAC_ROS_WS}/src/isaac_ros_common && "
        f"VIO_MODE={selected_camera_mode(cfg)} "
        f"VIO_CAMERA_SERIAL={shlex.quote(cfg.camera_serial)} "
        f"CAMERA_X_M={cfg.camera_x_m:.17g} "
        f"CAMERA_Y_M={cfg.camera_y_m:.17g} "
        f"CAMERA_Z_M={cfg.camera_z_m:.17g} "
        f"CAMERA_ROLL_RAD={math.radians(cfg.camera_roll_deg):.17g} "
        f"CAMERA_PITCH_RAD={math.radians(cfg.camera_pitch_deg):.17g} "
        f"CAMERA_YAW_RAD={math.radians(cfg.camera_yaw_deg):.17g} "
        "./scripts/start_vio.sh",
    )
    print("- Terminal 1 is rebuilding changed VSLAM files, then starting cuVSLAM...")
    try:
        odom_deadline = time.monotonic() + 2700.0
        while True:
            check_startup_current(cfg)
            try:
                if defer_health:
                    require_camera_ready(cfg, started_only=True)
                else:
                    require_ros_message(
                        cfg, raw_topic, "cuVSLAM body odometry", 3.0,
                        message_type="nav_msgs/msg/Odometry",
                    )
                break
            except (StartupCancelled, ProcessOperationBusy):
                raise
            except RuntimeError as exc:
                check_startup_current(cfg)
                recent = tmux_recent_output(ISAAC_TMUX_VIO, lines=120)
                launch_stopped = (
                    "process has finished cleanly" in recent
                    or "process has died" in recent
                    or "user interrupted with ctrl-c" in recent
                    or "VIO setup failed" in recent
                    or "VIO IMU FAILED" in recent
                    or ("requested device with" in recent and "is NOT found" in recent)
                )
                if launch_stopped or not tmux_session_exists(ISAAC_TMUX_VIO):
                    raise RuntimeError(
                        "cuVSLAM stopped before body odometry became ready. "
                        "Check the visible Isaac ROS VIO terminal."
                    ) from exc
                if time.monotonic() >= odom_deadline:
                    raise RuntimeError(
                        f"No body odometry arrived on {raw_topic} within 45 minutes"
                    ) from exc
        check_startup_current(cfg)
        if not defer_health:
            require_camera_ready(cfg)
        if cfg.rviz_enabled:
            start_rviz_session(cfg)
            print("- RViz is running.")
        else:
            print("- RViz auto-launch is OFF; continuing without RViz.")
        if not defer_health:
            print("- cuVSLAM is publishing vehicle-body odometry directly (drone_link).")
    except (StartupCancelled, ProcessOperationBusy):
        # Stop all owns teardown. An old startup must never kill or relaunch
        # sessions after its cancellation, including a subsequent new launch.
        raise
    except Exception as exc:
        check_startup_current(cfg)
        # Capture evidence before closing the auto-removed container/terminal.
        recent = tmux_recent_output(ISAAC_TMUX_VIO, lines=200)
        log_note = ""
        try:
            cfg.log_dir.mkdir(parents=True, exist_ok=True)
            failure_log = cfg.log_dir / "camera-startup-failure.log"
            failure_log.write_text(recent, encoding="utf-8")
            log_note = f"\nCamera startup log: {failure_log}"
        except OSError:
            pass  # Do not hide the actual startup error if storage is full.
        with process_operation(cfg):
            stop_visible_isaac_sessions()
        tail = "\n".join(recent.splitlines()[-30:])
        raise RuntimeError(f"{exc}{log_note}\nRecent camera output:\n{tail}") from exc
    print("- Camera started; health validation will run during spoofing. Keep the drone stationary."
          if defer_health else "- cuVSLAM body odometry ready.")


@serialized_process_operation
def stop_failed_spoof(cfg: Config, reason: str, *,
                      expected_proc: Optional[dict] = None) -> None:
    """GPS has already stopped in the bridge; stop it before camera teardown."""
    if expected_proc is not None:
        current = find_proc(cfg, "gps-bridge")
        if current is None or not same_managed_process(current, expected_proc):
            raise RuntimeError(
                "The original GPS bridge is no longer managed by this launch. "
                "Replacement processes and camera were left untouched.")
    print(f"- Spoof validation failed: {reason}", flush=True)
    stop_managed(cfg, "gps-bridge")
    # Save the camera evidence before the owning terminal/container is removed.
    try:
        cfg.log_dir.mkdir(parents=True, exist_ok=True)
        (cfg.log_dir / "camera-spoof-failure.log").write_text(
            f"Spoof validation failed: {reason}\n" + tmux_recent_output(ISAAC_TMUX_VIO, lines=200),
            encoding="utf-8",
        )
    except OSError as exc:
        print(f"- Could not save camera failure log: {exc}", flush=True)
    stop_visible_isaac_sessions()
    print("- GPS bridge and cuVSLAM/RViz sessions stopped. Start Path A again to retry.", flush=True)


@serialized_startup
def start_gps_bridge(cfg: Config) -> dict:
    """Own startup across preflight; lock only state mutations, not readiness."""
    require_current_cli()
    if find_proc(cfg, "gps-bridge"):
        raise RuntimeError("gps-bridge already running. Stop it first.")
    if find_proc(cfg, "ev-bridge") or find_proc(cfg, "mavros"):
        raise RuntimeError("A legacy EV/MAVROS process is already running. Stop it first.")
    # This command is also callable directly, without cmd_run/cmd_vio.
    # Verify the opened device before starting any HIL_GPS publisher.
    require_camera_ready(cfg, started_only=True)
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
        # Keep the stationary bootstrap active for the full configured duration.
        # PX4/QGroundControl receives STATUSTEXT when spoof starts and when VIO takes over.
        "spoof_until_vio": "false",
        "boot_accuracy_m": 0.1,
        "cruise_eph_m": 0.1,
        # Maximum numeric MAVLink value; 255 is reserved for "unknown".
        "satellites": 40,
        "horizontal_only_output": "true",
        "gps_id": 1,
        "validate_dual_gps_selection": "true",
        "expected_vio_gps_instance": -1,
        "rc_trigger_enabled": "true",
        "rc_channel": 6,
        "rc_low_pwm_max": 1300,
        "rc_high_pwm_min": 1700,
        "rc_request_rate_hz": 10.0,
        "rate_hz": 10.0,
        "odom_topic": cfg.odom_topic,
        "external_yaw_enabled": str(cfg.external_yaw_enabled).lower(),
        "external_yaw_noise_deg": cfg.external_yaw_noise_deg,
        "external_yaw_rate_hz": cfg.external_yaw_rate_hz,
        "external_yaw_max_age_s": cfg.external_yaw_max_age_s,
        "external_yaw_startup_tolerance_deg": cfg.external_yaw_startup_tolerance_deg,
        "heading_source": cfg.heading_source,
        "manual_heading_deg": cfg.manual_heading_deg,
        "mag_declination_source": cfg.mag_declination_source,
        "mag_declination_deg": cfg.mag_declination_deg,
        "child_to_body_yaw_deg": 0.0,  # body odometry; no second mounting correction
        "expected_child_frame": "drone_link",
        "continuity_max_gap_s": cfg.gate_max_gap_s,
        "continuity_recovery_samples": cfg.gate_recovery_samples,
        "continuity_confirmation_samples": cfg.gate_confirmation_samples,
        "continuity_max_speed_m_s": cfg.gate_max_speed_m_s,
        "continuity_max_acceleration_m_s2": cfg.gate_max_acceleration_m_s2,
        "continuity_max_yaw_rate_deg_s": cfg.gate_max_yaw_rate_deg_s,
        "inertial_max_message_age_s": cfg.inertial_max_message_age_s,
        "inertial_max_message_gap_s": cfg.inertial_max_message_gap_s,
        "inertial_reorder_tolerance_s": cfg.inertial_reorder_tolerance_s,
        "inertial_max_position_uncertainty_m": cfg.inertial_max_position_uncertainty_m,
        "recovery_velocity_agreement_m_s": cfg.recovery_velocity_agreement_m_s,
        "recovery_velocity_agreement_samples": cfg.recovery_velocity_agreement_samples,
        "recovery_yaw_agreement_deg": cfg.recovery_yaw_agreement_deg,
        "recovery_yaw_agreement_samples": cfg.recovery_yaw_agreement_samples,
        "recovery_accuracy_tighten_s": cfg.recovery_accuracy_tighten_s,
        "status_file": str(cfg.state_dir / "navigation-status.json"),
        "camera_imu_required": "true" if selected_camera_mode(cfg) == "vio" else "false",
    }
    param_args = " ".join(
        f"-p {shlex.quote(f'{name}:={value}')}" for name, value in params.items()
    )
    cmd = (
        f"{ros_prefix(cfg, [setup])} && ros2 run vio_px4_bridge vio_px4_gps_bridge --ros-args"
        f" {param_args}"
    )
    navigation_status = cfg.state_dir / "navigation-status.json"
    with process_operation(cfg):
        navigation_status.unlink(missing_ok=True)
    return start_managed(cfg, "gps-bridge", cmd, startup_check_s=6.0)


@serialized_startup
def cmd_gps(cfg: Config) -> None:
    proc = start_gps_bridge(cfg)
    navigation_status = cfg.state_dir / "navigation-status.json"
    # Do not hold the operation lock while waiting for the operator's RC switch.
    # Another terminal must remain able to stop the pipeline during this wait.
    print(f"- Path A connected to PX4 on {cfg.mavlink_url} (pid {proc['pid']}).")
    if cfg.external_yaw_enabled:
        print("- Heading time sync warms up before activation (normally about 15 seconds).")
        print("- If RC is triggered too early, wait for clock ready in Status, then toggle LOW/HIGH again.")
    print("- Waiting for spoof: move RC channel 6 LOW/MID → HIGH.", flush=True)

    saw_spoof = False
    while True:
        check_startup_current(cfg)
        if not managed_leader_alive(proc):
            stop_managed(cfg, "gps-bridge", expected_proc=proc)
            raise RuntimeError("GPS bridge stopped during the spoof sequence")
        try:
            status = json.loads(navigation_status.read_text())
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            status = {}
        navigation = status.get("navigation")
        if navigation == "spoof_active" and not saw_spoof:
            saw_spoof = True
            print(
                f"- Spoofing at {cfg.home_lat:.7f}, {cfg.home_lon:.7f} for 15 s; checking camera health.",
                flush=True,
            )
        if saw_spoof and navigation == "live_vio":
            print("- 15 s done. VIO takes over. See option 2) Status.", flush=True)
            return
        if status.get("spoof_failure_reason") or (saw_spoof and navigation == "waiting_for_rc"):
            reason = status.get("spoof_failure_reason") or "no healthy heading-aligned VIO handoff"
            stop_failed_spoof(cfg, reason, expected_proc=proc)
            raise RuntimeError(
                f"Spoof validation failed: {reason}. GPS and camera stopped."
            )
        time.sleep(0.2)


def show_navigation_status(cfg: Config) -> None:
    path = cfg.state_dir / "navigation-status.json"
    print("Status\n")
    try:
        status = json.loads(path.read_text())
    except FileNotFoundError:
        print("- Output: unknown (no status)")
        print("- Bridge: not running (no status received)")
        return
    except (OSError, json.JSONDecodeError) as exc:
        print("- Output: unknown (status unavailable)")
        print(f"- Status unavailable: {exc}")
        return

    age = max(0.0, time.time() - float(status.get("updated_unix_s", 0.0)))
    if age > 2.0:
        print("- Output: unknown (stale status)")
        print(f"- Bridge: not reporting (last update {age:.1f} s ago)")
        return

    navigation = status.get("navigation", "unknown")
    connection = "connected" if status.get("px4_connected") else "disconnected"
    vio_age = status.get("vio_age_s")
    accepted_age = f"{float(vio_age):.2f} s ago" if vio_age is not None else "unavailable"

    def output_health(active):
        if navigation == "waiting_for_rc":
            return "waiting for RC trigger"
        if status.get("pose_gate_quarantine"):
            reason = status.get("pose_gate_reason")
            if reason in ("position_jump", "yaw_jump", "speed_limit", "acceleration_limit",
                          "yaw_rate_limit", "frame_changed"):
                return "pose-jump detected"
            if reason in ("tracking_gap", "vio_stale", "heading_sample_stale"):
                return "stale VO"
            return "quarantined"
        if active:
            return "healthy"
        if navigation == "live_vio" and not status.get("vio_fresh"):
            return "stale VO"
        return "blocked"

    yaw_enabled = bool(status.get("external_yaw_enabled"))
    yaw_active = yaw_enabled and bool(status.get("external_yaw_output_active"))
    gps_active = navigation == "spoof_active" or bool(status.get("hil_gps_output_active"))
    gps_state = "ON (spoof)" if navigation == "spoof_active" else "ON" if gps_active else "OFF"
    yaw_state = "ON" if yaw_active else "OFF" if yaw_enabled else "disabled"
    print(f"- Output: GPS {gps_state} | Heading {yaw_state} | PX4 {connection}")

    gps_health = output_health(gps_active)
    if navigation == "spoof_active":
        gps_health = "spoofing"
        remaining = status.get("spoof_remaining_s")
        if remaining is not None:
            gps_health += f" ({float(remaining):.1f} s left)"
    lat, lon, eph = (status.get(key) for key in
                     ("hil_gps_lat_deg", "hil_gps_lon_deg", "hil_gps_eph_m"))
    gps_age = status.get("hil_gps_sample_age_s")
    gps_row = [f"- {'HIL_GPS:':<12}{gps_health}"]
    if lat is not None and lon is not None and eph is not None:
        prefix = "" if gps_active else "last sent "
        gps_age_text = "unavailable" if gps_age is None else f"{float(gps_age)*1000:.0f} ms"
        gps_row.extend([f"{prefix}{float(lat):.7f}, {float(lon):.7f}",
                        ("hAcc", f"{float(eph):.2f}", "m"), f"age {gps_age_text}"])
    elif "hil_gps_lat_deg" in status:
        gps_row.append("not sent yet")
    else:
        gps_row.append("sent values unavailable (restart bridge)")
    rows = [gps_row]
    if yaw_enabled:
        yaw_health = output_health(yaw_active)
        yaw_row = [f"- {'VO heading:':<12}{yaw_health}"]
        heading = status.get("external_yaw_heading_deg")
        variance = status.get("external_yaw_variance_rad2")
        if heading is not None and variance is not None:
            heading_text = f"{round(float(heading), 1) % 360:.1f}°"
            sample_age = status.get("external_yaw_sample_age_s")
            age_text = "age unavailable" if sample_age is None else f"age {float(sample_age)*1000:.0f} ms"
            prefix = "" if yaw_active else "last sent "
            # Use the covariance attached to that sent sample, not a pending setting.
            variance = float(variance)
            sigma_deg = (math.degrees(math.sqrt(variance))
                         if math.isfinite(variance) and variance >= 0 else None)
            uncertainty = ("yaw std", f"{sigma_deg:.2f}", "°") if sigma_deg is not None else (
                "yaw std", "unavailable", "")
            yaw_row.extend([f"{prefix}{heading_text}", uncertainty, age_text])
        elif "external_yaw_heading_deg" in status:
            yaw_row.append("not sent yet")
        else:
            yaw_row = ["- VO heading: unavailable (restart the bridge after updating)"]
        rows.append(yaw_row)
    # Align uncertainty labels, decimal points, and units inside their shared
    # column. The raw rad² covariance remains unchanged in the status snapshot.
    uncertainty_rows = [row for row in rows if len(row) > 2]
    if uncertainty_rows:
        label_width = max(len(row[2][0]) for row in uncertainty_rows)
        number_width = max(len(row[2][1]) for row in uncertainty_rows)
        for row in uncertainty_rows:
            label, number, unit = row[2]
            row[2] = f"{label:<{label_width}} {number:>{number_width}} {unit}".rstrip()
    # Pad shared columns together, including longer recovery labels/last values.
    # Never truncate coordinates, uncertainty, age, or a warning to force a fit.
    widths = [max((len(row[i]) for row in rows if i < len(row)), default=0)
              for i in range(max(map(len, rows)))]
    for row in rows:
        print(" | ".join(cell.ljust(widths[i]) if i < len(row) - 1 else cell
                         for i, cell in enumerate(row)))
    if not gps_active and not yaw_active:
        print(f"- Last trusted VO: {accepted_age}")

    # Keep routine clock/health details out of the healthy view. The full
    # diagnostic snapshot remains in navigation-status.json.
    blocked = []
    if (not status.get("hil_gps_output_active")
            and navigation not in ("spoof_active", "waiting_for_rc")):
        blocked.append(status.get("output_block_reason") or "GPS output stopped")
    if yaw_enabled and not yaw_active and navigation != "waiting_for_rc":
        reason = status.get("external_yaw_status")
        if reason and reason not in ("ready", "disabled"):
            blocked.append(reason)
        if not status.get("external_yaw_clock_ready"):
            blocked.append("heading clock warming/stale")
    if navigation in ("spoof_active", "waiting_for_rc"):
        check = status.get("handoff_check_reason")
        if check and check != "healthy":
            print(f"- Startup check: {check}")
    if status.get("spoof_failure_reason"):
        print(f"- Spoof failed: {status['spoof_failure_reason']}")
    if float(status.get("frozen_position_uncertainty_m", 0.0)) > 0:
        print("- Position warning: movement discarded; verify/reset origin.")
    inherited = float(status.get("inherited_position_uncertainty_m", 0.0))
    if inherited > 0:
        print(f"- Position uncertainty: ≥{inherited:.2f} m (from PX4)")
    if status.get("inertial_recovery_active"):
        if "recovery_position_safe" in status:
            position_safe = bool(status["recovery_position_safe"])
            heading_safe = bool(status.get("recovery_heading_safe"))
        else:
            # Older bridges report one combined safety flag.
            heading_safe = bool(status.get("inertial_recovery_safe"))
            position_safe = heading_safe and not status.get("inertial_fallback_to_frozen_pose")
        position = "following PX4" if position_safe else "frozen"
        heading_reference = "ready" if heading_safe else "waiting for FCU"
        print(f"- Recovery: position {position} | heading {heading_reference}")
        if not position_safe:
            print(f"- Position frozen: {status.get('inertial_recovery_failure') or 'unknown reason'}")
        if not heading_safe:
            blocked.append(status.get("recovery_heading_failure") or "fresh FCU heading needed")
        velocity_count = int(status.get("recovery_velocity_agreement_count", 0))
        yaw_state = status.get("recovery_yaw_agreement_state", "unknown")
        yaw_count = int(status.get("recovery_yaw_agreement_count", 0))
        yaw_note = " (disagreeing)" if yaw_state == "disagreeing" else ""
        if yaw_state == "waiting_for_time_sync":
            yaw_note = " (waiting for clock sync)"
        elif yaw_state == "waiting_for_time_matched_fcu":
            yaw_note = " (waiting for matching FCU sample)"
        elif yaw_state == "waiting_for_post_reset_fcu":
            yaw_note = " (waiting for post-reset FCU match)"
        print(f"- Recovery checks: velocity {velocity_count}/{cfg.recovery_velocity_agreement_samples}"
              f" | yaw {yaw_count}/{cfg.recovery_yaw_agreement_samples}{yaw_note}")
    if blocked:
        print("- Blocked: " + "; ".join(dict.fromkeys(blocked)))


def watch_navigation_status(cfg: Config) -> None:
    """Refresh the status screen until the operator presses Enter."""
    while True:
        _clear()
        show_navigation_status(cfg)
        print("\nRefresh: 0.5 s | Enter: back", flush=True)
        readable, _, _ = select.select([sys.stdin], [], [], 0.5)
        if readable:
            sys.stdin.readline()
            return


@serialized_startup
def cmd_run(
    cfg: Config,
    path_arg: str,
) -> None:
    require_current_cli()
    which = normalize_path(path_arg)
    print(f"Starting Path {which.upper()} (GPS spoof + VIO GPS)...\n", flush=True)

    # Migration guard: do not compete with a publisher left by an older CLI.
    if find_proc(cfg, "ev-bridge") or find_proc(cfg, "mavros"):
        raise RuntimeError("A legacy EV/MAVROS process is already running. Stop it first.")
    cmd_vio(cfg, defer_health=True)
    if not find_proc(cfg, "gps-bridge"):
        cmd_gps(cfg)


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


@serialized_process_operation
def cmd_stop(cfg: Config, name: Optional[str] = None) -> None:
    # Keep invalid ownership state fail-closed, before changing anything.
    read_state(cfg)
    if name in (None, "cuvslam", "gps-bridge"):
        # Invalidate the starter's token without allocating disk space. Stop
        # must still be possible when logging has exhausted the filesystem.
        (cfg.state_dir / "startup-generation.json").unlink(missing_ok=True)
    stopped = stop_managed(cfg, name)
    if name is None:
        stop_visible_isaac_sessions()
    if name in (None, "cuvslam"):
        cleanup = cfg.vio_px4_dir / "scripts/stop_cuvslam_body.sh"
        if cleanup.exists():
            subprocess.run(["bash", str(cleanup)], check=False, timeout=8.0)
    if not stopped:
        print(f"Nothing named '{name}' to stop." if name else "Nothing to stop.")
        return
    print(f"Stopped: {', '.join(stopped)}")


@serialized_process_operation
def cmd_shutdown(cfg: Config) -> None:
    """Stop this pipeline and its Isaac ROS container/session."""
    cmd_stop(cfg)
    if not isaac_container_running():
        print(f"Isaac container already stopped: {ISAAC_CONTAINER}")
        return
    print(f"Stopping Isaac container: {ISAAC_CONTAINER}...")
    result = subprocess.run(
        ["docker", "stop", "--time", "10", ISAAC_CONTAINER],
        capture_output=True,
        text=True,
        timeout=20.0,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"Could not stop Isaac container: {detail}")
    print("Isaac container stopped; the run_dev.sh session can now close.")


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
    print("Configure home / spoof origin (latitude and longitude)\n")
    print(f"Current: {cfg.home_lat}, {cfg.home_lon}")
    lat = prompt_line("Latitude deg", str(cfg.home_lat))
    lon = prompt_line("Longitude deg", str(cfg.home_lon))
    if lat is None:
        return cfg
    try:
        lat_value = float(lat)
        lon_value = float(lon or cfg.home_lon)
        if not all(math.isfinite(value) for value in (lat_value, lon_value)):
            raise ValueError("latitude and longitude must be finite numbers")
        if not -90.0 <= lat_value <= 90.0:
            raise ValueError("latitude must be between -90 and 90")
        if not -180.0 <= lon_value <= 180.0:
            raise ValueError("longitude must be between -180 and 180")
        cfg = replace(
            cfg,
            home_lat=lat_value,
            home_lon=lon_value,
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
    print("Configure startup heading alignment\n")
    print("PX4 mode inherits the flight controller's estimated heading.")
    print("The bridge does not add declination or calculate compass heading again.")
    print("Verify PX4 heading first; keep the drone stationary during spoofing.")
    print("Alignment stays fixed in normal tracking; recovery uses current PX4 heading.\n")
    print("Heading describes drone_link's forward direction, not the camera direction.")
    print("Camera mounting is handled only in Camera extrinsics; no extra yaw correction.\n")
    print("Optional Path A heading output: ODOMETRY yaw + HIL_GPS position.")
    print("One quarantine stops BOTH; barometer/rangefinder height is unchanged.")
    print("Camera capture-age protection applies with heading ON or OFF (also in Pose-jump limits).")
    enabled = prompt_line("Send VO heading to PX4 (on/off)",
                          "on" if cfg.external_yaw_enabled else "off")
    if enabled is None:
        return cfg
    if enabled.strip().lower() not in ("on", "off"):
        print("Use on or off.")
        pause()
        return cfg
    enabled = enabled.strip().lower() == "on"
    source = prompt_line("Initial heading reference (px4/manual)", cfg.heading_source)
    if source is None:
        return cfg
    source = (source or cfg.heading_source).strip().lower()
    if source not in ("px4", "manual"):
        print("Invalid source; use px4 or manual.")
        pause()
        return cfg
    manual = str(cfg.manual_heading_deg)
    if source == "manual":
        manual = prompt_line("Body true heading at alignment (deg clockwise from north)", manual)
    output_values = (cfg.external_yaw_noise_deg, cfg.external_yaw_rate_hz,
                     cfg.external_yaw_max_age_s, cfg.external_yaw_startup_tolerance_deg)
    if enabled:
        print("PX4: EKF2_EV_CTRL=8, EKF2_EV_NOISE_MD=0, EKF2_GPS_CTRL=1.")
        print("EKF2_MAG_TYPE=6 for PX4 startup heading, or 5 for a manual heading seed.")
        print("Manual seed: disarmed startup only. Reboot PX4 if already aligned to a different heading.")
        print("No forced heading reset; live GPS waits for matching PX4 heading readback.")
        print("Time sync warms up before heading output. Watch status before triggering spoof.")
        print("The heading noise setting is a minimum, not a fixed accuracy claim.")
        print("PX4 reference uncertainty is retained at startup/recovery; it does not reset to 1 degree.")
        prompts = ("Minimum heading uncertainty deg (1 sigma, 0.1..90)",
                   "Heading output rate Hz (30..50)",
                   "Maximum camera sample age s (0.02..0.5)",
                   "Startup heading readback tolerance deg (>0..20)")
        output_values = tuple(prompt_line(label, str(value))
                              for label, value in zip(prompts, output_values))
    try:
        manual_heading = float(manual)
        noise, rate, age, tolerance = tuple(float(x) for x in output_values)
        if not math.isfinite(manual_heading):
            raise ValueError("manual heading must be finite")
        if not (.1 <= noise <= 90 and 30 <= rate <= 50 and .02 <= age <= .5
                and 0 < tolerance <= 20):
            raise ValueError("heading output values outside displayed bounds")
        cfg = replace(
            cfg,
            heading_source=source,
            manual_heading_deg=manual_heading,
            external_yaw_enabled=enabled, external_yaw_noise_deg=noise,
            external_yaw_rate_hz=rate, external_yaw_max_age_s=age,
            external_yaw_startup_tolerance_deg=tolerance,
        )
    except (TypeError, ValueError) as exc:
        print(f"Invalid heading configuration: {exc}")
        pause()
        return cfg
    save_config(cfg)
    print("Saved heading alignment configuration. Restart Path A to apply.")
    if not enabled:
        print("GPS-only Path A: set PX4 EKF2_EV_CTRL=0 before use. No PX4 parameter was changed here.")
    pause()
    return cfg


def configure_camera_extrinsic(cfg: Config) -> Config:
    _clear()
    print("Configure drone_link → camera_link extrinsic\n")
    print("Use the rear D456 SDK serial from rs-enumerate-devices, not its USB serial.")
    serial = prompt_line("Rear D456 serial", cfg.camera_serial)
    if serial is None:
        return cfg
    serial = serial.strip().removeprefix('_')
    if not serial.isascii() or not serial.isdecimal():
        print("Invalid serial; automatic selection is disabled.")
        pause()
        return cfg
    print("ROS FLU convention (camera position measured from the vehicle origin):")
    print("  +X forward / -X backward")
    print("  +Y left    / -Y right")
    print("  +Z up      / -Z down")
    print("Positive rotations follow the right-hand rule:")
    print("  +roll about +X: right side down")
    print("  +pitch about +Y: camera points down")
    print("  +yaw about +Z: camera points left\n")
    prompts = (
        ("X translation m", cfg.camera_x_m),
        ("Y translation m", cfg.camera_y_m),
        ("Z translation m", cfg.camera_z_m),
        ("Roll deg", cfg.camera_roll_deg),
        ("Pitch deg", cfg.camera_pitch_deg),
        ("Yaw deg", cfg.camera_yaw_deg),
    )
    entered = []
    try:
        for label, current in prompts:
            value = prompt_line(label, str(current))
            if value is None:
                return cfg
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"{label} must be finite")
            entered.append(number)
    except ValueError as exc:
        print(f"Invalid camera extrinsic: {exc}")
        pause()
        return cfg
    cfg = replace(
        cfg,
        camera_serial=serial,
        camera_x_m=entered[0], camera_y_m=entered[1], camera_z_m=entered[2],
        camera_roll_deg=entered[3], camera_pitch_deg=entered[4],
        camera_yaw_deg=entered[5],
    )
    save_config(cfg)
    print(
        "Saved camera extrinsic: "
        f"xyz=({entered[0]:+g}, {entered[1]:+g}, {entered[2]:+g}) m, "
        f"rpy=({entered[3]:+g}, {entered[4]:+g}, {entered[5]:+g})°."
    )
    print("The new extrinsic takes effect the next time cuVSLAM is started.")
    pause()
    return cfg


def configure_pose_gate(cfg: Config) -> Config:
    _clear()
    print("Configure VIO continuity / pose-jump gate\n")
    print("A sample is quarantined when a physical or timing limit is exceeded.")
    print("Quarantined samples do not update the accepted local trajectory.\n")
    groups = (
        ("Physical rate limits", (
            ("speed", "Maximum speed m/s", cfg.gate_max_speed_m_s, float, 0.0),
            ("acceleration", "Maximum acceleration m/s^2",
             cfg.gate_max_acceleration_m_s2, float, 0.0),
            ("yaw_rate", "Maximum yaw rate deg/s",
             cfg.gate_max_yaw_rate_deg_s, float, 0.0),
        )),
        ("Timing", (
            ("capture_age", "Maximum camera capture age s (0.02..0.5, GPS and yaw)",
             cfg.external_yaw_max_age_s, float, 0.0),
            ("gap", "Maximum tracking gap s", cfg.gate_max_gap_s, float, 0.0),
        )),
        ("Coordinate-reset recovery", (
            ("confirmation", "Samples to confirm a coordinate reset",
             cfg.gate_confirmation_samples, int, 1),
            ("recovery", "Stable samples before output resumes",
             cfg.gate_recovery_samples, int, 0),
        )),
        ("GPS-silent PX4 inertial propagation", (
            ("inertial_age", "Maximum PX4 motion-message age s",
             cfg.inertial_max_message_age_s, float, 0.0),
            ("inertial_gap", "Maximum PX4 motion-message gap s",
             cfg.inertial_max_message_gap_s, float, 0.0),
            ("inertial_reorder", "Small timestamp-reorder tolerance s",
             cfg.inertial_reorder_tolerance_s, float, -1.0),
            ("inertial_uncertainty", "Maximum PX4 horizontal uncertainty m",
             cfg.inertial_max_position_uncertainty_m, float, 0.0),
        )),
        ("VIO handoff", (
            ("velocity_agreement", "Maximum VIO/PX4 velocity disagreement m/s",
             cfg.recovery_velocity_agreement_m_s, float, 0.0),
            ("velocity_samples", "Consecutive velocity-agreement samples",
             cfg.recovery_velocity_agreement_samples, int, 0),
            ("yaw_agreement", "Maximum VIO/PX4 yaw-change disagreement deg",
             cfg.recovery_yaw_agreement_deg, float, 0.0),
            ("yaw_samples", "Consecutive unique PX4 yaw-agreement samples",
             cfg.recovery_yaw_agreement_samples, int, 0),
        )),
    )
    values: dict[str, float | int] = {}
    try:
        for group_index, (heading, prompts) in enumerate(groups):
            if group_index:
                print("------------------------------------------------------------")
            print(f"{heading}\n")
            for key, label, current, kind, minimum in prompts:
                raw = prompt_line(label, str(current))
                if raw is None:
                    return cfg
                value = kind(raw)
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError(f"{label} must be finite")
                if value <= minimum:
                    raise ValueError(f"{label} must be greater than {minimum:g}")
                values[key] = value
        if not .02 <= values["capture_age"] <= .5:
            raise ValueError("Camera capture age must be 0.02..0.5 s")
    except ValueError as exc:
        print(f"Invalid gate configuration: {exc}")
        pause()
        return cfg
    cfg = replace(
        cfg,
        gate_max_speed_m_s=float(values["speed"]),
        gate_max_acceleration_m_s2=float(values["acceleration"]),
        gate_max_yaw_rate_deg_s=float(values["yaw_rate"]),
        gate_max_gap_s=float(values["gap"]),
        external_yaw_max_age_s=float(values["capture_age"]),
        gate_confirmation_samples=int(values["confirmation"]),
        gate_recovery_samples=int(values["recovery"]),
        inertial_max_message_age_s=float(values["inertial_age"]),
        inertial_max_message_gap_s=float(values["inertial_gap"]),
        inertial_reorder_tolerance_s=float(values["inertial_reorder"]),
        inertial_max_position_uncertainty_m=float(values["inertial_uncertainty"]),
        recovery_velocity_agreement_m_s=float(values["velocity_agreement"]),
        recovery_velocity_agreement_samples=int(values["velocity_samples"]),
        recovery_yaw_agreement_deg=float(values["yaw_agreement"]),
        recovery_yaw_agreement_samples=int(values["yaw_samples"]),
    )
    save_config(cfg)
    print("\nSaved pose-jump gate limits.")
    print(
        "Note: speed and yaw-rate limits are automatically converted into "
        "allowed position and heading changes using the measured time between "
        "arriving odometry messages."
    )
    print(
        "During quarantine, PX4 odometry is used only while HIL_GPS is silent. "
        "There is no fixed duration cutoff. "
        "If position propagation fails a guard, position remains frozen. "
        "Recovery uses current FCU heading after fresh camera/FCU agreement; "
        "it does not require uninterrupted heading during the disturbance."
    )
    print("The new limits take effect the next time Path A starts.")
    print("Recovery uncertainty is retained; stable VO alone cannot remove global reference error.")
    pause()
    return cfg


def toggle_rviz(cfg: Config) -> Config:
    enabled = not cfg.rviz_enabled
    cfg = replace(cfg, rviz_enabled=enabled)
    save_config(cfg)
    if not enabled:
        stop_rviz_session()
        print("RViz auto-launch is OFF; the existing RViz session was stopped.")
    elif tmux_session_exists(ISAAC_TMUX_VIO) and isaac_container_running():
        try:
            require_ros_message(
                cfg, "/visual_slam/tracking/odometry", "existing cuVSLAM body odometry",
                6.0, message_type="nav_msgs/msg/Odometry",
            )
            start_rviz_session(cfg)
            print("RViz auto-launch is ON and RViz is starting now.")
        except RuntimeError as exc:
            print(f"RViz auto-launch is ON; it will start with the next path ({exc}).")
    else:
        print("RViz auto-launch is ON; it will start with the next path.")
    pause()
    return cfg


def pick_log(cfg: Config) -> None:
    ensure_dirs(cfg)
    # Only show logs that exist for components used by the hardware paths.
    names = [
        name
        for name in ("gps-bridge",)
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
        choice = select_menu(
            "Main Menu",
            [
                MenuItem("", "h1", disabled=True),
                MenuItem("Flight paths", "hdr1", disabled=True),
                MenuItem(
                    "Path A — GPS spoof → live VIO (global)",
                    "run-a",
                ),
                MenuItem("Status", "nav-status"),
                MenuItem("", "h2", disabled=True),
                MenuItem("Setup", "hdr2", disabled=True),
                MenuItem(
                    "FC link (UART / UDP)",
                    "link",
                    cfg.mavlink_url,
                ),
                MenuItem(
                    "Home latitude/longitude",
                    "home",
                    f"{cfg.home_lat}, {cfg.home_lon}",
                ),
                MenuItem(
                    "Heading alignment",
                    "heading",
                    f"Initial reference: {heading_label(cfg)}",
                ),
                MenuItem(
                    "Camera extrinsic (ROS FLU XYZ/RPY)",
                    "camera-extrinsic",
                    f"xyz=({cfg.camera_x_m:+g},{cfg.camera_y_m:+g},{cfg.camera_z_m:+g}) m; "
                    f"rpy=({cfg.camera_roll_deg:+g},{cfg.camera_pitch_deg:+g},"
                    f"{cfg.camera_yaw_deg:+g})°",
                ),
                MenuItem(
                    "Pose-jump gate limits",
                    "pose-gate",
                    f"speed={cfg.gate_max_speed_m_s:g} m/s; "
                    f"accel={cfg.gate_max_acceleration_m_s2:g} m/s²; "
                    f"yaw rate={cfg.gate_max_yaw_rate_deg_s:g}°/s",
                ),
                MenuItem(
                    "RViz auto-launch (toggle ON/OFF)",
                    "rviz-toggle",
                    "ON" if cfg.rviz_enabled else "OFF",
                ),
                MenuItem("", "h4", disabled=True),
                MenuItem("Leave CLI (keep processes running)", "detach"),
                MenuItem("Stop all processes (including Isaac container)", "stop-all"),
            ],
            subtitle=(f"Hardware: Jetson + Cube · link={cfg.mavlink_url}\n"
                      f"Selected camera mode: {camera_mode_label(cfg)}"),
        )

        if choice is None:
            _clear()
            return 0

        if choice == "detach":
            _clear()
            print("Left vio-launch; managed processes are still running.")
            return 0

        if choice == "stop-all":
            _clear()
            print("Stopping all managed processes...\n")
            try:
                cmd_shutdown(cfg)
            except Exception as exc:  # noqa: BLE001
                print(f"Error during shutdown: {exc}")
                pause()
                continue
            pause()
            continue

        try:
            # Check AFTER the blocking menu prompt: an update may have happened
            # while waiting for input. Leave/stop actions remain available above.
            require_current_cli()
            if choice == "run-a":
                _clear()
                cmd_run(cfg, "a")
                pause()
            elif choice == "link":
                cfg = configure_link(cfg)
            elif choice == "home":
                cfg = configure_home(cfg)
            elif choice == "heading":
                cfg = configure_heading(cfg)
            elif choice == "camera-extrinsic":
                cfg = configure_camera_extrinsic(cfg)
            elif choice == "pose-gate":
                cfg = configure_pose_gate(cfg)
            elif choice == "rviz-toggle":
                cfg = toggle_rviz(cfg)
            elif choice == "nav-status":
                watch_navigation_status(cfg)
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
        prog="vio-launch",
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
    sub.add_parser("vio")
    sub.add_parser("gps")

    run_p = sub.add_parser("run")
    run_p.add_argument("path")

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
        if args.command == "vio":
            cmd_vio(cfg)
        elif args.command == "gps":
            cmd_gps(cfg)
        elif args.command == "run":
            cmd_run(cfg, args.path)
        else:
            parser.print_help()
            return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
