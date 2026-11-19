"""Receive-only local diagnostic stream. Never opens a serial port or sends data."""
from collections import Counter
import json
import math
import socket
import time

PORT = 14691


def validate_packet(raw):
    if len(raw) > 4096:
        raise ValueError("oversize packet")
    p = json.loads(raw)
    if p.get("schema") != 1 or p.get("source") != "vio_px4_bridge_diagnostic":
        raise ValueError("not the bridge diagnostic stream")
    if p.get("type") not in ("ATTITUDE", "ODOMETRY"):
        raise ValueError("unsupported telemetry")
    for key in ("received_unix_s", "received_monotonic_s"):
        if not math.isfinite(float(p[key])):
            raise ValueError("invalid timestamp")
    d = p["data"]
    fields = ("boot_s", "roll", "pitch", "yaw", "rollspeed", "pitchspeed", "yawspeed")
    if p["type"] == "ODOMETRY":
        fields = ("boot_s", "z", "reset_counter", "frame_id")
        if d["frame_id"] != 1:
            raise ValueError("odometry is not LOCAL_NED")
    if not all(math.isfinite(float(d[f])) for f in fields):
        raise ValueError("nonfinite telemetry")
    if not isinstance(p.get("quarantined"), bool) or not isinstance(p.get("phase"), str):
        raise ValueError("missing bridge state")
    return p


class PassiveTelemetry:
    def __init__(self, port=PORT):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # Intentionally exclusive: a second recorder must not steal packets.
            self.socket.bind(("127.0.0.1", port))
            self.socket.setblocking(False)
        except BaseException:
            self.socket.close()
            raise
        self.counts = Counter()
        self.last_received = None

    def drain(self, output=None):
        for _ in range(200):  # Bounded work per ROS spin, no blocking calls.
            try:
                raw, address = self.socket.recvfrom(4097)
            except BlockingIOError:
                return
            try:
                if address[0] != "127.0.0.1":
                    raise ValueError("nonlocal sender")
                p = validate_packet(raw)
                if abs(time.monotonic()-p["received_monotonic_s"]) > 1:
                    raise ValueError("stale diagnostic datagram")
                self.last_received = time.monotonic()
                self.counts[p["type"]] += 1
                if output is not None:
                    output.write(json.dumps(p, allow_nan=False, separators=(",", ":"))+"\n")
            except (ValueError, TypeError, KeyError):
                self.counts["invalid"] += 1

    def close(self):
        self.socket.close()
