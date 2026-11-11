"""Stationary startup alignment to PX4 ATTITUDE yaw (earth NED, radians).

This checks telemetry consistency, not whether PX4 has found true north.
ATTITUDE has no yaw-aligned flag; even ESTIMATOR_ATTITUDE in the deployed
PX4 only certifies tilt alignment. Operators must verify PX4 heading first.
"""

import math

from vio_px4_bridge.geo_utils import wrap_pi


def heading_source(value):
    """Migrate the former automatic compass mode; preserve explicit manual mode."""
    source = str(value).strip().lower()
    if source == "compass":
        source = "px4"
    if source not in ("px4", "manual"):
        raise ValueError("heading_source must be 'px4' or 'manual'")
    return source


class Px4StartupHeading:
    def __init__(self, samples=30, max_std_rad=math.radians(5),
                 max_speed=0.25, max_age=0.5):
        self.required_samples = samples
        self.max_std_rad = max_std_rad
        self.max_speed = max_speed
        self.max_age = max_age
        self.latest = None
        self.last_boot_time = None
        self.last_consumed = None
        self.reset()

    def reset(self, reason="waiting for fresh PX4 heading samples"):
        self.samples = []
        self.offset = None
        self.reason = reason
        # Do not reuse the latest packet after an RC activation/local reset.
        self.last_consumed = self.last_boot_time

    def observe(self, now, boot_time, yaw):
        if not all(math.isfinite(v) for v in (now, boot_time, yaw)) or boot_time < 0:
            self.latest = None
            self.reset("invalid PX4 attitude")
            return
        if self.last_boot_time is not None and boot_time <= self.last_boot_time:
            if boot_time < self.last_boot_time:
                self.latest = None
                self.last_boot_time = boot_time
                self.reset("PX4 attitude timestamp regressed; collecting new samples")
            # A repeated packet must not refresh the receive time or count twice.
            return
        if self.latest is not None:
            received, stamp, _ = self.latest
            if not 0 <= now - received <= self.max_age or boot_time - stamp > self.max_age:
                self.reset("PX4 attitude gap; collecting new samples")
        self.last_boot_time = boot_time
        self.latest = (now, boot_time, wrap_pi(yaw))

    def fresh(self, now):
        return (self.latest is not None
                and 0 <= now - self.latest[0] <= self.max_age)

    def update(self, vio_yaw_ned, horizontal_speed, now):
        if not self.fresh(now):
            self.reset("PX4 attitude missing or stale")
            return None
        if (not all(math.isfinite(v) for v in (vio_yaw_ned, horizontal_speed))
                or not 0 <= horizontal_speed <= self.max_speed):
            self.reset("keep vehicle stationary for PX4 heading alignment")
            return None
        _, boot_time, yaw = self.latest
        if boot_time == self.last_consumed:
            return self.offset
        self.last_consumed = boot_time
        candidate = wrap_pi(yaw - vio_yaw_ned)
        # A startup recalibration/reset must not leave an old locked offset.
        # The bridge stops calling this method once LIVE; normal tracking is fixed.
        if self.offset is not None:
            if abs(wrap_pi(candidate - self.offset)) <= self.max_std_rad:
                return self.offset
            self.reset("PX4/VIO heading changed before handoff; realigning")
        self.samples.append(candidate)
        if len(self.samples) < self.required_samples:
            self.reason = f"collecting PX4 heading samples {len(self.samples)}/{self.required_samples}"
            return None
        s = sum(math.sin(x) for x in self.samples) / len(self.samples)
        c = sum(math.cos(x) for x in self.samples) / len(self.samples)
        std = math.sqrt(max(0.0, -2 * math.log(max(math.hypot(s, c), 1e-12))))
        if std > self.max_std_rad:
            self.reset("PX4/VIO heading offset unstable; collecting new samples")
            return None
        self.offset = math.atan2(s, c)
        self.reason = "aligned to PX4 heading"
        return self.offset
