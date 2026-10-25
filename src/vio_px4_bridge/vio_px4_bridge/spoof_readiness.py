"""Recent-data readiness for the stationary spoof-to-live handoff, not flight control."""
from collections import deque
import math


class SampleWindow:
    def __init__(self, gap_s, minimum_samples, duration_s=2.0):
        self.gap_s = gap_s
        self.minimum_samples = minimum_samples
        self.duration_s = duration_s
        self.samples = deque(maxlen=4096)
        self.last_issue = "waiting for samples"

    def reset(self, reason="waiting for samples"):
        self.samples.clear()
        self.last_issue = reason

    def observe(self, stamp_ns, received, value=0.0, valid=True):
        if not valid or stamp_ns <= 0 or not math.isfinite(value):
            self.reset("invalid sample or frame")
            return
        if self.samples:
            previous = self.samples[-1]
            dt = (stamp_ns - previous[0]) * 1e-9
            delivery_dt = received - previous[1]
            if dt <= 0 or delivery_dt < 0:
                self.reset(f"timestamp discontinuity ({dt:.6f}s)")
                return  # Never count the offending sample as healthy evidence.
            if max(dt, delivery_dt) > self.gap_s:
                self.reset(f"sample gap (sensor={dt:.3f}s, delivery={delivery_dt:.3f}s)")
        self.samples.append((stamp_ns, received, value))
        while len(self.samples) > self.minimum_samples:
            second = self.samples[1]
            if min((stamp_ns - second[0]) * 1e-9, received - second[1]) < self.duration_s:
                break
            self.samples.popleft()

    def reason(self, now):
        if not self.samples:
            return self.last_issue
        if now - self.samples[-1][1] > self.gap_s:
            return "data stale"
        sensor_span = (self.samples[-1][0] - self.samples[0][0]) * 1e-9
        arrival_span = self.samples[-1][1] - self.samples[0][1]
        if (len(self.samples) < self.minimum_samples
                or min(sensor_span, arrival_span) < self.duration_s):
            return f"collecting {self.duration_s:g}s healthy window ({self.last_issue})"
        return None


class SpoofReadiness:
    """Startup faults reset evidence, not the camera. Deadline is owned by bridge."""
    def __init__(self, require_imu=True):
        self.require_imu = require_imu
        self.imu = SampleWindow(0.1, 300)
        self.odom = SampleWindow(0.5, 20)
        self.fusion = None
        self.fusion_time = None

    def reset(self):
        # Deliberately keep the separately refreshed fusion-mode response.
        self.imu.reset()
        self.odom.reset()

    def observe_imu(self, stamp_ns, now, values, frame):
        valid = (frame == "camera_imu_optical_frame"
                 and all(math.isfinite(v) for v in values))
        norm = math.hypot(*values[:3]) if valid else 0.0
        self.imu.observe(stamp_ns, now, norm, valid)

    def observe_odom(self, stamp_ns, now, values, frame, expected_frame):
        valid = (frame == expected_frame and all(math.isfinite(v) for v in values)
                 and 0.9 < math.hypot(*values[3:7]) < 1.1)
        self.odom.observe(stamp_ns, now, valid=valid)

    def observe_fusion(self, enabled, now):
        self.fusion = enabled
        self.fusion_time = now

    def reason(self, now):
        if self.fusion_time is None or now - self.fusion_time > 3.0:
            return "camera fusion setting not confirmed recently"
        if self.fusion is not self.require_imu:
            return "camera fusion setting differs from selected VO/VIO mode"
        reason = self.odom.reason(now)
        if reason:
            return f"odometry: {reason}"
        if self.require_imu:
            reason = self.imu.reason(now)
            if reason:
                return f"IMU: {reason}"
            gravity = sum(s[2] for s in self.imu.samples) / len(self.imu.samples)
            if not 7 < gravity < 12:
                return "IMU acceleration inconsistent with stationary startup"
        return None
