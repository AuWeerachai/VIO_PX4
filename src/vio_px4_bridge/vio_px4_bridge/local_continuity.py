"""Confirmed-reset continuity gate for VIO sources that can relocalize."""
from __future__ import annotations
from dataclasses import dataclass
import math
from vio_px4_bridge.geo_utils import wrap_pi

@dataclass(frozen=True)
class ContinuityResult:
    position: tuple[float, float, float]
    yaw: float
    velocity: tuple[float, float, float]
    recovering: bool
    reanchored: bool
    reason: str | None
    epoch: int
    event: str | None = None
    detail: str | None = None

class LocalPoseContinuity:
    """Map reset-prone VIO into a continuous frame using a quarantine gate."""
    def __init__(self, max_gap_s=1.0,
                 recovery_samples=10, confirmation_samples=3,
                 max_speed_mps=10.0, max_acceleration_mps2=5.0,
                 max_yaw_rate_rad_s=math.radians(360.0)):
        self.max_gap_s = max(0.05, max_gap_s)
        self.recovery_samples = max(1, recovery_samples)
        self.confirmation_samples = max(2, confirmation_samples)
        self.max_speed_mps = max(0.1, max_speed_mps)
        self.max_acceleration_mps2 = max(0.1, max_acceleration_mps2)
        self.max_yaw_rate_rad_s = max(math.radians(1.0), max_yaw_rate_rad_s)
        self.reset()

    def reset(self):
        self.rotation = 0.0
        self.translation = (0.0, 0.0, 0.0)
        self.last_raw_position = self.last_raw_yaw = None
        self.last_raw_velocity = self.last_yaw_rate = None
        self.last_timestamp = self.last_frame_key = None
        self.last_output_position = self.last_output_yaw = None
        self.stable_samples = self.epoch = 0
        self._clear_suspect()

    def _clear_suspect(self):
        self.suspect_reason = None
        self.suspect_detail = None
        self.suspect_samples = []

    @staticmethod
    def _rotate_xy(x, y, angle):
        c, s = math.cos(angle), math.sin(angle)
        return c*x-s*y, s*x+c*y

    @staticmethod
    def _norm(values):
        return math.sqrt(sum(x*x for x in values))

    def _transform_position(self, p):
        x, y = self._rotate_xy(p[0], p[1], self.rotation)
        return x+self.translation[0], y+self.translation[1], p[2]+self.translation[2]

    def _transform_velocity(self, v):
        x, y = self._rotate_xy(v[0], v[1], self.rotation)
        return x, y, v[2]

    def _motion_reason(self, old, new, enforce_gap=True):
        p0, yaw0, v0, rate0, t0, frame0 = old
        p1, yaw1, v1, rate1, t1, frame1 = new
        dt = t1-t0
        if dt <= 0: return "timestamp_regression"
        if frame1 != frame0: return "frame_changed"
        if enforce_gap and dt > self.max_gap_s: return "tracking_gap"
        if self._norm(v1) > self.max_speed_mps: return "speed_limit"
        if self._norm(tuple((v1[i]-v0[i])/dt for i in range(3))) > self.max_acceleration_mps2:
            return "acceleration_limit"
        if abs(rate1) > self.max_yaw_rate_rad_s: return "yaw_rate_limit"
        # Derive pose-change limits from the operator's physical rate limits
        # and the actual time between samples. This independently catches a
        # pose jump even when the velocity/rate fields remain plausible.
        delta = tuple(p1[i]-p0[i] for i in range(3))
        if self._norm(delta) / dt > self.max_speed_mps:
            return "position_jump"
        if abs(wrap_pi(yaw1-yaw0)) / dt > self.max_yaw_rate_rad_s:
            return "yaw_jump"
        return None

    def _gate_detail(self, old, new, reason):
        p0, yaw0, v0, rate0, t0, frame0 = old
        p1, yaw1, v1, rate1, t1, frame1 = new
        dt = t1-t0
        if reason == "timestamp_regression":
            return f"dt_s={dt:.6f} limit=>0"
        if reason == "frame_changed":
            return f"old_frame={frame0!r} new_frame={frame1!r}"
        if reason == "tracking_gap":
            return f"gap_s={dt:.3f} limit_s={self.max_gap_s:.3f}"
        if reason == "speed_limit":
            return f"speed_m_s={self._norm(v1):.3f} limit_m_s={self.max_speed_mps:.3f}"
        if reason == "acceleration_limit":
            accel = self._norm(tuple((v1[i]-v0[i])/dt for i in range(3)))
            return f"acceleration_m_s2={accel:.3f} limit_m_s2={self.max_acceleration_mps2:.3f}"
        if reason == "yaw_rate_limit":
            return (f"yaw_rate_deg_s={math.degrees(abs(rate1)):.3f} "
                    f"limit_deg_s={math.degrees(self.max_yaw_rate_rad_s):.3f}")
        if reason == "position_jump":
            delta = tuple(p1[i]-p0[i] for i in range(3))
            implied_speed = self._norm(delta) / dt
            return (f"pose_implied_speed_m_s={implied_speed:.3f} "
                    f"limit_m_s={self.max_speed_mps:.3f} dt_s={dt:.3f}")
        if reason == "yaw_jump":
            implied_rate = abs(wrap_pi(yaw1-yaw0)) / dt
            return (f"pose_implied_yaw_rate_deg_s={math.degrees(implied_rate):.3f} "
                    f"limit_deg_s={math.degrees(self.max_yaw_rate_rad_s):.3f} "
                    f"dt_s={dt:.3f}")
        return "explicit_source_event=true"

    def _accepted_sample(self):
        if self.last_timestamp is None: return None
        return (self.last_raw_position, self.last_raw_yaw, self.last_raw_velocity,
                self.last_yaw_rate, self.last_timestamp, self.last_frame_key)

    def _held(self, reason, event=None, detail=None):
        return ContinuityResult(self.last_output_position, self.last_output_yaw,
                                (0.0, 0.0, 0.0), True, False, reason, self.epoch,
                                event, detail)

    def _accept(self, sample, reanchored=False, reason=None, event=None,
                detail=None):
        p, yaw, v, rate, timestamp, frame = sample
        out_p, out_yaw, out_v = self._transform_position(p), wrap_pi(yaw+self.rotation), self._transform_velocity(v)
        self.last_raw_position, self.last_raw_yaw, self.last_raw_velocity = p, yaw, v
        self.last_yaw_rate, self.last_timestamp, self.last_frame_key = rate, timestamp, frame
        self.last_output_position, self.last_output_yaw = out_p, out_yaw
        if self.epoch and not reanchored: self.stable_samples += 1
        recovering = bool(self.epoch and self.stable_samples < self.recovery_samples)
        return ContinuityResult(out_p, out_yaw, out_v, recovering, reanchored,
                                reason, self.epoch, event, detail)

    def update(self, position, yaw, velocity, yaw_rate, timestamp, frame_key,
               force_reanchor_reason=None):
        sample = (position, yaw, velocity, yaw_rate, timestamp, frame_key)
        accepted = self._accepted_sample()
        if accepted is None: return self._accept(sample)

        if force_reanchor_reason:
            self.suspect_reason, self.suspect_samples = force_reanchor_reason, [sample]
            self.suspect_detail = "explicit_source_event=true"
            if self.confirmation_samples > 1:
                return self._held(force_reanchor_reason, "quarantine_started",
                                  self.suspect_detail)
        elif not self.suspect_samples:
            reason = self._motion_reason(accepted, sample)
            if reason is None: return self._accept(sample)
            self.suspect_reason, self.suspect_samples = reason, [sample]
            self.suspect_detail = self._gate_detail(accepted, sample, reason)
            return self._held(reason, "quarantine_started", self.suspect_detail)
        else:
            # Returning to the accepted trajectory means the prior point was
            # an isolated outlier; discard it without moving the origin.
            if self._motion_reason(accepted, sample, enforce_gap=False) is None:
                old_reason, old_detail = self.suspect_reason, self.suspect_detail
                self._clear_suspect()
                return self._accept(sample, reason=old_reason,
                                    event="isolated_outlier_rejected",
                                    detail=old_detail)
            candidate_reason = self._motion_reason(self.suspect_samples[-1], sample)
            if candidate_reason is None:
                self.suspect_samples.append(sample)
            else:
                self.suspect_reason = candidate_reason
                self.suspect_detail = self._gate_detail(
                    self.suspect_samples[-1], sample, candidate_reason
                )
                self.suspect_samples = [sample]
                return self._held(candidate_reason, "candidate_window_restarted",
                                  self.suspect_detail)

        if len(self.suspect_samples) < self.confirmation_samples:
            return self._held(self.suspect_reason or "suspect")

        # Anchor the first sample of the confirmed new epoch to the last good
        # output. Motion accumulated during confirmation is then retained.
        anchor_p, anchor_yaw = self.suspect_samples[0][0:2]
        self.rotation = wrap_pi(self.last_output_yaw-anchor_yaw)
        rx, ry = self._rotate_xy(anchor_p[0], anchor_p[1], self.rotation)
        self.translation = (self.last_output_position[0]-rx,
                            self.last_output_position[1]-ry,
                            self.last_output_position[2]-anchor_p[2])
        reason, detail = self.suspect_reason, self.suspect_detail
        self._clear_suspect()
        self.epoch += 1
        self.stable_samples = 0
        return self._accept(sample, True, reason, "reset_confirmed", detail)
