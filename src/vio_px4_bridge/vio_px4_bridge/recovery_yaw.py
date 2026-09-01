"""Recovery-only agreement check for reset-prone VIO yaw."""

from __future__ import annotations

from dataclasses import dataclass
import math

from vio_px4_bridge.geo_utils import wrap_pi


@dataclass(frozen=True)
class RecoveryYawResult:
    agreed: bool
    evaluated: bool
    residual_rad: float | None
    good_samples: int
    event: str | None = None


class RecoveryYawAgreement:
    """Compare relative VIO yaw with independent PX4 yaw propagation.

    Absolute yaw is intentionally ignored: a new VIO epoch may start at any
    numerical angle.  The first paired sample establishes a baseline, and only
    subsequent yaw changes are compared.  A disagreement starts a fresh
    baseline so a transiently corrupt recovery cannot taint the stable window.
    """

    def __init__(self, max_residual_rad=math.radians(20.0), required_samples=5):
        self.max_residual_rad = max(math.radians(0.1), float(max_residual_rad))
        self.required_samples = max(1, int(required_samples))
        self.reset()

    def reset(self):
        self.baseline_vio_yaw = None
        self.baseline_px4_delta = None
        self.last_px4_timestamp = None
        self.good_samples = 0
        self.last_residual_rad = None

    def _set_baseline(self, vio_yaw, px4_yaw_delta, px4_timestamp):
        self.baseline_vio_yaw = wrap_pi(float(vio_yaw))
        self.baseline_px4_delta = wrap_pi(float(px4_yaw_delta))
        self.last_px4_timestamp = float(px4_timestamp)

    def update(self, vio_yaw, px4_yaw_delta, px4_timestamp):
        values = (vio_yaw, px4_yaw_delta, px4_timestamp)
        if not all(math.isfinite(float(value)) for value in values):
            self.reset()
            return RecoveryYawResult(False, False, None, 0, "nonfinite_reset")

        px4_timestamp = float(px4_timestamp)
        if self.baseline_vio_yaw is None:
            self._set_baseline(vio_yaw, px4_yaw_delta, px4_timestamp)
            return RecoveryYawResult(False, False, None, 0, "baseline_started")

        # VIO normally runs faster than PX4 ATTITUDE. Do not count the same
        # independent PX4 sample multiple times merely because VIO updated.
        if px4_timestamp == self.last_px4_timestamp:
            return RecoveryYawResult(
                self.good_samples >= self.required_samples,
                False,
                self.last_residual_rad,
                self.good_samples,
            )
        if px4_timestamp < self.last_px4_timestamp:
            self.reset()
            self._set_baseline(vio_yaw, px4_yaw_delta, px4_timestamp)
            return RecoveryYawResult(False, False, None, 0, "timestamp_reset")

        vio_change = wrap_pi(float(vio_yaw) - self.baseline_vio_yaw)
        px4_change = wrap_pi(float(px4_yaw_delta) - self.baseline_px4_delta)
        residual = wrap_pi(vio_change - px4_change)
        self.last_px4_timestamp = px4_timestamp
        self.last_residual_rad = residual

        if abs(residual) <= self.max_residual_rad:
            self.good_samples += 1
            event = (
                "agreement_confirmed"
                if self.good_samples == self.required_samples else None
            )
            return RecoveryYawResult(
                self.good_samples >= self.required_samples,
                True,
                residual,
                self.good_samples,
                event,
            )

        # Begin a clean comparison window at the disagreeing sample. This lets
        # recovery proceed only after the source becomes consistently healthy.
        self._set_baseline(vio_yaw, px4_yaw_delta, px4_timestamp)
        self.good_samples = 0
        return RecoveryYawResult(False, True, residual, 0, "disagreement_reset")
