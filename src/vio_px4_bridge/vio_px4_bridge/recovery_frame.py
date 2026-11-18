"""Experimental, recovery-only full attitude alignment and timestamp pairing.

No ROS, MAVLink transport, publishing, camera extrinsics or estimator fusion.
PX4 time is mapped to companion monotonic time using echoed TIMESYNC requests.
Only interpolation is permitted: never pretend arrival time is capture time.
"""
from collections import deque
from dataclasses import dataclass
import math
from statistics import median

from vio_px4_bridge import frame_rotation as rot


class ClockSync:
    def __init__(self, max_rtt=.05, max_age=2.0, max_uncertainty=.03):
        self.max_rtt, self.max_age = max_rtt, max_age
        self.max_uncertainty = max_uncertainty
        self.pending = {}
        self.estimates = deque(maxlen=20)
        self.last_reply = None
        self.offset = None  # companion monotonic seconds minus PX4 boot seconds
        self.generation = 0

    def request(self, now):
        self.pending = {key: value for key, value in self.pending.items()
                        if 0 <= now-value <= self.max_rtt}
        token = int(now*1e9)
        self.pending[token] = now
        return token

    def observe(self, tc1, ts1, now):
        sent = self.pending.pop(ts1, None)
        if sent is None or tc1 <= 0 or not math.isfinite(now):
            return False
        rtt = now-sent
        if not 0 <= rtt <= self.max_rtt:
            return False
        offset = (sent+now)/2 - tc1*1e-9
        if self.offset is not None and abs(offset-self.offset) > self.max_uncertainty*2:
            self.estimates.clear()
            self.offset = None
            self.generation += 1
        self.estimates.append((offset, rtt/2))
        self.offset = median(x[0] for x in self.estimates)
        self.last_reply = now
        return True

    def valid(self, now):
        return (len(self.estimates) >= 5 and self.last_reply is not None
                and 0 <= now-self.last_reply <= self.max_age
                and max(abs(x-self.offset)+bound for x, bound in self.estimates)
                <= self.max_uncertainty)

    def boot_time(self, stamp_ros, now_ros, now):
        if stamp_ros is None:
            return None
        if not self.valid(now) or not all(math.isfinite(x) for x in (stamp_ros, now_ros, now)):
            return None
        return now-(now_ros-stamp_ros)-self.offset


@dataclass(frozen=True)
class FramePair:
    boot_time: float
    attitude: tuple
    attitude_key: float
    motion: tuple
    motion_key: float


class RecoveryFrame:
    def __init__(self, max_age=.25, max_gap=.25, max_skew=.03,
                 max_body_tilt=math.radians(10), agreement=math.radians(5),
                 samples=5, max_rtt=.05, sync_max_age=2.0):
        limits = (max_age, max_gap, max_skew, max_body_tilt, agreement,
                  max_rtt, sync_max_age)
        if not all(math.isfinite(v) and v > 0 for v in limits) or samples < 1:
            raise ValueError("3D recovery limits must be positive and finite")
        if max_body_tilt >= math.pi/2 or agreement >= math.pi/2:
            raise ValueError("3D recovery tilt/tolerance must be below 90 degrees")
        self.max_age, self.max_gap, self.max_skew = max_age, max_gap, max_skew
        self.max_body_tilt, self.agreement, self.samples = max_body_tilt, agreement, samples
        self.clock = ClockSync(max_rtt, sync_max_age, max_skew)
        self.attitudes = deque(maxlen=256)
        self.motions = deque(maxlen=256)
        self.reset()

    def reset(self, reason="waiting_for_paired_attitude"):
        self.candidate = None
        self.good_samples = 0
        self.last_key = None
        self.last_pair_received = None
        self.pair = None
        self.residual = None
        self.reason = reason

    def clear_history(self, reason):
        self.attitudes.clear()
        self.motions.clear()
        self.reset(reason)

    def observe_attitude(self, received, boot_time, q):
        q = rot.normalize(q)
        if not all(math.isfinite(x) for x in (received, boot_time)) or boot_time < 0:
            self.reset("invalid_fcu_attitude")
            return
        if self.attitudes and boot_time <= self.attitudes[-1][1]:
            return  # repeated/reordered packets never refresh freshness/evidence
        self.attitudes.append((received, boot_time, q))

    def observe_motion(self, sample):
        if self.motions and sample[1] <= self.motions[-1][1]:
            return
        if self.motions and sample[8] != self.motions[-1][8]:
            self.motions.clear()
            self.attitudes.clear()
            self.reset("px4_estimator_reset")
        self.motions.append(sample)

    def _bracket(self, history, stamp, now):
        if not history or not 0 <= now-history[-1][0] <= self.max_age:
            return None
        for previous, current in zip(history, list(history)[1:]):
            dt = current[1]-previous[1]
            if previous[1] <= stamp <= current[1] and 0 < dt <= self.max_gap:
                # Do not interpolate across a long hole even if the stream resumed.
                if min(stamp-previous[1], current[1]-stamp) > self.max_skew:
                    return None
                return previous, current, (stamp-previous[1])/dt
        return None

    def motion_at(self, stamp, now):
        bracket = self._bracket(self.motions, stamp, now)
        if bracket is None:
            return None
        a, b, t = bracket
        if a[8] != b[8]:
            return None
        return ((now, stamp, *(a[i]+t*(b[i]-a[i]) for i in range(2, 6)),
                 max(a[6], b[6]), max(a[7], b[7]), b[8]), b[1])

    def telemetry_fresh(self, now):
        return (self.clock.valid(now) and bool(self.attitudes) and bool(self.motions)
                and all(0 <= now-h[-1][0] <= self.max_age
                        for h in (self.attitudes, self.motions)))

    def waiting_for_next_pair(self, now):
        # Camera and FCU run at different rates. A missing upper interpolation
        # sample is not evidence of disagreement. Preserve the window briefly,
        # but NEVER authorize handoff without a new matched sample.
        return (self.reason == "waiting_for_time_matched_px4"
                and self.telemetry_fresh(now)
                and self.last_pair_received is not None
                and 0 <= now-self.last_pair_received <= self.max_gap)

    def match(self, stamp_ros, now_ros, now):
        self.pair = None
        if not 0 <= now_ros-stamp_ros <= self.max_age:
            self.reason = "camera_timestamp_not_fresh"
            return None
        stamp = self.clock.boot_time(stamp_ros, now_ros, now)
        if stamp is None:
            self.reason = "waiting_for_time_sync"
            return None
        attitude = self._bracket(self.attitudes, stamp, now)
        motion = self.motion_at(stamp, now)
        if attitude is None or motion is None:
            self.reason = "waiting_for_time_matched_px4"
            return None
        a, b, t = attitude
        self.pair = FramePair(stamp, rot.interpolate(a[2], b[2], t), b[1], *motion)
        return self.pair

    def check(self, raw_q, target_local_q, pair):
        if pair is None:
            self.reset(self.reason)
            return False
        if raw_q is None or target_local_q is None:
            self.reset("orientation_alignment_unavailable")
            return False
        if rot.tilt(pair.attitude) > self.max_body_tilt:
            self.reset("fcu_not_nearly_level")
            self.pair = pair
            return False
        correction = rot.multiply(target_local_q, rot.inverse(raw_q))
        if self.last_key is not None and pair.attitude_key < self.last_key:
            self.reset("paired_attitude_timestamp_regressed")
        self.last_pair_received = pair.motion[0]
        if self.candidate is None:
            self.candidate, self.last_key = correction, pair.attitude_key
            self.reason = "collecting_3d_attitude_agreement"
            return False
        self.residual = rot.distance(correction, self.candidate)
        # Validate against a FIXED candidate, never fit away each new error.
        if self.residual > self.agreement:
            self.reset("3d_attitude_disagreement")
            self.candidate, self.last_key = correction, pair.attitude_key
            self.pair = pair
            return False
        if pair.attitude_key != self.last_key:
            self.good_samples += 1
            self.last_key = pair.attitude_key
        self.reason = ("3d_attitude_ready" if self.good_samples >= self.samples
                       else "collecting_3d_attitude_agreement")
        return self.good_samples >= self.samples
