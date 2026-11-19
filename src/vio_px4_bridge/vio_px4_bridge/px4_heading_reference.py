"""Bounded, time-matched PX4 attitude history. Never use receipt time as capture time.

TIMESYNC follows https://mavlink.io/en/services/timesync.html. Only replies to
our own outstanding tokens are accepted. The caller filters system/component.
No gyro integration, heading output, or ROS dependency. Cached position is
available only to establish the one-time quarantine recovery boundary.
"""
from collections import deque
import math
from statistics import median

from vio_px4_bridge import frame_alignment as frames
from vio_px4_bridge.geo_utils import wrap_pi


class ClockSync:
    def __init__(self, max_skew_s=0.03):
        self.max_skew_s = max_skew_s
        self.pending = {}
        self.estimates = deque(maxlen=20)
        self.offset = self.last_reply = None
        self.generation = 0

    def clear(self):
        self.pending.clear()
        self.estimates.clear()
        self.offset = self.last_reply = None
        self.generation += 1

    def request(self, now):
        self.pending = {k: v for k, v in self.pending.items() if 0 <= now-v <= 2*self.max_skew_s}
        token = int(now*1e9)
        self.pending[token] = now
        return token

    def observe(self, tc1, ts1, now):
        sent = self.pending.pop(ts1, None)
        if sent is None or tc1 <= 0 or not math.isfinite(now):
            return False
        rtt = now-sent
        if not 0 <= rtt <= 2*self.max_skew_s:
            return False
        offset = (sent+now)*.5-tc1*1e-9
        if self.offset is not None and abs(offset-self.offset) > 2*self.max_skew_s:
            self.clear()
        self.estimates.append((offset, rtt*.5))
        self.offset = median(v[0] for v in self.estimates)
        self.last_reply = now
        return True

    def valid(self, now):
        return (len(self.estimates) >= 5 and self.last_reply is not None
                and 0 <= now-self.last_reply <= 2.0
                and max(abs(v-self.offset)+u for v, u in self.estimates) <= self.max_skew_s)


def interpolate(a, b, t):
    """Shortest-arc quaternion SLERP, including q/-q and yaw wrap."""
    a, b = frames.normalize(a), frames.normalize(b)
    dot = sum(x*y for x, y in zip(a, b))
    if dot < 0:
        b, dot = tuple(-v for v in b), -dot
    dot = min(1.0, dot)
    if dot > .9995:
        return frames.normalize(tuple((1-t)*x+t*y for x, y in zip(a, b)))
    angle = math.acos(dot)
    return frames.normalize(tuple((math.sin((1-t)*angle)*x+math.sin(t*angle)*y)/math.sin(angle)
                                  for x, y in zip(a, b)))


class Px4HeadingReference:
    def __init__(self, max_age_s=.25, max_gap_s=.25, max_skew_s=.03,
                 max_rotation_rate=math.tau, reorder_s=.05):
        if not all(math.isfinite(v) and v > 0 for v in
                   (max_age_s, max_gap_s, max_skew_s, max_rotation_rate)):
            raise ValueError("heading timing/rate limits must be finite and positive")
        if not 0 < max_skew_s <= .1 or max_skew_s > max_age_s:
            raise ValueError("heading time-match tolerance must be <=0.1 s and message age")
        self.max_age, self.max_gap, self.max_skew = max_age_s, max_gap_s, max_skew_s
        self.max_rotation_rate, self.reorder_s = max_rotation_rate, reorder_s
        self.clock = ClockSync(max_skew_s)
        self.history = deque(maxlen=256)
        self.reason = "waiting_for_px4_heading"
        self.last_ros_offset = None
        self.last_match = None
        self.generation = 0

    def invalidate(self, reason):
        self.generation += 1
        self.history.clear()
        self.last_match = None
        self.reason = reason

    def observe(self, arrival, boot_time, orientation, reset_counter, motion=None):
        """Return a discontinuity reason requiring quarantine, or None."""
        try:
            q = frames.normalize(orientation)
            if not all(math.isfinite(v) for v in (arrival, boot_time)) or boot_time <= 0:
                raise ValueError("invalid time")
        except ValueError:
            self.invalidate("invalid_px4_heading")
            return self.reason
        reason = None
        if self.history:
            old = self.history[-1]
            dt = boot_time-old[1]
            if -self.reorder_s <= dt <= 0 and reset_counter == old[3]:
                return None  # duplicates/reordering must never refresh freshness
            if dt < 0:
                self.clock.clear()
                reason = "px4_heading_clock_regression"
            elif reset_counter != old[3]:
                reason = "px4_estimator_reset"
            elif dt > self.max_gap:
                reason = "px4_heading_message_gap"
            elif frames.angle_between(old[2], q)/dt > self.max_rotation_rate:
                reason = "px4_attitude_rate_limit"
        if reason:
            self.invalidate(reason)
        self.history.append((arrival, boot_time, q, reset_counter, motion))
        # Keep enough support for any still-fresh camera/recovery boundary,
        # rather than scanning the full 256-packet memory cap each callback.
        oldest = boot_time-self.max_age-self.max_gap-2*self.max_skew
        while len(self.history) > 2 and self.history[1][1] < oldest:
            self.history.popleft()
        return reason

    def motion_at(self, monotonic_stamp, now):
        """Recovery boundary only; never use PX4 position in healthy tracking."""
        if monotonic_stamp is None or not self.fresh(now):
            return None
        stamp = monotonic_stamp-self.clock.offset
        for a, b in zip(self.history, list(self.history)[1:]):
            dt = b[1]-a[1]
            if (a[1] <= stamp <= b[1] and 0 < dt <= self.max_gap and a[3] == b[3]
                    and a[4] is not None and b[4] is not None):
                t, x, y = (stamp-a[1])/dt, a[4], b[4]
                return (monotonic_stamp, stamp, *(x[i]+t*(y[i]-x[i]) for i in range(2, 6)),
                        max(x[6], y[6]), max(x[7], y[7]), b[3])
        return None

    def fresh(self, now):
        return (self.clock.valid(now) and bool(self.history)
                and 0 <= now-self.history[-1][0] <= self.max_age
                # The clock offset is estimated with bounded uncertainty, not
                # exact. A few ms apparently in the future is not a reset.
                and -self.max_skew <= now-(self.history[-1][1]+self.clock.offset) <= self.max_age)

    def match(self, camera_stamp, now_ros, now):
        self.last_match = None
        if not all(math.isfinite(v) for v in (camera_stamp, now_ros, now)) or camera_stamp <= 0:
            self.reason = "invalid_camera_timestamp"
            return None
        ros_offset = now_ros-now
        if self.last_ros_offset is not None and abs(ros_offset-self.last_ros_offset) > 2*self.max_skew:
            self.last_ros_offset = ros_offset
            self.invalidate("camera_clock_changed")
            return None
        self.last_ros_offset = ros_offset
        age = now_ros-camera_stamp
        if not 0 <= age <= self.max_age:
            self.reason = "camera_timestamp_stale_or_future"
            return None
        if not self.clock.valid(now):
            self.reason = "waiting_for_time_sync"
            return None
        if not self.fresh(now):
            self.reason = "px4_heading_stale"
            return None
        stamp = now-age-self.clock.offset
        for a, b in zip(self.history, list(self.history)[1:]):
            dt = b[1]-a[1]
            if a[1] <= stamp <= b[1] and 0 < dt <= self.max_gap and a[3] == b[3]:
                if min(stamp-a[1], b[1]-stamp) > self.max_skew:
                    break
                q = interpolate(a[2], b[2], (stamp-a[1])/dt)
                self.last_match = (stamp, q)
                self.reason = "matched"
                return frames.yaw(q)
        self.reason = "waiting_for_time_matched_heading"
        return None
