"""Legacy rotation-evidence experiment; NOT used by the active GPS bridge.

The active recovery policy trusts current FCU heading after fresh agreement
windows. This helper is retained for comparison, not as a recovery gate.

ODOMETRY's counter does not identify the state that reset. Compare its attitude
with body-gyro rotation at the same sample timestamps, including tilt. This is
consistency evidence, not an absolute heading sensor or a reset-type decoder.
The sum of unexplained rotation is bounded for the entire quarantine; restarting
VIO candidates or resetting PX4 position never replenishes that budget.
"""
import math

from vio_px4_bridge.recovery_yaw import timestamp_order


def multiply(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return (w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
            w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v)


def unit_quaternion(q):
    if len(q) != 4 or not all(math.isfinite(x) for x in q):
        raise ValueError('invalid quaternion')
    norm = math.sqrt(sum(x*x for x in q))
    if norm < 1e-6:
        raise ValueError('zero quaternion')
    return tuple(x/norm for x in q)


class ResetHeadingEvidence:
    def __init__(self, max_unexplained_rad, max_gap_s, reorder_tolerance_s):
        self.limit = max_unexplained_rad
        self.max_gap = max_gap_s
        self.reorder_tolerance = reorder_tolerance_s
        self.last = None
        self.failure = None
        self.error_sum = 0.0
        self.intervals = 0

    def reject(self, reason):
        if self.failure is None:
            self.failure = reason

    def observe(self, sample):
        """sample = (arrival monotonic s, PX4 sample s, q wxyz, body rates)."""
        if self.failure is not None:
            return False
        arrival, stamp, q, rates = sample
        try:
            q = unit_quaternion(q)
            if len(rates) != 3 or not all(math.isfinite(v) for v in (arrival, stamp, *rates)):
                raise ValueError('nonfinite timestamp/rates')
        except (ValueError, TypeError):
            self.reject('invalid_rotation_evidence')
            return False
        if self.last is not None:
            old_arrival, old_stamp, old_q, old_rates = self.last
            dt = stamp - old_stamp
            order = timestamp_order(dt, self.reorder_tolerance)
            if order == 'reordered':
                return False
            if order != 'forward':
                self.reject('rotation_timestamp_regression')
                return False
            if dt > self.max_gap or arrival - old_arrival > self.max_gap:
                self.reject('rotation_evidence_gap')
                return False
            # Rates are body-FRD. Right-composition advances body orientation
            # in NED; it is not a rotation around the camera or world axes.
            rotation = tuple(.5*(a+b)*dt for a, b in zip(old_rates, rates))
            angle = math.sqrt(sum(v*v for v in rotation))
            scale = .5 if angle < 1e-9 else math.sin(angle/2)/angle
            dq = (math.cos(angle/2), *(v*scale for v in rotation))
            predicted = unit_quaternion(multiply(old_q, dq))
            residual = multiply((predicted[0], *(-v for v in predicted[1:])), q)
            unexplained = 2*math.atan2(math.sqrt(sum(v*v for v in residual[1:])),
                                     abs(residual[0]))
            self.error_sum += unexplained
            self.intervals += 1
            if self.error_sum > self.limit:
                self.reject('unexplained_rotation_budget_exceeded')
                return False
        self.last = (arrival, stamp, q, tuple(rates))
        return True

    def verifies(self, stamp, now, max_age):
        return (self.failure is None and self.last is not None and self.intervals > 0
                and self.last[1] >= stamp and 0 <= now-self.last[0] <= max_age)
