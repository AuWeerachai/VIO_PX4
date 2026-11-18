"""Optional yaw transmitter. It has no authority over GPS or pose continuity.

Only accepted, corrected drone_link poses enter this class. Timing failures
pause EV packets; they never declare a coordinate reset. MAVLink timestamps
stay in companion monotonic time, which PX4 maps through TIMESYNC.
"""
from collections import deque
import math
from statistics import median


def rpy(q):
    """Hamilton wxyz body-to-world rotation; reject undefined heading."""
    if len(q) != 4 or not all(math.isfinite(v) for v in q):
        raise ValueError("invalid quaternion")
    norm = math.sqrt(sum(v*v for v in q))
    if not .99 <= norm <= 1.01:
        raise ValueError("nonunit quaternion")
    w, x, y, z = (v/norm for v in q)
    pitch = math.asin(max(-1., min(1., 2*(w*y-z*x))))
    if abs(math.cos(pitch)) < .1:
        raise ValueError("heading undefined near vertical attitude")
    return (math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)), pitch,
            math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)))


def corrected_quaternion(q_xyzw, yaw):
    """ENU/FLU -> NED/FRD, retaining VO tilt and using corrected earth yaw.

    Camera extrinsics are already applied by cuVSLAM. Do not apply them again.
    """
    x, y, z, w = q_xyzw
    roll, pitch, _ = rpy((w, x, y, z))
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    cp, sp = math.cos(-pitch/2), math.sin(-pitch/2)
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    return (cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
            cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy)


class ClockSync:
    """Bidirectional MAVLink TIMESYNC, isolated from navigation state.

    The deployed PX4 Timesync filter requires 500 accepted exchanges. Allow
    600 prompt replies before EV starts; this is a conservative warmup, not
    a remote convergence ACK. An unavailable clock only pauses EV output.
    """
    MAX_RTT_S = .010
    WARMUP_REPLIES = 600

    def __init__(self):
        self.offsets = deque(maxlen=20)
        self.pending = {}
        self.last_good = self.last_request = self.last_px4_request = None
        self.replies = 0
        self.generation = 0

    def reset(self):
        generation = self.generation + 1
        self.__init__()
        self.generation = generation

    @property
    def offset(self):
        return None if not self.offsets else median(self.offsets)

    def request(self, now):
        if self.last_request is not None and now-self.last_request < .1:
            return None
        self.last_request = now
        self.pending = {k: v for k, v in self.pending.items() if now-v < 1.}
        key = int(now*1e9)
        self.pending[key] = now
        return 0, key

    def receive(self, tc1, ts1, now):
        if tc1 == 0 and ts1 > 0:
            remote = ts1*1e-9
            if self.last_px4_request is not None and remote < self.last_px4_request:
                self.reset()
            if remote != self.last_px4_request:
                if self.offset is not None and abs(now+self.offset-remote) < self.MAX_RTT_S/2:
                    self.replies += 1
                self.last_px4_request = remote
            return int(now*1e9), ts1
        sent = self.pending.pop(ts1, None)
        if sent is None or tc1 <= 0 or not 0 <= now-sent < self.MAX_RTT_S:
            return None
        offset = tc1*1e-9-(sent+now)/2
        if self.offset is not None and abs(offset-self.offset) > .1:
            self.reset()
        self.offsets.append(offset)
        self.last_good = now
        return None

    def ready(self, now):
        return (len(self.offsets) >= 10 and self.last_good is not None
                and 0 <= now-self.last_good < 1.
                and max(self.offsets)-min(self.offsets) < self.MAX_RTT_S
                and self.replies >= self.WARMUP_REPLIES)


class HeadingOutput:
    RATE_HZ = 30.0

    def __init__(self, std_deg, timeout_s):
        if not math.isfinite(std_deg) or not .1 <= std_deg <= 90:
            raise ValueError("Heading std must be 0.1..90 degrees (1 sigma)")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Invalid existing VIO timeout")
        self.variance = math.radians(std_deg)**2
        self.timeout_s = timeout_s  # existing GPS timeout; no extra 200 ms gate
        self.clock = ClockSync()
        self.pose = self.last_sent = self.last_output = None
        self.last_ros_stamp = self.last_ros_offset = None
        self.reference_covariance = None
        self.awaiting_recovery_covariance = False
        self.generation = 0
        self.block_reason = "waiting_for_accepted_pose"
        self.active = False

    def clear(self, reason):
        self.pose = None
        self.active = False
        self.block_reason = reason

    def begin_sample(self):
        # Accepting a newer pose is not an interruption of an active stream.
        # If the callback rejects this pose, reason() still blocks transmission.
        self.pose = None
        self.block_reason = "waiting_for_accepted_pose"

    def store(self, ros_stamp, ros_now, now, position, q_xyzw, yaw):
        age, offset = ros_now-ros_stamp, now-ros_now
        if not all(math.isfinite(v) for v in (ros_stamp, age, offset, yaw, *position)):
            self.clear("invalid_heading_sample")
            return
        previous_stamp, previous_offset = self.last_ros_stamp, self.last_ros_offset
        self.last_ros_stamp, self.last_ros_offset = ros_stamp, offset
        if (ros_stamp <= 0 or not 0 <= age <= self.timeout_s
                or (previous_stamp is not None and ros_stamp <= previous_stamp)
                or (previous_offset is not None and abs(offset-previous_offset) > .01)):
            self.clear("heading_timestamp_invalid_or_stale")
            return
        try:
            q = corrected_quaternion(q_xyzw, yaw)
        except ValueError:
            self.clear("invalid_heading_orientation")
            return
        self.pose = (now-age, tuple(position), q, yaw)
        self.generation = self.clock.generation
        self.block_reason = None

    def observe_covariance(self, msg, now):
        """Only FCU AUTOPILOT odometry may supply recovery uncertainty."""
        try:
            if msg.frame_id != 1 or msg.estimator_type != 8 or msg.quality < 0:
                return
            _, pitch, _ = rpy(msg.q)
            diag = [float(msg.pose_covariance[i]) for i in (15, 18, 20)]
            if not all(math.isfinite(v) and v >= 0 for v in diag):
                return
            # Conservative bound on Euler-yaw variance from small-angle
            # variances when cross terms are not available from PX4.
            variance = sum(diag)/math.cos(pitch)**2
        except (ValueError, TypeError, IndexError, AttributeError):
            return
        self.reference_covariance = (now, variance)
        if self.awaiting_recovery_covariance:
            self.variance = max(self.variance, variance)
            self.awaiting_recovery_covariance = False

    def recover(self, now, covariance_max_age):
        self.clear("pose_recovery")
        self.awaiting_recovery_covariance = True
        sample = self.reference_covariance
        if sample is not None and 0 <= now-sample[0] <= covariance_max_age:
            self.variance = max(self.variance, sample[1])
            self.awaiting_recovery_covariance = False

    def reason(self, now):
        if self.awaiting_recovery_covariance:
            return "waiting_for_px4_heading_uncertainty"
        if not self.clock.ready(now):
            return "heading_clock_sync_warming_or_stale"
        if self.generation != self.clock.generation:
            return "waiting_for_post_clock_reset_pose"
        if self.pose is None:
            return self.block_reason
        if not 0 <= now-self.pose[0] <= self.timeout_s:
            return "heading_sample_stale"
        return None

    def packet(self, now, position_std, height_std):
        reason = self.reason(now)
        if reason:
            self.block_reason, self.active = reason, False
            return None
        stamp, position, q, _ = self.pose
        if self.last_sent is not None and stamp <= self.last_sent:
            self.active = False
            self.block_reason = "waiting_for_new_heading_sample"
            return None
        covariance = [0.]*21
        covariance[0] = covariance[6] = position_std**2
        covariance[11] = height_std**2
        for i in (15, 18, 20):
            covariance[i] = self.variance
        return dict(time_usec=int(stamp*1e6), frame_id=1, child_frame_id=12,
                    x=position[0], y=position[1], z=position[2], q=q,
                    vx=math.nan, vy=math.nan, vz=math.nan,
                    rollspeed=math.nan, pitchspeed=math.nan, yawspeed=math.nan,
                    pose_covariance=covariance, velocity_covariance=[math.nan]*21,
                    reset_counter=0, estimator_type=2, quality=100)

    def mark_sent(self, now):
        self.last_sent = self.pose[0]
        self.last_output = (now, self.pose[0], self.pose[3], self.variance)
        self.active, self.block_reason = True, None

    def status(self, now, allowed):
        output = self.last_output
        return dict(vo_heading_enabled=True,
                    vo_heading_active=bool(allowed and self.active and not self.reason(now)),
                    vo_heading_reason=(self.reason(now) or self.block_reason) if allowed else self.block_reason,
                    vo_heading_deg=None if output is None else math.degrees(output[2]) % 360,
                    vo_heading_std_deg=None if output is None else math.degrees(math.sqrt(output[3])),
                    vo_heading_age_s=None if output is None else max(0., now-output[1]))
