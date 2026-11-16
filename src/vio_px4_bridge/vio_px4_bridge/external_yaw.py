"""Yaw-only EV aid for Path A. No ROS or serial ownership in this module.

ODOMETRY carries a real NED pose, but PX4 MUST be configured to fuse yaw only.
The corrected trajectory, not PX4's continuously changing yaw, drives output.
"""
from collections import deque
from dataclasses import dataclass, replace
import math
from statistics import median

from vio_px4_bridge.geo_utils import wrap_pi


def rpy(q):
    """Hamilton wxyz, body-to-world. Reject undefined/ill-conditioned yaw."""
    if len(q) != 4 or not all(math.isfinite(v) for v in q):
        raise ValueError("nonfinite quaternion")
    norm = math.sqrt(sum(v * v for v in q))
    if not 0.99 <= norm <= 1.01:
        raise ValueError("quaternion is not unit length")
    w, x, y, z = (v / norm for v in q)
    pitch = math.asin(max(-1., min(1., 2 * (w*y - z*x))))
    if abs(math.cos(pitch)) < 0.1:
        raise ValueError("heading undefined near vertical body attitude")
    return (math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)), pitch,
            math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)))


def quaternion(roll, pitch, yaw):
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    return (cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
            cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy)


def corrected_quaternion(q_xyzw, corrected_yaw, enu=True):
    """Preserve VO tilt, convert FLU/ENU to FRD/NED, replace world yaw only.

    cuVSLAM already includes the camera extrinsic. No second camera rotation.
    For non-ENU input the contract is body FRD in local NED.
    """
    x, y, z, w = q_xyzw
    roll, pitch, _ = rpy((w, x, y, z))
    if not math.isfinite(corrected_yaw):
        raise ValueError("nonfinite corrected yaw")
    return quaternion(roll, -pitch if enu else pitch, corrected_yaw)


class ClockSync:
    """Standard bidirectional MAVLink TIMESYNC, using companion MONOTONIC time.

    PX4 7cfb72fe uses 500 accepted exchanges before sync_stamp stops replacing
    sample time with arrival time. Request its TIMESYNC stream at 50 Hz and
    allow 600 low-delay replies before enabling EV. This is a warmup guard,
    not a claim of remote acknowledgement (TIMESYNC has no convergence ACK).
    """
    PX4_WARMUP_REPLIES = 600
    MAX_RTT_S = .010  # PX4 Timesync.hpp MAX_RTT_SAMPLE, not an aircraft limit

    def __init__(self):
        self.offsets = deque(maxlen=20)
        self.pending = {}
        self.last_good = None
        self.last_request = None
        self.last_px4_request = None
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
        if self.last_request is not None and now - self.last_request < .1:
            return None
        self.last_request = now
        self.pending = {k: v for k, v in self.pending.items() if now-v < 1.}
        key = int(now * 1e9)
        self.pending[key] = now
        return 0, key

    def receive(self, tc1, ts1, now):
        if tc1 == 0 and ts1 > 0:
            remote = ts1 * 1e-9
            if self.last_px4_request is not None and remote < self.last_px4_request:
                self.reset()  # reboot: do not carry old boot-clock alignment
            if remote != self.last_px4_request:
                if (self.offset is not None and
                        abs(now + self.offset - remote) < self.MAX_RTT_S / 2):
                    self.replies += 1
                self.last_px4_request = remote
            return int(now * 1e9), ts1
        sent = self.pending.pop(ts1, None)
        if sent is None or tc1 <= 0:
            return None
        rtt = now - sent
        if not 0 <= rtt < self.MAX_RTT_S:
            return None
        offset = tc1 * 1e-9 - (sent + now) / 2
        if self.offset is not None and abs(offset - self.offset) > .1:
            self.reset()
        self.offsets.append(offset)
        self.last_good = now
        return None

    def ready(self, now):
        return (len(self.offsets) >= 10 and self.last_good is not None
                and 0 <= now - self.last_good < 1.
                and max(self.offsets)-min(self.offsets) < self.MAX_RTT_S
                and self.replies >= self.PX4_WARMUP_REPLIES)


@dataclass(frozen=True)
class FcuPose:
    boot_s: float
    received_s: float
    north: float
    east: float
    vn: float
    ve: float
    yaw: float
    yaw_rate: float
    position_variance: float
    yaw_variance: float
    reset_counter: int
    evidence_s: float  # actual FCU sample time; extrapolation is not new evidence


class NavigationTiming:
    """Capture validation and matched FCU history, independent of yaw output."""
    def __init__(self, max_age_s=.2):
        if not math.isfinite(max_age_s) or not .02 <= max_age_s <= .2:
            raise ValueError("Camera capture age must be 0.02..0.2 s")
        self.max_age_s = max_age_s
        self.clock = ClockSync()
        self.history = deque(maxlen=100)
        self.last_ros_clock_offset = None
        self.last_ros_stamp = None
        self.trusted_capture_s = None
        self.input_error = None
        self.generation = 0
        self.boundary = None
        self.recovery_pose = None

    def observe_fcu(self, msg, now):
        """Only the caller's verified autopilot component may enter here."""
        if msg.frame_id != 1 or msg.estimator_type != 8 or msg.quality < 0:
            self.history.clear()
            return
        try:
            roll, pitch, yaw = rpy(msg.q)
            variances = [float(msg.pose_covariance[i]) for i in (0, 6, 15, 18, 20)]
            values = [msg.time_usec * 1e-6, now, msg.x, msg.y, msg.vx, msg.vy,
                      msg.pitchspeed, msg.yawspeed, *variances]
            if not all(math.isfinite(v) for v in values) or min(variances) < 0:
                raise ValueError("invalid FCU covariance or motion")
            # PX4 publishes body small-angle variances, not Euler yaw variance.
            # Trace/cos(pitch)^2 is conservative without cross covariances.
            yaw_var = sum(variances[2:]) / math.cos(pitch)**2
            yaw_rate = (msg.pitchspeed * math.sin(roll)
                        + msg.yawspeed * math.cos(roll)) / math.cos(pitch)
            # Bridge requests LOCAL_NED velocity. Don't silently interpret body velocity here.
            if msg.child_frame_id != 1:
                from vio_px4_bridge.geo_utils import mavlink_velocity_to_ned
                vn, ve, _ = mavlink_velocity_to_ned(
                    msg.frame_id, msg.child_frame_id, msg.q, (msg.vx, msg.vy, msg.vz))
            else:
                vn, ve = msg.vx, msg.vy
            stamp = msg.time_usec * 1e-6
            sample = FcuPose(stamp, now, msg.x, msg.y, vn, ve, yaw, yaw_rate,
                             max(variances[:2]), yaw_var, msg.reset_counter, stamp)
        except (ValueError, TypeError, IndexError):
            self.history.clear()
            return
        if self.history and sample.boot_s <= self.history[-1].boot_s:
            # Ignore small wire reorder; a real reboot invalidates clock and history.
            if self.history[-1].boot_s - sample.boot_s > .1:
                self.clock.reset()
                self.history.clear()
            return
        self.history.append(sample)

    def matched_fcu(self, mono_s, now, max_age_s):
        """Interpolate, or bounded constant-rate extrapolate to camera capture time."""
        if not self.clock.ready(now) or not self.history:
            return None
        target = mono_s + self.clock.offset
        latest = self.history[-1]
        if not 0 <= now - latest.received_s <= max_age_s:
            return None
        for a, b in zip(self.history, list(self.history)[1:]):
            if a.boot_s <= target <= b.boot_s:
                if a.reset_counter != b.reset_counter or b.boot_s-a.boot_s > max_age_s:
                    return None
                f = (target-a.boot_s)/(b.boot_s-a.boot_s)
                return replace(b, boot_s=target,
                               north=a.north+f*(b.north-a.north),
                               east=a.east+f*(b.east-a.east),
                               vn=a.vn+f*(b.vn-a.vn), ve=a.ve+f*(b.ve-a.ve),
                               yaw=wrap_pi(a.yaw+f*wrap_pi(b.yaw-a.yaw)),
                               yaw_variance=max(a.yaw_variance, b.yaw_variance),
                               position_variance=max(a.position_variance, b.position_variance))
        dt = target-latest.boot_s
        if not 0 <= dt <= max_age_s:
            return None
        return replace(latest, boot_s=target,
                       north=latest.north+latest.vn*dt, east=latest.east+latest.ve*dt,
                       yaw=wrap_pi(latest.yaw+latest.yaw_rate*dt),
                       yaw_variance=latest.yaw_variance+(latest.yaw_rate*dt)**2,
                       position_variance=latest.position_variance
                       + (math.hypot(latest.vn, latest.ve)*dt)**2)

    def sample_time(self, ros_stamp_s, ros_now_s, mono_now_s):
        """Convert capture time, never refresh an old pose with send time."""
        offset = mono_now_s-ros_now_s
        age = ros_now_s-ros_stamp_s
        reason = None
        if not all(math.isfinite(v) for v in (offset, ros_stamp_s, age)) or ros_stamp_s <= 0:
            reason = "invalid_camera_timestamp"
        elif not 0 <= age <= self.max_age_s:
            reason = "camera_capture_stale_or_future"
        elif self.last_ros_stamp is not None and ros_stamp_s <= self.last_ros_stamp:
            reason = "camera_timestamp_duplicate_or_regressed"
        elif self.last_ros_clock_offset is not None and abs(offset-self.last_ros_clock_offset) > .01:
            reason = "ROS_clock_jump"
        self.last_ros_stamp = ros_stamp_s
        self.last_ros_clock_offset = offset
        if reason:
            raise ValueError(reason)
        return ros_stamp_s+offset


class ExternalYaw(NavigationTiming):
    """Optional transmitter; disabling it does not disable navigation checks."""
    def __init__(self, noise_deg=1., rate_hz=30., max_age_s=.2):
        super().__init__(max_age_s)
        if (not all(math.isfinite(x) for x in (noise_deg, rate_hz))
                or not .1 <= noise_deg <= 90 or not 30 <= rate_hz <= 50):
            raise ValueError("EV yaw needs noise 0.1..90 deg, rate 30..50 Hz")
        self.noise_variance = math.radians(noise_deg) ** 2
        self.variance_floor = self.noise_variance
        self.rate_hz = rate_hz
        self.pose = None
        self.last_sent = None
        self.last_send_time = None
        self.last_output = None
        self.next_send_time = None
        self.startup_verified = False
        self.startup_agreement = 0
        self.startup_evidence = None
        self.sent = 0

    @property
    def reported_variance(self):
        return max(self.noise_variance, self.variance_floor)

    def store(self, sample_s, position, yaw, q_xyzw, enu):
        if not all(math.isfinite(v) for v in (*position, yaw, sample_s)):
            raise ValueError("nonfinite corrected pose")
        q = corrected_quaternion(q_xyzw, yaw, enu)
        self.pose = (sample_s, tuple(position), q, yaw)

    def reason(self, now):
        if self.input_error:
            return self.input_error
        if self.clock.generation != self.generation:
            return "PX4_clock_reset"
        if not self.clock.ready(now):
            return "heading_time_sync_warming_or_stale"
        if self.pose is None or not 0 <= now-self.pose[0] <= self.max_age_s:
            return "heading_sample_stale"
        return None

    def packet(self, now, position_sigma, height_sigma):
        if self.reason(now):
            return None
        stamp, position, q, _ = self.pose
        if self.last_sent is not None and stamp <= self.last_sent:
            return None
        if self.next_send_time is not None and now < self.next_send_time:
            return None
        covariance = [0.] * 21
        for index in (0, 6):
            covariance[index] = position_sigma**2
        covariance[11] = height_sigma**2
        for index in (15, 18, 20):
            covariance[index] = self.reported_variance
        # LOCAL_NED pose, BODY_FRD orientation; velocities unused (NaN).
        # reset_counter stays zero: the bridge repairs raw VO epochs itself.
        return dict(time_usec=int(stamp*1e6), frame_id=1, child_frame_id=12,
                    x=position[0], y=position[1], z=position[2], q=q,
                    vx=math.nan, vy=math.nan, vz=math.nan,
                    rollspeed=math.nan, pitchspeed=math.nan, yawspeed=math.nan,
                    pose_covariance=covariance, velocity_covariance=[math.nan]*21,
                    reset_counter=0, estimator_type=2, quality=100)

    def mark_sent(self, now):
        self.last_sent = self.pose[0]
        self.last_send_time = now
        self.last_output = (self.pose[0], self.pose[3], self.reported_variance)
        period = 1/self.rate_hz
        self.next_send_time = (now+period if self.next_send_time is None or
                              now-self.next_send_time >= period else
                              self.next_send_time+period)
        self.sent += 1

    def output_status(self, now, output_allowed):
        """Read-only snapshot of the last transmitted heading, not a candidate.

        A successful serial write is not proof of PX4 fusion. Preserve the last
        value through quarantine but explicitly mark it as paused, never live.
        """
        if self.last_output is None:
            return dict(external_yaw_heading_deg=None, external_yaw_variance_rad2=None,
                        external_yaw_sample_age_s=None, external_yaw_output_active=False)
        stamp, yaw, variance = self.last_output
        age = now - stamp
        return dict(external_yaw_heading_deg=math.degrees(yaw) % 360.,
                    external_yaw_variance_rad2=variance,
                    external_yaw_sample_age_s=age,
                    external_yaw_output_active=bool(output_allowed and 0 <= age <= self.max_age_s))
