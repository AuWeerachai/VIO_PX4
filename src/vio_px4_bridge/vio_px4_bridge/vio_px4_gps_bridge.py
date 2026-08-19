#!/usr/bin/env python3
"""Internship-style GPS spoof + VIO-as-GPS bridge for PX4.

ArduPilot internship stack uses MAVLink GPS_INPUT. Stock PX4 does not fuse
GPS_INPUT the same way; the supported inject path is HIL_GPS with
MAV_USEHILGPS=1 (and matching source sysid), which publishes into sensor_gps.

Phases (same idea as vns-sdk GpsSpoof -> BasaltGpsBridge):
  1) Boot spoof: stream static home HIL_GPS so EKF can init before VIO is ready
  2) Live VIO: convert local odometry offset from home into LLA and keep streaming
"""

from __future__ import annotations

import math
import time
from enum import Enum

import rclpy
from nav_msgs.msg import Odometry
from px4_msgs.msg import SensorGps
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy

from vio_px4_bridge.geo_utils import course_over_ground_cdeg
from vio_px4_bridge.geo_utils import enu_offset_to_ned
from vio_px4_bridge.geo_utils import ned_offset_to_lla
from vio_px4_bridge.geo_utils import rotate_ned_heading
from vio_px4_bridge.geo_utils import rotate_vector_by_quaternion_xyzw
from vio_px4_bridge.geo_utils import tilt_compensated_compass_yaw
from vio_px4_bridge.geo_utils import wrap_pi
from vio_px4_bridge.geo_utils import yaw_from_quaternion_xyzw
from vio_px4_bridge.local_continuity import LocalPoseContinuity


class Phase(Enum):
    WAITING = "waiting"
    SPOOF = "spoof"
    ALIGNING = "aligning"
    LIVE = "live"


def clamp_int(value: float, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(round(value))))


class VioPx4GpsBridge(Node):
    def __init__(self):
        super().__init__("vio_px4_gps_bridge")

        self.declare_parameter("odom_topic", "/visual_slam/tracking/odometry")
        self.declare_parameter("transport", "mavlink")  # mavlink | ros2
        self.declare_parameter("mavlink_url", "udpout:127.0.0.1:14540")
        self.declare_parameter("mavlink_sysid", 1)
        self.declare_parameter("mavlink_compid", 191)
        self.declare_parameter("heartbeat_timeout_s", 5.0)
        self.declare_parameter("ros2_gps_topic", "/fmu/in/sensor_gps")
        self.declare_parameter("rate_hz", 10.0)

        # Home / EKF origin (spoof target and LLA anchor for live VIO)
        self.declare_parameter("home_lat_deg", 40.4433)
        self.declare_parameter("home_lon_deg", -79.9436)
        self.declare_parameter("home_alt_m", 300.0)

        # Spoof phase (internship GpsSpoof analogue)
        self.declare_parameter("spoof_duration_s", 15.0)
        self.declare_parameter("spoof_until_vio", True)
        self.declare_parameter("spoof_eph_m", 0.5)
        self.declare_parameter("spoof_epv_m", 1.0)

        # Live / cruise accuracy (BasaltGpsBridge boot -> cruise analogue)
        self.declare_parameter("boot_accuracy_m", 0.5)
        self.declare_parameter("boot_duration_s", 60.0)
        self.declare_parameter("cruise_eph_m", 5.0)
        self.declare_parameter("cruise_epv_m", 1.5)
        self.declare_parameter("speed_accuracy_m_s", 0.25)

        self.declare_parameter("fix_type", 3)
        self.declare_parameter("satellites", 10)
        self.declare_parameter("gps_id", 0)
        self.declare_parameter("odom_is_enu", True)
        self.declare_parameter("anchor_on_first_odom", True)
        self.declare_parameter("twist_is_body_frame", True)
        self.declare_parameter("vio_timeout_s", 1.0)
        self.declare_parameter("send_set_gps_global_origin", True)
        self.declare_parameter("rc_trigger_enabled", True)
        self.declare_parameter("rc_channel", 6)
        self.declare_parameter("rc_low_pwm_max", 1300)
        self.declare_parameter("rc_high_pwm_min", 1700)
        self.declare_parameter("rc_request_rate_hz", 10.0)

        # Absolute heading alignment. Compass mode uses FC roll/pitch and the
        # calibrated body-frame magnetometer, never the FC's fused yaw.
        self.declare_parameter("heading_source", "compass")  # compass | manual
        self.declare_parameter("manual_heading_deg", 0.0)
        self.declare_parameter("mag_declination_deg", 0.0)
        self.declare_parameter("child_to_body_yaw_deg", 0.0)
        self.declare_parameter("mag_roll_offset_deg", 0.0)
        self.declare_parameter("mag_pitch_offset_deg", 0.0)
        self.declare_parameter("mag_yaw_offset_deg", 0.0)
        self.declare_parameter("alignment_samples", 30)
        self.declare_parameter("alignment_max_std_deg", 5.0)
        self.declare_parameter("alignment_max_speed_m_s", 0.25)
        self.declare_parameter("alignment_sensor_max_age_s", 0.5)
        self.declare_parameter("continuity_position_residual_m", 0.75)
        self.declare_parameter("continuity_yaw_residual_deg", 20.0)
        self.declare_parameter("continuity_max_gap_s", 1.0)
        self.declare_parameter("continuity_recovery_samples", 10)
        self.declare_parameter("continuity_confirmation_samples", 3)
        self.declare_parameter("continuity_max_speed_m_s", 10.0)
        self.declare_parameter("continuity_max_acceleration_m_s2", 5.0)
        self.declare_parameter("continuity_max_yaw_rate_deg_s", 200.0)

        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.transport = str(self.get_parameter("transport").value).lower()
        self.mavlink_url = str(self.get_parameter("mavlink_url").value)
        self.mavlink_sysid = int(self.get_parameter("mavlink_sysid").value)
        self.mavlink_compid = int(self.get_parameter("mavlink_compid").value)
        self.heartbeat_timeout_s = float(self.get_parameter("heartbeat_timeout_s").value)
        self.ros2_gps_topic = str(self.get_parameter("ros2_gps_topic").value)
        self.rate_hz = max(1.0, float(self.get_parameter("rate_hz").value))

        self.home_lat = float(self.get_parameter("home_lat_deg").value)
        self.home_lon = float(self.get_parameter("home_lon_deg").value)
        self.home_alt = float(self.get_parameter("home_alt_m").value)
        if not all(math.isfinite(v) for v in (self.home_lat, self.home_lon, self.home_alt)):
            raise ValueError("home latitude, longitude, and altitude must be finite")
        if not -90.0 <= self.home_lat <= 90.0:
            raise ValueError("home latitude must be between -90 and 90 degrees")
        if not -180.0 <= self.home_lon <= 180.0:
            raise ValueError("home longitude must be between -180 and 180 degrees")

        self.spoof_duration_s = max(0.0, float(self.get_parameter("spoof_duration_s").value))
        self.spoof_until_vio = bool(self.get_parameter("spoof_until_vio").value)
        self.spoof_eph_m = float(self.get_parameter("spoof_eph_m").value)
        self.spoof_epv_m = float(self.get_parameter("spoof_epv_m").value)

        self.boot_accuracy_m = float(self.get_parameter("boot_accuracy_m").value)
        self.boot_duration_s = float(self.get_parameter("boot_duration_s").value)
        self.cruise_eph_m = float(self.get_parameter("cruise_eph_m").value)
        self.cruise_epv_m = float(self.get_parameter("cruise_epv_m").value)
        self.speed_accuracy_m_s = float(self.get_parameter("speed_accuracy_m_s").value)

        self.fix_type = int(self.get_parameter("fix_type").value)
        self.satellites = int(self.get_parameter("satellites").value)
        self.gps_id = int(self.get_parameter("gps_id").value)
        self.odom_is_enu = bool(self.get_parameter("odom_is_enu").value)
        self.anchor_on_first_odom = bool(self.get_parameter("anchor_on_first_odom").value)
        self.twist_is_body_frame = bool(self.get_parameter("twist_is_body_frame").value)
        self.vio_timeout_s = max(0.1, float(self.get_parameter("vio_timeout_s").value))
        self.send_set_gps_global_origin = bool(
            self.get_parameter("send_set_gps_global_origin").value
        )
        self.rc_trigger_enabled = bool(self.get_parameter("rc_trigger_enabled").value)
        self.rc_channel = int(self.get_parameter("rc_channel").value)
        self.rc_low_pwm_max = int(self.get_parameter("rc_low_pwm_max").value)
        self.rc_high_pwm_min = int(self.get_parameter("rc_high_pwm_min").value)
        self.rc_request_rate_hz = max(
            1.0, float(self.get_parameter("rc_request_rate_hz").value)
        )
        if not 1 <= self.rc_channel <= 18:
            raise ValueError("rc_channel must be between 1 and 18")
        if self.rc_low_pwm_max >= self.rc_high_pwm_min:
            raise ValueError("rc_low_pwm_max must be below rc_high_pwm_min")

        self.heading_source = str(self.get_parameter("heading_source").value).lower()
        if self.heading_source not in ("compass", "manual"):
            raise ValueError("heading_source must be 'compass' or 'manual'")
        if self.transport == "ros2" and self.heading_source == "compass":
            raise ValueError(
                "heading_source=compass requires MAVLink ATTITUDE/HIGHRES_IMU; "
                "use transport=mavlink or an independently verified manual heading"
            )
        self.manual_heading_rad = math.radians(
            float(self.get_parameter("manual_heading_deg").value)
        )
        self.mag_declination_rad = math.radians(
            float(self.get_parameter("mag_declination_deg").value)
        )
        self.child_to_body_yaw_rad = math.radians(
            float(self.get_parameter("child_to_body_yaw_deg").value)
        )
        self.mag_mount_rpy = tuple(
            math.radians(float(self.get_parameter(name).value))
            for name in (
                "mag_roll_offset_deg",
                "mag_pitch_offset_deg",
                "mag_yaw_offset_deg",
            )
        )
        self.alignment_sample_count = max(
            5, int(self.get_parameter("alignment_samples").value)
        )
        self.alignment_max_std_rad = math.radians(
            max(0.1, float(self.get_parameter("alignment_max_std_deg").value))
        )
        self.alignment_max_speed = max(
            0.0, float(self.get_parameter("alignment_max_speed_m_s").value)
        )
        self.alignment_sensor_max_age_s = max(
            0.05, float(self.get_parameter("alignment_sensor_max_age_s").value)
        )
        self.continuity = LocalPoseContinuity(
            position_residual_limit_m=float(
                self.get_parameter("continuity_position_residual_m").value
            ),
            yaw_residual_limit_rad=math.radians(
                float(self.get_parameter("continuity_yaw_residual_deg").value)
            ),
            max_gap_s=float(self.get_parameter("continuity_max_gap_s").value),
            recovery_samples=int(self.get_parameter("continuity_recovery_samples").value),
            confirmation_samples=int(
                self.get_parameter("continuity_confirmation_samples").value
            ),
            max_speed_mps=float(
                self.get_parameter("continuity_max_speed_m_s").value
            ),
            max_acceleration_mps2=float(
                self.get_parameter("continuity_max_acceleration_m_s2").value
            ),
            max_yaw_rate_rad_s=math.radians(
                float(self.get_parameter("continuity_max_yaw_rate_deg_s").value)
            ),
        )

        self.phase = Phase.WAITING if self.rc_trigger_enabled else Phase.SPOOF
        self.start_time = None if self.rc_trigger_enabled else time.monotonic()
        self.live_start_time = None
        self.origin_sent = False
        self.anchor_ned = None  # first VIO pose in NED; treated as home unless disabled
        self.latest_fix = {
            "lat": self.home_lat,
            "lon": self.home_lon,
            "alt": self.home_alt,
            "vn": 0.0,
            "ve": 0.0,
            "vd": 0.0,
            "eph": self.spoof_eph_m,
            "epv": self.spoof_epv_m,
            "ignore_velocity": False,
        }
        self.have_vio = False
        self.last_vio_time = None
        self.vio_stale_warned = False
        self.last_rc_position = None
        self.messages_sent = 0
        self.heading_offset_rad = None
        self.heading_samples = []
        self.latest_attitude = None
        self.latest_mag = None
        self.last_alignment_mag_id = None
        self.logged_odom_frames = False
        self.continuity_recovering = False

        self._mav = None
        self._gps_pub = None
        self._init_transport()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.create_subscription(Odometry, self.odom_topic, self._odom_callback, qos)
        self.create_timer(1.0 / self.rate_hz, self._tick)

        self.get_logger().info(
            f"GPS bridge started transport={self.transport} "
            f"home=({self.home_lat:.7f},{self.home_lon:.7f},{self.home_alt:.1f}) "
            f"spoof_duration_s={self.spoof_duration_s} odom={self.odom_topic}"
        )
        self.get_logger().warn(
            "PX4 requires MAV_USEHILGPS=1 for mavlink HIL_GPS inject; "
            "disable/blend onboard GNSS carefully on Cube+."
        )
        if self.rc_trigger_enabled:
            self.get_logger().info(
                f"Waiting for RC channel {self.rc_channel} LOW/MID->HIGH trigger "
                f"(low<={self.rc_low_pwm_max}, high>={self.rc_high_pwm_min})"
            )
        self.get_logger().info(
            f"Heading alignment source={self.heading_source}; live GPS is gated until aligned"
        )

    def _init_transport(self):
        if self.transport == "mavlink":
            from pymavlink import mavutil

            # The CLI stores serial links as /dev/ttyTHS1:921600 for a compact,
            # human-readable setting. pymavlink expects the device and baud as
            # separate arguments; passing the combined string makes it try to
            # open a file literally named "ttyTHS1:921600".
            mavlink_device = self.mavlink_url
            mavlink_kwargs = {}
            if self.mavlink_url.startswith("/dev/"):
                candidate, separator, baud_text = self.mavlink_url.rpartition(":")
                if separator and baud_text.isdigit():
                    mavlink_device = candidate
                    mavlink_kwargs["baud"] = int(baud_text)

            self._mav = mavutil.mavlink_connection(
                mavlink_device,
                source_system=self.mavlink_sysid,
                source_component=self.mavlink_compid,
                **mavlink_kwargs,
            )
            heartbeat = None
            heartbeat_deadline = time.monotonic() + self.heartbeat_timeout_s
            while time.monotonic() < heartbeat_deadline:
                candidate = self._mav.recv_match(
                    type="HEARTBEAT", blocking=True, timeout=0.25
                )
                if (
                    candidate is not None
                    and int(candidate.autopilot) == mavutil.mavlink.MAV_AUTOPILOT_PX4
                ):
                    heartbeat = candidate
                    break
            if heartbeat is None:
                self._mav.close()
                raise RuntimeError(
                    f"No PX4 MAVLink heartbeat received on {self.mavlink_url} "
                    f"within {self.heartbeat_timeout_s:.1f}s"
                )
            px4_sysid = int(heartbeat.get_srcSystem())
            if self.mavlink_sysid != px4_sysid:
                self._mav.close()
                raise RuntimeError(
                    f"HIL_GPS sender sysid ({self.mavlink_sysid}) must equal PX4 "
                    f"MAV_SYS_ID ({px4_sysid}); PX4 rejects it otherwise"
                )
            self.get_logger().info(
                f"PX4 heartbeat received; MAVLink HIL_GPS -> {self.mavlink_url} "
                f"(sysid={self.mavlink_sysid} compid={self.mavlink_compid})"
            )
            use_hil_gps = self._read_px4_param(heartbeat, "MAV_USEHILGPS")
            gps_ctrl = self._read_px4_param(heartbeat, "EKF2_GPS_CTRL")
            if use_hil_gps is None:
                raise RuntimeError("PX4 did not return required parameter MAV_USEHILGPS")
            if int(round(use_hil_gps)) != 1:
                raise RuntimeError("PX4 MAV_USEHILGPS must be 1 before starting Path A")
            if gps_ctrl is None:
                raise RuntimeError("PX4 did not return required parameter EKF2_GPS_CTRL")
            gps_ctrl_bits = int(round(gps_ctrl))
            if (gps_ctrl_bits & 0b101) != 0b101:
                raise RuntimeError(
                    "EKF2_GPS_CTRL must enable longitude/latitude and 3D velocity "
                    f"fusion (current value {gps_ctrl_bits})"
                )
            if gps_ctrl_bits & 0b1000:
                raise RuntimeError(
                    "EKF2_GPS_CTRL dual-antenna heading bit must be off; HIL_GPS "
                    "does not provide independent GPS yaw"
                )
            self.get_logger().info(
                f"PX4 parameter gates passed: MAV_USEHILGPS=1 EKF2_GPS_CTRL={gps_ctrl_bits}"
            )
            if self.rc_trigger_enabled:
                self._request_mavlink_stream(
                    heartbeat, mavutil.mavlink.MAVLINK_MSG_ID_RC_CHANNELS,
                    self.rc_request_rate_hz,
                )
                self.get_logger().info(
                    f"Requested RC_CHANNELS at {self.rc_request_rate_hz:.1f} Hz"
                )
            if self.heading_source == "compass":
                self._request_mavlink_stream(
                    heartbeat, mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, 20.0
                )
                self._request_mavlink_stream(
                    heartbeat, mavutil.mavlink.MAVLINK_MSG_ID_HIGHRES_IMU, 20.0
                )
                self.get_logger().info("Requested ATTITUDE and HIGHRES_IMU at 20 Hz")
        elif self.transport == "ros2":
            qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )
            self._gps_pub = self.create_publisher(SensorGps, self.ros2_gps_topic, qos)
            self.get_logger().warn(
                f"Publishing SensorGps on {self.ros2_gps_topic}. "
                "Stock PX4 DDS yaml only has /fmu/out/vehicle_gps_position; "
                "add /fmu/in/sensor_gps subscription (see README) or use transport:=mavlink."
            )
        else:
            raise RuntimeError(f"Unknown transport '{self.transport}' (use mavlink|ros2)")

    def _request_mavlink_stream(self, heartbeat, message_id: int, rate_hz: float):
        from pymavlink import mavutil

        self._mav.mav.command_long_send(
            heartbeat.get_srcSystem(), heartbeat.get_srcComponent(),
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
            message_id, int(round(1_000_000.0 / rate_hz)), 0, 0, 0, 0, 0,
        )

    def _read_px4_param(self, heartbeat, name: str) -> float | None:
        encoded = name.encode("ascii")
        self._mav.mav.param_request_read_send(
            heartbeat.get_srcSystem(), heartbeat.get_srcComponent(), encoded, -1
        )
        deadline = time.monotonic() + self.heartbeat_timeout_s
        while time.monotonic() < deadline:
            msg = self._mav.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.25)
            if msg is None:
                continue
            param_id = msg.param_id
            if isinstance(param_id, bytes):
                param_id = param_id.decode("ascii", errors="ignore")
            if str(param_id).rstrip("\x00") == name:
                return float(msg.param_value)
        return None

    def _odom_callback(self, msg: Odometry):
        if not self.logged_odom_frames:
            self.get_logger().info(
                "cuVSLAM odometry frames: "
                f"frame_id='{msg.header.frame_id}' child_frame_id='{msg.child_frame_id}'. "
                "Confirm child_frame is the configured vehicle base frame"
            )
            self.logged_odom_frames = True
        pos = msg.pose.pose.position
        lin = msg.twist.twist.linear
        angular = msg.twist.twist.angular
        orientation = msg.pose.pose.orientation
        values = (
            pos.x,
            pos.y,
            pos.z,
            lin.x,
            lin.y,
            lin.z,
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
            angular.z,
        )
        if not all(math.isfinite(float(value)) for value in values):
            self.get_logger().error("Ignoring VIO odometry containing NaN/Inf")
            return
        if self.odom_is_enu:
            north, east, down = enu_offset_to_ned(pos.x, pos.y, pos.z)
            if self.twist_is_body_frame:
                try:
                    east_v, north_v, up_v = rotate_vector_by_quaternion_xyzw(
                        lin.x,
                        lin.y,
                        lin.z,
                        orientation.x,
                        orientation.y,
                        orientation.z,
                        orientation.w,
                    )
                except ValueError as exc:
                    self.get_logger().error(f"Ignoring invalid VIO orientation: {exc}")
                    return
                vn, ve, vd = enu_offset_to_ned(east_v, north_v, up_v)
            else:
                vn, ve, vd = enu_offset_to_ned(lin.x, lin.y, lin.z)
        else:
            north, east, down = float(pos.x), float(pos.y), float(pos.z)
            vn, ve, vd = float(lin.x), float(lin.y), float(lin.z)

        try:
            vio_child_yaw_enu = yaw_from_quaternion_xyzw(
                orientation.x, orientation.y, orientation.z, orientation.w
            )
        except ValueError as exc:
            self.get_logger().error(f"Ignoring invalid VIO orientation: {exc}")
            return
        raw_body_yaw_ned = wrap_pi(
            math.pi * 0.5 - (vio_child_yaw_enu + self.child_to_body_yaw_rad)
        )
        raw_yaw_rate_ned = -float(angular.z) if self.odom_is_enu else float(angular.z)
        stamp = msg.header.stamp
        sample_time = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        if sample_time <= 0.0:
            sample_time = time.monotonic()
        frame_key = f"{msg.header.frame_id}|{msg.child_frame_id}"
        continuity = self.continuity.update(
            (north, east, down), raw_body_yaw_ned, (vn, ve, vd),
            raw_yaw_rate_ned, sample_time, frame_key,
        )
        north, east, down = continuity.position
        vn, ve, vd = continuity.velocity
        if continuity.event:
            message = (
                f"VIO_CONTINUITY event={continuity.event} "
                f"reason={continuity.reason or 'none'} epoch={continuity.epoch} "
                f"detail={continuity.detail or 'none'}"
            )
            if continuity.event in ("quarantine_started", "reset_confirmed",
                                    "candidate_window_restarted"):
                self.get_logger().warn(message)
            else:
                self.get_logger().info(message)
        if continuity.recovering:
            self.continuity_recovering = True
            return
        if self.continuity_recovering:
            self.get_logger().info(
                f"VIO local continuity recovered at epoch={continuity.epoch}; GPS may resume"
            )
            self.continuity_recovering = False

        if self.anchor_on_first_odom and self.anchor_ned is None:
            self.anchor_ned = (north, east, down)
            self.get_logger().info(
                f"Anchored VIO origin at NED ({north:.3f},{east:.3f},{down:.3f}) -> home LLA"
            )

        if self.anchor_ned is not None:
            north -= self.anchor_ned[0]
            east -= self.anchor_ned[1]
            down -= self.anchor_ned[2]

        # A fixed ENU->NED swap does not align cuVSLAM's arbitrary world yaw to
        # north. Establish a one-time independent offset, then apply the same
        # rotation to displacement and world velocity.
        horizontal_speed = math.hypot(vn, ve)
        self._update_heading_alignment(continuity.yaw, horizontal_speed)
        if self.heading_offset_rad is not None:
            north, east, down = rotate_ned_heading(
                north, east, down, self.heading_offset_rad
            )
            vn, ve, vd = rotate_ned_heading(vn, ve, vd, self.heading_offset_rad)

        lat, lon, alt = ned_offset_to_lla(
            self.home_lat, self.home_lon, self.home_alt, north, east, down
        )

        eph, epv = self._select_accuracy()
        self.latest_fix = {
            "lat": lat,
            "lon": lon,
            "alt": alt,
            "vn": vn,
            "ve": ve,
            "vd": vd,
            "eph": eph,
            "epv": epv,
            "ignore_velocity": False,
        }
        self.have_vio = True
        self.last_vio_time = time.monotonic()
        if self.vio_stale_warned:
            self.get_logger().info("VIO odometry resumed; GPS updates resumed")
            self.vio_stale_warned = False

        if self.phase in (Phase.SPOOF, Phase.ALIGNING) and self.spoof_until_vio:
            if self.heading_offset_rad is not None:
                self._enter_live("vio_fresh_and_heading_aligned")
            elif self.phase == Phase.SPOOF:
                self.phase = Phase.ALIGNING
                self.get_logger().info(
                    "VIO ready; continuing static GPS while heading alignment completes"
                )

    def _update_heading_alignment(self, vio_body_yaw_ned: float, horizontal_speed: float):
        if self.heading_offset_rad is not None:
            return

        if self.heading_source == "manual":
            self.heading_offset_rad = wrap_pi(
                self.manual_heading_rad - vio_body_yaw_ned
            )
            self.get_logger().warn(
                "Heading aligned from manual true heading; verify independently before arming "
                f"(offset={math.degrees(self.heading_offset_rad):.2f} deg)"
            )
            return

        now = time.monotonic()
        if self.latest_attitude is None or self.latest_mag is None:
            return
        att_time, roll, pitch = self.latest_attitude
        mag_time, mag_id, mx, my, mz = self.latest_mag
        if now - att_time > self.alignment_sensor_max_age_s:
            return
        if now - mag_time > self.alignment_sensor_max_age_s:
            return
        if mag_id == self.last_alignment_mag_id:
            return
        if horizontal_speed > self.alignment_max_speed:
            return

        mx, my, mz = self._rotate_mag_mount(mx, my, mz)
        mag_norm = math.sqrt(mx * mx + my * my + mz * mz)
        if not math.isfinite(mag_norm) or mag_norm < 1e-6:
            return
        compass_yaw = tilt_compensated_compass_yaw(
            roll, pitch, mx, my, mz, self.mag_declination_rad
        )
        offset = wrap_pi(compass_yaw - vio_body_yaw_ned)
        self.last_alignment_mag_id = mag_id
        self.heading_samples.append(offset)
        if len(self.heading_samples) < self.alignment_sample_count:
            return

        sin_mean = sum(math.sin(x) for x in self.heading_samples) / len(self.heading_samples)
        cos_mean = sum(math.cos(x) for x in self.heading_samples) / len(self.heading_samples)
        resultant = math.hypot(sin_mean, cos_mean)
        circular_std = math.sqrt(max(0.0, -2.0 * math.log(max(resultant, 1e-12))))
        if circular_std > self.alignment_max_std_rad:
            self.get_logger().warn(
                "Compass/VIO heading samples unstable; restarting alignment "
                f"(std={math.degrees(circular_std):.1f} deg)"
            )
            self.heading_samples.clear()
            return
        self.heading_offset_rad = math.atan2(sin_mean, cos_mean)
        self.get_logger().info(
            "Independent compass heading alignment locked and frozen: "
            f"offset={math.degrees(self.heading_offset_rad):.2f} deg, "
            f"std={math.degrees(circular_std):.2f} deg, n={len(self.heading_samples)}"
        )

    def _rotate_mag_mount(self, x: float, y: float, z: float):
        """Apply configured Rx, then Ry, then Rz compass-to-body offsets."""
        roll, pitch, yaw = self.mag_mount_rpy
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        x1, y1, z1 = x, cr * y - sr * z, sr * y + cr * z
        x2, y2, z2 = cp * x1 + sp * z1, y1, -sp * x1 + cp * z1
        return cy * x2 - sy * y2, sy * x2 + cy * y2, z2

    def _select_accuracy(self) -> tuple[float, float]:
        if self.phase == Phase.SPOOF:
            return self.spoof_eph_m, self.spoof_epv_m
        if self.live_start_time is None:
            return self.boot_accuracy_m, self.spoof_epv_m
        if (time.monotonic() - self.live_start_time) < self.boot_duration_s:
            return self.boot_accuracy_m, self.spoof_epv_m
        return self.cruise_eph_m, self.cruise_epv_m

    def _enter_live(self, reason: str):
        if self.phase == Phase.LIVE:
            return
        previous_phase = self.phase
        self.phase = Phase.LIVE
        self.live_start_time = time.monotonic()
        if previous_phase in (Phase.SPOOF, Phase.ALIGNING):
            self.get_logger().info(f"GPS spoof ended -> live VIO GPS ({reason})")
        else:
            self.get_logger().info(f"RC activated live VIO GPS ({reason})")

    def _rc_position(self, pwm: int) -> str | None:
        if pwm in (0, 65535):
            return None
        if pwm <= self.rc_low_pwm_max:
            return "low"
        if pwm >= self.rc_high_pwm_min:
            return "high"
        return "mid"

    def _poll_mavlink(self):
        """Drain MAVLink once and dispatch without discarding other message types."""
        if self.transport != "mavlink" or self._mav is None:
            return
        while True:
            msg = self._mav.recv_match(blocking=False)
            if msg is None:
                return
            message_type = msg.get_type()
            now = time.monotonic()
            if message_type == "ATTITUDE":
                roll, pitch = float(msg.roll), float(msg.pitch)
                if math.isfinite(roll) and math.isfinite(pitch):
                    # Deliberately exclude msg.yaw: injected GPS must never feed
                    # back into the reference used to align injected GPS.
                    self.latest_attitude = (now, roll, pitch)
                continue
            if message_type == "HIGHRES_IMU":
                fields = int(getattr(msg, "fields_updated", 0))
                mag_mask = (1 << 6) | (1 << 7) | (1 << 8)
                if fields and (fields & mag_mask) != mag_mask:
                    continue
                mx, my, mz = float(msg.xmag), float(msg.ymag), float(msg.zmag)
                if all(math.isfinite(v) for v in (mx, my, mz)):
                    mag_id = int(getattr(msg, "time_usec", 0)) or time.monotonic_ns()
                    self.latest_mag = (now, mag_id, mx, my, mz)
                continue
            if message_type != "RC_CHANNELS" or not self.rc_trigger_enabled:
                continue
            pwm = int(getattr(msg, f"chan{self.rc_channel}_raw", 65535))
            position = self._rc_position(pwm)
            if position is None:
                continue
            rising_high = self.last_rc_position != "high" and position == "high"
            self.last_rc_position = position
            if not rising_high or self.phase != Phase.WAITING:
                continue

            vio_is_fresh = (
                self.have_vio
                and self.last_vio_time is not None
                and (now - self.last_vio_time) <= self.vio_timeout_s
            )
            self.get_logger().info(
                f"RC channel {self.rc_channel} pwm={pwm} triggered GPS bootstrap"
            )
            if vio_is_fresh and self.heading_offset_rad is not None:
                self._enter_live("rc_trigger_vio_already_ready")
            else:
                self.phase = Phase.SPOOF
                self.start_time = now
                self.get_logger().info(
                    f"GPS spoof started from RC channel {self.rc_channel} "
                    f"for up to {self.spoof_duration_s:.1f}s"
                )

    def _maybe_end_spoof_by_timer(self):
        if self.phase not in (Phase.SPOOF, Phase.ALIGNING):
            return
        if self.spoof_duration_s <= 0.0:
            return
        if self.start_time is None:
            return
        if (time.monotonic() - self.start_time) >= self.spoof_duration_s:
            if self.have_vio and self.heading_offset_rad is not None:
                self._enter_live("spoof_duration")
            else:
                self.phase = Phase.WAITING
                self.start_time = None
                self.get_logger().warn(
                    f"GPS spoof expired after {self.spoof_duration_s:.1f}s without "
                    "fresh, heading-aligned VIO; "
                    f"GPS output stopped. Toggle RC channel {self.rc_channel} LOW then HIGH to retry"
                )

    def _tick(self):
        self._poll_mavlink()
        if self.phase == Phase.WAITING:
            return
        self._maybe_end_spoof_by_timer()
        if self.phase == Phase.WAITING:
            return

        if (
            self.phase == Phase.LIVE
            and self.continuity_recovering
        ):
            return

        if (
            self.phase == Phase.LIVE
            and self.last_vio_time is not None
            and (time.monotonic() - self.last_vio_time) > self.vio_timeout_s
        ):
            if not self.vio_stale_warned:
                self.get_logger().error(
                    f"VIO odometry stale for >{self.vio_timeout_s:.1f}s; "
                    "stopping GPS updates so PX4 can time out the aid source"
                )
                self.vio_stale_warned = True
            return

        if self.phase in (Phase.SPOOF, Phase.ALIGNING) or not self.have_vio:
            fix = {
                "lat": self.home_lat,
                "lon": self.home_lon,
                "alt": self.home_alt,
                "vn": 0.0,
                "ve": 0.0,
                "vd": 0.0,
                "eph": self.spoof_eph_m,
                "epv": self.spoof_epv_m,
                "ignore_velocity": False,
            }
        else:
            eph, epv = self._select_accuracy()
            fix = dict(self.latest_fix)
            fix["eph"] = eph
            fix["epv"] = epv

        if self.send_set_gps_global_origin and not self.origin_sent:
            self._send_global_origin()
            self.origin_sent = True

        self._publish_fix(fix)
        self.messages_sent += 1
        if self.messages_sent == 1 or self.messages_sent % 50 == 0:
            self.get_logger().info(
                f"GPS [{self.phase.value}] n={self.messages_sent} "
                f"lat={fix['lat']:.7f} lon={fix['lon']:.7f} alt={fix['alt']:.2f} "
                f"eph={fix['eph']:.2f}"
            )

    def _send_global_origin(self):
        if self.transport != "mavlink" or self._mav is None:
            return
        try:
            # SET_GPS_GLOBAL_ORIGIN (msg id 48)
            self._mav.mav.set_gps_global_origin_send(
                self.mavlink_sysid,
                int(round(self.home_lat * 1e7)),
                int(round(self.home_lon * 1e7)),
                int(round(self.home_alt * 1000.0)),
            )
            self.get_logger().info("Sent SET_GPS_GLOBAL_ORIGIN (home)")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"SET_GPS_GLOBAL_ORIGIN failed: {exc}")

    def _publish_fix(self, fix: dict):
        if self.transport == "mavlink":
            self._publish_hil_gps(fix)
        else:
            self._publish_ros2_sensor_gps(fix)

    def _publish_hil_gps(self, fix: dict):
        vn = float(fix["vn"])
        ve = float(fix["ve"])
        vd = float(fix["vd"])
        speed = math.hypot(vn, ve, vd)
        cog = course_over_ground_cdeg(vn, ve)
        now_us = int(time.time() * 1e6)
        # Units: lat/lon degE7, alt mm, eph/epv cm, vel cm/s
        self._mav.mav.hil_gps_send(
            now_us,
            clamp_int(self.fix_type, 0, 255),
            clamp_int(fix["lat"] * 1e7, -2147483648, 2147483647),
            clamp_int(fix["lon"] * 1e7, -2147483648, 2147483647),
            clamp_int(fix["alt"] * 1000.0, -2147483648, 2147483647),
            clamp_int(max(0.0, fix["eph"]) * 100.0, 0, 65535),
            clamp_int(max(0.0, fix["epv"]) * 100.0, 0, 65535),
            clamp_int(speed * 100.0, 0, 65535),
            clamp_int(vn * 100.0, -32768, 32767),
            clamp_int(ve * 100.0, -32768, 32767),
            clamp_int(vd * 100.0, -32768, 32767),
            int(cog),
            clamp_int(self.satellites, 0, 255),
            clamp_int(self.gps_id, 0, 255),
            0,  # yaw unavailable (PX4 HIL_GPS receiver also forces heading NaN)
        )

    def _publish_ros2_sensor_gps(self, fix: dict):
        msg = SensorGps()
        now_us = int(self.get_clock().now().nanoseconds / 1000)
        msg.timestamp = now_us
        msg.timestamp_sample = now_us
        msg.device_id = 197388  # arbitrary stable id (MAVLink/sim-ish)
        msg.latitude_deg = float(fix["lat"])
        msg.longitude_deg = float(fix["lon"])
        msg.altitude_msl_m = float(fix["alt"])
        msg.altitude_ellipsoid_m = float(fix["alt"])
        msg.s_variance_m_s = float(self.speed_accuracy_m_s)
        msg.c_variance_rad = 0.5
        msg.fix_type = int(self.fix_type)
        msg.eph = float(fix["eph"])
        msg.epv = float(fix["epv"])
        msg.hdop = 0.8
        msg.vdop = 0.8
        msg.vel_n_m_s = float(fix["vn"])
        msg.vel_e_m_s = float(fix["ve"])
        msg.vel_d_m_s = float(fix["vd"])
        msg.vel_m_s = float(math.hypot(fix["vn"], fix["ve"], fix["vd"]))
        cog = course_over_ground_cdeg(fix["vn"], fix["ve"])
        msg.cog_rad = float("nan") if cog == 65535 else math.radians(cog / 100.0)
        msg.vel_ned_valid = True
        msg.satellites_used = int(self.satellites)
        msg.time_utc_usec = int(time.time() * 1e6)
        msg.heading = float("nan")
        msg.heading_offset = float("nan")
        self._gps_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VioPx4GpsBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
