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

import json
import math
import os
import struct
import time
from enum import Enum
from pathlib import Path

import rclpy
from nav_msgs.msg import Odometry
try:
    from px4_msgs.msg import SensorGps
except ImportError:
    SensorGps = None
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
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
from vio_px4_bridge.mag_declination import default_table_path
from vio_px4_bridge.mag_declination import lookup_declination_deg


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
        self.declare_parameter("expected_child_frame", "drone_link")
        self.declare_parameter("transport", "mavlink")  # mavlink | ros2
        self.declare_parameter("mavlink_url", "udpout:127.0.0.1:14540")
        self.declare_parameter("mavlink_sysid", 1)
        self.declare_parameter("mavlink_compid", 191)
        self.declare_parameter("heartbeat_timeout_s", 5.0)
        self.declare_parameter("ros2_gps_topic", "/fmu/in/sensor_gps")
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("status_file", "")

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
        self.declare_parameter("boot_accuracy_m", 0.1)
        self.declare_parameter("boot_duration_s", 60.0)
        self.declare_parameter("cruise_eph_m", 0.1)
        self.declare_parameter("cruise_epv_m", 1.5)
        self.declare_parameter("speed_accuracy_m_s", 0.25)
        # HIL_GPS requires altitude and velocity fields even when PX4 is
        # configured to fuse horizontal position only. In this mode those
        # required fields are stable fillers; only lat/lon carry VIO motion.
        self.declare_parameter("horizontal_only_output", True)

        self.declare_parameter("fix_type", 3)
        # Strong but realistic open-sky multi-constellation receiver count.
        self.declare_parameter("satellites", 40)
        # Keep the injected receiver distinct from the physical Here GNSS
        # (GPS1 / instance 0). HIL_GPS.id is carried into PX4's GPS device id.
        self.declare_parameter("gps_id", 1)
        self.declare_parameter("validate_dual_gps_selection", True)
        # -1 accepts either instance after checking that blending is disabled.
        # Instance numbering changes when the optional physical Here GPS is absent.
        self.declare_parameter("expected_vio_gps_instance", -1)
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
        self.declare_parameter("mag_declination_source", "table")  # table | manual
        self.declare_parameter("mag_declination_deg", 0.0)
        self.declare_parameter("mag_declination_table_path", "")
        self.declare_parameter("child_to_body_yaw_deg", 0.0)
        self.declare_parameter("mag_roll_offset_deg", 0.0)
        self.declare_parameter("mag_pitch_offset_deg", 0.0)
        self.declare_parameter("mag_yaw_offset_deg", 0.0)
        self.declare_parameter("alignment_samples", 30)
        self.declare_parameter("alignment_max_std_deg", 5.0)
        self.declare_parameter("alignment_max_speed_m_s", 0.25)
        self.declare_parameter("alignment_sensor_max_age_s", 0.5)
        self.declare_parameter("heading_disagreement_limit_deg", 20.0)
        self.declare_parameter("heading_disagreement_confirmation_samples", 5)
        self.declare_parameter("heading_disagreement_recovery_samples", 20)
        self.declare_parameter("continuity_max_gap_s", 1.0)
        self.declare_parameter("continuity_recovery_samples", 10)
        self.declare_parameter("continuity_confirmation_samples", 3)
        self.declare_parameter("continuity_max_speed_m_s", 10.0)
        self.declare_parameter("continuity_max_acceleration_m_s2", 5.0)
        self.declare_parameter("continuity_max_yaw_rate_deg_s", 360.0)
        self.declare_parameter("inertial_max_duration_s", 2.0)
        self.declare_parameter("inertial_max_message_age_s", 0.25)
        self.declare_parameter("inertial_max_message_gap_s", 0.25)
        self.declare_parameter("inertial_max_position_uncertainty_m", 3.0)
        self.declare_parameter("recovery_velocity_agreement_m_s", 2.0)
        self.declare_parameter("recovery_velocity_agreement_samples", 3)
        self.declare_parameter("recovery_accuracy_tighten_s", 5.0)

        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.expected_child_frame = str(self.get_parameter("expected_child_frame").value)
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
        self.horizontal_only_output = bool(
            self.get_parameter("horizontal_only_output").value
        )

        self.fix_type = int(self.get_parameter("fix_type").value)
        self.satellites = clamp_int(
            int(self.get_parameter("satellites").value), 0, 254
        )
        self.gps_id = int(self.get_parameter("gps_id").value)
        self.validate_dual_gps_selection = bool(
            self.get_parameter("validate_dual_gps_selection").value
        )
        self.expected_vio_gps_instance = int(
            self.get_parameter("expected_vio_gps_instance").value
        )
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
        manual_declination_deg = float(self.get_parameter("mag_declination_deg").value)
        self.mag_declination_source = str(
            self.get_parameter("mag_declination_source").value
        ).lower()
        if self.mag_declination_source not in ("table", "manual"):
            raise ValueError("mag_declination_source must be 'table' or 'manual'")
        table_path_param = str(
            self.get_parameter("mag_declination_table_path").value
        ).strip()
        resolved_declination_deg = manual_declination_deg
        if self.heading_source == "compass" and self.mag_declination_source == "table":
            table_path = table_path_param or str(default_table_path())
            try:
                resolved_declination_deg = lookup_declination_deg(
                    self.home_lat, self.home_lon, table_path
                )
                self.get_logger().info(
                    "MAG_DECLINATION_RESOLVED "
                    f"source=initial_home lat_deg={self.home_lat:.7f} "
                    f"lon_deg={self.home_lon:.7f} "
                    f"declination_deg={resolved_declination_deg:.3f} "
                    f"table={table_path}; frozen_for_flight=true"
                )
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                self.get_logger().error(
                    "MAG_DECLINATION_TABLE_FAILED "
                    f"table={table_path} error={exc}; heading alignment blocked"
                )
                raise RuntimeError(
                    "magnetic declination table lookup failed; repair the table "
                    "or explicitly select mag_declination_source=manual"
                ) from exc
        self.mag_declination_deg = resolved_declination_deg
        self.mag_declination_rad = math.radians(resolved_declination_deg)
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
        self.heading_disagreement_limit_rad = math.radians(max(
            1.0, float(self.get_parameter("heading_disagreement_limit_deg").value)
        ))
        self.heading_disagreement_confirmation_samples = max(
            1, int(self.get_parameter("heading_disagreement_confirmation_samples").value)
        )
        self.heading_disagreement_recovery_samples = max(
            1, int(self.get_parameter("heading_disagreement_recovery_samples").value)
        )
        self.continuity = LocalPoseContinuity(
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
        self.inertial_max_duration_s = max(
            0.1, float(self.get_parameter("inertial_max_duration_s").value)
        )
        self.inertial_max_message_age_s = max(
            0.05, float(self.get_parameter("inertial_max_message_age_s").value)
        )
        self.inertial_max_message_gap_s = max(
            0.02, float(self.get_parameter("inertial_max_message_gap_s").value)
        )
        self.inertial_max_position_uncertainty_m = max(
            self.cruise_eph_m,
            float(self.get_parameter("inertial_max_position_uncertainty_m").value),
        )
        self.recovery_velocity_agreement_m_s = max(
            0.0, float(self.get_parameter("recovery_velocity_agreement_m_s").value)
        )
        self.recovery_velocity_agreement_samples = max(
            1, int(self.get_parameter("recovery_velocity_agreement_samples").value)
        )
        self.recovery_accuracy_tighten_s = max(
            0.0, float(self.get_parameter("recovery_accuracy_tighten_s").value)
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
        self.rejected_odom_frame = None
        self.last_vio_time = None
        self.vio_stale_warned = False
        self.last_rc_position = None
        self.messages_sent = 0
        self.heading_offset_rad = None
        self.heading_samples = []
        self.heading_disagreement_rad = None
        self.heading_disagreement_bad_count = 0
        self.heading_disagreement_good_count = 0
        self.heading_quarantine = False
        self.heading_last_attitude_arrival = None
        self.latest_attitude = None
        self.latest_mag = None
        self.latest_px4_velocity = None
        self.latest_px4_yaw = None
        self.latest_px4_odometry = None
        self.inertial_active = False
        self.inertial_safe = False
        self.inertial_failure_reason = None
        self.inertial_start_time = None
        self.inertial_displacement_ned = [0.0, 0.0, 0.0]
        self.inertial_integrated_duration_s = 0.0
        self.inertial_yaw_delta = 0.0
        self.inertial_last_yaw = None
        self.inertial_start_odometry = None
        self.inertial_last_odometry = None
        self.inertial_fallback_to_frozen_pose = False
        self.recovery_velocity_agreement_count = 0
        self.recovery_resume_eph_m = None
        self.recovery_resume_time = None
        self.last_alignment_mag_id = None
        self.logged_odom_frames = False
        self.continuity_recovering = False
        self.live_output_active = False
        self.live_output_block_reason = "not_live"
        self.last_px4_heartbeat_time = None
        self.px4_target_sysid = None
        self.status_file = str(self.get_parameter("status_file").value).strip()

        self._mav = None
        self._gps_pub = None
        self._hil_gps_extensions_supported = None
        self._init_transport()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.create_subscription(Odometry, self.odom_topic, self._odom_callback, qos)
        self.create_timer(1.0 / self.rate_hz, self._tick)
        if self.status_file:
            self.create_timer(0.5, self._write_status_file)

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
            self._send_status_text(
                f"VIO GPS: waiting for RC channel {self.rc_channel}", 6
            )
        self.get_logger().info(
            f"Heading alignment source={self.heading_source}; live GPS is gated until aligned"
        )

    def _write_status_file(self):
        """Publish a small atomic snapshot for the companion launcher UI."""
        now = time.monotonic()
        vio_age = None if self.last_vio_time is None else max(0.0, now - self.last_vio_time)
        vio_fresh = bool(
            self.have_vio and vio_age is not None and vio_age <= self.vio_timeout_s
        )
        spoof_remaining = None
        if self.phase in (Phase.SPOOF, Phase.ALIGNING) and self.start_time is not None:
            spoof_remaining = max(0.0, self.spoof_duration_s - (now - self.start_time))
        if self.live_output_active:
            navigation = "live_vio"
        elif self.phase in (Phase.SPOOF, Phase.ALIGNING):
            navigation = "spoof_active"
        elif self.phase == Phase.WAITING:
            navigation = "waiting_for_rc"
        else:
            navigation = "gps_output_stopped"
        payload = {
            "updated_unix_s": time.time(),
            "pid": os.getpid(),
            "phase": self.phase.value,
            "navigation": navigation,
            "rc_position": self.last_rc_position,
            "spoof_remaining_s": spoof_remaining,
            "vio_received": self.have_vio,
            "vio_fresh": vio_fresh,
            "vio_age_s": vio_age,
            "px4_connected": bool(
                self.last_px4_heartbeat_time is not None
                and (now - self.last_px4_heartbeat_time) <= self.heartbeat_timeout_s
            ),
            "heading_aligned": self.heading_offset_rad is not None,
            "heading_quarantine": self.heading_quarantine,
            "heading_disagreement_deg": (
                None if self.heading_disagreement_rad is None else
                math.degrees(self.heading_disagreement_rad)
            ),
            "heading_disagreement_limit_deg": math.degrees(
                self.heading_disagreement_limit_rad
            ),
            "pose_gate_quarantine": self.continuity_recovering,
            "inertial_recovery_active": self.inertial_active,
            "inertial_recovery_safe": self.inertial_safe,
            "inertial_fallback_to_frozen_pose": self.inertial_fallback_to_frozen_pose,
            "inertial_recovery_failure": self.inertial_failure_reason,
            "inertial_displacement_n_m": self.inertial_displacement_ned[0],
            "inertial_displacement_e_m": self.inertial_displacement_ned[1],
            "px4_position_uncertainty_m": (
                None if self.latest_px4_odometry is None else
                math.sqrt(max(0.0, self.latest_px4_odometry[6], self.latest_px4_odometry[7]))
            ),
            "recovery_velocity_agreement_count": self.recovery_velocity_agreement_count,
            "hil_gps_output_active": self.live_output_active,
            "output_block_reason": self.live_output_block_reason,
            "px4_fallback": (
                "not_needed" if self.live_output_active or navigation == "spoof_active"
                else "PX4 selects physical GPS if healthy; otherwise dead reckoning/failsafe"
            ),
        }
        target = Path(self.status_file).expanduser()
        temporary = target.with_suffix(target.suffix + ".tmp")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(json.dumps(payload, indent=2) + "\n")
            temporary.replace(target)
        except OSError as exc:
            self.get_logger().warn(f"Could not write navigation status: {exc}")

    def _init_transport(self):
        if self.transport == "mavlink":
            # ODOMETRY is MAVLink message 331 and therefore requires MAVLink 2.
            # pymavlink otherwise defaults to its v1 dialect on this Jetson.
            os.environ["MAVLINK20"] = "1"
            from pymavlink import mavutil
            mavutil.set_dialect("ardupilotmega")
            odometry_message_id = getattr(
                mavutil.mavlink, "MAVLINK_MSG_ID_ODOMETRY", 331
            )
            if str(mavutil.mavlink.WIRE_PROTOCOL_VERSION) != "2.0":
                raise RuntimeError(
                    "Path A recovery requires MAVLink 2 for PX4 ODOMETRY (message 331)"
                )

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
            self.px4_target_sysid = px4_sysid
            self.last_px4_heartbeat_time = time.monotonic()
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
            noaid_tout_us = self._read_px4_param(heartbeat, "EKF2_NOAID_TOUT")
            if use_hil_gps is None:
                raise RuntimeError("PX4 did not return required parameter MAV_USEHILGPS")
            if int(round(use_hil_gps)) != 1:
                raise RuntimeError("PX4 MAV_USEHILGPS must be 1 before starting Path A")
            if gps_ctrl is None:
                raise RuntimeError("PX4 did not return required parameter EKF2_GPS_CTRL")
            if noaid_tout_us is None:
                raise RuntimeError("PX4 did not return required parameter EKF2_NOAID_TOUT")
            gps_ctrl_bits = int(round(gps_ctrl))
            if gps_ctrl_bits != 0b0001:
                raise RuntimeError(
                    "EKF2_GPS_CTRL must be 1: longitude/latitude only; altitude "
                    "uses the configured height source, velocity is not fused, "
                    f"and heading stays on compass (current={gps_ctrl_bits})"
                )
            noaid_tout_s = float(noaid_tout_us) * 1e-6
            if self.inertial_max_duration_s >= noaid_tout_s:
                raise RuntimeError(
                    "inertial_max_duration_s must be shorter than PX4 "
                    f"EKF2_NOAID_TOUT ({noaid_tout_s:.3f}s); current="
                    f"{self.inertial_max_duration_s:.3f}s"
                )
            self.get_logger().info(
                "PX4 parameter gates passed: MAV_USEHILGPS=1 "
                f"EKF2_GPS_CTRL={gps_ctrl_bits} EKF2_NOAID_TOUT={noaid_tout_s:.3f}s"
            )
            if self.validate_dual_gps_selection:
                self._validate_dual_gps_selection(heartbeat)
            if self.rc_trigger_enabled:
                self._request_mavlink_stream(
                    heartbeat, mavutil.mavlink.MAVLINK_MSG_ID_RC_CHANNELS,
                    self.rc_request_rate_hz,
                )
                self.get_logger().info(
                    f"Requested RC_CHANNELS at {self.rc_request_rate_hz:.1f} Hz"
                )
            self._request_mavlink_stream(
                heartbeat, mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, 20.0
            )
            self._request_mavlink_stream(
                heartbeat, odometry_message_id, 20.0
            )
            self.get_logger().info(
                "Requested ATTITUDE and ODOMETRY at 20 Hz for guarded GPS-silent "
                "recovery (PX4 position/velocity variance and reset counter included)"
            )
            if self.heading_source == "compass":
                self._request_mavlink_stream(
                    heartbeat, mavutil.mavlink.MAVLINK_MSG_ID_HIGHRES_IMU, 20.0
                )
                self.get_logger().info("Requested HIGHRES_IMU at 20 Hz")
            self._send_status_text("VIO GPS: PX4 link ready", 6)
        elif self.transport == "ros2":
            if SensorGps is None:
                raise RuntimeError(
                    "transport=ros2 requires px4_msgs; use transport=mavlink on the Jetson"
                )
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
                # PX4 uses MAVLink's byte-wise parameter encoding. Integer
                # bits occupy the PARAM_VALUE float payload directly, so an
                # INT32 value of 1 appears to Python as 1.401298e-45 unless
                # decoded according to param_type.
                param_type = int(msg.param_type)
                raw = struct.pack("<f", float(msg.param_value))
                if param_type == 5:  # MAV_PARAM_TYPE_UINT32
                    return float(struct.unpack("<I", raw)[0])
                if param_type == 6:  # MAV_PARAM_TYPE_INT32
                    return float(struct.unpack("<i", raw)[0])
                return float(msg.param_value)
        return None

    def _send_status_text(self, message: str, severity: int):
        """Send one short operational state message to PX4 and QGroundControl."""
        if self.transport != "mavlink" or self._mav is None:
            return
        # MAVLink STATUSTEXT is 50 bytes. Keep messages ASCII and unchunked so
        # older PX4/QGC versions display them consistently.
        payload = message.encode("ascii", errors="replace")[:50]
        try:
            self._mav.mav.statustext_send(severity, payload, 0, 0)
        except TypeError:
            # MAVLink 1 dialects omit the MAVLink 2 id/chunk_seq extensions.
            self._mav.mav.statustext_send(severity, payload)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"Could not send QGC status text: {exc}")

    def _validate_dual_gps_selection(self, heartbeat):
        """Validate the legacy PX4 selector when this firmware exposes it."""
        blend_mask = self._read_px4_param(heartbeat, "SENS_GPS_MASK")
        primary = self._read_px4_param(heartbeat, "SENS_GPS_PRIME")
        if blend_mask is None and primary is None:
            self.get_logger().warn(
                "PX4_DUAL_GPS_SELECTION_UNVERIFIED: firmware exposes neither "
                "SENS_GPS_MASK nor SENS_GPS_PRIME; verify sensor_gps instances "
                "and this firmware's GNSS selector before flight"
            )
            return
        if blend_mask is None or primary is None:
            raise RuntimeError(
                "PX4 exposes only part of the legacy dual-GPS selector "
                "(SENS_GPS_MASK/SENS_GPS_PRIME); source priority is ambiguous"
            )
        mask = int(round(blend_mask))
        selected = int(round(primary))
        if mask != 0:
            raise RuntimeError(
                f"SENS_GPS_MASK must be 0 (no blending) for VIO-first failover; current={mask}"
            )
        if (
            self.expected_vio_gps_instance >= 0
            and selected != self.expected_vio_gps_instance
        ):
            raise RuntimeError(
                "SENS_GPS_PRIME must select the verified VIO sensor_gps instance "
                f"{self.expected_vio_gps_instance}; current={selected}"
            )
        self.get_logger().info(
            "PX4 dual-GPS policy verified: blending=off "
            f"preferred_instance={selected}; verify this instance is VIO before flight"
        )

    def _set_live_output_state(self, active: bool, reason: str):
        if active == self.live_output_active and reason == self.live_output_block_reason:
            return
        was_active = self.live_output_active
        self.live_output_active = active
        self.live_output_block_reason = reason
        if active:
            self.get_logger().info(
                f"VIO_GPS_OUTPUT_RESUMED reason={reason}; PX4 may reselect preferred VIO GPS"
            )
            self._send_status_text("VIO GPS: live output resumed", 6)
        elif was_active:
            self.get_logger().error(
                f"VIO_GPS_OUTPUT_STOPPED reason={reason}; waiting for PX4 GPS timeout/fallback"
            )
            if reason == "vio_stale":
                text = "VIO GPS stopped: VIO stale; GPS1 fallback"
            elif reason == "continuity_quarantine":
                text = "VIO GPS stopped: pose gate; GPS1 fallback"
            else:
                text = f"VIO GPS stopped: {reason}"
            self._send_status_text(text, 3)

    def _fail_inertial_propagation(self, reason: str):
        if self.inertial_failure_reason is not None:
            return
        self.inertial_safe = False
        self.inertial_failure_reason = reason
        self.inertial_fallback_to_frozen_pose = True
        self.inertial_displacement_ned = [0.0, 0.0, 0.0]
        self.inertial_yaw_delta = 0.0
        self.get_logger().error(
            f"VIO_INERTIAL_RECOVERY_REJECTED reason={reason}; propagated movement "
            "discarded; recovery target=last trusted frozen VIO pose"
        )
        self._send_status_text("VIO GPS: PX4 propagation rejected; pose frozen", 3)

    def _start_inertial_propagation(self):
        now = time.monotonic()
        self.inertial_active = True
        self.inertial_safe = True
        self.inertial_failure_reason = None
        self.inertial_start_time = now
        self.inertial_displacement_ned = [0.0, 0.0, 0.0]
        self.inertial_integrated_duration_s = 0.0
        self.inertial_yaw_delta = 0.0
        self.inertial_last_yaw = None
        self.inertial_start_odometry = None
        self.inertial_last_odometry = None
        self.inertial_fallback_to_frozen_pose = False
        self.recovery_velocity_agreement_count = 0
        if self.latest_px4_odometry is None or self.latest_px4_yaw is None:
            self._fail_inertial_propagation("px4_motion_stream_unavailable_at_quarantine")
        elif (now - self.latest_px4_odometry[0] > self.inertial_max_message_age_s
              or now - self.latest_px4_yaw[0] > self.inertial_max_message_age_s):
            self._fail_inertial_propagation("px4_motion_stream_stale_at_quarantine")
        else:
            (odom_arrival, odom_time, north, east, vn, ve,
             var_n, var_e, reset_counter) = self.latest_px4_odometry
            # Move the cached sample to the quarantine boundary using its
            # velocity. This prevents motion just before GPS was stopped from
            # being counted a second time.
            age = max(0.0, now - odom_arrival)
            boundary = (
                now, odom_time + age, north + vn * age, east + ve * age,
                vn, ve, var_n, var_e, reset_counter,
            )
            self.inertial_start_odometry = boundary
            self.inertial_last_odometry = boundary
            self.latest_px4_velocity = (now, odom_time + age, vn, ve)
            yaw_arrival, yaw_boot, yaw = self.latest_px4_yaw
            self.inertial_last_yaw = (
                now, yaw_boot + (now - yaw_arrival), yaw
            )
        self.get_logger().warn(
            "VIO_INERTIAL_RECOVERY_STARTED source=PX4_ODOMETRY_position+variance+"
            "reset_counter+ATTITUDE_yaw gps_output=silent"
        )

    def _update_px4_odometry(self, sample):
        """Track PX4 EKF displacement while synthetic GPS output is silent."""
        self.latest_px4_odometry = sample
        _, odom_time, north, east, vn, ve, var_n, var_e, reset_counter = sample
        self.latest_px4_velocity = (sample[0], odom_time, vn, ve)
        if not self.inertial_active or not self.inertial_safe:
            return
        if self.inertial_start_odometry is None:
            self.inertial_start_odometry = sample
            self.inertial_last_odometry = sample
            return
        previous = self.inertial_last_odometry
        dt = odom_time - previous[1]
        if dt <= 0.0:
            self._fail_inertial_propagation(
                f"odometry_timestamp_regression dt_s={dt:.6f}"
            )
            return
        if dt > self.inertial_max_message_gap_s:
            self._fail_inertial_propagation(
                f"odometry_message_gap gap_s={dt:.3f} "
                f"limit_s={self.inertial_max_message_gap_s:.3f}"
            )
            return
        if reset_counter != self.inertial_start_odometry[8]:
            self._fail_inertial_propagation(
                f"px4_estimator_reset old={self.inertial_start_odometry[8]} "
                f"new={reset_counter}"
            )
            return
        uncertainty = math.sqrt(max(0.0, var_n, var_e))
        if uncertainty > self.inertial_max_position_uncertainty_m:
            self._fail_inertial_propagation(
                f"position_uncertainty uncertainty_m={uncertainty:.3f} "
                f"limit_m={self.inertial_max_position_uncertainty_m:.3f}"
            )
            return
        speed = math.hypot(vn, ve)
        if speed > self.continuity.max_speed_mps:
            self._fail_inertial_propagation(
                f"velocity_limit speed_m_s={speed:.3f} "
                f"limit_m_s={self.continuity.max_speed_mps:.3f}"
            )
            return
        old_vn, old_ve = previous[4], previous[5]
        acceleration = math.hypot(vn - old_vn, ve - old_ve) / dt
        if acceleration > self.continuity.max_acceleration_mps2:
            self._fail_inertial_propagation(
                f"acceleration_limit acceleration_m_s2={acceleration:.3f} "
                f"limit_m_s2={self.continuity.max_acceleration_mps2:.3f}"
            )
            return
        # Integrate PX4 EKF velocity rather than copying its absolute local
        # position. This preserves real motion without importing small PX4
        # position corrections into the recovered VIO trajectory.
        self.inertial_displacement_ned[0] += 0.5 * (old_vn + vn) * dt
        self.inertial_displacement_ned[1] += 0.5 * (old_ve + ve) * dt
        self.inertial_integrated_duration_s += dt
        self.inertial_last_odometry = sample

    def _integrate_px4_yaw(self, sample):
        self.latest_px4_yaw = sample
        if not self.inertial_active or not self.inertial_safe:
            return
        if self.inertial_last_yaw is None:
            self.inertial_last_yaw = sample
            return
        _, old_t, old_yaw = self.inertial_last_yaw
        _, new_t, yaw = sample
        dt = new_t - old_t
        if dt <= 0.0:
            self._fail_inertial_propagation(f"yaw_timestamp_regression dt_s={dt:.6f}")
            return
        if dt > self.inertial_max_message_gap_s:
            self._fail_inertial_propagation(
                f"yaw_message_gap gap_s={dt:.3f} "
                f"limit_s={self.inertial_max_message_gap_s:.3f}"
            )
            return
        delta = wrap_pi(yaw - old_yaw)
        yaw_rate = abs(delta) / dt
        if yaw_rate > self.continuity.max_yaw_rate_rad_s:
            self._fail_inertial_propagation(
                f"yaw_rate_limit yaw_rate_deg_s={math.degrees(yaw_rate):.3f} "
                f"limit_deg_s={math.degrees(self.continuity.max_yaw_rate_rad_s):.3f}"
            )
            return
        self.inertial_yaw_delta = wrap_pi(self.inertial_yaw_delta + delta)
        self.inertial_last_yaw = sample

    def _inertial_recovery_target(self):
        if not self.inertial_active:
            return None
        if self.inertial_fallback_to_frozen_pose:
            return (0.0, 0.0, 0.0), 0.0
        now = time.monotonic()
        elapsed = now - self.inertial_start_time
        if elapsed > self.inertial_max_duration_s:
            self._fail_inertial_propagation(
                f"duration_limit elapsed_s={elapsed:.3f} "
                f"limit_s={self.inertial_max_duration_s:.3f}"
            )
            return (0.0, 0.0, 0.0), 0.0
        if (self.latest_px4_odometry is None
                or now - self.latest_px4_odometry[0] > self.inertial_max_message_age_s):
            self._fail_inertial_propagation("odometry_message_stale")
            return (0.0, 0.0, 0.0), 0.0
        if (self.latest_px4_yaw is None
                or now - self.latest_px4_yaw[0] > self.inertial_max_message_age_s):
            self._fail_inertial_propagation("attitude_message_stale")
            return (0.0, 0.0, 0.0), 0.0
        horizontal_distance = math.hypot(
            self.inertial_displacement_ned[0], self.inertial_displacement_ned[1]
        )
        physical_bound = (
            self.continuity.max_speed_mps * self.inertial_integrated_duration_s
        )
        if horizontal_distance > physical_bound + 1e-6:
            self._fail_inertial_propagation(
                f"displacement_limit displacement_m={horizontal_distance:.3f} "
                f"physical_bound_m={physical_bound:.3f}"
            )
            return (0.0, 0.0, 0.0), 0.0
        # Continuity operates before the fixed local-world-to-true-NED heading
        # rotation. Convert the independently propagated NED displacement back
        # into that local frame before establishing the recovered epoch.
        if self.heading_offset_rad is None:
            self._fail_inertial_propagation("heading_alignment_unavailable")
            return (0.0, 0.0, 0.0), 0.0
        local_delta = rotate_ned_heading(
            self.inertial_displacement_ned[0],
            self.inertial_displacement_ned[1],
            0.0,
            -self.heading_offset_rad,
        )
        return local_delta, self.inertial_yaw_delta

    def _finish_inertial_propagation(self):
        displacement = tuple(self.inertial_displacement_ned)
        elapsed = time.monotonic() - self.inertial_start_time
        if self.inertial_fallback_to_frozen_pose:
            resume_eph = self.cruise_eph_m
            source = "frozen_pose"
        else:
            variance = 0.0 if self.latest_px4_odometry is None else max(
                0.0, self.latest_px4_odometry[6], self.latest_px4_odometry[7]
            )
            resume_eph = max(self.cruise_eph_m, math.sqrt(variance))
            source = "px4_odometry"
        self.recovery_resume_eph_m = resume_eph
        self.recovery_resume_time = time.monotonic()
        self.get_logger().info(
            "VIO_INERTIAL_RECOVERY_APPLIED "
            f"north_m={displacement[0]:.3f} east_m={displacement[1]:.3f} "
            f"yaw_deg={math.degrees(self.inertial_yaw_delta):.3f} "
            f"elapsed_s={elapsed:.3f} source={source} resume_eph_m={resume_eph:.3f}"
        )
        self.inertial_active = False

    def _cancel_inertial_propagation(self, reason: str):
        if not self.inertial_active:
            return
        self.get_logger().info(
            f"VIO_INERTIAL_RECOVERY_CANCELLED reason={reason}; "
            "no propagated displacement applied"
        )
        self.inertial_active = False
        self.inertial_safe = False

    def _update_heading_consistency(self, vio_body_yaw_ned: float):
        """Compare independent VIO-global yaw with PX4 yaw; never feed PX4 yaw back."""
        if self.heading_offset_rad is None or self.latest_attitude is None:
            return
        attitude_arrival, _, _, px4_yaw = self.latest_attitude
        if time.monotonic() - attitude_arrival > self.alignment_sensor_max_age_s:
            return
        if attitude_arrival == self.heading_last_attitude_arrival:
            return
        self.heading_last_attitude_arrival = attitude_arrival
        vio_global_yaw = wrap_pi(vio_body_yaw_ned + self.heading_offset_rad)
        disagreement = wrap_pi(px4_yaw - vio_global_yaw)
        self.heading_disagreement_rad = disagreement
        bad = abs(disagreement) > self.heading_disagreement_limit_rad
        if bad:
            self.heading_disagreement_bad_count += 1
            self.heading_disagreement_good_count = 0
            if (not self.heading_quarantine
                    and self.heading_disagreement_bad_count
                    >= self.heading_disagreement_confirmation_samples):
                self.heading_quarantine = True
                self.get_logger().error(
                    "VIO_HEADING_QUARANTINE_STARTED "
                    f"disagreement_deg={math.degrees(disagreement):.2f} "
                    f"limit_deg={math.degrees(self.heading_disagreement_limit_rad):.2f}"
                )
                self._send_status_text("VIO GPS stopped: PX4/VIO heading disagree", 3)
        else:
            self.heading_disagreement_bad_count = 0
            if self.heading_quarantine:
                self.heading_disagreement_good_count += 1
                if (self.heading_disagreement_good_count
                        >= self.heading_disagreement_recovery_samples):
                    self.heading_quarantine = False
                    self.heading_disagreement_good_count = 0
                    self.get_logger().info(
                        "VIO_HEADING_QUARANTINE_CLEARED "
                        f"disagreement_deg={math.degrees(disagreement):.2f}"
                    )
                    self._send_status_text("VIO GPS: heading agreement restored", 6)

    def _recovery_velocity_agrees(self, local_velocity) -> bool:
        """Require recovered VIO motion to agree with PX4 before handoff."""
        if self.inertial_fallback_to_frozen_pose:
            # PX4 propagation was rejected, so it is not a trusted comparison
            # source. Stable VIO samples still gate recovery in LocalPoseContinuity.
            return True
        if self.latest_px4_odometry is None or self.heading_offset_rad is None:
            self.recovery_velocity_agreement_count = 0
            return False
        vx, vy = self.continuity._rotate_xy(
            local_velocity[0], local_velocity[1], self.continuity.rotation
        )
        vio_n, vio_e, _ = rotate_ned_heading(
            vx, vy, 0.0, self.heading_offset_rad
        )
        px4_n, px4_e = self.latest_px4_odometry[4:6]
        residual = math.hypot(vio_n - px4_n, vio_e - px4_e)
        if residual <= self.recovery_velocity_agreement_m_s:
            self.recovery_velocity_agreement_count += 1
        else:
            if self.recovery_velocity_agreement_count:
                self.get_logger().warn(
                    "VIO_RECOVERY_VELOCITY_AGREEMENT_RESET "
                    f"residual_m_s={residual:.3f} "
                    f"limit_m_s={self.recovery_velocity_agreement_m_s:.3f}"
                )
            self.recovery_velocity_agreement_count = 0
        return (
            self.recovery_velocity_agreement_count
            >= self.recovery_velocity_agreement_samples
        )

    def _odom_callback(self, msg: Odometry):
        if msg.child_frame_id != self.expected_child_frame:
            if msg.child_frame_id != self.rejected_odom_frame:
                self.rejected_odom_frame = msg.child_frame_id
                self.get_logger().error(
                    "VIO_FRAME_REJECTED "
                    f"received={msg.child_frame_id!r} required={self.expected_child_frame!r}; "
                    "GPS output remains gated"
                )
            return
        self.rejected_odom_frame = None
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
        recovery_agrees = (
            self._recovery_velocity_agrees((vn, ve, vd))
            if self.inertial_active else False
        )
        inertial_target = (
            self._inertial_recovery_target()
            if self.inertial_active and recovery_agrees else None
        )
        recovery_displacement = None if inertial_target is None else inertial_target[0]
        recovery_yaw_delta = 0.0 if inertial_target is None else inertial_target[1]
        continuity = self.continuity.update(
            (north, east, down), raw_body_yaw_ned, (vn, ve, vd),
            raw_yaw_rate_ned, sample_time, frame_key,
            recovery_displacement=recovery_displacement,
            recovery_yaw_delta=recovery_yaw_delta,
        )
        # Before live handoff there is no VIO GPS trajectory to preserve. A
        # startup relocalization should establish a fresh local origin rather
        # than invoke PX4-aided in-flight recovery.
        if continuity.recovering and self.phase != Phase.LIVE:
            self.continuity.reset()
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
            if continuity.event == "quarantine_started" and self.phase == Phase.LIVE:
                self._start_inertial_propagation()
            elif continuity.event == "recovery_completed":
                self._finish_inertial_propagation()
            elif continuity.event == "isolated_outlier_rejected":
                self._cancel_inertial_propagation("isolated_outlier")
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
        self._update_heading_consistency(continuity.yaw)
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
                self._send_status_text("VIO GPS: VIO ready, aligning heading", 6)

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
        att_time, roll, pitch, _yaw = self.latest_attitude
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
        self._send_status_text("VIO GPS: heading aligned", 6)

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
        if self.recovery_resume_eph_m is not None and self.recovery_resume_time is not None:
            if self.recovery_accuracy_tighten_s <= 0.0:
                self.recovery_resume_eph_m = None
                self.recovery_resume_time = None
            else:
                fraction = min(
                    1.0,
                    (time.monotonic() - self.recovery_resume_time)
                    / self.recovery_accuracy_tighten_s,
                )
                eph = (
                    self.recovery_resume_eph_m
                    + (self.cruise_eph_m - self.recovery_resume_eph_m) * fraction
                )
                if fraction >= 1.0:
                    self.recovery_resume_eph_m = None
                    self.recovery_resume_time = None
                return max(self.cruise_eph_m, eph), self.cruise_epv_m
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
        self._send_status_text("VIO GPS: live VIO active", 6)

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
            if (self.px4_target_sysid is not None
                    and int(msg.get_srcSystem()) != self.px4_target_sysid):
                continue
            if (
                message_type == "HEARTBEAT"
                and self.px4_target_sysid is not None
                and int(msg.get_srcSystem()) == self.px4_target_sysid
            ):
                self.last_px4_heartbeat_time = now
                continue
            if message_type == "ATTITUDE":
                roll, pitch, yaw = float(msg.roll), float(msg.pitch), float(msg.yaw)
                boot_time = float(msg.time_boot_ms) * 1e-3
                if all(math.isfinite(value) for value in (roll, pitch, yaw, boot_time)):
                    # PX4 yaw is excluded from initial earth-heading alignment.
                    # Its delta is used only while HIL_GPS is silent in quarantine.
                    self.latest_attitude = (now, roll, pitch, yaw)
                    self._integrate_px4_yaw((now, boot_time, yaw))
                continue
            if message_type == "ODOMETRY":
                sample_time = float(msg.time_usec) * 1e-6
                north, east = float(msg.x), float(msg.y)
                vn, ve = float(msg.vx), float(msg.vy)
                pose_covariance = list(msg.pose_covariance)
                var_n = float(pose_covariance[0])
                var_e = float(pose_covariance[6])
                reset_counter = int(msg.reset_counter)
                # This PX4 stream publishes estimator output in LOCAL_NED and
                # marks it as MAV_ESTIMATOR_TYPE_AUTOPILOT. Reject any other
                # producer/frame instead of silently mixing conventions.
                valid_source = (
                    int(msg.frame_id) == 1
                    and int(msg.estimator_type) == 8
                )
                values = (sample_time, north, east, vn, ve, var_n, var_e)
                if not valid_source or not all(math.isfinite(value) for value in values):
                    if self.inertial_active:
                        self._fail_inertial_propagation(
                            "invalid_px4_odometry_source_frame_or_covariance"
                        )
                else:
                    self._update_px4_odometry(
                        (now, sample_time, north, east, vn, ve,
                         var_n, var_e, reset_counter)
                    )
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
            # `spoof_until_vio=true` permits an immediate handoff when VIO is
            # already healthy. With it disabled, always provide the complete
            # timed stationary bootstrap before switching to live VIO.
            if (
                self.spoof_until_vio
                and vio_is_fresh
                and self.heading_offset_rad is not None
            ):
                self._enter_live("rc_trigger_vio_already_ready")
            else:
                self.phase = Phase.SPOOF
                self.start_time = now
                self.get_logger().info(
                    f"GPS spoof started from RC channel {self.rc_channel} "
                    f"for up to {self.spoof_duration_s:.1f}s"
                )
                self._send_status_text("VIO GPS: home spoof active (15 s max)", 6)

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
                self._send_status_text("VIO GPS: spoof expired; output stopped", 4)

    def _tick(self):
        self._poll_mavlink()
        if self.inertial_active:
            # Enforce duration and stream-freshness guards even if cuVSLAM has
            # stopped producing callbacks during quarantine.
            self._inertial_recovery_target()
        if self.phase == Phase.WAITING:
            self._set_live_output_state(False, "waiting_for_activation")
            return
        self._maybe_end_spoof_by_timer()
        if self.phase == Phase.WAITING:
            self._set_live_output_state(False, "spoof_expired")
            return

        if (
            self.phase == Phase.LIVE
            and self.continuity_recovering
        ):
            self._set_live_output_state(False, "continuity_quarantine")
            return

        if self.phase == Phase.LIVE and self.heading_quarantine:
            self._set_live_output_state(False, "heading_disagreement")
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
            self._set_live_output_state(False, "vio_stale")
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
            if self.horizontal_only_output:
                fix["alt"] = self.home_alt
                fix["vn"] = 0.0
                fix["ve"] = 0.0
                fix["vd"] = 0.0
            self._set_live_output_state(True, "all_health_gates_passed")

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
        base_args = (
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
            clamp_int(self.satellites, 0, 254),
        )
        if self._hil_gps_extensions_supported is not False:
            try:
                self._mav.mav.hil_gps_send(
                    *base_args,
                    clamp_int(self.gps_id, 0, 255),
                    0,  # yaw unavailable; PX4 also forces heading NaN
                )
                self._hil_gps_extensions_supported = True
                return
            except TypeError:
                self._hil_gps_extensions_supported = False
                self.get_logger().warn(
                    "Jetson pymavlink HIL_GPS has no id/yaw extensions; "
                    "using compatible base message (PX4 ignores those fields)"
                )
        self._mav.mav.hil_gps_send(*base_args)

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
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
