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
from sensor_msgs.msg import Imu
from rcl_interfaces.srv import GetParameters
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
from vio_px4_bridge.geo_utils import mavlink_velocity_to_ned
from vio_px4_bridge.geo_utils import rotate_ned_heading
from vio_px4_bridge.geo_utils import rotate_vector_by_quaternion_xyzw
from vio_px4_bridge.geo_utils import wrap_pi
from vio_px4_bridge.geo_utils import yaw_from_quaternion_xyzw
from vio_px4_bridge.local_continuity import LocalPoseContinuity
from vio_px4_bridge.global_origin import GlobalOriginSetup
from vio_px4_bridge.recovery_yaw import RecoveryYawAgreement
from vio_px4_bridge.recovery_yaw import timestamp_order
from vio_px4_bridge.spoof_readiness import SpoofReadiness
from vio_px4_bridge.startup_heading import Px4StartupHeading, heading_source
from vio_px4_bridge.external_yaw import ExternalYaw, NavigationTiming, rpy


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
        self.declare_parameter("camera_imu_required", True)

        # Home / EKF origin (spoof target and LLA anchor for live VIO)
        self.declare_parameter("home_lat_deg", 40.4433)
        self.declare_parameter("home_lon_deg", -79.9436)
        self.declare_parameter("home_alt_m", 200.0)

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

        # PX4 owns earth heading at startup and recovery. Never add declination
        # to ATTITUDE yaw: PX4 has already applied its heading corrections.
        self.declare_parameter("heading_source", "px4")  # px4 | manual
        self.declare_parameter("manual_heading_deg", 0.0)
        self.declare_parameter("external_yaw_enabled", False)
        self.declare_parameter("external_yaw_noise_deg", 1.0)
        self.declare_parameter("external_yaw_rate_hz", 30.0)
        self.declare_parameter("external_yaw_max_age_s", 0.2)
        self.declare_parameter("external_yaw_startup_tolerance_deg", 10.0)
        # Legacy launch parameters accepted but no longer used for alignment.
        self.declare_parameter("mag_declination_source", "table")
        self.declare_parameter("mag_declination_deg", 0.0)
        self.declare_parameter("mag_declination_table_path", "")
        # Compatibility only: nonzero legacy mounting corrections are rejected.
        # cuVSLAM applies full extrinsics before publishing the body pose.
        self.declare_parameter("child_to_body_yaw_deg", 0.0)
        self.declare_parameter("mag_roll_offset_deg", 0.0)
        self.declare_parameter("mag_pitch_offset_deg", 0.0)
        self.declare_parameter("mag_yaw_offset_deg", 0.0)
        self.declare_parameter("alignment_samples", 30)
        self.declare_parameter("alignment_max_std_deg", 5.0)
        self.declare_parameter("alignment_max_speed_m_s", 0.25)
        self.declare_parameter("alignment_sensor_max_age_s", 0.5)
        self.declare_parameter("continuity_max_gap_s", 1.0)
        self.declare_parameter("continuity_recovery_samples", 10)
        self.declare_parameter("continuity_confirmation_samples", 3)
        self.declare_parameter("continuity_max_speed_m_s", 10.0)
        self.declare_parameter("continuity_max_acceleration_m_s2", 5.0)
        self.declare_parameter("continuity_max_yaw_rate_deg_s", 360.0)
        # Legacy parameter accepted but ignored: recovery has no fixed time cutoff.
        self.declare_parameter("inertial_max_duration_s", 2.0)
        self.declare_parameter("inertial_max_message_age_s", 0.25)
        self.declare_parameter("inertial_max_message_gap_s", 0.25)
        self.declare_parameter("inertial_reorder_tolerance_s", 0.05)
        self.declare_parameter("inertial_max_position_uncertainty_m", 3.0)
        self.declare_parameter("recovery_velocity_agreement_m_s", 2.0)
        self.declare_parameter("recovery_velocity_agreement_samples", 3)
        self.declare_parameter("recovery_yaw_agreement_deg", 20.0)
        self.declare_parameter("recovery_yaw_agreement_samples", 5)
        # Accepted for old launch/config compatibility; no longer a recovery gate.
        self.declare_parameter("recovery_reset_rotation_budget_deg", 2.0)
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

        requested_heading_source = self.get_parameter("heading_source").value
        self.heading_source = heading_source(requested_heading_source)
        if str(requested_heading_source).strip().lower() == "compass":
            self.get_logger().warn(
                "Legacy heading_source=compass migrated to px4; bridge declination "
                "and magnetometer offsets are ignored. Verify PX4 heading before starting."
            )
        if self.transport == "ros2" and self.heading_source == "px4":
            raise ValueError(
                "heading_source=px4 requires MAVLink ATTITUDE; "
                "use transport=mavlink or an independently verified manual heading"
            )
        self.manual_heading_rad = math.radians(
            float(self.get_parameter("manual_heading_deg").value)
        )
        legacy_child_yaw = float(self.get_parameter("child_to_body_yaw_deg").value)
        if not math.isfinite(self.manual_heading_rad):
            raise ValueError("Manual heading must be finite")
        if not math.isfinite(legacy_child_yaw) or legacy_child_yaw != 0.0:
            raise ValueError(
                "child_to_body_yaw_deg must be zero: camera mounting belongs to "
                "cuVSLAM extrinsics; the bridge requires body-frame odometry"
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
        self.startup_heading = Px4StartupHeading(
            self.alignment_sample_count, self.alignment_max_std_rad,
            self.alignment_max_speed, self.alignment_sensor_max_age_s,
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
        self.inertial_max_message_age_s = max(
            0.05, float(self.get_parameter("inertial_max_message_age_s").value)
        )
        self.inertial_max_message_gap_s = max(
            0.02, float(self.get_parameter("inertial_max_message_gap_s").value)
        )
        self.inertial_reorder_tolerance_s = max(
            0.0, float(self.get_parameter("inertial_reorder_tolerance_s").value)
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
        self.recovery_yaw_agreement_deg = max(
            0.1, float(self.get_parameter("recovery_yaw_agreement_deg").value)
        )
        self.recovery_yaw_agreement_samples = max(
            1, int(self.get_parameter("recovery_yaw_agreement_samples").value)
        )
        self.recovery_yaw_agreement = RecoveryYawAgreement(
            math.radians(self.recovery_yaw_agreement_deg),
            self.recovery_yaw_agreement_samples,
        )
        self.recovery_accuracy_tighten_s = max(
            0.0, float(self.get_parameter("recovery_accuracy_tighten_s").value)
        )

        self.external_yaw = None
        if self.transport == "mavlink" and bool(self.get_parameter("use_sim_time").value):
            raise ValueError("Hardware navigation cannot use ROS simulated time")
        if bool(self.get_parameter("external_yaw_enabled").value):
            if self.transport != "mavlink":
                raise ValueError("External yaw requires MAVLink")
            if bool(self.get_parameter("use_sim_time").value):
                raise ValueError("External yaw on hardware cannot use ROS simulated time")
            self.external_yaw = ExternalYaw(
                float(self.get_parameter("external_yaw_noise_deg").value),
                float(self.get_parameter("external_yaw_rate_hz").value),
                float(self.get_parameter("external_yaw_max_age_s").value),
            )
        # Camera protection and FCU time matching are mandatory even when the
        # operator disables the optional yaw transmitter. Keep the existing
        # parameter name for compatibility with saved launcher configurations.
        self.navigation_timing = self.external_yaw or NavigationTiming(
            float(self.get_parameter("external_yaw_max_age_s").value))
        self.external_yaw_startup_tolerance = math.radians(
            float(self.get_parameter("external_yaw_startup_tolerance_deg").value))
        if not math.isfinite(self.external_yaw_startup_tolerance) or not (
                0 < self.external_yaw_startup_tolerance <= math.radians(20)):
            raise ValueError("External yaw startup tolerance must be >0 and <=20 degrees")
        self.external_sample_time = None

        self.phase = Phase.WAITING if self.rc_trigger_enabled else Phase.SPOOF
        self.start_time = None if self.rc_trigger_enabled else time.monotonic()
        self.live_start_time = None
        self.origin_setup = GlobalOriginSetup(self.home_lat, self.home_lon, self.home_alt)
        self.px4_armed = None
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
        # Display-only metadata. Never substitute an unsent candidate or a
        # configured accuracy for the values actually written to the transport.
        self.latest_fix_ros_stamp_s = None
        self.last_gps_output = None
        self.vio_stale_warned = False
        self.last_rc_position = None
        self.messages_sent = 0
        self.heading_offset_rad = None
        self.latest_px4_velocity = None
        self.latest_px4_yaw = None
        self.latest_px4_odometry = None
        self.recovery_reset_stamp = None
        self.inertial_active = False
        self.inertial_position_safe = False
        self.inertial_heading_safe = False
        self.inertial_heading_failure = None
        self.inertial_failure_reason = None
        self.inertial_start_time = None
        self.inertial_displacement_ned = [0.0, 0.0, 0.0]
        self.inertial_integrated_duration_s = 0.0
        self.inertial_position_uncertainty_m = 0.0
        self.inherited_position_uncertainty_m = 0.0
        self.inertial_yaw_delta = 0.0
        self.inertial_last_yaw = None
        self.inertial_start_odometry = None
        self.inertial_last_odometry = None
        self.inertial_fallback_to_frozen_pose = False
        self.inertial_reordered_odometry_drops = 0
        self.inertial_reordered_yaw_drops = 0
        self.recovery_velocity_agreement_count = 0
        self.recovery_velocity_last_timestamp = None
        self.recovery_velocity_sample_valid = False
        self.recovery_yaw_agreement_state = "idle"
        self.recovery_resume_eph_m = None
        self.frozen_position_uncertainty_m = 0.0
        self.recovery_resume_time = None
        self.logged_odom_frames = False
        self.continuity_recovering = False
        self.live_output_active = False
        self.live_output_block_reason = "not_live"
        self.last_px4_heartbeat_time = None
        self.px4_target_sysid = None
        self.status_file = str(self.get_parameter("status_file").value).strip()
        self.camera_readiness = SpoofReadiness(
            require_imu=bool(self.get_parameter("camera_imu_required").value)
        )
        self.spoof_failure_reason = None
        self._camera_param_future = None
        self._camera_param_sent = 0.0
        self._last_handoff_reason = None

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
        self.create_subscription(Imu, "/imu", self._camera_imu_callback, QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=100,
        ))
        self._camera_param_client = self.create_client(
            GetParameters, "/visual_slam_node/get_parameters"
        )
        self.create_timer(0.5, self._check_camera_fusion)
        self.create_timer(1.0 / self.rate_hz, self._tick)
        if self.transport == "mavlink":
            # Fast dispatch keeps TIMESYNC replies inside PX4's 10 ms RTT limit.
            # Measurement rate remains separately limited to 30..50 Hz.
            self.create_timer(0.005, self._external_yaw_tick)
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
        yaw_reason = "disabled" if self.external_yaw is None else self._external_yaw_reason()
        yaw_output = ({} if self.external_yaw is None else self.external_yaw.output_status(
            now, output_allowed=self.phase != Phase.WAITING and yaw_reason is None))
        gps_output = self._gps_output_status(now)
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
            "heading_source": self.heading_source,
            "external_yaw_enabled": self.external_yaw is not None,
            "external_yaw_messages": 0 if self.external_yaw is None else self.external_yaw.sent,
            "external_yaw_sigma_deg": (None if self.external_yaw is None else
                                       math.degrees(math.sqrt(self.external_yaw.reported_variance))),
            "external_yaw_reference_policy": (None if self.external_yaw is None else
                                              "inherited_reference_floor"),
            "external_yaw_status": yaw_reason or "ready",
            **yaw_output,
            "external_yaw_clock_ready": (False if self.external_yaw is None else
                                         self.external_yaw.clock.ready(now)),
            "startup_heading_status": (
                self.startup_heading.reason if self.heading_source == "px4"
                else "explicit manual heading"
            ),
            "pose_gate_quarantine": self.continuity_recovering,
            "pose_gate_reason": (
                self.continuity.pending_quarantine or self.continuity.suspect_reason
                if self.continuity_recovering else None
            ),
            "inertial_recovery_active": self.inertial_active,
            "inertial_recovery_safe": self.inertial_position_safe and self.inertial_heading_safe,
            "recovery_position_safe": self.inertial_position_safe,
            "recovery_heading_safe": self.inertial_heading_safe,
            "recovery_heading_failure": self.inertial_heading_failure,
            "recovery_reset_stamp": self.recovery_reset_stamp,
            "recovery_heading_reference": "current_fcu_heading",
            "inertial_fallback_to_frozen_pose": self.inertial_fallback_to_frozen_pose,
            "inertial_recovery_failure": self.inertial_failure_reason,
            "frozen_position_uncertainty_m": self.frozen_position_uncertainty_m,
            "inherited_position_uncertainty_m": self.inherited_position_uncertainty_m,
            "inertial_reordered_odometry_drops": self.inertial_reordered_odometry_drops,
            "inertial_reordered_yaw_drops": self.inertial_reordered_yaw_drops,
            "inertial_displacement_n_m": self.inertial_displacement_ned[0],
            "inertial_displacement_e_m": self.inertial_displacement_ned[1],
            "px4_position_uncertainty_m": (
                None if self.latest_px4_odometry is None else
                math.sqrt(max(0.0, self.latest_px4_odometry[6], self.latest_px4_odometry[7]))
            ),
            "recovery_velocity_agreement_count": self.recovery_velocity_agreement_count,
            "recovery_velocity_sample_valid": self.recovery_velocity_sample_valid,
            "recovery_yaw_agreement_state": self.recovery_yaw_agreement_state,
            "recovery_yaw_agreement_count": self.recovery_yaw_agreement.good_samples,
            "recovery_yaw_residual_deg": (
                None if self.recovery_yaw_agreement.last_residual_rad is None else
                math.degrees(self.recovery_yaw_agreement.last_residual_rad)
            ),
            "hil_gps_output_active": self.live_output_active,
            **gps_output,
            "output_block_reason": self.live_output_block_reason,
            "spoof_failure_reason": self.spoof_failure_reason,
            "global_origin_verified": self.origin_setup.confirmed,
            "global_origin_altitude_m": self.home_alt,
            "handoff_check_reason": self._handoff_block_reason(),
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

    def _gps_output_status(self, now):
        """Read the last successful GPS write without changing navigation state."""
        output = self.last_gps_output
        return {
            "hil_gps_lat_deg": None if output is None else output["lat"],
            "hil_gps_lon_deg": None if output is None else output["lon"],
            "hil_gps_eph_m": None if output is None else output["eph"],
            "hil_gps_sample_age_s": (
                None if output is None or output["sample_time"] is None
                else now - output["sample_time"]
            ),
            "hil_gps_send_age_s": None if output is None else now - output["send_time"],
            "hil_gps_sample_source": None if output is None else output["source"],
        }

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
            self.px4_armed = bool(int(heartbeat.base_mode) & 128)
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
            height_ref = self._read_px4_param(heartbeat, "EKF2_HGT_REF")
            noaid_tout_us = self._read_px4_param(heartbeat, "EKF2_NOAID_TOUT")
            ev_ctrl = self._read_px4_param(heartbeat, "EKF2_EV_CTRL")
            expected_ev_ctrl = 8 if self.external_yaw is not None else 0
            if ev_ctrl is None or ev_ctrl != expected_ev_ctrl:
                raise RuntimeError(
                    f"Path A requires EKF2_EV_CTRL={expected_ev_ctrl}; current={ev_ctrl}. "
                    "Do not fuse VO position/height twice or leave EV yaw enabled in GPS-only mode.")
            if self.external_yaw is not None:
                required = {"EKF2_EV_NOISE_MD": 0,
                            "EKF2_MAG_TYPE": 5 if self.heading_source == "manual" else 6}
                if self.px4_armed is not False:
                    raise RuntimeError("Start GPS + external yaw only while DISARMED")
                for name, expected in required.items():
                    value = self._read_px4_param(heartbeat, name)
                    if value is None or not math.isfinite(value) or value != expected:
                        raise RuntimeError(f"GPS + external yaw requires {name}={expected}; current={value}")
                self.get_logger().warn(
                    "External yaw enabled: EV yaw ONLY, message covariance, no continuous mag fusion. "
                    "Both position and yaw stop together on quarantine. Bench verification required."
                )
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
                    f"and GPS yaw is not fused (current={gps_ctrl_bits})"
                )
            noaid_tout_s = float(noaid_tout_us) * 1e-6
            if height_ref is None or int(round(height_ref)) not in (0, 2):
                raise RuntimeError(
                    "Path A requires EKF2_HGT_REF=0 (barometer) or 2 (rangefinder); "
                    "bridge altitude is only a global reference, not a height sensor"
                )
            self.get_logger().info(
                f"Height source remains PX4 {'barometer' if int(height_ref) == 0 else 'rangefinder'}; "
                f"GPS height fusion disabled; global reference altitude={self.home_alt:.1f} m AMSL"
            )
            self.get_logger().info(
                "PX4 parameter gates passed: MAV_USEHILGPS=1 "
                f"EKF2_GPS_CTRL={gps_ctrl_bits} EKF2_NOAID_TOUT={noaid_tout_s:.3f}s"
            )
            self.get_logger().info(
                "Recovery uses guarded PX4 position displacement without a fixed "
                "duration cutoff; PX4 position-loss failsafes remain unchanged"
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
            self.px4_target_compid = heartbeat.get_srcComponent()
            self._request_mavlink_stream(heartbeat, 111, 50.0)  # TIMESYNC
            if self.heading_source == "px4":
                self.get_logger().info(
                    "Startup heading uses PX4 ATTITUDE yaw; no bridge compass/declination correction"
                )
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
            if (msg.get_srcSystem() != heartbeat.get_srcSystem()
                    or msg.get_srcComponent() != heartbeat.get_srcComponent()):
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
            self._send_status_text("VIO GPS+yaw: live output resumed" if
                                   getattr(self, "external_yaw", None) is not None else
                                   "VIO GPS: live output resumed", 6)
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
            if getattr(self, "external_yaw", None) is not None:
                text = f"VIO GPS+yaw stopped: {reason}"
            self._send_status_text(text, 3)

    def _invalidate_navigation(self, reason):
        """One latch for both measurement streams; telemetry reception continues."""
        aid = self.navigation_timing
        changed = aid.input_error != reason
        aid.input_error = reason
        if self.phase != Phase.LIVE:
            self.camera_readiness.odom.reset(reason)
        if self.phase == Phase.LIVE and changed and reason != "continuity_quarantine":
            self.continuity.quarantine(reason)
        if self.phase == Phase.LIVE and not self.continuity_recovering:
            self.continuity.quarantine(reason)
            self.continuity_recovering = True
            self._start_inertial_propagation()
            self._reset_recovery_agreement()
        self._set_live_output_state(False, reason)

    def _navigation_reason(self):
        """Shared GPS/yaw gate; a frozen accepted pose cannot become fresh again."""
        timing = self.navigation_timing
        now = time.monotonic()
        if self.continuity_recovering:
            return "continuity_quarantine"
        if timing.input_error:
            return timing.input_error
        if timing.clock.generation != timing.generation:
            return "PX4_clock_reset"
        if self.transport == "mavlink" and (
                self.last_px4_heartbeat_time is None or
                not 0 <= now-self.last_px4_heartbeat_time <= self.heartbeat_timeout_s):
            return "PX4_heartbeat_stale"
        if self.rejected_odom_frame is not None:
            return "invalid_body_frame"
        if (timing.trusted_capture_s is None or
                not 0 <= now-timing.trusted_capture_s <= timing.max_age_s):
            return "camera_capture_stale"
        return None

    def _external_yaw_reason(self):
        aid = self.external_yaw
        now = time.monotonic()
        reason = self._navigation_reason()
        if reason:
            return reason
        if self.heading_offset_rad is None:
            return "heading_alignment_incomplete"
        if self.rejected_odom_frame is not None:
            return "invalid_body_frame"
        if self.phase != Phase.LIVE:
            if self.px4_armed is not False:
                return "external_heading_start_requires_disarmed"
            if not getattr(self, "external_startup_stationary", False):
                return "external_heading_start_requires_stationary"
            reason = self.camera_readiness.reason(now)
            if reason:
                return reason
        return aid.reason(now)

    def _verify_external_yaw_startup(self, speed):
        aid = self.external_yaw
        self.external_startup_stationary = speed <= self.alignment_max_speed
        if not self.external_startup_stationary or aid.pose is None:
            aid.startup_agreement = 0
            aid.startup_verified = False
            return
        sample = aid.matched_fcu(aid.pose[0], time.monotonic(), self.inertial_max_message_age_s)
        if sample is None:
            aid.startup_verified = False
            return
        if self.heading_source == "px4":
            # Accurate relative turns do not remove uncertainty in the adopted
            # absolute reference. Keep the prior conservative inheritance.
            aid.variance_floor = max(aid.variance_floor, sample.yaw_variance)
        if self.heading_source == "manual" and aid.sent == 0:
            return  # need a readback AFTER supplying the manual seed
        if abs(wrap_pi(sample.yaw-aid.pose[3])) > self.external_yaw_startup_tolerance:
            aid.startup_agreement = 0
            aid.startup_verified = False
        elif aid.startup_evidence != sample.evidence_s:
            aid.startup_agreement += 1
        aid.startup_evidence = sample.evidence_s
        if aid.startup_agreement >= self.recovery_yaw_agreement_samples:
            if not aid.startup_verified:
                self.get_logger().info(
                    f"VIO_YAW_STARTUP_VERIFIED source={self.heading_source} "
                    f"sigma_deg={math.degrees(math.sqrt(aid.reported_variance)):.2f}; "
                    "retaining heading reference uncertainty")
            aid.startup_verified = True

    def _external_yaw_tick(self):
        """Common fast TIMESYNC/watchdog service, with optional rate-limited EV."""
        self._poll_mavlink()
        now = time.monotonic()
        aid = self.navigation_timing
        request = aid.clock.request(now)
        if request is not None:
            self._mav.mav.timesync_send(*request)
        if aid.generation != aid.clock.generation:
            self._invalidate_navigation("PX4_clock_reset")
            aid.generation = aid.clock.generation
            aid.history.clear()
        if self.phase == Phase.LIVE:
            reason = self._navigation_reason()
            if reason:
                self._invalidate_navigation(reason)
                return
        aid = self.external_yaw
        if aid is None:
            return
        if self.phase == Phase.WAITING:
            return
        reason = self._external_yaw_reason()
        if reason:
            if self.phase == Phase.LIVE:
                self._invalidate_navigation(reason)
            return
        # During spoofing, GPS is deliberately stationary. EV starts only
        # after camera validation; this also permits a MANUAL heading seed
        # with MAG_TYPE=5 before requiring PX4 heading readback for live GPS.
        eph, epv = self._select_accuracy()
        packet = aid.packet(now, eph, epv)
        if packet is not None:
            try:
                self._mav.mav.odometry_send(**packet)
            except Exception as exc:  # transport failure must stop GPS as well
                self._invalidate_navigation(f"external_yaw_transport_error: {exc}")
                return
            aid.mark_sent(now)

    def _matched_recovery_target(self, velocity, raw_yaw):
        """Time-matched FCU position AND yaw; no output while checks accumulate."""
        aid = self.navigation_timing
        # Keep existing position/reset/physical guards. They may explicitly
        # discard translation, but do not permanently latch healthy yaw out.
        aid.recovery_pose = None
        if self._inertial_recovery_target() is None:
            self._reset_recovery_agreement()
            return None
        sample = aid.matched_fcu(self.external_sample_time, time.monotonic(),
                                 self.inertial_max_message_age_s)
        if sample is None:
            self._reset_recovery_agreement()
            self.recovery_yaw_agreement_state = (
                "waiting_for_time_matched_fcu" if aid.clock.ready(time.monotonic())
                else "waiting_for_time_sync")
            return None
        # A delayed camera capture can still match buffered poses from BEFORE
        # the latest FCU reset. Those poses are individually valid but no longer
        # define the current heading reference. Do not count them as recovery
        # evidence or feed the old reference back to PX4. The time check also
        # handles reset-counter wraparound/repeated resets during one loss.
        if (sample.reset_counter != aid.history[-1].reset_counter
                or (self.recovery_reset_stamp is not None
                    and sample.boot_s < self.recovery_reset_stamp)):
            self._reset_recovery_agreement()
            self.recovery_yaw_agreement_state = "waiting_for_post_reset_fcu"
            return None
        self.inertial_heading_safe = True
        self.inertial_heading_failure = None
        yaw_delta = wrap_pi(sample.yaw - self.heading_offset_rad
                           - self.continuity.recovery_anchor_yaw)
        rotation = wrap_pi(sample.yaw - raw_yaw)
        vn, ve, _ = rotate_ned_heading(*velocity, rotation)
        residual = math.hypot(vn-sample.vn, ve-sample.ve)
        if not self.recovery_velocity_sample_valid or residual > self.recovery_velocity_agreement_m_s:
            if self.recovery_velocity_agreement_count:
                self.get_logger().warn(
                    f"VIO_RECOVERY_VELOCITY_AGREEMENT_RESET residual_m_s={residual:.3f} "
                    f"limit_m_s={self.recovery_velocity_agreement_m_s:.3f} time_matched=true")
            self.recovery_velocity_agreement_count = 0
        elif sample.evidence_s != self.recovery_velocity_last_timestamp:
            self.recovery_velocity_agreement_count += 1
        self.recovery_velocity_last_timestamp = sample.evidence_s
        yaw_result = self.recovery_yaw_agreement.update(raw_yaw, yaw_delta, sample.evidence_s)
        self.recovery_yaw_agreement_state = "confirmed" if yaw_result.agreed else "collecting"
        if yaw_result.event == "disagreement_reset":
            self.recovery_yaw_agreement_state = "disagreeing"
            self.get_logger().warn(
                f"VIO_RECOVERY_YAW_AGREEMENT_RESET residual_deg={math.degrees(yaw_result.residual_rad):.3f} "
                f"limit_deg={self.recovery_yaw_agreement_deg:.3f} time_matched=true")
        # Both windows always update; never short-circuit one behind the other.
        if (self.recovery_velocity_agreement_count < self.recovery_velocity_agreement_samples
                or not yaw_result.agreed):
            return None
        if self.inertial_position_safe:
            boundary = aid.boundary
            if boundary is None or boundary.reset_counter != sample.reset_counter:
                self._reject_position_propagation("time_matched_boundary_unavailable_or_reset")
            else:
                self.inertial_displacement_ned = [
                    sample.north-boundary.north, sample.east-boundary.east, 0.]
                self.inertial_integrated_duration_s = sample.boot_s-boundary.boot_s
                if (self.inertial_integrated_duration_s < 0 or
                        math.hypot(*self.inertial_displacement_ned[:2]) >
                        self.continuity.max_speed_mps*self.inertial_integrated_duration_s + 1e-6):
                    self._reject_position_propagation("matched_displacement_limit")
        self.inertial_yaw_delta = yaw_delta
        self.inertial_position_uncertainty_m = max(
            self.inertial_position_uncertainty_m, math.sqrt(sample.position_variance))
        aid.recovery_pose = sample
        delta = rotate_ned_heading(*self.inertial_displacement_ned, -self.heading_offset_rad)
        return delta, yaw_delta

    def _reject_position_propagation(self, reason: str):
        """Discard translation only; current FCU heading can still be used."""
        if self.inertial_failure_reason is not None:
            return
        self.inertial_position_safe = False
        self.inertial_failure_reason = reason
        self.inertial_fallback_to_frozen_pose = True
        self.inertial_displacement_ned = [0.0, 0.0, 0.0]
        self.get_logger().error(
            f"VIO_RECOVERY_POSITION_FROZEN reason={reason}; propagated movement "
            "discarded; fresh FCU heading and camera agreement still required"
        )
        self._send_status_text("VIO recovery: position frozen; checking heading", 4)

    def _reject_heading_recovery(self, reason: str):
        """Pause recovery until fresh valid FCU heading is available again."""
        changed = self.inertial_heading_failure != reason
        self.inertial_heading_safe = False
        self.inertial_heading_failure = reason
        self._reset_recovery_agreement()
        if changed:
            self.get_logger().warn(
                f"VIO_RECOVERY_HEADING_WAIT reason={reason}; waiting for fresh FCU heading"
            )
            self._send_status_text("VIO recovery: waiting for FCU heading", 4)

    def _fail_inertial_propagation(self, reason: str):
        # Position propagation remains rejected for this quarantine. Fresh
        # telemetry may rebuild recovery agreement, without restoring movement.
        self._reject_position_propagation(reason)
        self.recovery_velocity_sample_valid = False
        self._reset_recovery_agreement()

    def _start_inertial_propagation(self):
        # Another candidate can fail during the same quarantine. Keep the
        # original position reference and latched position failure.
        if self.inertial_active:
            return
        now = time.monotonic()
        self.inertial_active = True
        self.inertial_position_safe = True
        self.inertial_heading_safe = True
        self.inertial_failure_reason = None
        self.inertial_heading_failure = None
        self.recovery_reset_stamp = None
        self.inertial_start_time = now
        self.inertial_displacement_ned = [0.0, 0.0, 0.0]
        self.inertial_integrated_duration_s = 0.0
        self.inertial_position_uncertainty_m = 0.0
        self.inertial_yaw_delta = 0.0
        self.inertial_last_yaw = None
        self.inertial_start_odometry = None
        self.inertial_last_odometry = None
        self.inertial_fallback_to_frozen_pose = False
        self.inertial_reordered_odometry_drops = 0
        self.inertial_reordered_yaw_drops = 0
        self.recovery_velocity_agreement_count = 0
        self.recovery_velocity_last_timestamp = None
        self.recovery_velocity_sample_valid = False
        self.recovery_yaw_agreement.reset()
        self.recovery_yaw_agreement_state = "waiting_for_baseline"
        if self.latest_px4_odometry is None:
            self._fail_inertial_propagation("odometry_unavailable_at_quarantine")
        elif now - self.latest_px4_odometry[0] > self.inertial_max_message_age_s:
            self._fail_inertial_propagation("odometry_stale_at_quarantine")
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
            self.inertial_position_uncertainty_m = math.sqrt(max(var_n, var_e))
            if math.sqrt(max(var_n, var_e)) > self.inertial_max_position_uncertainty_m:
                self._reject_position_propagation("position_uncertainty_at_quarantine")
        # Do not extrapolate the timestamp without extrapolating orientation:
        # doing so shortens the next dt and can manufacture a high yaw rate.
        self.inertial_last_yaw = self.latest_px4_yaw
        self._refresh_recovery_heading()
        aid = self.navigation_timing
        if self.transport == "mavlink":
            aid.boundary = (None if aid.trusted_capture_s is None else aid.matched_fcu(
                aid.trusted_capture_s, now, self.inertial_max_message_age_s))
            if aid.trusted_capture_s is not None:
                self.inertial_start_time = min(now, aid.trusted_capture_s)
            aid.recovery_pose = None
            if aid.boundary is None:
                self._reject_position_propagation("time_matched_boundary_unavailable")
        self.get_logger().warn(
            "VIO_INERTIAL_RECOVERY_STARTED source=PX4_ODOMETRY_position+variance+"
            "reset_counter heading_reference=current_FCU_ATTITUDE gps_output=silent"
        )

    def _update_px4_odometry(self, sample):
        """Track PX4 EKF displacement while synthetic GPS output is silent."""
        if not all(math.isfinite(value) for value in sample) or min(sample[6:8]) < 0:
            self.latest_px4_odometry = None
            self.recovery_velocity_sample_valid = False
            self.recovery_velocity_agreement_count = 0
            if self.inertial_active:
                self._fail_inertial_propagation("invalid_px4_odometry_sample")
            return
        _, odom_time, north, east, vn, ve, var_n, var_e, reset_counter = sample
        if self.inertial_active and self.inertial_last_odometry is not None:
            dt = odom_time - self.inertial_last_odometry[1]
            ordering = timestamp_order(dt, self.inertial_reorder_tolerance_s)
            if ordering == "reordered":
                self.inertial_reordered_odometry_drops += 1
                self.get_logger().warn(
                    "VIO_INERTIAL_SAMPLE_DROPPED stream=ODOMETRY "
                    f"reason=small_timestamp_reorder dt_s={dt:.6f} "
                    f"tolerance_s={self.inertial_reorder_tolerance_s:.3f}"
                )
                return
            if ordering == "regressed":
                self.recovery_velocity_sample_valid = False
                self.recovery_velocity_agreement_count = 0
                self._fail_inertial_propagation(
                    f"odometry_timestamp_regression dt_s={dt:.6f}"
                )
                # Establish a new clock baseline; do not remain stuck comparing
                # every later packet against a timestamp from before a reboot.
                self.inertial_last_odometry = sample
                self.latest_px4_odometry = None
                self.recovery_reset_stamp = odom_time
                return
        self.latest_px4_odometry = sample
        self.latest_px4_velocity = (sample[0], odom_time, vn, ve)
        if not self.inertial_active:
            return
        self.recovery_velocity_sample_valid = False
        if self.inertial_start_odometry is None:
            self.inertial_start_odometry = sample
            self.inertial_last_odometry = sample
            return
        previous = self.inertial_last_odometry
        dt = odom_time - previous[1]
        # Continue monitoring reset counters and velocity after translation is
        # discarded. Do not leave the previous timestamp stuck at the failure.
        self.inertial_last_odometry = sample
        if dt > self.inertial_max_message_gap_s:
            self.recovery_velocity_agreement_count = 0
            self._fail_inertial_propagation(
                f"odometry_message_gap gap_s={dt:.3f} "
                f"limit_s={self.inertial_max_message_gap_s:.3f}"
            )
            return
        if reset_counter != previous[8]:
            # Keep the original position anchor. Current FCU heading
            # is authoritative; require new post-reset agreement windows.
            self._reject_position_propagation(
                f"px4_estimator_reset old={previous[8]} new={reset_counter}")
            self.recovery_reset_stamp = odom_time
            self._reset_recovery_agreement()
            self.get_logger().warn(
                f"VIO_RECOVERY_PX4_RESET old={previous[8]} new={reset_counter}; "
                "position=frozen; using current FCU heading with fresh agreement windows"
            )
            self._refresh_recovery_heading()
            # This packet becomes the velocity baseline, not an agreement
            # sample. Keep the original quarantine anchor.
            return
        uncertainty = math.sqrt(max(0.0, var_n, var_e))
        self.inertial_position_uncertainty_m = max(
            self.inertial_position_uncertainty_m, uncertainty)
        if uncertainty > self.inertial_max_position_uncertainty_m:
            self._reject_position_propagation(
                f"position_uncertainty uncertainty_m={uncertainty:.3f} "
                f"limit_m={self.inertial_max_position_uncertainty_m:.3f}"
            )
        speed = math.hypot(vn, ve)
        if speed > self.continuity.max_speed_mps:
            self.recovery_velocity_agreement_count = 0
            self._reject_position_propagation(
                f"velocity_limit speed_m_s={speed:.3f} "
                f"limit_m_s={self.continuity.max_speed_mps:.3f}"
            )
            return
        old_vn, old_ve = previous[4], previous[5]
        acceleration = math.hypot(vn - old_vn, ve - old_ve) / dt
        if acceleration > self.continuity.max_acceleration_mps2:
            self.recovery_velocity_agreement_count = 0
            self._reject_position_propagation(
                f"acceleration_limit acceleration_m_s2={acceleration:.3f} "
                f"limit_m_s2={self.continuity.max_acceleration_mps2:.3f}"
            )
            return
        # Use the actual PX4 local-position displacement, not another integral
        # of its velocity. Subtract the quarantine-boundary position so PX4's
        # arbitrary local origin never replaces the bridge's global home.
        # Unannounced/impossible position steps also invalidate translation.
        self.recovery_velocity_sample_valid = True
        if self.inertial_position_safe:
            step = math.hypot(north - previous[2], east - previous[3])
            if step > self.continuity.max_speed_mps * dt + 1e-6:
                self._reject_position_propagation(
                    f"position_step_limit displacement_m={step:.3f} dt_s={dt:.3f} "
                    f"limit_m_s={self.continuity.max_speed_mps:.3f}")
            else:
                start = self.inertial_start_odometry
                self.inertial_displacement_ned = [north - start[2], east - start[3], 0.0]
                self.inertial_integrated_duration_s = odom_time - start[1]
        self.inertial_last_odometry = sample

    def _integrate_px4_yaw(self, sample):
        """Receive absolute FCU yaw (legacy method name, no yaw integration)."""
        if not all(math.isfinite(value) for value in sample):
            self.latest_px4_yaw = None
            if self.inertial_active:
                self._reject_heading_recovery("invalid_px4_attitude_sample")
            return
        if self.inertial_active and self.inertial_last_yaw is not None:
            dt = sample[1] - self.inertial_last_yaw[1]
            ordering = timestamp_order(dt, self.inertial_reorder_tolerance_s)
            if ordering == "reordered":
                self.inertial_reordered_yaw_drops += 1
                self.get_logger().warn(
                    "VIO_INERTIAL_SAMPLE_DROPPED stream=ATTITUDE "
                    f"reason=small_timestamp_reorder dt_s={dt:.6f} "
                    f"tolerance_s={self.inertial_reorder_tolerance_s:.3f}"
                )
                return
            if ordering == "regressed":
                self.latest_px4_yaw = None
                self.inertial_last_yaw = sample
                self._reject_heading_recovery(
                    f"yaw_timestamp_regression dt_s={dt:.6f}"
                )
                return
        if self.inertial_active and self.inertial_last_yaw is not None:
            dt = sample[1] - self.inertial_last_yaw[1]
            yaw_rate = abs(wrap_pi(sample[2] - self.inertial_last_yaw[2])) / dt
            if (dt > self.inertial_max_message_gap_s
                    or yaw_rate > self.continuity.max_yaw_rate_rad_s):
                # A discontinuity invalidates OLD agreement samples, not the
                # FCU's new heading. Never latch it unsafe until restart.
                self._reset_recovery_agreement()
                self.get_logger().warn(
                    "VIO_RECOVERY_FCU_HEADING_REBASE "
                    f"gap_s={dt:.3f} yaw_rate_deg_s={math.degrees(yaw_rate):.3f}; "
                    "using current FCU heading; restarting agreement windows"
                )
        self.latest_px4_yaw = sample
        self.inertial_last_yaw = sample
        if self.inertial_active:
            self._refresh_recovery_heading()

    def _refresh_recovery_heading(self):
        """Align the recovery target to CURRENT FCU heading, not old yaw deltas.

        Corrected local yaw + fixed heading_offset = earth/NED yaw. Therefore
        anchor_yaw + recovery_delta + heading_offset must equal FCU yaw.
        Position remains anchored; changing this delta never rotates the old
        global path. This deliberately trusts FCU heading, not gyro evidence.
        """
        now = time.monotonic()
        sample = self.latest_px4_yaw
        aid = self.navigation_timing
        if self.transport == "mavlink":
            # GPS-only and GPS+yaw use FCU ODOMETRY for recovery heading;
            # a missing parallel ATTITUDE stream cannot deadlock its recovery.
            sample = None if not aid.history else (
                aid.history[-1].received_s, aid.history[-1].boot_s, aid.history[-1].yaw)
        if sample is None or not all(math.isfinite(v) for v in sample):
            self._reject_heading_recovery("attitude_unavailable")
            return False
        if not 0.0 <= now - sample[0] <= self.inertial_max_message_age_s:
            self._reject_heading_recovery("attitude_message_stale")
            return False
        if self.recovery_reset_stamp is not None and sample[1] < self.recovery_reset_stamp:
            self._reject_heading_recovery("waiting_for_post_reset_attitude")
            return False
        anchor_yaw = self.continuity.recovery_anchor_yaw
        if (anchor_yaw is None or self.heading_offset_rad is None
                or not all(math.isfinite(v) for v in (anchor_yaw, self.heading_offset_rad))):
            self._reject_heading_recovery("heading_alignment_unavailable")
            return False
        self.inertial_yaw_delta = wrap_pi(sample[2] - self.heading_offset_rad - anchor_yaw)
        if not self.inertial_heading_safe:
            self._reset_recovery_agreement()
            self.get_logger().info(
                "VIO_RECOVERY_FCU_HEADING_AVAILABLE; collecting fresh agreement windows"
            )
        self.inertial_heading_safe = True
        self.inertial_heading_failure = None
        return True

    def _inertial_recovery_target(self):
        if not self.inertial_active:
            return None
        now = time.monotonic()
        if (self.latest_px4_odometry is None
                or now - self.latest_px4_odometry[0] > self.inertial_max_message_age_s):
            self._fail_inertial_propagation("odometry_message_stale")
        if not self._refresh_recovery_heading():
            return None
        if not self.inertial_position_safe:
            return (0.0, 0.0, 0.0), self.inertial_yaw_delta
        horizontal_distance = math.hypot(
            self.inertial_displacement_ned[0], self.inertial_displacement_ned[1]
        )
        physical_bound = (
            self.continuity.max_speed_mps * self.inertial_integrated_duration_s
        )
        if horizontal_distance > physical_bound + 1e-6:
            self._reject_position_propagation(
                f"displacement_limit displacement_m={horizontal_distance:.3f} "
                f"physical_bound_m={physical_bound:.3f}"
            )
            return (0.0, 0.0, 0.0), self.inertial_yaw_delta
        # Continuity operates before the fixed local-world-to-true-NED heading
        # rotation. Convert the independently propagated NED displacement back
        # into that local frame before establishing the recovered epoch.
        if self.heading_offset_rad is None:
            self._reject_heading_recovery("heading_alignment_unavailable")
            return None
        local_delta = rotate_ned_heading(
            self.inertial_displacement_ned[0],
            self.inertial_displacement_ned[1],
            0.0,
            -self.heading_offset_rad,
        )
        return local_delta, self.inertial_yaw_delta

    def _finish_inertial_propagation(self):
        aid = getattr(self, "external_yaw", None)
        if aid is not None:
            # No timed tightening: relative VO tracking does not independently
            # verify the absolute heading adopted from PX4 after a loss.
            aid.variance_floor = max(aid.variance_floor, aid.recovery_pose.yaw_variance)
            self.get_logger().info(
                f"VIO_YAW_RECOVERY_APPLIED sigma_deg={math.degrees(math.sqrt(aid.reported_variance)):.2f}; "
                "retaining inherited PX4 heading uncertainty")
        displacement = tuple(self.inertial_displacement_ned)
        elapsed = time.monotonic() - self.inertial_start_time
        variance = 0.0 if self.latest_px4_odometry is None else max(
            0.0, self.latest_px4_odometry[6], self.latest_px4_odometry[7]
        )
        resume_eph = max(self.cruise_eph_m, math.sqrt(variance),
                         self.inertial_position_uncertainty_m)
        source = "px4_position_displacement"
        if self.inertial_fallback_to_frozen_pose:
            # Freezing loses unknown real movement. Do not claim 0.1 m accuracy
            # on resumption; bound the possible loss using the configured speed.
            resume_eph = max(resume_eph, self.continuity.max_speed_mps * elapsed)
            self.frozen_position_uncertainty_m += self.continuity.max_speed_mps * elapsed
            source = "frozen_pose_current_fcu_heading"
        else:
            # Aligning to PX4 is not a new independent absolute measurement.
            # Stable camera tracking cannot remove the inherited global error.
            self.inherited_position_uncertainty_m = max(
                self.inherited_position_uncertainty_m, resume_eph)
        self.recovery_resume_eph_m = resume_eph
        self.recovery_resume_time = time.monotonic()
        self.get_logger().info(
            "VIO_INERTIAL_RECOVERY_APPLIED "
            f"north_m={displacement[0]:.3f} east_m={displacement[1]:.3f} "
            f"yaw_deg={math.degrees(self.inertial_yaw_delta):.3f} "
            f"elapsed_s={elapsed:.3f} source={source} resume_eph_m={resume_eph:.3f}"
        )
        self.inertial_active = False

    def _recovery_velocity_agrees(self, local_velocity, raw_vio_yaw) -> bool:
        """Require recovered VIO motion to agree with PX4 before handoff."""
        if (not self.recovery_velocity_sample_valid or not self.inertial_heading_safe
                or self.latest_px4_odometry is None or self.heading_offset_rad is None
                or self.continuity.recovery_anchor_yaw is None
                or time.monotonic() - self.latest_px4_odometry[0] > self.inertial_max_message_age_s
                or not all(math.isfinite(v) for v in (*local_velocity, raw_vio_yaw))):
            self.recovery_velocity_agreement_count = 0
            return False
        # Use the rotation that the handoff will actually apply, not the
        # provisional epoch rotation (which omits motion during quarantine).
        rotation = wrap_pi(self.continuity.recovery_anchor_yaw
                           + self.inertial_yaw_delta - raw_vio_yaw)
        vx, vy = self.continuity._rotate_xy(
            local_velocity[0], local_velocity[1], rotation
        )
        vio_n, vio_e, _ = rotate_ned_heading(
            vx, vy, 0.0, self.heading_offset_rad
        )
        px4_n, px4_e = self.latest_px4_odometry[4:6]
        residual = math.hypot(vio_n - px4_n, vio_e - px4_e)
        stamp = self.latest_px4_odometry[1]
        if (residual <= self.recovery_velocity_agreement_m_s
                and stamp != self.recovery_velocity_last_timestamp):
            self.recovery_velocity_agreement_count += 1
        elif residual > self.recovery_velocity_agreement_m_s:
            if self.recovery_velocity_agreement_count:
                self.get_logger().warn(
                    "VIO_RECOVERY_VELOCITY_AGREEMENT_RESET "
                    f"residual_m_s={residual:.3f} "
                    f"limit_m_s={self.recovery_velocity_agreement_m_s:.3f}"
                )
            self.recovery_velocity_agreement_count = 0
        self.recovery_velocity_last_timestamp = stamp
        return (
            self.recovery_velocity_agreement_count
            >= self.recovery_velocity_agreement_samples
        )

    def _recovery_yaw_agrees(self, raw_vio_yaw: float) -> bool:
        """Gate reset recovery on VIO/PX4 relative yaw-change agreement."""
        if not self.inertial_heading_safe:
            # A current heading reference is required even with frozen position.
            self.recovery_yaw_agreement_state = "waiting_for_fcu_heading"
            return False
        if (self.latest_px4_yaw is None
                or time.monotonic() - self.latest_px4_yaw[0] > self.inertial_max_message_age_s):
            self._reject_heading_recovery("attitude_message_stale")
            return False
        result = self.recovery_yaw_agreement.update(
            raw_vio_yaw,
            self.inertial_yaw_delta,
            self.latest_px4_yaw[1],
        )
        if result.event == "baseline_started":
            self.recovery_yaw_agreement_state = "collecting"
            self.get_logger().info(
                "VIO_RECOVERY_YAW_BASELINE_STARTED comparison=relative_change_only"
            )
        elif result.event == "disagreement_reset":
            self.recovery_yaw_agreement_state = "disagreeing"
            self.get_logger().warn(
                "VIO_RECOVERY_YAW_AGREEMENT_RESET "
                f"residual_deg={math.degrees(result.residual_rad):.3f} "
                f"limit_deg={self.recovery_yaw_agreement_deg:.3f}"
            )
        elif result.event == "agreement_confirmed":
            self.recovery_yaw_agreement_state = "confirmed"
            self.get_logger().info(
                "VIO_RECOVERY_YAW_AGREEMENT_CONFIRMED "
                f"samples={result.good_samples} "
                f"residual_deg={math.degrees(result.residual_rad):.3f}"
            )
        elif not result.agreed:
            self.recovery_yaw_agreement_state = "collecting"
        return result.agreed

    def _validated_recovery_target(self, local_velocity, raw_vio_yaw):
        """Combine independent checks only at the final recovery decision."""
        if self.transport == "mavlink":
            return self._matched_recovery_target(local_velocity, raw_vio_yaw)
        target = self._inertial_recovery_target()
        velocity_agrees = self._recovery_velocity_agrees(local_velocity, raw_vio_yaw)
        yaw_agrees = self._recovery_yaw_agrees(raw_vio_yaw)
        return target if velocity_agrees and yaw_agrees else None

    def _reset_recovery_agreement(self):
        self.recovery_yaw_agreement.reset()
        self.recovery_yaw_agreement_state = (
            "waiting_for_baseline" if self.inertial_heading_safe
            else "waiting_for_fcu_heading"
        )
        self.recovery_velocity_agreement_count = 0
        self.recovery_velocity_last_timestamp = None

    def _camera_imu_callback(self, msg: Imu):
        stamp = msg.header.stamp
        self.camera_readiness.observe_imu(
            stamp.sec * 1000000000 + stamp.nanosec, time.monotonic(),
            (msg.linear_acceleration.x, msg.linear_acceleration.y,
             msg.linear_acceleration.z, msg.angular_velocity.x,
             msg.angular_velocity.y, msg.angular_velocity.z), msg.header.frame_id,
        )

    def _check_camera_fusion(self):
        now = time.monotonic()
        future = self._camera_param_future
        if future is not None:
            if future.done():
                try:
                    response = future.result()
                    if response and len(response.values) == 1 and response.values[0].type == 1:
                        self.camera_readiness.observe_fusion(response.values[0].bool_value, now)
                except Exception as exc:
                    self.get_logger().warn(f"Camera fusion query failed: {exc}")
                self._camera_param_future = None
            elif now - self._camera_param_sent > 2.0:
                self._camera_param_client.remove_pending_request(future)
                future.cancel()
                self._camera_param_future = None
            else:
                return
        if self._camera_param_client.service_is_ready():
            request = GetParameters.Request()
            request.names = ["enable_imu_fusion"]
            self._camera_param_future = self._camera_param_client.call_async(request)
            self._camera_param_sent = now

    def _handoff_block_reason(self):
        now = time.monotonic()
        if (self.phase != Phase.LIVE and self.transport == "mavlink"
                and self.send_set_gps_global_origin):
            if (self.px4_armed is not False or self.last_px4_heartbeat_time is None
                    or now - self.last_px4_heartbeat_time > self.heartbeat_timeout_s):
                return "global origin handoff requires fresh DISARMED PX4 heartbeat"
            reason = self.origin_setup.reason(now)
            if reason:
                return reason
        if not self.have_vio or self.last_vio_time is None or now - self.last_vio_time > self.vio_timeout_s:
            return "accepted VIO odometry not fresh"
        if self.heading_offset_rad is None:
            if self.heading_source == "px4":
                return f"heading alignment incomplete: {self.startup_heading.reason}"
            return "heading alignment incomplete"
        if self.heading_source == "px4" and (
            self.startup_heading.offset is None or not self.startup_heading.fresh(now)
        ):
            return "PX4 startup heading missing, stale, or collecting fresh samples"
        if self.continuity_recovering:
            return "pose-jump gate recovering"
        reason = self._navigation_reason()
        if reason:
            return reason
        if getattr(self, "external_yaw", None) is not None:
            reason = self._external_yaw_reason()
            if reason:
                return reason
            if not self.external_yaw.startup_verified:
                return "waiting for PX4 heading readback agreement (no forced yaw reset)"
        return self.camera_readiness.reason(now)

    def _odom_callback(self, msg: Odometry):
        pos, orientation = msg.pose.pose.position, msg.pose.pose.orientation
        stamp = msg.header.stamp
        aid = getattr(self, "external_yaw", None)
        timing = self.navigation_timing
        try:
            if not msg.header.frame_id:
                raise ValueError("invalid_world_frame")
            self.external_sample_time = timing.sample_time(
                stamp.sec + stamp.nanosec * 1e-9,
                self.get_clock().now().nanoseconds * 1e-9, time.monotonic())
            rpy((orientation.w, orientation.x, orientation.y, orientation.z))
        except ValueError as exc:
            self._invalidate_navigation(str(exc))
            return
        if (self.phase == Phase.LIVE and not self.continuity_recovering
                and timing.trusted_capture_s is not None
                and self.external_sample_time-timing.trusted_capture_s > timing.max_age_s):
            # Also catch an outage if a busy executor delivers this callback
            # before the watchdog timer gets to run.
            self._invalidate_navigation("camera_capture_gap")
        timing.input_error = None
        self.camera_readiness.observe_odom(
            stamp.sec * 1000000000 + stamp.nanosec, time.monotonic(),
            (pos.x, pos.y, pos.z, orientation.x, orientation.y, orientation.z, orientation.w),
            msg.child_frame_id, self.expected_child_frame,
        )
        if msg.child_frame_id != self.expected_child_frame:
            self._invalidate_navigation("invalid_body_frame")
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
            angular.x,
            angular.y,
            angular.z,
        )
        if not all(math.isfinite(float(value)) for value in values):
            self._invalidate_navigation("invalid_odometry")
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
                    self._invalidate_navigation("invalid_orientation")
                    self.get_logger().error(f"Ignoring invalid VIO orientation: {exc}")
                    return
                vn, ve, vd = enu_offset_to_ned(east_v, north_v, up_v)
            else:
                vn, ve, vd = enu_offset_to_ned(lin.x, lin.y, lin.z)
        else:
            north, east, down = float(pos.x), float(pos.y), float(pos.z)
            vn, ve, vd = float(lin.x), float(lin.y), float(lin.z)

        try:
            vio_body_yaw = yaw_from_quaternion_xyzw(
                orientation.x, orientation.y, orientation.z, orientation.w
            )
        except ValueError as exc:
            self._invalidate_navigation("invalid_orientation")
            self.get_logger().error(f"Ignoring invalid VIO orientation: {exc}")
            return
        raw_body_yaw_ned = wrap_pi(
            math.pi * 0.5 - vio_body_yaw if self.odom_is_enu else vio_body_yaw
        )
        raw_yaw_rate_ned = -float(angular.z) if self.odom_is_enu else float(angular.z)
        stamp = msg.header.stamp
        sample_time = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        frame_key = f"{msg.header.frame_id}|{msg.child_frame_id}"
        inertial_target = None
        if self.inertial_active:
            # Evaluate freshness/reset/uncertainty guards even while the
            # agreement windows are still collecting samples.
            inertial_target = self._validated_recovery_target((vn, ve, vd), raw_body_yaw_ned)
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
            self.camera_readiness.odom.reset("pose-jump gate restarted startup trajectory")
            self.anchor_ned = None
            self.heading_offset_rad = None
            self.startup_heading.reset()
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
            if continuity.event in ("quarantine_started", "reset_confirmed", "candidate_window_restarted"):
                # Validate only the current candidate epoch. Samples from an
                # earlier failed candidate must not count toward recovery.
                self._reset_recovery_agreement()
            if continuity.event == "quarantine_started" and self.phase == Phase.LIVE:
                self._start_inertial_propagation()
            elif continuity.event == "recovery_completed":
                self._finish_inertial_propagation()
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
        timing.trusted_capture_s = self.external_sample_time
        if aid is not None and self.heading_offset_rad is not None:
            aid.store(self.external_sample_time, (north, east, down),
                      wrap_pi(continuity.yaw + self.heading_offset_rad),
                      (orientation.x, orientation.y, orientation.z, orientation.w),
                      self.odom_is_enu)
            if self.phase != Phase.LIVE:
                self._verify_external_yaw_startup(horizontal_speed)
        self.have_vio = True
        self.last_vio_time = time.monotonic()
        self.latest_fix_ros_stamp_s = stamp.sec + stamp.nanosec * 1e-9
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
        # Normal tracking and quarantine recovery retain the original fixed
        # offset. Recovery aligns its new local epoch using current FCU yaw.
        if self.phase == Phase.LIVE:
            return

        if self.heading_source == "manual":
            if self.heading_offset_rad is not None:
                return
            self.heading_offset_rad = wrap_pi(
                self.manual_heading_rad - vio_body_yaw_ned
            )
            self.get_logger().warn(
                "Heading aligned from manual true heading; verify independently before arming "
                f"(offset={math.degrees(self.heading_offset_rad):.2f} deg)"
            )
            return

        previous = self.heading_offset_rad
        self.heading_offset_rad = self.startup_heading.update(
            vio_body_yaw_ned, horizontal_speed, time.monotonic()
        )
        if self.heading_offset_rad is None and previous is not None:
            self.get_logger().warn(
                f"PX4_STARTUP_HEADING_WAIT: {self.startup_heading.reason}"
            )
        elif self.heading_offset_rad is not None and previous is None:
            self.get_logger().info(
                "PX4_STARTUP_HEADING_ALIGNED "
                f"offset_deg={math.degrees(self.heading_offset_rad):.2f} "
                f"samples={self.alignment_sample_count}; "
                "no extra declination; fixed during live tracking"
            )
            self._send_status_text("VIO GPS: aligned to PX4 heading", 6)

    def _select_accuracy(self) -> tuple[float, float]:
        if self.phase == Phase.SPOOF:
            return self.spoof_eph_m, self.spoof_epv_m
        accuracy_floor = max(self.cruise_eph_m, self.inherited_position_uncertainty_m)
        if self.frozen_position_uncertainty_m > 0.0:
            # Stable relative vision cannot recover an unknown global offset.
            # Require a new verified origin/restart before claiming tight GPS.
            return max(accuracy_floor, self.frozen_position_uncertainty_m,
                       self.recovery_resume_eph_m or 0.0), self.cruise_epv_m
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
                return max(accuracy_floor, eph), self.cruise_epv_m
        if self.live_start_time is None:
            return max(accuracy_floor, self.boot_accuracy_m), self.spoof_epv_m
        if (time.monotonic() - self.live_start_time) < self.boot_duration_s:
            return max(accuracy_floor, self.boot_accuracy_m), self.spoof_epv_m
        return accuracy_floor, self.cruise_epv_m

    def _enter_live(self, reason: str):
        if self.phase == Phase.LIVE:
            return True
        blocked = self._handoff_block_reason()
        if blocked:
            self.live_output_block_reason = blocked
            return False
        previous_phase = self.phase
        self.phase = Phase.LIVE
        self.live_start_time = time.monotonic()
        if previous_phase in (Phase.SPOOF, Phase.ALIGNING):
            self.get_logger().info(f"GPS spoof ended -> live VIO GPS ({reason})")
        else:
            self.get_logger().info(f"RC activated live VIO GPS ({reason})")
        self._send_status_text("VIO GPS: live VIO active", 6)
        return True

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
            if message_type == "TIMESYNC":
                if int(msg.get_srcComponent()) == self.px4_target_compid:
                    response = self.navigation_timing.clock.receive(int(msg.tc1), int(msg.ts1), now)
                    if response is not None:
                        self._mav.mav.timesync_send(*response)
                continue
            if (
                message_type == "HEARTBEAT"
                and self.px4_target_sysid is not None
                and int(msg.get_srcSystem()) == self.px4_target_sysid
                and int(msg.get_srcComponent()) == self.px4_target_compid
            ):
                self.last_px4_heartbeat_time = now
                self.px4_armed = bool(int(msg.base_mode) & 128)
                continue
            if message_type == "GPS_GLOBAL_ORIGIN":
                if (int(msg.get_srcComponent()) == self.px4_target_compid
                        and self.phase in (Phase.SPOOF, Phase.ALIGNING)):
                    was_confirmed = self.origin_setup.confirmed
                    self.origin_setup.observe(now, msg.latitude, msg.longitude, msg.altitude)
                    if self.origin_setup.confirmed and not was_confirmed:
                        self.get_logger().info(
                            f"PX4 global origin verified: {self.home_alt:.1f} m AMSL; "
                            "height still comes from PX4 rangefinder/barometer"
                        )
                continue
            if message_type == "ATTITUDE":
                # Do not use another component's attitude on the same system.
                if int(msg.get_srcComponent()) != self.px4_target_compid:
                    continue
                roll, pitch, yaw = float(msg.roll), float(msg.pitch), float(msg.yaw)
                boot_time = float(msg.time_boot_ms) * 1e-3
                if all(math.isfinite(value) for value in (roll, pitch, yaw, boot_time)):
                    if self.phase != Phase.LIVE:
                        self.startup_heading.observe(now, boot_time, yaw)
                    self._integrate_px4_yaw((now, boot_time, yaw))
                else:
                    if self.phase != Phase.LIVE:
                        self.startup_heading.observe(now, boot_time, float("nan"))
                    if self.inertial_active:
                        self.latest_px4_yaw = None
                        self._reject_heading_recovery("invalid_px4_attitude_sample")
                continue
            if message_type == "ODOMETRY":
                if int(msg.get_srcComponent()) != self.px4_target_compid:
                    continue
                self.navigation_timing.observe_fcu(msg, now)
                sample_time = float(msg.time_usec) * 1e-6
                north, east = float(msg.x), float(msg.y)
                try:
                    vn, ve, _ = mavlink_velocity_to_ned(
                        int(msg.frame_id), int(msg.child_frame_id), msg.q,
                        (float(msg.vx), float(msg.vy), float(msg.vz)),
                    )
                except ValueError as exc:
                    self.latest_px4_odometry = None
                    if self.inertial_active:
                        self._fail_inertial_propagation(f"invalid_px4_velocity_frame: {exc}")
                    continue
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
                    and int(getattr(msg, "quality", 0)) >= 0
                )
                values = (sample_time, north, east, vn, ve, var_n, var_e)
                if not valid_source or not all(math.isfinite(value) for value in values):
                    self.latest_px4_odometry = None
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
            if message_type != "RC_CHANNELS" or not self.rc_trigger_enabled:
                continue
            pwm = int(getattr(msg, f"chan{self.rc_channel}_raw", 65535))
            position = self._rc_position(pwm)
            if position is None:
                continue
            rising_high = self.last_rc_position != "high" and position == "high"
            self.last_rc_position = position
            if (not rising_high or self.phase != Phase.WAITING
                    or self.spoof_failure_reason is not None):
                continue
            if (getattr(self, "external_yaw", None) is not None
                    and not self.external_yaw.clock.ready(now)):
                self.get_logger().warn(
                    "External heading time sync not ready. Wait for Status clock=ready, "
                    "then move RC LOW/MID -> HIGH again; no GPS/heading has started.")
                self._send_status_text("VIO: time sync warming; retry RC when ready", 4)
                continue
            self.get_logger().info(
                f"RC channel {self.rc_channel} pwm={pwm} triggered GPS bootstrap"
            )
            # Evidence must be collected during this activation, not reused
            # from a previous successful run or a prior spoof attempt.
            self.camera_readiness.reset()
            self.continuity.reset()
            self.continuity_recovering = False
            self.have_vio = False
            self.last_vio_time = None
            self.navigation_timing.trusted_capture_s = None
            self.anchor_ned = None
            self.heading_offset_rad = None
            self.startup_heading.reset()
            if getattr(self, "external_yaw", None) is not None:
                self.external_yaw.startup_verified = False
                self.external_yaw.startup_agreement = 0
                self.external_yaw.startup_evidence = None
                self.external_yaw.pose = None
            self.spoof_failure_reason = None
            self._last_handoff_reason = None
            self.phase = Phase.SPOOF
            self.start_time = now
            self.get_logger().info(
                f"GPS spoof started from RC channel {self.rc_channel} for up to "
                f"{self.spoof_duration_s:.1f}s; validating recent camera data"
            )
            self._send_status_text("VIO GPS: spoofing; checking camera health", 6)

    def _maybe_end_spoof_by_timer(self):
        if self.phase not in (Phase.SPOOF, Phase.ALIGNING):
            return
        if self.spoof_duration_s <= 0.0:
            return
        if self.start_time is None:
            return
        if (time.monotonic() - self.start_time) >= self.spoof_duration_s:
            if not self._enter_live("spoof_duration_health_passed"):
                self.spoof_failure_reason = self._handoff_block_reason()
                self.phase = Phase.WAITING
                self.start_time = None
                self._set_live_output_state(False, "spoof_validation_failed")
                self.get_logger().warn(
                    f"GPS spoof expired after {self.spoof_duration_s:.1f}s; "
                    f"validation failed: {self.spoof_failure_reason}. "
                    "GPS output stopped; launcher will stop the camera session."
                )
                self._send_status_text("VIO GPS: camera check failed; GPS stopped", 4)

    def _tick(self):
        self._poll_mavlink()
        if self.phase == Phase.LIVE:
            reason = self._navigation_reason()
            if reason:
                self._invalidate_navigation(reason)
        if self.inertial_active:
            # Enforce stream-freshness guards even if cuVSLAM has
            # stopped producing callbacks during quarantine.
            self._inertial_recovery_target()
        if self.phase == Phase.WAITING:
            self._set_live_output_state(False, "spoof_validation_failed" if self.spoof_failure_reason
                                        else "waiting_for_activation")
            return
        if self.phase in (Phase.SPOOF, Phase.ALIGNING):
            if self.transport == "mavlink" and self.send_set_gps_global_origin:
                self._update_global_origin()
                if self.origin_setup.failure:
                    self.spoof_failure_reason = self.origin_setup.failure
                    self.phase = Phase.WAITING
                    self.start_time = None
                    self._set_live_output_state(False, "global_origin_setup_failed")
                    self.get_logger().error(self.spoof_failure_reason)
                    self._send_status_text("VIO GPS: origin setup failed; GPS stopped", 4)
                    return
            reason = self._handoff_block_reason() or "healthy; waiting for spoof timer"
            if reason != self._last_handoff_reason:
                self.get_logger().info(f"SPOOF_HEALTH_CHECK: {reason}")
                self._last_handoff_reason = reason
        self._maybe_end_spoof_by_timer()
        if self.phase == Phase.WAITING:
            self._set_live_output_state(False, "spoof_expired")
            return

        if self.phase == Phase.LIVE and getattr(self, "external_yaw", None) is not None:
            reason = self._external_yaw_reason()
            if reason:
                self._set_live_output_state(False, reason)
                return

        if (
            self.phase == Phase.LIVE
            and self.continuity_recovering
        ):
            self._set_live_output_state(False, "continuity_quarantine")
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

        self._publish_fix(fix)
        self.messages_sent += 1
        if self.messages_sent == 1 or self.messages_sent % 50 == 0:
            self.get_logger().info(
                f"GPS [{self.phase.value}] n={self.messages_sent} "
                f"lat={fix['lat']:.7f} lon={fix['lon']:.7f} alt={fix['alt']:.2f} "
                f"eph={fix['eph']:.2f}"
            )

    def _update_global_origin(self):
        # Never change an origin in LIVE/recovery, or with unknown/armed status.
        if self.phase not in (Phase.SPOOF, Phase.ALIGNING):
            return
        now = time.monotonic()
        fresh = (self.last_px4_heartbeat_time is not None
                 and now - self.last_px4_heartbeat_time <= self.heartbeat_timeout_s)
        disarmed = not self.px4_armed if fresh and self.px4_armed is not None else None
        set_origin, request = self.origin_setup.step(now, disarmed)
        try:
            if set_origin:
                self._mav.mav.set_gps_global_origin_send(
                    self.px4_target_sysid, *self.origin_setup.target
                )
                self.get_logger().info(
                    f"Setting PX4 global reference (attempt {self.origin_setup.attempts}/5); "
                    "waiting for readback"
                )
            if request:
                # REQUEST_MESSAGE=512, GPS_GLOBAL_ORIGIN=49. Unlike the periodic
                # stream, this forces a response even when the origin is unchanged.
                self._mav.mav.command_long_send(
                    self.px4_target_sysid, self.px4_target_compid, 512, 0,
                    49, 0, 0, 0, 0, 0, 0,
                )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"Global origin setup transport error: {exc}")

    def _publish_fix(self, fix: dict):
        if self.transport == "mavlink":
            packet = self._publish_hil_gps(fix)
            # Preserve the actual MAVLink quantization (degE7 and cm).
            lat, lon, eph = packet[2] * 1e-7, packet[3] * 1e-7, packet[5] * .01
        else:
            self._publish_ros2_sensor_gps(fix)
            lat, lon, eph = float(fix["lat"]), float(fix["lon"]), float(fix["eph"])
        now = time.monotonic()
        is_live = self.phase == Phase.LIVE and self.have_vio
        sample_time = now  # static spoof measurement is generated at publication
        if is_live:
            sample_time = None
            stamp = self.latest_fix_ros_stamp_s
            if stamp is not None and stamp > 0:
                sample_age = self.get_clock().now().nanoseconds * 1e-9 - stamp
                if math.isfinite(sample_age) and sample_age >= 0:
                    sample_time = now - sample_age
        # This executes only after a successful write. Failed writes leave the
        # previous snapshot intact; it is not evidence of PX4 reception/fusion.
        self.last_gps_output = dict(lat=lat, lon=lon, eph=eph, sample_time=sample_time,
                                    send_time=now, source="vio" if is_live else "spoof")

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
                return base_args
            except TypeError:
                self._hil_gps_extensions_supported = False
                self.get_logger().warn(
                    "Jetson pymavlink HIL_GPS has no id/yaw extensions; "
                    "using compatible base message (PX4 ignores those fields)"
                )
        self._mav.mav.hil_gps_send(*base_args)
        return base_args

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
