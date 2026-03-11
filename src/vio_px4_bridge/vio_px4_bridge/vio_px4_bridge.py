#!/usr/bin/env python3

import rclpy
from nav_msgs.msg import Odometry
from numpy import isfinite
from px4_msgs.msg import VehicleOdometry
from px4_msgs.msg import TimesyncStatus
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from scipy.spatial.transform import Rotation


class VisualOdometryBridge(Node):
    """
    Input convention (ROS2):
    - World frame: ENU
    - Body frame: FLU

    Output convention (PX4):
    - World frame: NED
    - Body frame: FRD
    """

    def __init__(self):
        super().__init__("vio_px4_bridge")

        # Declare parameters.
        self.declare_parameter("odom_topic", "/visual_slam/tracking/odometry")
        self.declare_parameter("px4_topic", "/fmu/in/vehicle_visual_odometry")
        self.declare_parameter("quality", 100)
        self.declare_parameter("reset_counter", 0)

        # Suggestion 5: use message timestamp by default.
        self.declare_parameter("use_msg_timestamp", True)

        self.declare_parameter("use_timesync", True)
        self.declare_parameter("timesync_topic", "/fmu/out/timesync_status")
        self.declare_parameter("timesync_timeout_us", 5_000_000)

        # Suggestion 3: use more realistic nonzero variance floors.
        self.declare_parameter("position_variance_floor", 1e-2)
        self.declare_parameter("orientation_variance_floor", 1e-2)
        self.declare_parameter("velocity_variance_floor", 1e-2)
        self.declare_parameter("expected_world_frame", "")
        self.declare_parameter("expected_body_frame", "")

        # Read parameters.
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.px4_topic = str(self.get_parameter("px4_topic").value)
        self.quality = int(self.get_parameter("quality").value)
        self.reset_counter = int(self.get_parameter("reset_counter").value)
        self.use_msg_timestamp = bool(self.get_parameter("use_msg_timestamp").value)
        self.use_timesync = bool(self.get_parameter("use_timesync").value)
        self.timesync_topic = str(self.get_parameter("timesync_topic").value)
        self.timesync_timeout_us = int(self.get_parameter("timesync_timeout_us").value)

        self.position_variance_floor = float(self.get_parameter("position_variance_floor").value)
        self.orientation_variance_floor = float(self.get_parameter("orientation_variance_floor").value)
        self.velocity_variance_floor = float(self.get_parameter("velocity_variance_floor").value)
        self.expected_world_frame = str(self.get_parameter("expected_world_frame").value).strip()
        self.expected_body_frame = str(self.get_parameter("expected_body_frame").value).strip()

        sub_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        pub_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.subscription = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            sub_qos_profile,
        )
        self.publisher = self.create_publisher(
            VehicleOdometry,
            self.px4_topic,
            pub_qos_profile,
        )
        self.timesync_subscription = self.create_subscription(
            TimesyncStatus,
            self.timesync_topic,
            self.timesync_callback,
            sub_qos_profile,
        )

        self.get_logger().info(f"Subscribed odometry: {self.odom_topic}")
        self.get_logger().info(f"Publishing PX4 vision odometry: {self.px4_topic}")
        self.get_logger().info(f"use_msg_timestamp: {self.use_msg_timestamp}")
        self.get_logger().info(f"use_timesync: {self.use_timesync}")
        self.get_logger().info(f"timesync_timeout_us: {self.timesync_timeout_us}")
        self.get_logger().info(
            f"variance floors -> position: {self.position_variance_floor}, "
            f"orientation: {self.orientation_variance_floor}, "
            f"velocity: {self.velocity_variance_floor}"
        )
        if self.expected_world_frame:
            self.get_logger().info(f"Expected odometry world frame: {self.expected_world_frame}")
        if self.expected_body_frame:
            self.get_logger().info(f"Expected odometry body frame: {self.expected_body_frame}")
        if self.use_timesync:
            self.get_logger().info(f"Subscribed timesync: {self.timesync_topic}")

        self.timesync_offset_us = None
        self.timesync_last_update_us = 0
        self.timesync_warned = False

    def odom_callback(self, msg: Odometry):
        if not self.validate_frames(msg):
            return

        px4_odom = VehicleOdometry()

        now_us = int(self.get_clock().now().nanoseconds / 1000)
        px4_now_us = self.get_px4_time_us(now_us)
        px4_odom.timestamp = px4_now_us

        if self.use_msg_timestamp:
            # Use the timestamp provided by the ROS message header.
            # Convert it to PX4 time domain using timesync (if available).
            ros_sample_us = self.ros_stamp_to_microseconds(
                msg.header.stamp.sec,
                msg.header.stamp.nanosec,
            )
            px4_odom.timestamp_sample = self.to_px4_time_us(ros_sample_us, now_us)
        else:
            # Use "now" for both timestamp and sample time.
            px4_odom.timestamp_sample = px4_now_us

        # Output pose frame (world frame): PX4 expects NED.
        px4_odom.pose_frame = VehicleOdometry.POSE_FRAME_NED

        # Position: ROS ENU -> PX4 NED.
        ros_position = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        ]
        px4_position = self.enu_to_ned_vector(ros_position)
        px4_odom.position = [
            float(px4_position[0]),
            float(px4_position[1]),
            float(px4_position[2]),
        ]

        # Orientation: ROS quaternion (ENU, FLU) -> PX4 quaternion (NED, FRD).
        ros_quat_wxyz = [
            msg.pose.pose.orientation.w,
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
        ]
        px4_quat_wxyz = self.convert_orientation_ros2_to_px4(ros_quat_wxyz)
        if px4_quat_wxyz is None:
            self.get_logger().warn("Dropping odometry with invalid quaternion.", throttle_duration_sec=5.0)
            return
        px4_odom.q = [
            float(px4_quat_wxyz[0]),
            float(px4_quat_wxyz[1]),
            float(px4_quat_wxyz[2]),
            float(px4_quat_wxyz[3]),
        ]

        # Linear velocity: assume ROS twist is in body frame (FLU).
        # Convert to PX4 body frame (FRD).
        ros_linear_velocity = [
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z,
        ]
        px4_linear_velocity = self.flu_to_frd_vector(ros_linear_velocity)
        px4_odom.velocity_frame = VehicleOdometry.VELOCITY_FRAME_BODY_FRD
        px4_odom.velocity = [
            float(px4_linear_velocity[0]),
            float(px4_linear_velocity[1]),
            float(px4_linear_velocity[2]),
        ]

        # Angular velocity: ROS body frame FLU -> PX4 body frame FRD.
        ros_angular_velocity = [
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z,
        ]
        px4_angular_velocity = self.flu_to_frd_vector(ros_angular_velocity)
        px4_odom.angular_velocity = [
            float(px4_angular_velocity[0]),
            float(px4_angular_velocity[1]),
            float(px4_angular_velocity[2]),
        ]

        # Variances (diagonal only).
        # ENU -> NED swaps X/Y, so swap X/Y variances for position and orientation.
        pose_covariance = msg.pose.covariance
        twist_covariance = msg.twist.covariance

        position_variance_enu = self.covariance_diag_with_floor(
            pose_covariance,
            0, 7, 14,
            self.position_variance_floor,
        )
        orientation_variance_enu = self.covariance_diag_with_floor(
            pose_covariance,
            21, 28, 35,
            self.orientation_variance_floor,
        )
        velocity_variance_body = self.covariance_diag_with_floor(
            twist_covariance,
            0, 7, 14,
            self.velocity_variance_floor,
        )

        px4_odom.position_variance = self.swap_xy_variance(position_variance_enu)
        px4_odom.orientation_variance = self.swap_xy_variance(orientation_variance_enu)
        px4_odom.velocity_variance = velocity_variance_body

        px4_odom.reset_counter = int(self.reset_counter)

        quality_value = int(self.quality)
        if quality_value < 0:
            quality_value = 0
        if quality_value > 100:
            quality_value = 100
        px4_odom.quality = quality_value

        self.publisher.publish(px4_odom)

    def enu_to_ned_vector(self, vector_enu):
        # World frame transform: ROS ENU -> PX4 NED
        x_enu = vector_enu[0]
        y_enu = vector_enu[1]
        z_enu = vector_enu[2]

        x_ned = y_enu
        y_ned = x_enu
        z_ned = -z_enu

        return [x_ned, y_ned, z_ned]

    def flu_to_frd_vector(self, vector_flu):
        # Body frame transform: ROS FLU -> PX4 FRD
        x_flu = vector_flu[0]
        y_flu = vector_flu[1]
        z_flu = vector_flu[2]

        x_frd = x_flu
        y_frd = -y_flu
        z_frd = -z_flu

        return [x_frd, y_frd, z_frd]

    def convert_orientation_ros2_to_px4(self, quat_wxyz):
        if not self.is_valid_quaternion(quat_wxyz):
            return None

        # ROS world frame: ENU  (X=East, Y=North, Z=Up)
        # PX4 world frame: NED  (X=North, Y=East, Z=Down)
        rotation_ned_from_enu = Rotation.from_matrix([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
        ])

        # ROS body frame: FLU   (X=Forward, Y=Left, Z=Up)
        # PX4 body frame: FRD   (X=Forward, Y=Right, Z=Down)
        # This matrix is its own inverse.
        rotation_flu_from_frd = Rotation.from_matrix([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ])

        quat_xyzw = self.quat_wxyz_to_xyzw(quat_wxyz)
        rotation_enu_flu = Rotation.from_quat(quat_xyzw)

        # Change world frame on the left, body frame on the right.
        rotation_ned_frd = (
            rotation_ned_from_enu
            * rotation_enu_flu
            * rotation_flu_from_frd
        )

        output_xyzw = rotation_ned_frd.as_quat()
        output_wxyz = self.quat_xyzw_to_wxyz(output_xyzw)
        return output_wxyz

    def is_valid_quaternion(self, quat_wxyz):
        if len(quat_wxyz) != 4:
            return False

        if not all(isfinite(value) for value in quat_wxyz):
            return False

        norm_squared = sum(value * value for value in quat_wxyz)
        return norm_squared > 1e-12

    def quat_wxyz_to_xyzw(self, quat_wxyz):
        w = quat_wxyz[0]
        x = quat_wxyz[1]
        y = quat_wxyz[2]
        z = quat_wxyz[3]
        return [x, y, z, w]

    def quat_xyzw_to_wxyz(self, quat_xyzw):
        x = quat_xyzw[0]
        y = quat_xyzw[1]
        z = quat_xyzw[2]
        w = quat_xyzw[3]
        return [w, x, y, z]

    def covariance_diag_with_floor(self, covariance, index_a, index_b, index_c, floor):
        value_a = covariance[index_a]
        if value_a < floor:
            value_a = floor

        value_b = covariance[index_b]
        if value_b < floor:
            value_b = floor

        value_c = covariance[index_c]
        if value_c < floor:
            value_c = floor

        return [float(value_a), float(value_b), float(value_c)]

    def swap_xy_variance(self, variance_xyz):
        # ENU -> NED swaps X and Y. Variance remains positive, so only swap.
        return [
            float(variance_xyz[1]),
            float(variance_xyz[0]),
            float(variance_xyz[2]),
        ]

    def ros_stamp_to_microseconds(self, sec, nanosec):
        microseconds_from_sec = sec * 1_000_000
        microseconds_from_nanosec = nanosec / 1000
        return int(microseconds_from_sec + microseconds_from_nanosec)

    def validate_frames(self, msg: Odometry):
        if self.expected_world_frame and msg.header.frame_id != self.expected_world_frame:
            self.get_logger().warn(
                f"Unexpected world frame '{msg.header.frame_id}', expected '{self.expected_world_frame}'. "
                "Dropping odometry.",
                throttle_duration_sec=5.0,
            )
            return False

        if self.expected_body_frame and msg.child_frame_id != self.expected_body_frame:
            self.get_logger().warn(
                f"Unexpected body frame '{msg.child_frame_id}', expected '{self.expected_body_frame}'. "
                "Dropping odometry.",
                throttle_duration_sec=5.0,
            )
            return False

        return True

    def timesync_callback(self, msg: TimesyncStatus):
        # Suggestion 1:
        # Use PX4's filtered estimated_offset directly.
        if not self.use_timesync:
            return

        self.timesync_offset_us = int(msg.estimated_offset)
        self.timesync_last_update_us = int(self.get_clock().now().nanoseconds / 1000)

    def get_px4_time_us(self, now_us):
        return self.to_px4_time_us(now_us, now_us)

    def to_px4_time_us(self, ros_time_us, now_us):
        # Convert ROS time -> PX4 time using the latest timesync offset.
        # If timesync is missing, fall back to ROS time.
        # If timesync is stale but an offset was received before, keep using the
        # last known offset because it is usually better than switching clocks.
        if not self.use_timesync:
            return int(ros_time_us)

        if self.timesync_offset_us is None:
            if not self.timesync_warned:
                self.get_logger().warn(
                    "No timesync offset yet; using ROS time for PX4 timestamps."
                )
                self.timesync_warned = True
            return int(ros_time_us)

        if (now_us - self.timesync_last_update_us) > self.timesync_timeout_us:
            if not self.timesync_warned:
                self.get_logger().warn(
                    "Timesync stale; using last known PX4 offset."
                )
                self.timesync_warned = True
            return int(ros_time_us + self.timesync_offset_us)

        self.timesync_warned = False
        return int(ros_time_us + self.timesync_offset_us)


def main():
    rclpy.init()
    node = VisualOdometryBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
