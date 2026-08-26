#!/usr/bin/env python3
"""Convert cuVSLAM camera-frame odometry into a level vehicle-body frame."""

import math

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.executors import ExternalShutdownException
from scipy.spatial.transform import Rotation


class CuvslamBodyRelay(Node):
    def __init__(self):
        super().__init__("cuvslam_body_relay")
        self.declare_parameter("input_topic", "/visual_slam/tracking/odometry")
        self.declare_parameter("output_topic", "/vio/body/odometry")
        # ROS camera_link is FLU; positive rotation about +Y pitches the
        # forward +X axis toward -Z (physically downward).
        self.declare_parameter("camera_pitch_deg", 30.0)
        self.declare_parameter("camera_x_m", 0.0)
        self.declare_parameter("camera_y_m", 0.0)
        self.declare_parameter("camera_z_m", 0.0)

        input_topic = str(self.get_parameter("input_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        pitch = math.radians(float(self.get_parameter("camera_pitch_deg").value))
        translation = np.array([
            float(self.get_parameter("camera_x_m").value),
            float(self.get_parameter("camera_y_m").value),
            float(self.get_parameter("camera_z_m").value),
        ])
        # T_body_camera: physical camera pose expressed in the body frame.
        self.r_bc = Rotation.from_euler("y", pitch)
        self.t_bc = translation
        self.r_cb = self.r_bc.inv()
        self.t_cb = -self.r_cb.apply(self.t_bc)

        self.publisher = self.create_publisher(Odometry, output_topic, qos_profile_sensor_data)
        self.subscription = self.create_subscription(
            Odometry, input_topic, self._callback, qos_profile_sensor_data
        )
        self.get_logger().info(
            f"camera odometry {input_topic} -> body odometry {output_topic}; "
            f"pitch={math.degrees(pitch):.1f} deg"
        )

    def _callback(self, source: Odometry) -> None:
        p_oc = np.array([
            source.pose.pose.position.x,
            source.pose.pose.position.y,
            source.pose.pose.position.z,
        ])
        q = source.pose.pose.orientation
        try:
            r_oc = Rotation.from_quat([q.x, q.y, q.z, q.w])
        except ValueError:
            self.get_logger().error("Rejected invalid cuVSLAM quaternion")
            return

        # Change both the initial world basis and tracked child from camera to
        # body: T_world_body = T_body_camera * T_odom_camera * T_camera_body.
        r_wb = self.r_bc * r_oc * self.r_cb
        p_wb = self.r_bc.apply(p_oc) + self.t_bc + (self.r_bc * r_oc).apply(self.t_cb)
        quat = r_wb.as_quat()

        target = Odometry()
        target.header = source.header
        target.header.frame_id = "odom_body"
        target.child_frame_id = "drone_link"
        target.pose.pose.position.x, target.pose.pose.position.y, target.pose.pose.position.z = p_wb
        target.pose.pose.orientation.x = float(quat[0])
        target.pose.pose.orientation.y = float(quat[1])
        target.pose.pose.orientation.z = float(quat[2])
        target.pose.pose.orientation.w = float(quat[3])
        target.pose.covariance = source.pose.covariance

        linear = np.array([
            source.twist.twist.linear.x,
            source.twist.twist.linear.y,
            source.twist.twist.linear.z,
        ])
        angular = np.array([
            source.twist.twist.angular.x,
            source.twist.twist.angular.y,
            source.twist.twist.angular.z,
        ])
        linear_body = self.r_bc.apply(linear)
        angular_body = self.r_bc.apply(angular)
        target.twist.twist.linear.x, target.twist.twist.linear.y, target.twist.twist.linear.z = linear_body
        target.twist.twist.angular.x, target.twist.twist.angular.y, target.twist.twist.angular.z = angular_body
        target.twist.covariance = source.twist.covariance
        self.publisher.publish(target)


def main(args=None):
    rclpy.init(args=args)
    node = CuvslamBodyRelay()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
