#!/usr/bin/env python3

import copy
import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry


class RotatedOdometryRelay(Node):
    def __init__(self):
        super().__init__('rotated_odometry_relay')

        self.declare_parameter('input_topic', '/visual_slam/tracking/odometry')
        self.declare_parameter('output_topic', '/visual_slam/tracking/odometry_rotated')
        self.declare_parameter('pitch_deg', -45.0)
        self.declare_parameter('output_header_frame_id', '')
        self.declare_parameter('output_child_frame_id', '')

        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.pitch_deg = self.get_parameter('pitch_deg').value
        self.output_header_frame_id = self.get_parameter('output_header_frame_id').value
        self.output_child_frame_id = self.get_parameter('output_child_frame_id').value

        self.r_rot = R.from_euler('y', self.pitch_deg, degrees=True)
        self.R3 = self.r_rot.as_matrix()
        self.R6 = np.block([
            [self.R3, np.zeros((3, 3))],
            [np.zeros((3, 3)), self.R3]
        ])

        self.sub = self.create_subscription(
            Odometry,
            self.input_topic,
            self.odom_callback,
            qos_profile_sensor_data
        )

        self.pub = self.create_publisher(
            Odometry,
            self.output_topic,
            qos_profile_sensor_data
        )

        self.get_logger().info(
            f'Rotating odometry from {self.input_topic} to {self.output_topic} '
            f'with pitch {self.pitch_deg} deg'
        )

    def odom_callback(self, msg: Odometry):
        out = copy.deepcopy(msg)

        if self.output_header_frame_id:
            out.header.frame_id = self.output_header_frame_id

        if self.output_child_frame_id:
            out.child_frame_id = self.output_child_frame_id

        p = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])
        p_rot = self.r_rot.apply(p)
        out.pose.pose.position.x = float(p_rot[0])
        out.pose.pose.position.y = float(p_rot[1])
        out.pose.pose.position.z = float(p_rot[2])

        q = [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ]
        r_in = R.from_quat(q)
        r_out = self.r_rot * r_in
        q_rot = r_out.as_quat()

        out.pose.pose.orientation.x = float(q_rot[0])
        out.pose.pose.orientation.y = float(q_rot[1])
        out.pose.pose.orientation.z = float(q_rot[2])
        out.pose.pose.orientation.w = float(q_rot[3])

        v = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ])
        v_rot = self.r_rot.apply(v)
        out.twist.twist.linear.x = float(v_rot[0])
        out.twist.twist.linear.y = float(v_rot[1])
        out.twist.twist.linear.z = float(v_rot[2])

        w = np.array([
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z
        ])
        w_rot = self.r_rot.apply(w)
        out.twist.twist.angular.x = float(w_rot[0])
        out.twist.twist.angular.y = float(w_rot[1])
        out.twist.twist.angular.z = float(w_rot[2])

        pose_cov = np.array(msg.pose.covariance, dtype=np.float64).reshape(6, 6)
        pose_cov_rot = self.R6 @ pose_cov @ self.R6.T
        out.pose.covariance = pose_cov_rot.reshape(-1).tolist()

        twist_cov = np.array(msg.twist.covariance, dtype=np.float64).reshape(6, 6)
        twist_cov_rot = self.R6 @ twist_cov @ self.R6.T
        out.twist.covariance = twist_cov_rot.reshape(-1).tolist()

        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = RotatedOdometryRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
