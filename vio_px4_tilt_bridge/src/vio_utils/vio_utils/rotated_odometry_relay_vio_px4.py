#!/usr/bin/env python3
 
import copy
import numpy as np
from scipy.spatial.transform import Rotation as R
 
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    qos_profile_sensor_data,
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)
from nav_msgs.msg import Odometry
 
 
class RotatedOdometryRelay(Node):
    def __init__(self):
        super().__init__('rotated_odometry_relay_vio_px4')
 
        self.declare_parameter('input_topic', '/visual_slam/tracking/odometry')
        self.declare_parameter('rotated_topic', '/visual_slam/tracking/odometry_rotated')
        self.declare_parameter('mavros_topic', '/mavros/odometry/out')
 
        # Set pitch_deg to the physical camera tilt angle.
        # Camera pointing 30 degrees DOWN from horizontal = -30.0
        # ROS right-hand rule: negative Y rotation = pitch DOWN
        self.declare_parameter('pitch_deg', -30.0)
 
        self.declare_parameter('output_header_frame_id', 'odom_tilt')
        self.declare_parameter('output_child_frame_id', 'drone_link')
 
        self.input_topic            = self.get_parameter('input_topic').value
        self.rotated_topic          = self.get_parameter('rotated_topic').value
        self.mavros_topic           = self.get_parameter('mavros_topic').value
        self.pitch_deg              = float(self.get_parameter('pitch_deg').value)
        self.output_header_frame_id = self.get_parameter('output_header_frame_id').value
        self.output_child_frame_id  = self.get_parameter('output_child_frame_id').value
 
        # r_rot : rotation representing the camera tilt (e.g. -30 deg around Y)
        # r_inv : inverse (+30 deg) — applied everywhere to UNDO the camera tilt
        #
        # Position / linear velocity / angular velocity:
        #   Multiplied by R3 = r_inv.as_matrix()
        #   Verified correct: walking forward, z collapsed 1.769 → 0.088 ✓
        #
        # Orientation:
        #   r_out = r_inv * r_in  (LEFT multiply in world frame)
        #   Verified correct: path flat and parallel to grid in RVIZ ✓
        #   Note: output w may be negative (quaternion double-cover) — normalised below
        self.r_rot = R.from_euler('y', self.pitch_deg, degrees=True)   # -30 deg
        self.r_inv = self.r_rot.inv()                                   # +30 deg
        self.R3    = self.r_inv.as_matrix()                             # 3x3 for pos/vel
        self.R6    = np.block([
            [self.R3, np.zeros((3, 3))],
            [np.zeros((3, 3)), self.R3]
        ])
 
        debug_qos  = qos_profile_sensor_data
        mavros_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
 
        self.sub = self.create_subscription(
            Odometry,
            self.input_topic,
            self.odom_callback,
            qos_profile_sensor_data
        )
 
        self.pub_rotated = self.create_publisher(
            Odometry,
            self.rotated_topic,
            debug_qos
        )
 
        self.pub_mavros = self.create_publisher(
            Odometry,
            self.mavros_topic,
            mavros_qos
        )
 
        self.get_logger().info(
            f'Input: {self.input_topic} | Rotated: {self.rotated_topic} | '
            f'MAVROS: {self.mavros_topic} | Camera pitch: {self.pitch_deg} deg | '
            f'Correction: {-self.pitch_deg} deg (r_inv) | '
            f'Frames: {self.output_header_frame_id} -> {self.output_child_frame_id}'
        )
 
    def odom_callback(self, msg: Odometry):
        out = copy.deepcopy(msg)
 
        out.header.frame_id = self.output_header_frame_id
        out.child_frame_id  = self.output_child_frame_id
 
        # ── Position ──────────────────────────────────────────────────────────
        # r_inv matrix projects tilted camera position back to level drone frame.
        # Verified: walking forward z collapsed 1.769 → 0.088 ✓
        p = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        ], dtype=np.float64)
        p_rot = self.R3 @ p
 
        out.pose.pose.position.x = float(p_rot[0])
        out.pose.pose.position.y = float(p_rot[1])
        out.pose.pose.position.z = float(p_rot[2])
 
        # ── Orientation ───────────────────────────────────────────────────────
        # LEFT multiply with r_inv undoes the camera tilt in world frame.
        # Verified: RVIZ path flat and parallel to grid ✓
        # Normalise to positive w — quaternion double-cover means
        # (x,y,z,w) and (-x,-y,-z,-w) are the same rotation,
        # but some consumers (PX4 EKF2) prefer positive w convention.
        q = np.array([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ], dtype=np.float64)
 
        r_in  = R.from_quat(q)
        r_out = self.r_inv * r_in
        q_rot = r_out.as_quat()
 
        # Ensure positive w for PX4 convention
        if q_rot[3] < 0:
            q_rot = -q_rot
 
        out.pose.pose.orientation.x = float(q_rot[0])
        out.pose.pose.orientation.y = float(q_rot[1])
        out.pose.pose.orientation.z = float(q_rot[2])
        out.pose.pose.orientation.w = float(q_rot[3])
 
        # ── Linear velocity ───────────────────────────────────────────────────
        # Same r_inv rotation as position.
        v = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z,
        ], dtype=np.float64)
        v_rot = self.R3 @ v
 
        out.twist.twist.linear.x = float(v_rot[0])
        out.twist.twist.linear.y = float(v_rot[1])
        out.twist.twist.linear.z = float(v_rot[2])
 
        # ── Angular velocity ──────────────────────────────────────────────────
        # Same r_inv rotation as position.
        w = np.array([
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z,
        ], dtype=np.float64)
        w_rot = self.R3 @ w
 
        out.twist.twist.angular.x = float(w_rot[0])
        out.twist.twist.angular.y = float(w_rot[1])
        out.twist.twist.angular.z = float(w_rot[2])
 
        # ── Covariances ───────────────────────────────────────────────────────
        pose_cov  = np.array(msg.pose.covariance,  dtype=np.float64).reshape(6, 6)
        twist_cov = np.array(msg.twist.covariance, dtype=np.float64).reshape(6, 6)
 
        out.pose.covariance  = (self.R6 @ pose_cov  @ self.R6.T).reshape(-1).tolist()
        out.twist.covariance = (self.R6 @ twist_cov @ self.R6.T).reshape(-1).tolist()
 
        self.pub_rotated.publish(out)
        self.pub_mavros.publish(out)
 
 
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
