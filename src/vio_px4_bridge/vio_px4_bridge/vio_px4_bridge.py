#!/usr/bin/env python3

import rclpy
from nav_msgs.msg import Odometry
from px4_msgs.msg import VehicleOdometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from scipy.spatial.transform import Rotation


class VisualOdometryBridge(Node): #python class name
    """
    Bridge ROS odometry to PX4 external vision odometry.

    Input convention:
    - World frame: ENU
    - Body frame: FLU

    Output convention for PX4:
    - World frame: NED
    - Body frame: FRD
    """

    def __init__(self):
        super().__init__("vio_px4_bridge") #node name

        # Declare parameters.
        self.declare_parameter("odom_topic", "/visual_slam/tracking/odometry")
        self.declare_parameter("px4_topic", "/fmu/in/vehicle_visual_odometry")
        self.declare_parameter("quality", 100)
        self.declare_parameter("reset_counter", 0)
        self.declare_parameter("twist_in_body_frame", True)
        self.declare_parameter("use_msg_timestamp", False)

        # Read parameters.
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.px4_topic = str(self.get_parameter("px4_topic").value)
        self.quality = int(self.get_parameter("quality").value)
        self.reset_counter = int(self.get_parameter("reset_counter").value)
        self.twist_in_body_frame = bool(self.get_parameter("twist_in_body_frame").value)
        self.use_msg_timestamp = bool(self.get_parameter("use_msg_timestamp").value)



        # Precompute frame conversion rotations.
        # Goal: convert ROS conventions (ENU world, FLU body) 
        self.rotation_ned_from_enu = Rotation.from_matrix([  # World-frame conversion: ROS ENU -> PX4 NED
            [0.0, 1.0, 0.0],                                 
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
        ])
        self.rotation_flu_from_frd = Rotation.from_matrix([  # Body-frame conversion matrix between PX4 FRD and ROS FLU (same numbers both directions)
            [1.0, 0.0, 0.0],                                 
            [0.0, -1.0, 0.0],                                
            [0.0, 0.0, -1.0],
        ])


        sub_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        pub_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.subscription = self.create_subscription(
            Odometry, self.odom_topic, self.odom_callback, sub_qos_profile
        ) #This node subscribe from this
        self.publisher = self.create_publisher(
            VehicleOdometry, self.px4_topic, pub_qos_profile
        ) #This node publish this
        

        self.get_logger().info("Subscribed odometry: " + self.odom_topic)
        self.get_logger().info("Publishing PX4 vision odometry: " + self.px4_topic)










    def enu_to_ned_vector(self, vector_enu): #world frame transform ros -> px4
        x_enu = vector_enu[0]
        y_enu = vector_enu[1]
        z_enu = vector_enu[2]

        x_ned = y_enu
        y_ned = x_enu
        z_ned = -z_enu

        return [x_ned, y_ned, z_ned]

    def flu_to_frd_vector(self, vector_flu): #body frame transform ros -> px4
        x_flu = vector_flu[0]
        y_flu = vector_flu[1]
        z_flu = vector_flu[2]

        x_frd = x_flu
        y_frd = -y_flu
        z_frd = -z_flu

        return [x_frd, y_frd, z_frd]

    def quat_wxyz_to_xyzw(self, quat_wxyz):
        w_value = quat_wxyz[0]
        x_value = quat_wxyz[1]
        y_value = quat_wxyz[2]
        z_value = quat_wxyz[3]

        quat_xyzw = [x_value, y_value, z_value, w_value]
        return quat_xyzw

    def quat_xyzw_to_wxyz(self, quat_xyzw):
        x_value = quat_xyzw[0]
        y_value = quat_xyzw[1]
        z_value = quat_xyzw[2]
        w_value = quat_xyzw[3]

        quat_wxyz = [w_value, x_value, y_value, z_value]
        return quat_wxyz

    def convert_orientation_enu_flu_to_ned_frd(self, quat_wxyz): #tranform from ros -> px4
        quat_xyzw = self.quat_wxyz_to_xyzw(quat_wxyz)
        rotation_enu_flu = Rotation.from_quat(quat_xyzw)

        rotation_ned_frd = (
            self.rotation_ned_from_enu
            * rotation_enu_flu
            * self.rotation_flu_from_frd
        )

        output_xyzw = rotation_ned_frd.as_quat()
        output_wxyz = self.quat_xyzw_to_wxyz(output_xyzw)
        return output_wxyz



    def ros_stamp_to_microseconds(self, sec, nanosec):
        microseconds_from_sec = sec * 1_000_000
        microseconds_from_nanosec = nanosec / 1000
        timestamp_us = int(microseconds_from_sec + microseconds_from_nanosec)
        return timestamp_us

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





    def odom_callback(self, msg):
        out = VehicleOdometry()
        now_us = int(self.get_clock().now().nanoseconds / 1000)
        out.timestamp = now_us


        if self.use_msg_timestamp:
            out.timestamp_sample = self.ros_stamp_to_microseconds(
                msg.header.stamp.sec,
                msg.header.stamp.nanosec,
            )
        else:
            out.timestamp_sample = now_us
        
        
        #out pose framne
        out.pose_frame = VehicleOdometry.POSE_FRAME_NED


        #out position
        ros_position = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        ]
        px4_position = self.enu_to_ned_vector(ros_position) 
        out.position = [
            float(px4_position[0]),
            float(px4_position[1]),
            float(px4_position[2]),
        ]


        #out quaternion
        ros_quat_wxyz = [
            msg.pose.pose.orientation.w,
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
        ]
        px4_quat_wxyz = self.convert_orientation_enu_flu_to_ned_frd(ros_quat_wxyz)
        out.q = [
            float(px4_quat_wxyz[0]),
            float(px4_quat_wxyz[1]),
            float(px4_quat_wxyz[2]),
            float(px4_quat_wxyz[3]),
        ]


        #out velocity
        ros_linear_velocity = [
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z,
        ]
        if self.twist_in_body_frame:
            out.velocity_frame = VehicleOdometry.VELOCITY_FRAME_BODY_FRD
            px4_linear_velocity = self.flu_to_frd_vector(ros_linear_velocity)
        else:
            out.velocity_frame = VehicleOdometry.VELOCITY_FRAME_NED
            px4_linear_velocity = self.enu_to_ned_vector(ros_linear_velocity)

        out.velocity = [
            float(px4_linear_velocity[0]),
            float(px4_linear_velocity[1]),
            float(px4_linear_velocity[2]),
        ]


        #out angular velocity
        ros_angular_velocity = [
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z,
        ]
        px4_angular_velocity = self.flu_to_frd_vector(ros_angular_velocity)
        out.angular_velocity = [
            float(px4_angular_velocity[0]),
            float(px4_angular_velocity[1]),
            float(px4_angular_velocity[2]),
        ]


        # out covariances
        floor = 1e-6
        pose_covariance = msg.pose.covariance
        twist_covariance = msg.twist.covariance
        out.position_variance = self.covariance_diag_with_floor(
            pose_covariance, 0, 7, 14, floor
        )
        out.orientation_variance = self.covariance_diag_with_floor(
            pose_covariance, 21, 28, 35, floor
        )
        out.velocity_variance = self.covariance_diag_with_floor(
            twist_covariance, 0, 7, 14, floor
        )



        #reset counter
        out.reset_counter = self.reset_counter

        quality_value = self.quality
        if quality_value < 1:
            quality_value = 1
        if quality_value > 100:
            quality_value = 100
        out.quality = int(quality_value)

        self.publisher.publish(out)




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
