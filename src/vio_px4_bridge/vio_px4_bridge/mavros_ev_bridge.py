#!/usr/bin/env python3
"""Validated cuVSLAM body odometry relay to MAVROS/PX4 ODOMETRY."""

import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


class MavrosEvBridge(Node):
    def __init__(self):
        super().__init__("mavros_ev_bridge")
        self.declare_parameter("odom_topic", "/visual_slam/tracking/odometry")
        self.declare_parameter("mavros_topic", "/mavros/odometry/out")
        self.declare_parameter("expected_child_frame", "drone_link")
        self.declare_parameter("output_parent_frame", "odom")
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.mavros_topic = str(self.get_parameter("mavros_topic").value)
        self.expected_child_frame = str(self.get_parameter("expected_child_frame").value)
        self.output_parent_frame = str(self.get_parameter("output_parent_frame").value)
        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                durability=DurabilityPolicy.VOLATILE,
                                history=HistoryPolicy.KEEP_LAST, depth=5)
        output_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                durability=DurabilityPolicy.VOLATILE,
                                history=HistoryPolicy.KEEP_LAST, depth=10)
        self.publisher = self.create_publisher(Odometry, self.mavros_topic, output_qos)
        self.subscription = self.create_subscription(
            Odometry, self.odom_topic, self._callback, sensor_qos
        )
        self.rejected_frame = None
        self.get_logger().info(
            f"Validated MAVROS EV relay {self.odom_topic} -> {self.mavros_topic}; "
            f"required child_frame_id={self.expected_child_frame}"
        )

    def _callback(self, msg: Odometry):
        if msg.child_frame_id != self.expected_child_frame:
            if msg.child_frame_id != self.rejected_frame:
                self.rejected_frame = msg.child_frame_id
                self.get_logger().error(
                    "EV_FRAME_REJECTED "
                    f"received={msg.child_frame_id!r} required={self.expected_child_frame!r}; "
                    "launch cuVSLAM with base_frame:=drone_link and a calibrated "
                    "camera_link->drone_link static transform"
                )
            return
        self.rejected_frame = None
        msg.header.frame_id = self.output_parent_frame
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MavrosEvBridge()
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
