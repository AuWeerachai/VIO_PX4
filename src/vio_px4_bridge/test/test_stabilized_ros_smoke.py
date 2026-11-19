"""Real ROS node/message construction; no camera, serial, FCU, or GPS socket.

Run with ROS Humble sourced. The sole mocked boundary is MAVLink transport.
An isolated ROS domain avoids discovering or controlling existing processes.
"""
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

rclpy = pytest.importorskip('rclpy')
from nav_msgs.msg import Odometry
from vio_px4_bridge import frame_alignment as frames
from vio_px4_bridge.vio_px4_gps_bridge import VioPx4GpsBridge, Phase


def test_real_node_initialization_callback_and_hil_gps(monkeypatch, tmp_path):
    monkeypatch.setenv('ROS_DOMAIN_ID', '227')
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
    monkeypatch.setenv('ROS_LOG_DIR', str(tmp_path))
    mav = Mock()

    def connect(node):
        node._mav = SimpleNamespace(mav=mav, recv_match=lambda **kwargs: None)
        node.px4_target_sysid = node.px4_target_compid = 1

    rclpy.init(args=['--ros-args', '-p', 'camera_imu_required:=false'])
    node = None
    try:
        with patch.object(VioPx4GpsBridge, '_init_transport', connect):
            node = VioPx4GpsBridge()
        assert node.stabilized_heading_enabled
        rclpy.spin_once(node, timeout_sec=.02)
        assert not mav.hil_gps_send.called
        node.phase = Phase.LIVE
        node.heading_offset_rad = 0.
        now = time.monotonic()
        for i in range(5):
            sent = now-.1+i*.01
            token = node.heading_reference.clock.request(sent)
            assert node.heading_reference.clock.observe(
                int((sent-5+.001)*1e9), token, sent+.002)
        node.heading_reference.observe(now-.08, now-.08-5, frames.IDENTITY, 0)
        node.heading_reference.observe(now, now-5, frames.IDENTITY, 0)
        msg = Odometry()
        msg.header.frame_id, msg.child_frame_id = 'odom', 'drone_link'
        stamp = node.get_clock().now().nanoseconds-20_000_000
        msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(stamp, 1_000_000_000)
        q = frames.ros_to_ned_frd(frames.IDENTITY)
        orientation = msg.pose.pose.orientation
        orientation.w, orientation.x, orientation.y, orientation.z = q
        node._odom_callback(msg)
        node._tick()
        mav.hil_gps_send.assert_called_once()
        args = mav.hil_gps_send.call_args.args
        assert args[2:4] == (round(node.home_lat*1e7), round(node.home_lon*1e7))
        assert args[7:11] == (0, 0, 0, 0)
        assert args[14] == 0  # no heading sent to PX4
        assert not mav.odometry_send.called
        assert node.pose_jump_count == 0
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
