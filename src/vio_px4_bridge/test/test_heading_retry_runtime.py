"""Real ROS timers and MAVLink2 serialization; no hardware/serial connection.

FCU connection and clock convergence are simulated. This does not verify PX4
fusion or a Jetson's scheduling performance; those remain bench-test items.
"""
import math
import time
from types import SimpleNamespace

import pytest


def test_real_node_runs_both_outputs_at_30hz_and_cuts_both_on_jump(monkeypatch):
    rclpy = pytest.importorskip('rclpy')
    mavlink = pytest.importorskip('pymavlink.dialects.v20.common')
    from nav_msgs.msg import Odometry
    from rclpy.executors import SingleThreadedExecutor
    from vio_px4_bridge.vio_px4_gps_bridge import VioPx4GpsBridge, Phase

    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
    monkeypatch.setenv('ROS_DOMAIN_ID', '231')
    frames = []
    parser = mavlink.MAVLink(None)

    class Sink:
        def write(self, data):
            for msg in parser.parse_buffer(data) or []:
                frames.append((time.monotonic(), msg))
            return len(data)

    def fake_connection(self):
        self._mav = SimpleNamespace(
            mav=mavlink.MAVLink(Sink(), srcSystem=1, srcComponent=191),
            recv_match=lambda **kwargs: None)
        self.px4_target_sysid = self.px4_target_compid = 1
        self.px4_armed = False
        self.last_px4_heartbeat_time = time.monotonic()

    monkeypatch.setattr(VioPx4GpsBridge, '_init_transport', fake_connection)
    rclpy.init(args=['--ros-args', '-p', 'send_vo_heading:=true',
                    '-p', 'camera_imu_required:=false'])
    executor = SingleThreadedExecutor()
    node = None
    try:
        node = VioPx4GpsBridge()
        assert node.rate_hz == 30
        node.phase = Phase.LIVE
        node.heading_offset_rad = math.radians(66)  # local +90 -> true +156
        node.live_start_time = time.monotonic()
        executor.add_node(node)
        start = time.monotonic()
        next_camera = start
        while time.monotonic()-start < 2.1:
            now = time.monotonic()
            if now >= next_camera:
                msg = Odometry()
                capture_ns = node.get_clock().now().nanoseconds-300_000_000
                msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(capture_ns, 1_000_000_000)
                msg.header.frame_id = 'odom'
                msg.child_frame_id = 'drone_link'
                msg.pose.pose.orientation.w = 1.
                node.heading_output.clock.offsets.extend([0.]*20)
                node.heading_output.clock.last_good = now
                node.heading_output.clock.replies = 600
                node._odom_callback(msg)
                next_camera += 1/90
            executor.spin_once(timeout_sec=.002)
        gps = [(t, m) for t, m in frames if m.get_type() == 'HIL_GPS']
        yaw = [(t, m) for t, m in frames if m.get_type() == 'ODOMETRY']
        for label, stream in (('HIL_GPS', gps), ('VO heading', yaw)):
            assert len(stream) >= 50
            measured_hz = (len(stream)-1)/(stream[-1][0]-stream[0][0])
            print(f'{label}: {len(stream)} packets, {measured_hz:.2f} Hz, capture age ~300 ms')
            assert 28 <= measured_hz <= 32
        assert len(set(m.time_usec for _, m in yaw)) == len(yaw)
        assert not node.continuity_recovering
        assert node.continuity.epoch == 0
        assert node.frozen_position_uncertainty_m == 0
        assert all(m.eph == 10 for _, m in gps)
        assert all(m.pose_covariance[20] == pytest.approx(math.radians(1)**2) for _, m in yaw)
        assert all(.25 < t-m.time_usec*1e-6 < .5 for t, m in yaw)

        # Actual callback rejects a spatial discontinuity, then both real timers stop.
        capture_ns = node.get_clock().now().nanoseconds-300_000_000
        msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(capture_ns, 1_000_000_000)
        msg.pose.pose.position.x = 100.
        node._odom_callback(msg)
        assert node.continuity_recovering
        before = (len(gps), len(yaw))
        deadline = time.monotonic()+.2
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=.002)
        assert before == (sum(m.get_type() == 'HIL_GPS' for _, m in frames),
                          sum(m.get_type() == 'ODOMETRY' for _, m in frames))
    finally:
        executor.shutdown()
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
