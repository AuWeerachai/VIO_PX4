"""Optional real ROS node startup check; FCU transport is always mocked.

Run after sourcing ROS Humble, with src/vio_px4_bridge on PYTHONPATH.
No serial/network MAVLink connection, camera launch, or PX4 writes occur.
"""
from unittest.mock import patch

import pytest

rclpy = pytest.importorskip("rclpy", reason="source ROS Humble for node smoke tests")
from vio_px4_bridge.vio_px4_gps_bridge import VioPx4GpsBridge


@pytest.mark.parametrize("enabled", [True, False])
def test_real_ros_node_startup_and_timers_without_fcu(monkeypatch, tmp_path, enabled):
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    monkeypatch.setenv("ROS_DOMAIN_ID", "220")
    monkeypatch.setenv("ROS_LOG_DIR", str(tmp_path))
    rclpy.init(args=["--ros-args", "-p", f"dynamic_repositioning_enabled:={str(enabled).lower()}"])
    node = None
    try:
        with patch.object(VioPx4GpsBridge, "_init_transport") as transport:
            node = VioPx4GpsBridge()
        transport.assert_called_once()
        assert node._mav is None
        assert node.dynamic_repositioning_enabled is enabled
        assert node.continuity.full_attitude is enabled
        for _ in range(3):
            rclpy.spin_once(node, timeout_sec=.05)
        assert node.messages_sent == 0
        assert not node.live_output_active
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
