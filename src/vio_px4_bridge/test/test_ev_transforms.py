import math

from vio_px4_bridge.vio_px4_bridge import VisualOdometryBridge


def bridge_without_ros_node():
    return VisualOdometryBridge.__new__(VisualOdometryBridge)


def test_local_flu_position_becomes_local_frd():
    bridge = bridge_without_ros_node()
    assert bridge.flu_to_frd_vector([1, 2, 3]) == [1, -2, -3]


def test_earth_enu_position_becomes_ned():
    bridge = bridge_without_ros_node()
    assert bridge.enu_to_ned_vector([1, 2, 3]) == [2, 1, -3]


def test_identity_local_flu_attitude_remains_identity_in_frd():
    bridge = bridge_without_ros_node()
    q = bridge.convert_orientation_ROS2_to_PX4([1, 0, 0, 0], "local_flu")
    assert math.isclose(abs(q[0]), 1, abs_tol=1e-12)
    assert all(math.isclose(value, 0, abs_tol=1e-12) for value in q[1:])
