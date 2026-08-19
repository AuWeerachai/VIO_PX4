import math

from vio_px4_bridge.geo_utils import rotate_ned_heading
from vio_px4_bridge.geo_utils import tilt_compensated_compass_yaw
from vio_px4_bridge.geo_utils import wrap_pi
from vio_px4_bridge.geo_utils import yaw_from_quaternion_xyzw


def test_level_compass_north_and_east():
    assert tilt_compensated_compass_yaw(0, 0, 1, 0, 0, 0) == 0
    assert math.isclose(
        tilt_compensated_compass_yaw(0, 0, 0, -1, 0, 0), math.pi / 2
    )


def test_heading_rotation_north_to_east():
    north, east, down = rotate_ned_heading(1, 0, 2, math.pi / 2)
    assert math.isclose(north, 0, abs_tol=1e-12)
    assert math.isclose(east, 1, abs_tol=1e-12)
    assert down == 2


def test_enu_body_north_maps_to_zero_ned_yaw():
    yaw_enu = yaw_from_quaternion_xyzw(
        0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4)
    )
    assert math.isclose(wrap_pi(math.pi / 2 - yaw_enu), 0, abs_tol=1e-12)


def test_declination_is_applied():
    declination = math.radians(-9)
    result = tilt_compensated_compass_yaw(0, 0, 1, 0, 0, declination)
    assert math.isclose(result, declination)
