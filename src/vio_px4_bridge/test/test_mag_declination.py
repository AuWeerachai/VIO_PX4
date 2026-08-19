import math

from vio_px4_bridge.mag_declination import load_table
from vio_px4_bridge.mag_declination import lookup_declination_deg


def test_grid_node_is_exact():
    table = load_table()
    row, col = 9, 18  # lat=0, lon=0
    expected = table["declination_deg"][row][col]
    assert lookup_declination_deg(0.0, 0.0) == expected


def test_home_lookup_matches_vns_grid_interpolation():
    value = lookup_declination_deg(40.44334, -79.94363)
    assert math.isclose(value, -9.283724733239664, abs_tol=1e-9)


def test_longitude_wraps():
    assert math.isclose(
        lookup_declination_deg(20.0, 190.0),
        lookup_declination_deg(20.0, -170.0),
        abs_tol=1e-12,
    )
