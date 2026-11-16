"""GPS status records successful writes, not candidates or configured defaults."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vio_px4_bridge.geo_utils import course_over_ground_cdeg


@pytest.fixture
def bridge():
    source = Path(__file__).parents[1] / 'vio_px4_bridge/vio_px4_gps_bridge.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'VioPx4GpsBridge')
    names = {'_publish_fix', '_publish_hil_gps', '_publish_ros2_sensor_gps', '_gps_output_status'}
    definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'clamp_int']
    definitions.extend(n for n in cls.body if getattr(n, 'name', '') in names)
    clock = SimpleNamespace(now=10.)
    clock.monotonic = lambda: clock.now
    clock.time = lambda: 1700000000. + clock.now
    ns = dict(time=clock, math=math, SensorGps=SimpleNamespace,
              Phase=SimpleNamespace(LIVE='live'), course_over_ground_cdeg=course_over_ground_cdeg)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), 'exec'), ns)
    obj = type('GpsStatusHarness', (), {name: ns[name] for name in names})()
    obj.clock = clock
    obj.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(
        nanoseconds=round(clock.time()*1e9)))
    obj.get_logger = Mock(return_value=Mock())
    obj.transport = 'mavlink'
    obj.phase = 'live'
    obj.have_vio = True
    obj.latest_fix_ros_stamp_s = clock.time() - .03
    obj.last_gps_output = None
    obj._mav = Mock()
    obj._gps_pub = Mock()
    obj._hil_gps_extensions_supported = None
    obj.fix_type, obj.satellites, obj.gps_id = 3, 40, 1
    obj.speed_accuracy_m_s = .5
    return obj


@pytest.fixture
def fix():
    return dict(lat=40.413589648, lon=-79.946294148, alt=200.,
                eph=.126, epv=5., vn=0., ve=0., vd=0.)


def test_before_first_send_has_no_invented_home_or_accuracy(bridge):
    assert all(value is None for value in bridge._gps_output_status(10.).values())


@pytest.mark.parametrize('transport', ['mavlink', 'ros2'])
def test_actual_sent_position_accuracy_and_capture_age(bridge, fix, transport):
    bridge.transport = transport
    bridge._publish_fix(fix)
    status = bridge._gps_output_status(10.)
    if transport == 'mavlink':
        packet = bridge._mav.mav.hil_gps_send.call_args.args
        assert status['hil_gps_lat_deg'] == packet[2] * 1e-7
        assert status['hil_gps_lon_deg'] == packet[3] * 1e-7
        assert status['hil_gps_eph_m'] == .13  # the cm-rounded wire value, not .126
        assert packet[12:15] == (40, 1, 0)
    else:
        packet = bridge._gps_pub.publish.call_args.args[0]
        assert status['hil_gps_lat_deg'] == packet.latitude_deg
        assert status['hil_gps_lon_deg'] == packet.longitude_deg
        assert status['hil_gps_eph_m'] == packet.eph
    assert status['hil_gps_sample_age_s'] == pytest.approx(.03, abs=1e-6)
    assert status['hil_gps_send_age_s'] == 0.
    assert status['hil_gps_sample_source'] == 'vio'


def test_pending_fix_and_accuracy_changes_do_not_replace_sent_snapshot(bridge, fix):
    bridge._publish_fix(fix)
    before = bridge._gps_output_status(10.)
    fix.update(lat=10., lon=20., eph=99.)
    bridge.latest_fix = fix
    bridge.latest_fix_ros_stamp_s = bridge.clock.time() + .1
    after = bridge._gps_output_status(10.1)
    for key in ('hil_gps_lat_deg', 'hil_gps_lon_deg', 'hil_gps_eph_m'):
        assert before[key] == after[key]
    assert after['hil_gps_sample_age_s'] == pytest.approx(.13, abs=1e-6)


def test_repeated_send_does_not_refresh_the_original_camera_sample(bridge, fix):
    bridge._publish_fix(fix)
    bridge.clock.now += .05
    bridge._publish_fix(fix)
    status = bridge._gps_output_status(bridge.clock.now)
    assert status['hil_gps_sample_age_s'] == pytest.approx(.08, abs=1e-6)
    assert status['hil_gps_send_age_s'] == 0.


@pytest.mark.parametrize('phase', ['spoof', 'aligning'])
def test_static_spoof_uses_generation_age_not_camera_time(bridge, fix, phase):
    bridge.phase = phase
    bridge.have_vio = False
    bridge.latest_fix_ros_stamp_s = None
    bridge._publish_fix(fix)
    status = bridge._gps_output_status(10.05)
    assert status['hil_gps_sample_source'] == 'spoof'
    assert status['hil_gps_sample_age_s'] == pytest.approx(.05)


@pytest.mark.parametrize('transport', ['mavlink', 'ros2'])
@pytest.mark.parametrize('had_success', [True, False])
def test_failed_write_does_not_claim_a_new_output(bridge, fix, transport, had_success):
    bridge.transport = transport
    if had_success:
        bridge._publish_fix(fix)
    previous = bridge.last_gps_output
    fix.update(lat=0., lon=0., eph=100.)
    bridge.clock.now += .5
    sender = bridge._mav.mav.hil_gps_send if transport == 'mavlink' else bridge._gps_pub.publish
    sender.side_effect = OSError('link failed')
    with pytest.raises(OSError, match='link failed'):
        bridge._publish_fix(fix)
    assert bridge.last_gps_output is previous


def test_old_pymavlink_fallback_still_records_one_success(bridge, fix):
    bridge._mav.mav.hil_gps_send.side_effect = [TypeError('no extensions'), None]
    bridge._publish_fix(fix)
    assert bridge._hil_gps_extensions_supported is False
    assert len(bridge._mav.mav.hil_gps_send.call_args.args) == 13
    assert bridge._gps_output_status(10.)['hil_gps_eph_m'] == .13


@pytest.mark.parametrize('stamp', [None, 0., -1., math.nan, math.inf, 1700000011.])
def test_unusable_camera_stamp_is_unavailable_not_zero_age(bridge, fix, stamp):
    bridge.latest_fix_ros_stamp_s = stamp
    bridge._publish_fix(fix)
    status = bridge._gps_output_status(10.)
    assert status['hil_gps_sample_age_s'] is None
    assert status['hil_gps_send_age_s'] == 0.
