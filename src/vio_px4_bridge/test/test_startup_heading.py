"""Startup math and real bridge dispatch, without ROS or a flight controller."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vio_px4_bridge.geo_utils import wrap_pi, rotate_ned_heading
from vio_px4_bridge.startup_heading import Px4StartupHeading, heading_source


def feed(gate, yaw=0.0, vio_yaw=math.pi/2, start=1.0, count=30):
    for i in range(count):
        now = start + i * .05
        gate.observe(now, now, yaw)
        result = gate.update(vio_yaw, 0.0, now)
    return result


@pytest.mark.parametrize('heading_deg', [0, 90, 158, 180, 270, 359])
def test_px4_heading_rotates_local_forward_to_correct_earth_direction(heading_deg):
    gate = Px4StartupHeading()
    heading = math.radians(heading_deg)
    # VIO starts with body along local ENU +X, equivalent NED yaw +90 deg.
    offset = feed(gate, heading)
    assert wrap_pi(offset - (heading - math.pi/2)) == pytest.approx(0, abs=1e-12)
    north, east, down = rotate_ned_heading(0, 1, 0, offset)
    assert (north, east, down) == pytest.approx((math.cos(heading), math.sin(heading), 0))


def test_angle_wrap_is_not_instability():
    gate = Px4StartupHeading()
    for i in range(30):
        t = 1 + i * .05
        gate.observe(t, t, math.radians(179.9 if i % 2 else -179.9))
        result = gate.update(0, 0, t)
    assert abs(result) == pytest.approx(math.pi)


def test_unique_samples_not_ros_callback_count():
    gate = Px4StartupHeading()
    gate.observe(1, 1, 0)
    for _ in range(100):
        gate.observe(1.01, 1, 0)
        assert gate.update(0, 0, 1.01) is None
    assert len(gate.samples) == 1
    assert not gate.fresh(1.51)


@pytest.mark.parametrize('problem', ['stale', 'moving', 'nan_yaw', 'nan_speed', 'regression', 'gap'])
def test_bad_evidence_clears_window_and_recovers_without_restart(problem):
    gate = Px4StartupHeading()
    assert feed(gate) is not None
    if problem == 'stale':
        assert gate.update(math.pi/2, 0, 4) is None
    elif problem == 'moving':
        assert gate.update(math.pi/2, .5, 2.46) is None
    elif problem == 'nan_yaw':
        gate.observe(2.5, 2.5, float('nan'))
    elif problem == 'nan_speed':
        assert gate.update(math.pi/2, float('nan'), 2.46) is None
    elif problem == 'regression':
        gate.observe(2.5, .1, 0)
    else:
        gate.observe(4, 4, 0)
    assert gate.offset is None
    assert feed(gate, start=5) is not None


def test_unstable_heading_cannot_align_then_stable_can():
    gate = Px4StartupHeading()
    for i in range(30):
        t = 1 + i * .05
        gate.observe(t, t, .5 if i % 2 else -.5)
        assert gate.update(0, 0, t) is None
    assert feed(gate, start=3) is not None


def test_calibration_change_during_spoof_requires_new_window():
    gate = Px4StartupHeading()
    feed(gate)
    gate.observe(2.5, 2.5, 1)
    assert gate.update(math.pi/2, 0, 2.5) is None
    assert feed(gate, yaw=1, start=2.55, count=28) is None
    assert feed(gate, yaw=1, start=3.95, count=1) == pytest.approx(1 - math.pi/2)


def test_new_rc_attempt_cannot_reuse_old_alignment_packet():
    gate = Px4StartupHeading()
    feed(gate)
    gate.reset()
    assert gate.update(math.pi/2, 0, 2.46) is None
    assert gate.samples == []


@pytest.mark.parametrize('saved,expected', [('compass', 'px4'), ('PX4', 'px4'), ('manual', 'manual')])
def test_source_migration(saved, expected):
    assert heading_source(saved) == expected


@pytest.fixture
def bridge():
    source = Path(__file__).parents[1] / 'vio_px4_bridge/vio_px4_gps_bridge.py'
    cls = next(n for n in ast.parse(source.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == 'VioPx4GpsBridge')
    names = {'_poll_mavlink', '_update_heading_alignment', '_handoff_block_reason'}
    clock = SimpleNamespace(now=1.)
    clock.monotonic = lambda: clock.now
    phases = SimpleNamespace(LIVE='live', WAITING='waiting', SPOOF='spoof')
    ns = dict(math=math, time=clock, wrap_pi=wrap_pi, Phase=phases)
    exec(compile(ast.Module(body=[n for n in cls.body if getattr(n, 'name', '') in names],
                            type_ignores=[]), str(source), 'exec'), ns)
    obj = type('StartupHarness', (), {name: ns[name] for name in names})()
    obj.startup_heading = Px4StartupHeading()
    obj.heading_source = 'px4'
    obj.heading_offset_rad = None
    obj.alignment_sample_count = 30
    obj.phase = 'spoof'
    obj.clock = clock
    obj.get_logger = Mock(return_value=Mock())
    obj._send_status_text = Mock()
    obj.transport = 'mavlink'
    obj.px4_target_sysid = obj.px4_target_compid = 1
    obj.inertial_active = False
    obj.rc_trigger_enabled = False
    obj._integrate_px4_yaw = Mock()
    obj.have_vio = True
    obj.last_vio_time = 1.
    obj.vio_timeout_s = 1.
    obj.continuity_recovering = False
    obj.camera_readiness = Mock()
    obj.camera_readiness.reason.return_value = None
    return obj


def attitude(t, yaw, sysid=1, compid=1):
    return SimpleNamespace(get_type=lambda: 'ATTITUDE', get_srcSystem=lambda: sysid,
                           get_srcComponent=lambda: compid, roll=0., pitch=0.,
                           yaw=yaw, time_boot_ms=int(t*1000))


def dispatch(bridge, msg):
    bridge._mav = Mock()
    bridge._mav.recv_match.side_effect = [msg, None]
    bridge._poll_mavlink()


def test_bridge_uses_px4_yaw_without_mag_or_declination_and_freezes_live(bridge):
    # Deliberately inconsistent legacy correction must have no effect.
    bridge.mag_declination_rad = 1.2
    for i in range(30):
        bridge.clock.now = 1 + i*.05
        dispatch(bridge, attitude(bridge.clock.now, math.radians(158)))
        bridge._update_heading_alignment(math.pi/2, 0)
    expected = math.radians(68)
    assert bridge.heading_offset_rad == pytest.approx(expected)
    bridge.phase = 'live'
    dispatch(bridge, attitude(3., 0))
    bridge._update_heading_alignment(1, 0)
    assert bridge.heading_offset_rad == pytest.approx(expected)
    # Recovery still gets the current FCU yaw, not the frozen startup yaw.
    assert bridge._integrate_px4_yaw.call_args.args[0][2] == 0


@pytest.mark.parametrize('sysid,compid', [(2, 1), (1, 191)])
def test_other_vehicle_or_component_cannot_supply_heading(bridge, sysid, compid):
    dispatch(bridge, attitude(1., 0, sysid, compid))
    assert bridge.startup_heading.latest is None
    bridge._integrate_px4_yaw.assert_not_called()


def test_handoff_blocks_when_locked_heading_becomes_stale_or_invalid(bridge):
    bridge.heading_offset_rad = feed(bridge.startup_heading)
    bridge.clock.now = bridge.last_vio_time = 2.46
    assert bridge._handoff_block_reason() is None
    bridge.clock.now = bridge.last_vio_time = 4
    assert 'PX4 startup heading' in bridge._handoff_block_reason()
    dispatch(bridge, attitude(4, float('nan')))
    assert bridge.startup_heading.offset is None


def test_manual_override_still_works(bridge):
    bridge.heading_source = 'manual'
    bridge.manual_heading_rad = math.pi
    bridge._update_heading_alignment(math.pi/2, 0)
    assert bridge.heading_offset_rad == pytest.approx(math.pi/2)
