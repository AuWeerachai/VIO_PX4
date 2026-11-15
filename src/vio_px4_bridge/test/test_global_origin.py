"""Origin setup tests without an FCU: no arming, parameter writes or GPS height fusion."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vio_px4_bridge.global_origin import GlobalOriginSetup


def test_requires_delayed_matching_readbacks_not_send_success():
    gate = GlobalOriginSetup(40.4, -79.9, 200)
    assert gate.step(0, True) == (False, True)
    gate.observe(.1, 404000000, -799000000, 0)  # old missing-altitude origin
    assert gate.step(1, True) == (True, True)
    assert gate.reason(1)
    gate.observe(1.1, *gate.target)
    assert gate.reason(1.1)
    assert gate.step(2, True) == (False, True)
    gate.observe(2.2, *gate.target)
    assert gate.reason(2.2) is None
    assert gate.reason(5)  # stale replies cannot pass handoff


def test_already_correct_origin_is_not_rewritten_and_zero_is_valid():
    gate = GlobalOriginSetup(40, -79, 0)
    gate.step(0, True)
    gate.observe(.1, *gate.target)
    assert gate.step(1, True) == (False, True)
    gate.observe(1.2, *gate.target)
    assert gate.reason(1.2) is None
    assert gate.attempts == 0


def test_mismatch_revokes_confirmation():
    gate = GlobalOriginSetup(40, -79, 200)
    gate.step(0, True)
    gate.observe(.1, *gate.target)
    gate.observe(1.2, *gate.target)
    gate.observe(2, gate.target[0], gate.target[1], 0)
    assert gate.reason(2)
    assert gate.step(2, True)[0]


def test_missing_readback_bounds_retries_and_latches_failure():
    gate = GlobalOriginSetup(40, -79, 200)
    writes = [gate.step(t / 10, True)[0] for t in range(201)]
    assert sum(writes) == 5
    assert '10 seconds' in gate.reason(20)
    gate.observe(21, *gate.target)
    gate.observe(23, *gate.target)
    assert gate.reason(23)


def test_armed_or_unknown_never_sends_and_stale_heartbeat_can_recover():
    gate = GlobalOriginSetup(40, -79, 200)
    assert gate.step(0, None) == (False, False)
    assert gate.step(1, True) == (True, True)
    assert gate.step(2, False) == (False, False)
    assert gate.step(3, True) == (False, False)
    assert 'DISARMED' in gate.reason(3)


@pytest.fixture
def bridge():
    source = Path(__file__).parents[1] / 'vio_px4_bridge/vio_px4_gps_bridge.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'VioPx4GpsBridge')
    names = {'_update_global_origin', '_handoff_block_reason', '_poll_mavlink'}
    clock = SimpleNamespace(now=0)
    clock.monotonic = lambda: clock.now
    phases = SimpleNamespace(SPOOF='spoof', ALIGNING='aligning', LIVE='live', WAITING='waiting')
    ns = dict(time=clock, Phase=phases)
    exec(compile(ast.Module(body=[n for n in cls.body if getattr(n, 'name', '') in names],
                            type_ignores=[]), str(source), 'exec'), ns)
    obj = type('OriginHarness', (), {name: ns[name] for name in names})()
    obj.clock = clock
    obj.phase = 'spoof'
    obj.transport = 'mavlink'
    obj.send_set_gps_global_origin = True
    obj.origin_setup = GlobalOriginSetup(40, -79, 200)
    obj.home_alt = 200
    obj.px4_target_sysid = 1
    obj.px4_target_compid = 1
    obj.px4_armed = False
    obj.last_px4_heartbeat_time = 0
    obj.heartbeat_timeout_s = 5
    obj._mav = Mock()
    obj.get_logger = Mock(return_value=Mock())
    obj.have_vio = True
    obj.last_vio_time = 0
    obj.vio_timeout_s = 1
    obj.heading_offset_rad = 0
    obj.heading_source = 'px4'
    obj.startup_heading = SimpleNamespace(offset=0, fresh=lambda now: True)
    obj.continuity_recovering = False
    obj.camera_readiness = Mock()
    obj.camera_readiness.reason.return_value = None
    obj.rc_trigger_enabled = False
    return obj


def test_wire_units_and_no_parameter_writes(bridge):
    bridge._update_global_origin()
    bridge.clock.now = 1
    bridge._update_global_origin()
    bridge._mav.mav.set_gps_global_origin_send.assert_called_once_with(
        1, 400000000, -790000000, 200000)
    bridge._mav.mav.param_set_send.assert_not_called()
    assert bridge._handoff_block_reason() == 'waiting for PX4 global-origin readback'


@pytest.mark.parametrize('phase,armed,heartbeat', [('live', False, 0), ('waiting', False, 0),
                                                   ('spoof', True, 0), ('spoof', False, -10)])
def test_no_origin_mutation_live_armed_or_stale(bridge, phase, armed, heartbeat):
    bridge.phase, bridge.px4_armed, bridge.last_px4_heartbeat_time = phase, armed, heartbeat
    for t in range(4):
        bridge.clock.now = t
        bridge._update_global_origin()
    bridge._mav.mav.set_gps_global_origin_send.assert_not_called()


def test_transport_failure_does_not_mark_success(bridge):
    bridge._mav.mav.set_gps_global_origin_send.side_effect = OSError('link lost')
    bridge._update_global_origin()
    bridge.clock.now = 1
    bridge._update_global_origin()
    assert bridge.origin_setup.reason(1)
    assert not bridge.origin_setup.confirmed


def test_readback_only_from_autopilot_and_only_during_startup(bridge):
    bridge._update_global_origin()
    def message(component):
        return SimpleNamespace(get_type=lambda: 'GPS_GLOBAL_ORIGIN', get_srcSystem=lambda: 1,
                               get_srcComponent=lambda: component,
                               latitude=400000000, longitude=-790000000, altitude=200000)
    for t, component in [(0.1, 191), (1.2, 191)]:
        bridge.clock.now = t
        bridge._mav.recv_match.side_effect = [message(component), None]
        bridge._poll_mavlink()
    assert not bridge.origin_setup.confirmed
    for t in [2, 3.1]:
        bridge.clock.now = t
        bridge._mav.recv_match.side_effect = [message(1), None]
        bridge._poll_mavlink()
    assert bridge.origin_setup.confirmed
    bridge.last_vio_time = bridge.clock.now
    assert bridge._handoff_block_reason() is None
    bridge.px4_armed = True
    assert 'DISARMED' in bridge._handoff_block_reason()
    bridge.phase = 'live'
    bridge.clock.now = bridge.last_vio_time = 30
    assert bridge._handoff_block_reason() is None  # setup gate is not a flight gate


@pytest.mark.parametrize('gps_ctrl,height_ref,allowed', [
    (1, 2, True), (1, 0, True), (3, 2, False), (7, 0, False),
    (1, 1, False), (1, 3, False), (1, None, False), (None, 2, False),
])
def test_actual_startup_rejects_gps_height_fusion_and_wrong_height_reference(
        monkeypatch, gps_ctrl, height_ref, allowed):
    from pymavlink import mavutil
    source = Path(__file__).parents[1] / 'vio_px4_bridge/vio_px4_gps_bridge.py'
    cls = next(n for n in ast.parse(source.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == 'VioPx4GpsBridge')
    method = next(n for n in cls.body if getattr(n, 'name', '') == '_init_transport')
    ns = dict(os=os, time=SimpleNamespace(monotonic=lambda: 0))
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), ns)
    obj = Mock()
    obj.transport = 'mavlink'
    obj.mavlink_url = '/dev/never-opened:921600'
    obj.mavlink_sysid, obj.mavlink_compid = 1, 191
    obj.heartbeat_timeout_s = 5
    obj.home_alt = 200
    obj.validate_dual_gps_selection = obj.rc_trigger_enabled = False
    obj.heading_source = 'px4'
    values = dict(MAV_USEHILGPS=1, EKF2_GPS_CTRL=gps_ctrl, EKF2_HGT_REF=height_ref,
                  EKF2_NOAID_TOUT=5000000)
    obj._read_px4_param.side_effect = lambda heartbeat, name: values[name]
    connection = Mock()
    connection.recv_match.return_value = SimpleNamespace(
        autopilot=12, base_mode=0, get_srcSystem=lambda: 1, get_srcComponent=lambda: 1)
    monkeypatch.setattr(mavutil, 'mavlink_connection', Mock(return_value=connection))
    monkeypatch.setattr(mavutil, 'set_dialect', lambda name: None)
    monkeypatch.setattr(mavutil.mavlink, 'WIRE_PROTOCOL_VERSION', '2.0')
    monkeypatch.setenv('MAVLINK20', '1')
    if allowed:
        ns['_init_transport'](obj)
    else:
        with pytest.raises(RuntimeError, match='EKF2_'):
            ns['_init_transport'](obj)
    connection.mav.param_set_send.assert_not_called()
    connection.mav.hil_gps_send.assert_not_called()
