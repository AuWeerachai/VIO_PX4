import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vio_px4_bridge.spoof_readiness import SpoofReadiness

IMU = (0, 0, 9.81, 0, 0, 0)
POSE = (0, 0, 0, 0, 0, 0, 1)


def sample(gate, i, stamp=None, imu=IMU, pose=POSE):
    now = i * .005
    stamp = 1000000000 + i * 5000000 if stamp is None else stamp
    gate.observe_imu(stamp, now, imu, 'camera_imu_optical_frame')
    gate.observe_odom(stamp, now, pose, 'drone_link', 'drone_link')
    gate.observe_fusion(gate.require_imu, now)
    return now


def healthy(gate, start=0, count=403):
    for i in range(start, start + count):
        now = sample(gate, i)
    return now


def test_healthy_vio_and_vo():
    for mode in (True, False):
        gate = SpoofReadiness(mode)
        now = healthy(gate)
        assert gate.reason(now) is None


@pytest.mark.parametrize('bad_stamp', [1000000000 + 200 * 5000000, 100000000])
def test_duplicate_or_regression_discards_window_but_can_recover(bad_stamp):
    gate = SpoofReadiness()
    healthy(gate, count=201)
    sample(gate, 201, stamp=bad_stamp)
    assert 'timestamp discontinuity' in gate.reason(1.005)
    assert len(gate.imu.samples) == 0
    now = healthy(gate, start=202)
    assert gate.reason(now) is None


def test_late_bad_sample_revokes_ready():
    gate = SpoofReadiness()
    now = healthy(gate)
    assert gate.reason(now) is None
    now = sample(gate, 403, imu=(float('nan'), 0, 9.81, 0, 0, 0))
    assert 'IMU' in gate.reason(now)


def test_persistent_bad_timestamps_never_ready():
    gate = SpoofReadiness()
    for i in range(3001):
        sample(gate, i, stamp=1000000000)
        assert gate.reason(i * .005)


def test_new_attempt_requires_new_samples():
    gate = SpoofReadiness()
    now = healthy(gate)
    gate.reset()
    assert gate.reason(now)


def test_gaps_stale_frames_and_invalid_quaternion():
    gate = SpoofReadiness()
    now = healthy(gate)
    assert 'stale' in gate.reason(now + .6)
    sample(gate, 500)
    assert gate.reason(2.5)
    now = healthy(gate, start=501)
    gate.observe_imu(9990000000, now, IMU, 'wrong_frame')
    assert 'IMU' in gate.reason(now)
    gate.observe_odom(9990000000, now, (0,) * 7, 'drone_link', 'drone_link')
    assert 'odometry' in gate.reason(now)


def test_fusion_mode_must_match_and_remain_fresh():
    gate = SpoofReadiness()
    now = healthy(gate)
    gate.observe_fusion(False, now)
    assert 'differs' in gate.reason(now)
    gate.fusion_time = now - 4
    assert 'not confirmed' in gate.reason(now)


def test_queued_burst_does_not_count_as_healthy_wall_time():
    gate = SpoofReadiness()
    for i in range(450):
        stamp = 1000000000 + i * 5000000
        now = i * .0001
        gate.observe_imu(stamp, now, IMU, 'camera_imu_optical_frame')
        gate.observe_odom(stamp, now, POSE, 'drone_link', 'drone_link')
        gate.observe_fusion(True, now)
    assert gate.reason(now)


@pytest.fixture
def bridge():
    # Execute actual production phase methods without requiring ROS/hardware.
    source = Path(__file__).parents[1] / 'vio_px4_bridge/vio_px4_gps_bridge.py'
    tree = ast.parse(source.read_text())
    phase_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Phase')
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'VioPx4GpsBridge')
    names = {'_enter_live', '_handoff_block_reason', '_maybe_end_spoof_by_timer', '_tick'}
    nodes = [n for n in cls.body if getattr(n, 'name', '') in names]
    from enum import Enum
    clock = SimpleNamespace(monotonic=lambda: 15.0)
    ns = dict(Enum=Enum, time=clock)
    exec(compile(ast.Module(body=[phase_class, *nodes], type_ignores=[]), str(source), 'exec'), ns)
    obj = type('HandoffHarness', (), {name: ns[name] for name in names})()
    obj.Phase = ns['Phase']
    obj.phase = obj.Phase.SPOOF
    obj.start_time = 0
    obj.spoof_duration_s = 15
    obj.have_vio = True
    obj.last_vio_time = 15
    obj.vio_timeout_s = .5
    obj.heading_offset_rad = 0.0
    obj.continuity_recovering = False
    obj.camera_readiness = Mock()
    obj.camera_readiness.reason.return_value = None
    obj.spoof_failure_reason = None
    obj.get_logger = Mock(return_value=Mock())
    obj._send_status_text = Mock()
    obj._set_live_output_state = Mock()
    obj._publish_fix = Mock()
    obj._poll_mavlink = Mock()
    obj._last_handoff_reason = None
    obj.inertial_active = False
    return obj


def test_deadline_hands_off_only_when_all_checks_pass(bridge):
    bridge._maybe_end_spoof_by_timer()
    assert bridge.phase == bridge.Phase.LIVE
    assert bridge.spoof_failure_reason is None


@pytest.mark.parametrize('problem', ['IMU timestamps unstable', 'camera fusion setting not confirmed recently'])
def test_failed_deadline_sends_no_more_gps(bridge, problem):
    bridge.camera_readiness.reason.return_value = problem
    bridge._tick()
    assert bridge.phase == bridge.Phase.WAITING
    assert bridge.spoof_failure_reason == problem
    bridge._publish_fix.assert_not_called()
    bridge._tick()
    bridge._publish_fix.assert_not_called()


@pytest.mark.parametrize('field,value', [('have_vio', False), ('last_vio_time', 10),
                                        ('heading_offset_rad', None), ('continuity_recovering', True)])
def test_no_early_handoff_around_other_gates(bridge, field, value):
    setattr(bridge, field, value)
    assert not bridge._enter_live('callback_shortcut')
    assert bridge.phase == bridge.Phase.SPOOF
