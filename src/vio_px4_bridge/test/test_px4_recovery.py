"""Exercise bridge recovery without ROS, serial hardware, or a running FCU."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vio_px4_bridge.geo_utils import mavlink_velocity_to_ned, rotate_ned_heading, wrap_pi
from vio_px4_bridge.recovery_yaw import RecoveryYawAgreement, timestamp_order


def test_velocity_frame_is_independent_of_position_frame():
    # Aircraft faces east: body-forward is east, not north.
    q = (math.sqrt(0.5), 0, 0, math.sqrt(0.5))
    assert mavlink_velocity_to_ned(1, 12, q, (1, 0, 0)) == pytest.approx((0, 1, 0))
    assert mavlink_velocity_to_ned(1, 1, q, (1, 0, 0)) == (1, 0, 0)


@pytest.mark.parametrize('pose,child,q,v', [
    (4, 1, (1, 0, 0, 0), (1, 0, 0)),
    (1, 20, (1, 0, 0, 0), (1, 0, 0)),
    (1, 12, (0, 0, 0, 0), (1, 0, 0)),
    (1, 12, (float('nan'), 0, 0, 0), (1, 0, 0)),
    (1, 1, (1, 0, 0, 0), (float('inf'), 0, 0)),
])
def test_invalid_velocity_rejected(pose, child, q, v):
    with pytest.raises(ValueError):
        mavlink_velocity_to_ned(pose, child, q, v)


@pytest.fixture
def bridge():
    # Compile the production methods only, leaving ROS imports/node creation out.
    source = Path(__file__).parents[1] / 'vio_px4_bridge/vio_px4_gps_bridge.py'
    cls = next(n for n in ast.parse(source.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == 'VioPx4GpsBridge')
    names = {'_start_inertial_propagation', '_inertial_recovery_target', '_integrate_px4_yaw',
             '_update_px4_odometry', '_fail_inertial_propagation',
             '_recovery_yaw_agrees', '_finish_inertial_propagation', '_select_accuracy'}
    module = ast.Module(body=[n for n in cls.body if getattr(n, 'name', '') in names],
                        type_ignores=[])
    clock = SimpleNamespace(monotonic=lambda: 10.0)
    ns = dict(time=clock, math=math, wrap_pi=wrap_pi,
              rotate_ned_heading=rotate_ned_heading, timestamp_order=timestamp_order,
              Phase=SimpleNamespace(SPOOF='spoof'))
    exec(compile(module, str(source), 'exec'), ns)
    obj = type('RecoveryHarness', (), {name: ns[name] for name in names})()
    obj.get_logger = Mock(return_value=Mock())
    obj._send_status_text = Mock()
    obj.inertial_active = obj.inertial_safe = True
    obj.inertial_fallback_to_frozen_pose = False
    obj.inertial_failure_reason = None
    obj.inertial_start_time = 7.0
    obj.inertial_max_duration_s = 2.0
    obj.inertial_max_message_age_s = obj.inertial_max_message_gap_s = 0.25
    obj.inertial_reorder_tolerance_s = 0.05
    obj.inertial_max_position_uncertainty_m = 3.0
    obj.inertial_displacement_ned = [1.0, 2.0, 0.0]
    obj.inertial_integrated_duration_s = 1.0
    obj.inertial_yaw_delta = 0.4
    obj.inertial_last_yaw = obj.latest_px4_yaw = (10.0, 10.0, 0.4)
    odom = (10.0, 10.0, 0, 0, 0, 0, 0.1, 0.1, 0)
    obj.latest_px4_odometry = obj.inertial_start_odometry = obj.inertial_last_odometry = odom
    obj.continuity = SimpleNamespace(max_speed_mps=10.0, max_acceleration_mps2=5.0,
                                    max_yaw_rate_rad_s=math.tau)
    obj.heading_offset_rad = 0.0
    obj.recovery_yaw_agreement = RecoveryYawAgreement(required_samples=2)
    obj.cruise_eph_m = 0.1
    obj.cruise_epv_m = 5.0
    obj.phase = 'live'
    obj.frozen_position_uncertainty_m = 0.0
    return obj


def test_repeated_quarantine_preserves_original_propagation_boundary(bridge):
    boundary = bridge.inertial_start_odometry
    bridge._start_inertial_propagation()
    assert bridge.inertial_start_time == 7.0
    assert bridge.inertial_start_odometry == boundary
    assert bridge.inertial_displacement_ned == [1.0, 2.0, 0.0]
    assert bridge.inertial_yaw_delta == 0.4
    # The original two-second deadline has still expired.
    assert bridge._inertial_recovery_target() == ((0, 0, 0), 0.4)


def test_repeated_quarantine_cannot_clear_latched_propagation_failure(bridge):
    bridge._fail_inertial_propagation('px4_estimator_reset')
    bridge._start_inertial_propagation()
    assert bridge.inertial_active
    assert not bridge.inertial_safe
    assert bridge.inertial_failure_reason == 'px4_estimator_reset'
    assert bridge.inertial_start_time == 7.0
    assert bridge._inertial_recovery_target() == ((0, 0, 0), 0)
    assert not bridge._recovery_yaw_agrees(2.0)


def test_timeout_freezes_translation_but_preserves_guarded_yaw(bridge):
    assert bridge._inertial_recovery_target() == ((0, 0, 0), 0.4)
    assert bridge.inertial_safe
    assert not bridge._recovery_yaw_agrees(2.0)
    for t, yaw in [(10.05, 0.45), (10.10, 0.50)]:
        bridge._integrate_px4_yaw((t, t, yaw))
        agreed = bridge._recovery_yaw_agrees(2.0 + yaw - 0.4)
    assert agreed
    assert bridge._inertial_recovery_target() == ((0, 0, 0), pytest.approx(0.5))
    bridge._finish_inertial_propagation()
    assert bridge.recovery_resume_eph_m >= 30.0  # lost movement is not 0.1m accurate
    bridge.recovery_resume_time = -100.0
    assert bridge._select_accuracy()[0] >= 30.0


def test_stale_yaw_after_timeout_still_blocks(bridge):
    bridge._inertial_recovery_target()
    bridge.latest_px4_yaw = (9.0, 9.0, 0.4)
    bridge._inertial_recovery_target()
    assert not bridge.inertial_safe
    assert not bridge._recovery_yaw_agrees(2.0)


def test_reset_counter_after_timeout_still_blocks(bridge):
    bridge._inertial_recovery_target()
    bridge._update_px4_odometry((10.1, 10.1, 0, 0, 0, 0, 0.1, 0.1, 1))
    assert not bridge.inertial_safe
    assert 'px4_estimator_reset' in bridge.inertial_failure_reason


def test_yaw_jump_after_timeout_still_blocks(bridge):
    bridge._inertial_recovery_target()
    bridge._integrate_px4_yaw((10.05, 10.05, 2.0))
    assert not bridge.inertial_safe
    assert 'yaw_rate_limit' in bridge.inertial_failure_reason
