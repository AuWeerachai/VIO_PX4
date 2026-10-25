"""Exercise bridge recovery without ROS, serial hardware, or a running FCU."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vio_px4_bridge.geo_utils import (
    mavlink_velocity_to_ned, rotate_ned_heading, wrap_pi, enu_offset_to_ned,
    yaw_from_quaternion_xyzw, ned_offset_to_lla,
)
from vio_px4_bridge.recovery_yaw import RecoveryYawAgreement, timestamp_order
from vio_px4_bridge.local_continuity import LocalPoseContinuity
from vio_px4_bridge.reset_heading import ResetHeadingEvidence


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
             '_reject_position_propagation', '_reject_heading_recovery',
             '_validated_recovery_target', '_reset_recovery_agreement',
             '_observe_reset_heading', '_reset_heading_ready',
             '_poll_mavlink',
             '_recovery_velocity_agrees', '_tick', '_odom_callback',
             '_recovery_yaw_agrees', '_finish_inertial_propagation', '_select_accuracy'}
    module = ast.Module(body=[n for n in cls.body if getattr(n, 'name', '') in names],
                        type_ignores=[])
    clock = SimpleNamespace(now=10.0)
    clock.monotonic = lambda: clock.now
    ns = dict(time=clock, math=math, wrap_pi=wrap_pi,
              rotate_ned_heading=rotate_ned_heading, timestamp_order=timestamp_order,
              Odometry=SimpleNamespace, enu_offset_to_ned=enu_offset_to_ned,
              yaw_from_quaternion_xyzw=yaw_from_quaternion_xyzw,
              ned_offset_to_lla=ned_offset_to_lla,
              ResetHeadingEvidence=ResetHeadingEvidence,
              mavlink_velocity_to_ned=mavlink_velocity_to_ned,
              Phase=SimpleNamespace(SPOOF='spoof', LIVE='live', ALIGNING='aligning', WAITING='waiting'))
    exec(compile(module, str(source), 'exec'), ns)
    obj = type('RecoveryHarness', (), {name: ns[name] for name in names})()
    obj.get_logger = Mock(return_value=Mock())
    obj._send_status_text = Mock()
    obj.clock = clock
    obj.inertial_active = obj.inertial_position_safe = obj.inertial_heading_safe = True
    obj.inertial_heading_failure = None
    obj.recovery_reset_rotation_budget_deg = 2.
    obj.recovery_reset_stamp = None
    obj.recovery_reset_reported_stamp = None
    obj.latest_px4_rotation_evidence = None
    obj.reset_heading_evidence = None
    obj.inertial_reordered_odometry_drops = obj.inertial_reordered_yaw_drops = 0
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
                                    max_yaw_rate_rad_s=math.tau, recovery_anchor_yaw=0,
                                    _rotate_xy=LocalPoseContinuity._rotate_xy)
    obj.heading_offset_rad = 0.0
    obj.recovery_yaw_agreement = RecoveryYawAgreement(required_samples=2)
    obj.recovery_yaw_agreement_deg = 20.0
    obj.recovery_velocity_sample_valid = True
    obj.recovery_velocity_agreement_count = 0
    obj.recovery_velocity_last_timestamp = None
    obj.recovery_velocity_agreement_m_s = .2
    obj.recovery_velocity_agreement_samples = 2
    obj.cruise_eph_m = 0.1
    obj.cruise_epv_m = 5.0
    obj.phase = 'live'
    obj.frozen_position_uncertainty_m = 0.0
    obj._poll_mavlink = Mock()
    obj.poll_real = type(obj)._poll_mavlink.__get__(obj)
    obj.transport = 'mavlink'
    obj.px4_target_sysid = 1
    obj._maybe_end_spoof_by_timer = Mock()
    obj._set_live_output_state = Mock()
    obj._publish_fix = Mock()
    obj.continuity_recovering = True
    obj.last_vio_time = 10.0
    obj.have_vio = True
    obj.vio_timeout_s = .5
    obj.horizontal_only_output = True
    obj.home_alt = 300.0
    obj.latest_fix = dict(lat=40.0, lon=-79.0, alt=300.0)
    obj.send_set_gps_global_origin = False
    obj.messages_sent = 1
    obj.camera_readiness = Mock()
    obj.expected_child_frame = 'drone_link'
    obj.rejected_odom_frame = None
    obj.logged_odom_frames = True
    obj.odom_is_enu = True
    obj.twist_is_body_frame = False
    obj.child_to_body_yaw_rad = 0.
    obj.anchor_on_first_odom = True
    obj.anchor_ned = None
    obj._update_heading_alignment = Mock()
    obj.home_lat, obj.home_lon = 40., -79.
    obj.vio_stale_warned = False
    obj.recovery_resume_eph_m = None
    obj.live_start_time = 0.
    obj.boot_duration_s = 0.
    obj.recovery_accuracy_tighten_s = 5.
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
    assert not bridge.inertial_position_safe
    assert not bridge.inertial_heading_safe
    assert bridge.inertial_failure_reason == 'px4_estimator_reset'
    assert bridge.inertial_start_time == 7.0
    assert bridge._inertial_recovery_target() is None
    assert not bridge._recovery_yaw_agrees(2.0)


def test_timeout_freezes_translation_but_preserves_guarded_yaw(bridge):
    assert bridge._inertial_recovery_target() == ((0, 0, 0), 0.4)
    assert not bridge.inertial_position_safe
    assert bridge.inertial_heading_safe
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
    assert not bridge.inertial_heading_safe
    assert not bridge._recovery_yaw_agrees(2.0)


def test_reset_counter_after_timeout_still_blocks(bridge):
    bridge._inertial_recovery_target()
    bridge._update_px4_odometry((10.1, 10.1, 0, 0, 0, 0, 0.1, 0.1, 1))
    assert not bridge.inertial_heading_safe
    assert 'px4_reset_rotation_unverified' in bridge.inertial_heading_failure


def test_yaw_jump_after_timeout_still_blocks(bridge):
    bridge._inertial_recovery_target()
    bridge._integrate_px4_yaw((10.05, 10.05, 2.0))
    assert not bridge.inertial_heading_safe
    assert 'yaw_rate_limit' in bridge.inertial_heading_failure


def feed_px4(bridge, t, yaw=.4, variance=.1, vn=0., ve=0., reset=0):
    bridge.clock.now = t
    bridge._update_px4_odometry((t, t, 0, 0, vn, ve, variance, variance, reset))
    bridge._integrate_px4_yaw((t, t, yaw))


@pytest.mark.parametrize('variance,expired', [(3.109 ** 2, False), (.1, True)])
def test_full_frozen_recovery_then_gps_resumption(bridge, variance, expired):
    """Reported failure: stable VIO recovers despite rejected PX4 translation."""
    bridge.inertial_start_time = 7.0 if expired else 10.0
    gate = LocalPoseContinuity(confirmation_samples=3, recovery_samples=3)
    bridge.continuity = gate
    gate.update((4, 2, 0), .4, (0, 0, 0), 0, 10., 'odom')
    # A 100m camera coordinate jump starts quarantine, not a vehicle movement.
    result = gate.update((104, 2, 0), 2., (0, 0, 0), 0, 10.01, 'odom')
    assert result.recovering
    for i in range(1, 30):
        t = 10 + i * .05
        feed_px4(bridge, t, variance=variance)
        target = bridge._validated_recovery_target((0, 0, 0), 2.)
        result = gate.update((104, 2, 0), 2., (0, 0, 0), 0, t, 'odom',
                             recovery_displacement=None if target is None else target[0],
                             recovery_yaw_delta=0 if target is None else target[1])
        if result.event in ('reset_confirmed', 'candidate_window_restarted'):
            bridge._reset_recovery_agreement()
        if result.recovering:
            bridge._tick()
            bridge._publish_fix.assert_not_called()
        else:
            break
    assert result.event == 'recovery_completed'
    assert result.position == (4, 2, 0)
    assert bridge.inertial_heading_safe
    assert not bridge.inertial_position_safe
    assert bridge.recovery_yaw_agreement.good_samples >= 2
    assert bridge.recovery_velocity_agreement_count >= 2
    bridge._finish_inertial_propagation()
    bridge.continuity_recovering = False
    bridge.last_vio_time = bridge.clock.now
    bridge._tick()
    bridge._publish_fix.assert_called_once()
    assert bridge._publish_fix.call_args.args[0]['eph'] > .1


def test_heading_failure_does_not_erase_independent_position_state(bridge):
    bridge.inertial_start_time = 10.
    bridge._integrate_px4_yaw((10.05, 10.05, 2.))
    assert bridge.inertial_position_safe
    assert not bridge.inertial_heading_safe
    assert bridge.inertial_displacement_ned == [1., 2., 0.]
    for i in range(1, 15):
        feed_px4(bridge, 10 + i * .05, yaw=2.)
        assert bridge._validated_recovery_target((0, 0, 0), 2.) is None
        bridge._tick()
    bridge._publish_fix.assert_not_called()


def test_uncertainty_failure_does_not_hide_later_reset(bridge):
    bridge.inertial_start_time = 10.
    feed_px4(bridge, 10.05, variance=3.109 ** 2)
    assert bridge.inertial_heading_safe
    feed_px4(bridge, 10.10, variance=3.109 ** 2, reset=1)
    assert not bridge.inertial_heading_safe
    assert 'px4_reset_rotation_unverified' in bridge.inertial_heading_failure
    assert 'position_uncertainty' in bridge.inertial_failure_reason


def test_high_uncertainty_at_entry_keeps_heading_initialized(bridge):
    bridge.inertial_active = False
    bridge.latest_px4_odometry = (10., 10., 0, 0, 0, 0, 10., 10., 0)
    bridge._start_inertial_propagation()
    assert not bridge.inertial_position_safe
    assert bridge.inertial_heading_safe
    assert bridge.inertial_last_yaw is not None


def test_velocity_comparison_uses_final_heading_not_provisional_rotation(bridge):
    bridge.inertial_yaw_delta = math.pi / 2
    bridge.continuity.recovery_anchor_yaw = 0
    bridge.latest_px4_odometry = (10., 10., 0, 0, 0, 1, .1, .1, 0)
    assert not bridge._recovery_velocity_agrees((1, 0, 0), 0)
    for _ in range(20):
        assert not bridge._recovery_velocity_agrees((1, 0, 0), 0)
    assert bridge.recovery_velocity_agreement_count == 1  # unique telemetry only
    bridge.latest_px4_odometry = (10.05, 10.05, 0, 0, 0, 1, .1, .1, 0)
    assert bridge._recovery_velocity_agrees((1, 0, 0), 0)
    bridge._reset_recovery_agreement()
    assert bridge.recovery_velocity_agreement_count == 0
    assert bridge.recovery_yaw_agreement.good_samples == 0


def test_bad_velocity_during_frozen_recovery_requires_fresh_good_window(bridge):
    bridge._reject_position_propagation('position_uncertainty')
    bridge.recovery_velocity_agreement_count = 2
    feed_px4(bridge, 10.05, vn=20.)
    assert bridge.recovery_velocity_agreement_count == 0
    assert not bridge._recovery_velocity_agrees((0, 0, 0), 0)
    feed_px4(bridge, 10.10)  # excessive deceleration still invalid
    assert not bridge.recovery_velocity_sample_valid
    feed_px4(bridge, 10.15)
    assert bridge.recovery_velocity_sample_valid
    assert bridge.inertial_heading_safe
    assert not bridge.inertial_position_safe


@pytest.mark.parametrize('stream,kind', [('odom', 'gap'), ('odom', 'regression'),
                                       ('yaw', 'gap'), ('yaw', 'regression'),
                                       ('odom', 'nan'), ('yaw', 'nan')])
def test_broken_reference_after_position_failure_still_blocks(bridge, stream, kind):
    bridge._reject_position_propagation('position_uncertainty')
    t = {'gap': 10.5, 'regression': 9., 'nan': float('nan')}[kind]
    if stream == 'odom':
        bridge._update_px4_odometry((t, t, 0, 0, 0, 0, .1, .1, 0))
    else:
        bridge._integrate_px4_yaw((t, t, .4))
    assert not bridge.inertial_heading_safe
    assert bridge._validated_recovery_target((0, 0, 0), 0) is None


def vio_message(t, north, yaw=0., vn=0.):
    """ENU odometry with an arbitrary local yaw origin after reset."""
    yaw_enu = math.pi / 2 - yaw
    vector = lambda x, y, z: SimpleNamespace(x=x, y=y, z=z)
    pose = SimpleNamespace(position=vector(0., north, 0.),
                           orientation=SimpleNamespace(x=0., y=0.,
                               z=math.sin(yaw_enu/2), w=math.cos(yaw_enu/2)))
    twist = SimpleNamespace(linear=vector(0., vn, 0.), angular=vector(0., 0., 0.))
    return SimpleNamespace(
        header=SimpleNamespace(frame_id='odom', stamp=SimpleNamespace(
            sec=int(t), nanosec=round((t-int(t))*1e9))),
        child_frame_id='drone_link', pose=SimpleNamespace(pose=pose),
        twist=SimpleNamespace(twist=twist))


@pytest.mark.parametrize('failure', ['uncertainty', 'timeout', 'yaw_jump', 'reset', 'none'])
def test_actual_odometry_callback_and_output_gate(bridge, failure):
    """Run production callback, continuity, telemetry checks and publish timer."""
    bridge.continuity = LocalPoseContinuity(confirmation_samples=3, recovery_samples=3)
    bridge.inertial_active = False
    bridge.continuity_recovering = False
    bridge._odom_callback(vio_message(10., 0.))
    initial_fix = dict(bridge.latest_fix)
    feed_px4(bridge, 10.05, yaw=0.)
    bridge._odom_callback(vio_message(10.05, 100., yaw=2.))
    assert bridge.continuity_recovering
    assert bridge.inertial_active
    if failure == 'timeout':
        bridge.inertial_max_duration_s = .1
    blocked = failure in ('yaw_jump', 'reset')
    for i in range(2, 30):
        t = 10. + i * .05
        feed_px4(bridge, t, yaw=2. if failure == 'yaw_jump' else 0.,
                  variance=3.109**2 if failure == 'uncertainty' else .1,
                  reset=1 if failure == 'reset' else 0)
        bridge._odom_callback(vio_message(t, 100., yaw=2.))
        bridge._tick()
        if bridge.continuity_recovering:
            bridge._publish_fix.assert_not_called()
            assert bridge.latest_fix == initial_fix
        else:
            break
    assert bridge.continuity_recovering == blocked
    if blocked:
        bridge._publish_fix.assert_not_called()
        assert not bridge.inertial_heading_safe
    else:
        bridge._publish_fix.assert_called_once()
        assert not bridge.inertial_active
        assert bridge.latest_fix['lat'] == initial_fix['lat']
        assert bridge.latest_fix['lon'] == initial_fix['lon']
        if failure in ('uncertainty', 'timeout'):
            assert bridge.frozen_position_uncertainty_m > 0
        # Relative tracking must continue after handoff, not remain stuck.
        bridge.clock.now += .05
        bridge._odom_callback(vio_message(bridge.clock.now, 100.01, yaw=2.))
        assert not bridge.continuity_recovering
        assert (bridge.latest_fix['lat'], bridge.latest_fix['lon']) != (
            initial_fix['lat'], initial_fix['lon'])


def test_real_motion_uses_propagated_target_when_position_remains_safe(bridge):
    bridge.continuity = LocalPoseContinuity(confirmation_samples=3, recovery_samples=3)
    bridge.inertial_active = False
    bridge.continuity_recovering = False
    bridge.latest_px4_odometry = (10., 10., 0, 0, .5, 0, .1, .1, 0)
    bridge._odom_callback(vio_message(10., 0., vn=.5))
    feed_px4(bridge, 10.05, yaw=0., vn=.5)
    bridge._odom_callback(vio_message(10.05, 100., vn=.5))
    for i in range(2, 30):
        t = 10 + .05 * i
        feed_px4(bridge, t, yaw=0., vn=.5)
        bridge._odom_callback(vio_message(t, 100 + .5*(t-10.05), vn=.5))
        if not bridge.continuity_recovering:
            break
    assert not bridge.continuity_recovering
    assert bridge.inertial_position_safe
    assert bridge.continuity.last_output_position[0] == pytest.approx(.5*(t-10.05))
    assert bridge.frozen_position_uncertainty_m == 0.


def telemetry(bridge, t, counter, yaw=0., gyro_z=0., attitude_first=True):
    """Feed actual MAVLink dispatch with synchronized ODOMETRY q/body gyro."""
    bridge.clock.now = t
    odom = SimpleNamespace(
        get_type=lambda: 'ODOMETRY', get_srcSystem=lambda: 1,
        time_usec=round(t*1e6), frame_id=1, child_frame_id=1, estimator_type=8,
        x=0., y=0., vx=0., vy=0., vz=0., q=(math.cos(yaw/2), 0., 0., math.sin(yaw/2)),
        rollspeed=0., pitchspeed=0., yawspeed=gyro_z,
        pose_covariance=[.1]*21, reset_counter=counter)
    attitude = SimpleNamespace(
        get_type=lambda: 'ATTITUDE', get_srcSystem=lambda: 1,
        time_boot_ms=round(t*1000), roll=0., pitch=0., yaw=yaw)
    queue = [attitude, odom, None] if attitude_first else [odom, attitude, None]
    bridge._mav = SimpleNamespace(recv_match=Mock(side_effect=queue))
    bridge.poll_real()


@pytest.mark.parametrize('heading_reset', [False, True])
def test_45_second_loss_and_aggregate_resets_through_actual_dispatch(bridge, heading_reset):
    bridge.continuity = LocalPoseContinuity(confirmation_samples=3, recovery_samples=3)
    bridge.inertial_active = False
    bridge.continuity_recovering = False
    telemetry(bridge, 10., 9)
    bridge._odom_callback(vio_message(10., 0.))
    original = dict(bridge.latest_fix)
    telemetry(bridge, 10.05, 9)
    bridge._odom_callback(vio_message(10.05, 100., yaw=2.))
    anchor = bridge.continuity.recovery_anchor_position
    start = bridge.inertial_start_time
    # No camera samples during the loss. PX4 position resets twice. The second
    # condition inserts a small yaw reset below the old physical-rate gate:
    # 3deg / .05s = 60deg/s, plausible as motion but not supported by zero gyro.
    for i in range(2, 902):
        t = 10. + i*.05
        counter = 9 if i < 200 else (11 if i < 600 else 12)
        yaw = math.radians(3) if heading_reset and i >= 200 else 0.
        telemetry(bridge, t, counter, yaw, attitude_first=i % 2 == 0)
        bridge._tick()
        bridge._publish_fix.assert_not_called()
        assert bridge.continuity.recovery_anchor_position == anchor
        assert bridge.inertial_start_time == start
    assert not bridge.inertial_position_safe
    assert bridge.inertial_heading_safe != heading_reset
    for i in range(902, 932):
        t = 10. + i*.05
        telemetry(bridge, t, 12, math.radians(3) if heading_reset else 0.)
        bridge._odom_callback(vio_message(t, 100., yaw=2.))
        bridge._tick()
        if not bridge.continuity_recovering:
            break
        bridge._publish_fix.assert_not_called()
    if heading_reset:
        assert bridge.continuity_recovering
        assert 'rotation_unverified' in bridge.inertial_heading_failure
    else:
        assert not bridge.continuity_recovering
        bridge._publish_fix.assert_called_once()
        assert bridge.latest_fix['lat'] == original['lat']
        assert bridge.latest_fix['lon'] == original['lon']
        assert bridge.latest_fix['eph'] >= 450.  # uncertainty not hidden
        assert bridge.recovery_velocity_agreement_count >= 2
        assert bridge.recovery_yaw_agreement.good_samples >= 2


def test_counter_wrap_restarts_velocity_window_but_not_propagation(bridge):
    bridge.inertial_active = False
    telemetry(bridge, 10., 255)
    bridge._start_inertial_propagation()
    telemetry(bridge, 10.05, 0)
    assert bridge.inertial_heading_safe
    assert not bridge.inertial_position_safe
    assert not bridge.recovery_velocity_sample_valid
    assert bridge.recovery_velocity_agreement_count == 0
    telemetry(bridge, 10.1, 0)
    assert bridge.recovery_velocity_sample_valid
    assert bridge.inertial_start_time == 10.
