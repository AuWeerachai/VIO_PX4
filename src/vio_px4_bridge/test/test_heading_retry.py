"""Heading is a consumer of stable GPS continuity, never a new reset trigger."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vio_px4_bridge.heading_output import HeadingOutput, ClockSync, corrected_quaternion, rpy
from vio_px4_bridge.local_continuity import LocalPoseContinuity
from test_px4_recovery import bridge as recovery_fixture, vio_message, feed_px4


def ready(aid, now):
    aid.clock.offsets.extend([0.]*20)
    aid.clock.last_good = now
    aid.clock.replies = 600


@pytest.fixture
def bridge():
    obj = recovery_fixture.__wrapped__()
    class PhaseString(str):
        @property
        def value(self):
            return str(self)
    obj.phase = PhaseString('live')
    path = Path(__file__).parents[1]/'vio_px4_bridge/vio_px4_gps_bridge.py'
    cls = next(n for n in ast.parse(path.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == 'VioPx4GpsBridge')
    names = {'_heading_tick', '_heading_service', '_validate_heading_configuration'}
    ns = dict(obj._odom_callback.__func__.__globals__)
    exec(compile(ast.Module(body=[n for n in cls.body if getattr(n, 'name', '') in names],
                            type_ignores=[]), str(path), 'exec'), ns)
    obj.__class__ = type('HeadingHarness', (type(obj),), {n: ns[n] for n in names})
    obj.heading_output = HeadingOutput(1., 1.)
    obj.vio_timeout_s = 1.
    obj.continuity = LocalPoseContinuity(confirmation_samples=3, recovery_samples=3)
    obj.inertial_active = obj.continuity_recovering = False
    obj.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(
        nanoseconds=int(obj.clock.now*1e9)))
    obj.last_px4_heartbeat_time = 10.
    obj.heartbeat_timeout_s = 5.
    obj._mav = Mock()
    obj.px4_armed = False
    return obj


def accepted(bridge, t, north=0., yaw=0., delay=.3, vn=0.):
    bridge.clock.now = t+delay
    bridge.last_px4_heartbeat_time = bridge.clock.now
    if bridge.heading_output is not None:
        ready(bridge.heading_output, bridge.clock.now)
    bridge._odom_callback(vio_message(t, north, yaw=yaw, vn=vn))


@pytest.mark.parametrize('delay', [.127, .219, .294, .5, .8])
def test_smooth_delayed_odometry_never_starts_quarantine(bridge, delay):
    for i in range(180):
        accepted(bridge, 10+i/90, north=i/180, delay=delay, vn=.5)
        if i % 3 == 0:
            bridge._tick()
            bridge._heading_tick()
        assert not bridge.continuity_recovering
        assert not bridge.inertial_active
        assert bridge.continuity.epoch == 0
        assert bridge.frozen_position_uncertainty_m == 0
        assert bridge.inherited_position_uncertainty_m == 0
    assert bridge._mav.mav.odometry_send.call_count == 60
    assert bridge._publish_fix.call_count == 60
    assert bridge.latest_fix['eph'] == .1
    for call in bridge._mav.mav.odometry_send.call_args_list:
        assert call.kwargs['time_usec'] < int(bridge.clock.now*1e6)
        assert call.kwargs['pose_covariance'][20] == pytest.approx(math.radians(1)**2)


@pytest.mark.parametrize('fault', ['sync', 'too_old', 'ros_clock', 'send_failure'])
def test_heading_only_failure_cannot_change_gps_or_recovery(bridge, fault):
    accepted(bridge, 10.)
    previous = (dict(bridge.latest_fix), dict(bridge.continuity.__dict__),
                bridge.heading_offset_rad, bridge.frozen_position_uncertainty_m)
    aid = bridge.heading_output
    if fault == 'sync':
        aid.clock.reset()
    elif fault == 'too_old':
        stamp, p, q, y = aid.pose
        aid.pose = (stamp-2, p, q, y)
    elif fault == 'ros_clock':
        aid.store(10.1, 10.4, 11., (0, 0, 0), (0, 0, 0, 1), 0)
    else:
        bridge._mav.mav.odometry_send.side_effect = OSError('wire error')
    bridge._heading_tick()
    assert not aid.active
    bridge._tick()
    bridge._publish_fix.assert_called_once()
    assert previous == (bridge.latest_fix, bridge.continuity.__dict__,
                        bridge.heading_offset_rad, bridge.frozen_position_uncertainty_m)
    assert not bridge.continuity_recovering and not bridge.inertial_active


@pytest.mark.parametrize('north,yaw', [(100., 0.), (0., 2.), (100., 2.)])
def test_actual_position_or_yaw_jump_stops_both_and_resumes_after_existing_checks(bridge, north, yaw):
    accepted(bridge, 10., delay=0)
    bridge._tick()
    bridge._heading_tick()
    assert bridge.heading_output.active
    bridge._publish_fix.reset_mock()
    bridge._mav.mav.odometry_send.reset_mock()
    feed_px4(bridge, 10.05, yaw=0.)
    accepted(bridge, 10.05, north=north, yaw=yaw, delay=0)
    assert bridge.continuity_recovering
    for i in range(2, 60):
        t = 10+i*.05
        feed_px4(bridge, t, yaw=0.)
        bridge.heading_output.reference_covariance = (t, math.radians(3)**2)
        accepted(bridge, t, north=north, yaw=yaw, delay=0)
        bridge._tick()
        bridge._heading_tick()
        if not bridge.continuity_recovering:
            break
        bridge._publish_fix.assert_not_called()
        bridge._mav.mav.odometry_send.assert_not_called()
    assert not bridge.continuity_recovering
    bridge._publish_fix.assert_called_once()
    bridge._mav.mav.odometry_send.assert_called_once()
    assert bridge.heading_output.variance == pytest.approx(math.radians(3)**2)
    assert bridge.heading_output.last_output[2] == pytest.approx(0., abs=1e-9)


def test_camera_blackout_stops_both_without_added_reset(bridge):
    accepted(bridge, 10., delay=0)
    bridge.clock.now = 11.1
    ready(bridge.heading_output, bridge.clock.now)
    bridge._tick()
    bridge._heading_tick()
    bridge._publish_fix.assert_not_called()
    bridge._mav.mav.odometry_send.assert_not_called()
    assert bridge.heading_output.block_reason == 'vio_stale'
    assert not bridge.inertial_active  # stable bridge's existing gap handling
    accepted(bridge, 11.2, delay=0)
    assert bridge.continuity_recovering  # genuine sample gap -> existing gate


def test_startup_or_quarantine_never_sends_heading(bridge):
    accepted(bridge, 10.)
    for phase in ('waiting', 'spoof', 'aligning'):
        bridge.phase = phase
        bridge._heading_tick()
        bridge._mav.mav.odometry_send.assert_not_called()
    bridge.phase = 'live'
    bridge.continuity_recovering = True
    bridge._heading_tick()
    bridge._mav.mav.odometry_send.assert_not_called()


def test_corrected_heading_not_continuously_dragged_toward_px4(bridge):
    for i in range(20):
        bridge.latest_px4_yaw = (10., 10., 2.5)
        accepted(bridge, 10+i*.05, yaw=i*.01, delay=.3)
        bridge._heading_tick()
        assert bridge.heading_output.last_output[2] == pytest.approx(i*.01, abs=1e-9)


def test_timestamps_are_real_capture_and_duplicate_is_not_relabelled(bridge):
    accepted(bridge, 10., delay=.3)
    bridge._heading_tick()
    sent = bridge._mav.mav.odometry_send.call_args.kwargs
    assert sent['time_usec'] == 10_000_000
    bridge.clock.now += .033
    bridge._heading_tick()
    bridge._mav.mav.odometry_send.assert_called_once()


def test_new_accepted_pose_does_not_flicker_stream_off_in_status(bridge):
    accepted(bridge, 10.)
    bridge._heading_tick()
    accepted(bridge, 10.01)
    status = bridge.heading_output.status(10.31, True)
    assert status['vo_heading_active']
    assert status['vo_heading_deg'] == pytest.approx(0, abs=1e-9)
    assert status['vo_heading_std_deg'] == pytest.approx(1.)
    assert status['vo_heading_age_s'] == pytest.approx(.31)


def test_invalid_pose_never_reuses_previous_heading(bridge):
    accepted(bridge, 10.)
    msg = vio_message(10.01, 0.)
    msg.pose.pose.orientation.w = float('nan')
    bridge.clock.now = 10.31
    bridge._odom_callback(msg)
    bridge._heading_tick()
    bridge._mav.mav.odometry_send.assert_not_called()


@pytest.mark.parametrize('yaw', [0, 90, 179.9, 180, 270, 359.9])
def test_enu_body_quaternion_uses_earth_yaw_and_preserves_tilt(yaw):
    # Front/rear camera mounting is not part of this conversion: drone_link
    # already contains that extrinsic. A +32-degree ROS pitch becomes -32 FRD.
    pitch = math.radians(32)
    q = (0., math.sin(pitch/2), 0., math.cos(pitch/2))
    roll, got_pitch, got_yaw = rpy(corrected_quaternion(q, math.radians(yaw)))
    assert roll == pytest.approx(0, abs=1e-9)
    assert got_pitch == pytest.approx(-pitch)
    assert math.sin(got_yaw) == pytest.approx(math.sin(math.radians(yaw)), abs=1e-9)
    assert math.cos(got_yaw) == pytest.approx(math.cos(math.radians(yaw)), abs=1e-9)


@pytest.mark.parametrize('field,value', [('EKF2_EV_CTRL', 9), ('EKF2_EV_CTRL', None),
                                      ('EKF2_EV_NOISE_MD', 1), ('EKF2_MAG_TYPE', 0),
                                      ('EKF2_EVA_NOISE', .2)])
def test_startup_rejects_conflicting_px4_fusion_without_param_writes(bridge, field, value):
    params = dict(EKF2_EV_CTRL=8, EKF2_EV_NOISE_MD=0, EKF2_MAG_TYPE=6, EKF2_EVA_NOISE=.01)
    params[field] = value
    bridge._read_px4_param = lambda _, name: params[name]
    with pytest.raises(RuntimeError):
        bridge._validate_heading_configuration(Mock())
    bridge._mav.mav.param_set_send.assert_not_called()


def test_matching_params_and_disarmed_required(bridge):
    params = dict(EKF2_EV_CTRL=8, EKF2_EV_NOISE_MD=0, EKF2_MAG_TYPE=6, EKF2_EVA_NOISE=.01)
    bridge._read_px4_param = lambda _, name: params[name]
    bridge._validate_heading_configuration(Mock())
    bridge.px4_armed = True
    with pytest.raises(RuntimeError, match='DISARMED'):
        bridge._validate_heading_configuration(Mock())


def test_echoed_heading_odometry_cannot_poison_px4_recovery_data(bridge):
    previous = bridge.latest_px4_odometry
    echo = SimpleNamespace(get_type=lambda: 'ODOMETRY', get_srcSystem=lambda: 1,
                           get_srcComponent=lambda: 191)
    bridge._mav.recv_match.side_effect = [echo, None]
    bridge.poll_real()
    assert bridge.latest_px4_odometry == previous
    assert not bridge.inertial_active


def test_timesync_warmup_bad_roundtrip_and_reboot():
    clock = ClockSync()
    for i in range(602):
        now = 10+i*.02
        request = clock.request(now)
        if request:
            clock.receive(int((now+.001)*1e9), request[1], now+.002)
        assert clock.receive(0, int(now*1e9), now) == (int(now*1e9), int(now*1e9))
    assert clock.ready(now+.003)
    last_good = clock.last_good
    request = clock.request(now+.2)
    clock.receive(int((now+.2)*1e9), request[1], now+.3)
    assert clock.last_good == last_good
    clock.receive(0, 1000000000, now+.4)
    assert not clock.ready(now+.4)
    assert clock.generation == 1


def test_recovery_does_not_invent_uncertainty_when_px4_covariance_missing():
    aid = HeadingOutput(1., 1.)
    aid.recover(10., .25)
    assert aid.reason(10.) == 'waiting_for_px4_heading_uncertainty'
    msg = SimpleNamespace(frame_id=1, estimator_type=8, quality=100,
                          q=(1, 0, 0, 0), pose_covariance=[math.radians(2)**2]*21)
    aid.observe_covariance(msg, 10.1)
    assert not aid.awaiting_recovery_covariance
    assert aid.variance == pytest.approx(3*math.radians(2)**2)


def test_gps_outputs_and_continuity_identical_with_heading_on_and_off():
    enabled = bridge.__wrapped__()
    disabled = bridge.__wrapped__()
    disabled.heading_output = None
    for i in range(240):
        t = 10+i*.02
        # 250..350 ms old, continuously arriving, then a real coordinate jump.
        delay = .3 + .05*math.sin(i*.1)
        north = i*.01 if i < 120 else 100 + i*.01
        for b in (enabled, disabled):
            feed_px4(b, t+delay, yaw=0., north=i*.01, vn=.5)
            accepted(b, t, north=north, vn=.5, delay=delay)
            b._tick()
        enabled._heading_tick()
        assert enabled.latest_fix == disabled.latest_fix
        assert enabled.continuity.__dict__ == disabled.continuity.__dict__
        assert enabled._publish_fix.call_args_list == disabled._publish_fix.call_args_list
        assert enabled.frozen_position_uncertainty_m == disabled.frozen_position_uncertainty_m
        assert enabled.inherited_position_uncertainty_m == disabled.inherited_position_uncertainty_m


def test_real_mavlink2_packet_roundtrip_preserves_yaw_covariance_and_capture_stamp():
    mavlink = pytest.importorskip('pymavlink.dialects.v20.common')
    aid = HeadingOutput(1., 1.)
    ready(aid, 10.3)
    aid.store(10, 10.3, 10.3, (1, 2, 3), (0, 0, 0, 1), math.radians(156))
    packet = aid.packet(10.3, .1, 1.5)
    tx, rx = mavlink.MAVLink(None), mavlink.MAVLink(None)
    encoded = mavlink.MAVLink_odometry_message(**packet).pack(tx)
    assert encoded[0] == 0xfd
    decoded = rx.parse_buffer(encoded)[0]
    assert decoded.time_usec == 10_000_000
    assert decoded.frame_id == 1 and decoded.child_frame_id == 12
    assert decoded.pose_covariance[20] == pytest.approx(math.radians(1)**2)
    assert math.degrees(rpy(decoded.q)[2]) == pytest.approx(156, abs=1e-5)
    assert math.isnan(decoded.vx) and math.isnan(decoded.velocity_covariance[0])
