"""Wire format, timing, and production bridge callbacks without a live aircraft."""
import ast
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pymavlink.dialects.v20 import common as mavlink

from vio_px4_bridge.external_yaw import ClockSync, ExternalYaw, NavigationTiming, corrected_quaternion, quaternion, rpy
from vio_px4_bridge.geo_utils import wrap_pi
from vio_px4_bridge.local_continuity import LocalPoseContinuity
from test_px4_recovery import bridge as legacy_bridge, vio_message


def test_helpers_are_loaded_from_working_tree_not_colcon_install():
    import vio_px4_bridge.external_yaw as helper
    assert Path(helper.__file__).resolve() == (
        Path(__file__).parents[1] / 'vio_px4_bridge' / 'external_yaw.py').resolve()


def synchronize(aid, now):
    aid.clock.offsets.extend([0.] * 20)
    aid.clock.replies = aid.clock.PX4_WARMUP_REPLIES
    aid.clock.last_good = now


@pytest.mark.parametrize('helper', [NavigationTiming, ExternalYaw])
def test_half_second_capture_limit_is_explicit_and_preserves_timestamp(helper):
    aid = helper(max_age_s=.5)
    assert aid.sample_time(100., 100.5, 10.5) == pytest.approx(10.)
    with pytest.raises(ValueError, match='stale_or_future'):
        helper(max_age_s=.5).sample_time(100., 100.501, 10.501)
    with pytest.raises(ValueError, match='stale_or_future'):
        helper(max_age_s=.5).sample_time(100.01, 100., 10.)
    # Existing installations keep their conservative default until selected.
    assert helper().max_age_s == .2
    for invalid in (.019, .501, 1., math.nan, math.inf):
        with pytest.raises(ValueError, match='0.02..0.5'):
            helper(max_age_s=invalid)


def test_half_second_heading_packet_keeps_real_capture_time_and_expires():
    aid = ExternalYaw(max_age_s=.5)
    synchronize(aid, 10.5)
    aid.store(10., (0, 0, 0), 0., (0, 0, 0, 1), True)
    packet = aid.packet(10.5, .1, 1.)
    assert packet['time_usec'] == 10000000
    assert aid.packet(10.501, .1, 1.) is None


def fcu(t, north=0., yaw=0., vn=0., yaw_variance=.01, reset=0, component=1):
    covariance = [0.] * 21
    covariance[0] = covariance[6] = .25
    covariance[15] = covariance[18] = .001
    covariance[20] = yaw_variance
    return SimpleNamespace(get_type=lambda: 'ODOMETRY', get_srcSystem=lambda: 1,
                           get_srcComponent=lambda: component,
                           time_usec=round(t*1e6), frame_id=1, child_frame_id=1,
                           estimator_type=8, quality=0, x=north, y=0., z=0.,
                           vx=vn, vy=0., vz=0., rollspeed=0., pitchspeed=0., yawspeed=0.,
                           q=quaternion(0, 0, yaw), pose_covariance=covariance,
                           reset_counter=reset)


@pytest.fixture
def bridge(legacy_bridge):
    """Bind ALL production methods, excluding only ROS initialization."""
    source = Path(__file__).parents[1] / 'vio_px4_bridge/vio_px4_gps_bridge.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'VioPx4GpsBridge')
    cls.body = [n for n in cls.body if getattr(n, 'name', '') != '__init__']
    # Reuse the regression suite's controlled clock and pure helper namespace.
    ns = dict(legacy_bridge.poll_real.__func__.__globals__)
    ns.update(Node=object, Imu=SimpleNamespace, ExternalYaw=ExternalYaw, rpy=rpy)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), 'exec'), ns)
    obj = ns['VioPx4GpsBridge']()
    obj.__dict__.update(legacy_bridge.__dict__)
    obj._poll_mavlink = type(obj)._poll_mavlink.__get__(obj)
    obj.external_yaw = ExternalYaw()
    obj.navigation_timing = obj.external_yaw
    obj.transport = 'mavlink'
    obj.external_sample_time = None
    obj.external_yaw_startup_tolerance = math.radians(10)
    obj.recovery_yaw_agreement_samples = 2
    obj.last_px4_heartbeat_time = obj.clock.now
    obj.heartbeat_timeout_s = 5.
    obj.px4_armed = False
    obj.heading_source = 'px4'
    obj.alignment_max_speed = .25
    obj.spoof_eph_m, obj.spoof_epv_m = .5, 1.
    obj.inertial_active = obj.continuity_recovering = False
    obj.continuity = LocalPoseContinuity(confirmation_samples=3, recovery_samples=4)
    obj.camera_readiness.reason.return_value = None
    obj._mav = Mock()
    obj._mav.recv_match.return_value = None
    obj.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(
        nanoseconds=round(obj.clock.now*1e9)))
    obj._select_accuracy = type(obj)._select_accuracy.__get__(obj)
    return obj


def feed(bridge, t, **kwargs):
    bridge.clock.now = t
    synchronize(bridge.navigation_timing, t)
    bridge.last_px4_heartbeat_time = t
    # ODOMETRY only: no separate ATTITUDE is needed by this recovery path.
    bridge._mav.recv_match.side_effect = [fcu(t, **kwargs), None]
    bridge._poll_mavlink()
    bridge._mav.recv_match.side_effect = None
    bridge._mav.recv_match.return_value = None


def outputs(bridge):
    bridge._external_yaw_tick()
    bridge._tick()


def silence(bridge):
    bridge._publish_fix.assert_not_called()
    bridge._mav.mav.odometry_send.assert_not_called()


def clear_sends(bridge):
    bridge._publish_fix.reset_mock()
    bridge._mav.mav.odometry_send.reset_mock()


@pytest.mark.parametrize('kind', ['position', 'yaw', 'both', 'nan', 'frame', 'world_frame', 'timestamp', 'zero_q'])
def test_either_invalid_input_stops_both_outputs(bridge, kind):
    feed(bridge, 10.)
    bridge._odom_callback(vio_message(10., 0.))
    outputs(bridge)
    assert bridge._publish_fix.call_count == bridge._mav.mav.odometry_send.call_count == 1
    assert bridge.latest_fix_ros_stamp_s == 10.
    clear_sends(bridge)
    feed(bridge, 10.05)
    msg = vio_message(10.05, 100. if kind in ('position', 'both') else 0.,
                      yaw=2. if kind in ('yaw', 'both') else 0.)
    if kind == 'nan':
        msg.pose.pose.position.x = math.nan
    if kind == 'frame':
        msg.child_frame_id = 'camera_link'
    if kind == 'world_frame':
        msg.header.frame_id = ''
    if kind == 'timestamp':
        msg.header.stamp.nanosec = 0
    if kind == 'zero_q':
        msg.pose.pose.orientation = SimpleNamespace(x=0., y=0., z=0., w=0.)
    bridge._odom_callback(msg)
    outputs(bridge)
    assert bridge.continuity_recovering
    assert bridge.latest_fix_ros_stamp_s == 10.  # rejected input never refreshes GPS display age
    silence(bridge)


def test_blackout_stops_both_and_same_pose_must_pass_full_recovery(bridge):
    feed(bridge, 10.)
    bridge._odom_callback(vio_message(10., 0.))
    outputs(bridge)
    clear_sends(bridge)
    for i in range(1, 12):
        feed(bridge, 10+i*.05)
        outputs(bridge)
    assert bridge.continuity_recovering
    # The old pose may have been sent at the first few fresh ticks; clear now.
    clear_sends(bridge)
    bridge._odom_callback(vio_message(10.55, 0.))
    outputs(bridge)
    silence(bridge)
    assert bridge.continuity_recovering
    for i in range(12, 40):
        feed(bridge, 10+i*.05)
        bridge._odom_callback(vio_message(bridge.clock.now, 0.))
        outputs(bridge)
        if not bridge.continuity_recovering:
            break
    assert not bridge.continuity_recovering
    assert bridge._mav.mav.odometry_send.call_count == 1
    assert bridge._publish_fix.call_count == 1


@pytest.mark.parametrize('duration', [2., 30., 90.])
@pytest.mark.parametrize('speed', [0., .5])
def test_long_loss_recovers_with_fcu_position_yaw_and_uncertainty(bridge, duration, speed):
    feed(bridge, 10., vn=speed)
    bridge._odom_callback(vio_message(10., 0., vn=speed))
    outputs(bridge)
    clear_sends(bridge)
    steps = round(duration/.05)
    for i in range(1, steps+1):
        t = 10 + i*.05
        feed(bridge, t, north=speed*(t-10), vn=speed, yaw=.4, yaw_variance=.09)
        bridge._odom_callback(vio_message(t, 100. if i % 2 else -100., yaw=2., vn=speed))
        outputs(bridge)
        silence(bridge)
    for i in range(1, 40):
        t = 10+duration+i*.05
        feed(bridge, t, north=speed*(t-10), vn=speed, yaw=.4, yaw_variance=.09)
        # Camera's new arbitrary yaw is 2; no physical turn needed for recovery.
        # Velocity points along earth north after applying the new yaw offset.
        msg = vio_message(t, 100., yaw=2.)
        angle = 2.-.4
        msg.twist.twist.linear.x = speed*math.sin(angle)
        msg.twist.twist.linear.y = speed*math.cos(angle)
        bridge._odom_callback(msg)
        outputs(bridge)
        if not bridge.continuity_recovering:
            break
        silence(bridge)
    assert not bridge.continuity_recovering
    assert bridge.continuity.last_output_position[0] == pytest.approx(speed*(t-10))
    assert bridge.continuity.last_output_yaw == pytest.approx(.4)
    assert bridge.external_yaw.noise_variance == pytest.approx(math.radians(1)**2)
    assert bridge.external_yaw.reported_variance >= .092
    assert bridge._select_accuracy()[0] >= .5
    assert bridge._mav.mav.odometry_send.call_count == 1
    assert bridge._publish_fix.call_count == 1
    packet = bridge._mav.mav.odometry_send.call_args.kwargs
    assert rpy(packet['q'])[2] == pytest.approx(.4)
    assert packet['pose_covariance'][20] >= .092
    assert packet['reset_counter'] == 0
    # Normal tracking MUST NOT follow fresh PX4 yaw continually.
    feed(bridge, t+.1, north=speed*(t-10), yaw=1.0, vn=speed, yaw_variance=.5)
    bridge._odom_callback(vio_message(t+.1, 100., yaw=2.))
    outputs(bridge)
    assert rpy(bridge._mav.mav.odometry_send.call_args.kwargs['q'])[2] == pytest.approx(.4)
    assert bridge._mav.mav.odometry_send.call_args.kwargs['pose_covariance'][20] == packet['pose_covariance'][20]


def test_bad_velocity_does_not_prevent_yaw_window_collecting(bridge):
    feed(bridge, 10.)
    bridge._odom_callback(vio_message(10., 0.))
    feed(bridge, 10.05)
    bridge._odom_callback(vio_message(10.05, 100., yaw=2.))
    for i in range(2, 20):
        feed(bridge, 10+i*.05)
        bridge.recovery_velocity_sample_valid = False
        bridge.external_sample_time = bridge.clock.now
        assert bridge._matched_recovery_target((0, 0, 0), 2.) is None
    assert bridge.recovery_yaw_agreement.good_samples >= 2
    assert bridge.recovery_velocity_agreement_count == 0


def test_recovered_heading_wraps_without_an_extra_rotation(bridge):
    feed(bridge, 10.)
    bridge._odom_callback(vio_message(10., 0.))
    outputs(bridge)
    for i in range(1, 30):
        t = 10+i*.05
        feed(bridge, t, yaw=math.radians(179))
        bridge._odom_callback(vio_message(t, 100., yaw=2.))
        outputs(bridge)
        if not bridge.continuity_recovering:
            break
    assert not bridge.continuity_recovering
    feed(bridge, t+.05, yaw=math.radians(-179))
    bridge._odom_callback(vio_message(t+.05, 100., yaw=2.+math.radians(2)))
    outputs(bridge)
    assert not bridge.continuity_recovering
    packet = bridge._mav.mav.odometry_send.call_args.kwargs
    assert rpy(packet['q'])[2] == pytest.approx(math.radians(-179))


def test_missing_clock_blocks_both_then_valid_clock_allows_full_recovery(bridge):
    feed(bridge, 10.)
    bridge._odom_callback(vio_message(10., 0.))
    outputs(bridge)
    clear_sends(bridge)
    bridge.external_yaw.clock.last_good = None
    outputs(bridge)
    silence(bridge)
    assert bridge.continuity_recovering
    for i in range(1, 30):
        feed(bridge, 10+i*.05)
        bridge._odom_callback(vio_message(bridge.clock.now, 0.))
        outputs(bridge)
        if not bridge.continuity_recovering:
            break
    assert not bridge.continuity_recovering
    assert bridge._mav.mav.odometry_send.call_count == 1
    assert bridge._publish_fix.call_count == 1


def test_bad_yaw_does_not_prevent_velocity_window_collecting(bridge):
    feed(bridge, 10.)
    bridge._odom_callback(vio_message(10., 0.))
    feed(bridge, 10.05)
    bridge._odom_callback(vio_message(10.05, 100., yaw=2.))
    for i in range(2, 20):
        feed(bridge, 10+i*.05)
        bridge.external_sample_time = bridge.clock.now
        assert bridge._matched_recovery_target((0, 0, 0), 1. if i % 2 else -1.) is None
    assert bridge.recovery_velocity_agreement_count >= 2
    assert bridge.recovery_yaw_agreement.good_samples < 2


def test_wrong_component_cannot_supply_recovery_or_heading(bridge):
    bridge._mav.recv_match.side_effect = [fcu(10., component=191), None]
    bridge._poll_mavlink()
    assert not bridge.external_yaw.history


@pytest.mark.parametrize('r,p,y', [(0,0,0), (.2,.4,math.pi), (-.1,-.5,-3.13)])
def test_quaternion_matches_full_basis_conversion_and_wire_decode(r, p, y):
    import numpy as np
    from scipy.spatial.transform import Rotation
    q = Rotation.from_euler('xyz', [r,p,y])
    target_yaw = wrap_pi(math.pi/2-y)
    corrected = corrected_quaternion(q.as_quat(), target_yaw)
    result = Rotation.from_quat([*corrected[1:], corrected[0]]).as_matrix()
    expected = np.array([[0,1,0],[1,0,0],[0,0,-1]]) @ q.as_matrix() @ np.diag([1,-1,-1])
    assert result == pytest.approx(expected)
    aid = ExternalYaw()
    synchronize(aid, 10.)
    aid.store(9.98, (1,2,3), target_yaw, q.as_quat(), True)
    packet = aid.packet(10., .1, 1.)
    wire = mavlink.MAVLink_odometry_message(**packet).pack(mavlink.MAVLink(None))
    decoded = mavlink.MAVLink(None).parse_buffer(wire)[0]
    assert decoded.frame_id == 1 and decoded.child_frame_id == 12
    assert decoded.estimator_type == 2 and decoded.quality == 100
    assert decoded.time_usec == 9980000  # capture, NOT send time
    assert decoded.q == pytest.approx(corrected)
    assert math.isfinite(decoded.x) and math.isnan(decoded.vx)
    assert decoded.pose_covariance[20] == pytest.approx(math.radians(1)**2)


@pytest.mark.parametrize('noise_deg', [1., 5., 10.])
def test_rate_limit_dedup_stale_and_configured_uncertainty(noise_deg):
    aid = ExternalYaw(noise_deg=noise_deg)
    sends = 0
    for i in range(201):
        t = 10+i*.005
        synchronize(aid, t)
        aid.store(t, (0,0,0), 0., (0,0,0,1), True)
        if aid.packet(t, .1, 1.) is not None:
            aid.mark_sent(t)
            sends += 1
        assert aid.packet(t, .1, 1.) is None
    assert 29 <= sends <= 31
    assert aid.packet(11.3, .1, 1.) is None
    synchronize(aid, 100.)
    aid.store(100., (0,0,0), 0., (0,0,0,1), True)
    assert aid.packet(100., .1, 1.)['pose_covariance'][20] == pytest.approx(math.radians(noise_deg)**2)
    aid.variance_floor = .5
    assert aid.packet(100., .1, 1.)['pose_covariance'][20] == .5


@pytest.mark.parametrize('source', ['px4', 'manual'])
@pytest.mark.parametrize('noise_deg', [1., 5.])
def test_startup_inherits_uncertainty_only_when_px4_supplies_reference(bridge, source, noise_deg):
    bridge.phase = 'spoof'
    bridge.heading_source = source
    bridge.external_yaw = ExternalYaw(noise_deg=noise_deg)
    bridge.navigation_timing = bridge.external_yaw
    bridge.external_yaw.sent = 1  # manual seed has already been sent
    for i in range(6):
        t = 10 + i * .05
        feed(bridge, t, yaw=.4, yaw_variance=.5)
        bridge.external_yaw.store(t, (0, 0, 0), .4, (0, 0, 0, 1), True)
        bridge._verify_external_yaw_startup(0.)
    assert bridge.external_yaw.startup_verified
    packet = bridge.external_yaw.packet(t, .7, 2.)
    expected = .502 if source == 'px4' else math.radians(noise_deg)**2
    assert [packet['pose_covariance'][i] for i in (15, 18, 20)] == pytest.approx([expected] * 3)
    assert packet['pose_covariance'][0] == pytest.approx(.7**2)
    assert packet['pose_covariance'][11] == pytest.approx(2.**2)
    assert bridge.external_yaw.history[-1].yaw_variance > .5
    # A lower later covariance must not silently erase inherited uncertainty.
    feed(bridge, t+.05, yaw=.4, yaw_variance=.001)
    bridge.external_yaw.store(t+.05, (0, 0, 0), .4, (0, 0, 0, 1), True)
    bridge._verify_external_yaw_startup(0.)
    assert bridge.external_yaw.reported_variance == pytest.approx(expected)
    # Inheritance does not bypass the fresh matching readback check.
    feed(bridge, t+.1, yaw=1.4, yaw_variance=.001)
    bridge.external_yaw.store(t+.1, (0, 0, 0), .4, (0, 0, 0, 1), True)
    bridge._verify_external_yaw_startup(0.)
    assert not bridge.external_yaw.startup_verified


def test_capture_clock_conversion_and_faults():
    aid = ExternalYaw()
    assert aid.sample_time(1700000000., 1700000000.05, 10.) == pytest.approx(9.95)
    with pytest.raises(ValueError, match='duplicate'):
        aid.sample_time(1700000000., 1700000000.05, 10.)
    with pytest.raises(ValueError, match='stale'):
        aid.sample_time(1700000000.1, 1700000001., 10.95)
    with pytest.raises(ValueError, match='ROS_clock_jump'):
        aid.sample_time(1700000001., 1700000001., 12.)


@pytest.mark.parametrize('yaw_deg,compass_deg', [(0., 0.), (156., 156.), (-170., 190.),
                                               (-180., 180.), (360., 0.)])
def test_output_status_tracks_sent_pose_not_pending_pose(yaw_deg, compass_deg):
    aid = ExternalYaw(noise_deg=1.)
    aid.variance_floor = .09
    synchronize(aid, 10.)
    aid.store(9.98, (0, 0, 0), math.radians(yaw_deg), (0, 0, 0, 1), True)
    assert aid.output_status(10., True)['external_yaw_heading_deg'] is None
    packet = aid.packet(10., .1, 1.)
    assert packet is not None
    aid.mark_sent(10.)
    aid.variance_floor = .16  # later reference uncertainty is not yet on the wire
    # The next camera observation has NOT been sent. Status must not claim it was.
    aid.store(10.01, (0, 0, 0), .75, (0, 0, 0, 1), True)
    snapshot = aid.output_status(10.02, True)
    assert snapshot['external_yaw_heading_deg'] == pytest.approx(compass_deg)
    assert snapshot['external_yaw_variance_rad2'] == packet['pose_covariance'][20]
    assert snapshot['external_yaw_variance_rad2'] == .09
    assert snapshot['external_yaw_sample_age_s'] == pytest.approx(.04)
    assert snapshot['external_yaw_output_active'] is True
    assert not aid.output_status(10.02, False)['external_yaw_output_active']
    assert not aid.output_status(10.3, True)['external_yaw_output_active']
    assert aid.sent == 1  # status is read-only


def test_timesync_roundtrip_offset_reboot_and_delayed_packets():
    sync = ClockSync()
    offset = -100.
    # Simulated FCU clock is companion monotonic - 100 seconds.
    for i in range(800):
        t = 200.+i*.02
        request = sync.request(t)
        if request:
            sync.receive(int((t+.001+offset)*1e9), request[1], t+.002)
        reply = sync.receive(0, int((t+offset)*1e9), t+.001)
        assert reply[1] == int((t+offset)*1e9)
    assert sync.ready(t+.002)
    assert sync.offset == pytest.approx(offset)
    assert not sync.ready(t+2.)
    sync.receive(0, int(1e9), t+2.)  # FCU reboot
    assert not sync.ready(t+2.) and sync.generation == 1
    sync.receive(123, 987654321, t+2.)  # unsolicited reply cannot align us
    assert sync.offset is None
    request = sync.request(t+3.)
    sync.receive(int((t+3.01)*1e9), request[1], t+3.1)
    assert sync.offset is None  # RTT too high


def test_fcu_time_matching_wrap_reset_and_uncertainty():
    aid = ExternalYaw()
    synchronize(aid, 10.1)
    aid.observe_fcu(fcu(10., yaw=math.radians(179)), 10.)
    aid.observe_fcu(fcu(10.1, north=.1, yaw=math.radians(-179)), 10.1)
    matched = aid.matched_fcu(10.05, 10.1, .25)
    assert matched.north == pytest.approx(.05)
    assert abs(matched.yaw) == pytest.approx(math.pi)
    assert matched.yaw_variance >= .01
    aid.observe_fcu(fcu(10.2, reset=1), 10.2)
    assert aid.matched_fcu(10.15, 10.2, .25) is None
    assert aid.matched_fcu(10.6, 10.6, .25) is None


@pytest.mark.parametrize('source,mag', [('px4', 6), ('manual', 5)])
@pytest.mark.parametrize('fault', [None, 'ev_position', 'covariance_override', 'mag', 'armed'])
def test_real_startup_rejects_conflicting_px4_settings(monkeypatch, source, mag, fault):
    from pymavlink import mavutil
    path = Path(__file__).parents[1] / 'vio_px4_bridge/vio_px4_gps_bridge.py'
    cls = next(n for n in ast.parse(path.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == 'VioPx4GpsBridge')
    method = next(n for n in cls.body if getattr(n, 'name', '') == '_init_transport')
    ns = dict(os=os, math=math, time=SimpleNamespace(monotonic=lambda: 0))
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), ns)
    obj = Mock()
    obj.transport = 'mavlink'
    obj.external_yaw = ExternalYaw()
    obj.mavlink_url = '/dev/never-opened:921600'
    obj.mavlink_sysid, obj.mavlink_compid = 1, 191
    obj.heartbeat_timeout_s = 5
    obj.home_alt = 200
    obj.validate_dual_gps_selection = obj.rc_trigger_enabled = False
    obj.heading_source = source
    values = dict(MAV_USEHILGPS=1, EKF2_GPS_CTRL=1, EKF2_HGT_REF=2,
                  EKF2_NOAID_TOUT=5000000, EKF2_EV_CTRL=8, EKF2_EV_NOISE_MD=0,
                  EKF2_MAG_TYPE=mag)
    if fault == 'ev_position': values['EKF2_EV_CTRL'] = 9
    if fault == 'covariance_override': values['EKF2_EV_NOISE_MD'] = 1
    if fault == 'mag': values['EKF2_MAG_TYPE'] = 0
    obj._read_px4_param.side_effect = lambda heartbeat, name: values[name]
    connection = Mock()
    connection.recv_match.return_value = SimpleNamespace(
        autopilot=12, base_mode=128 if fault == 'armed' else 0,
        get_srcSystem=lambda: 1, get_srcComponent=lambda: 1)
    monkeypatch.setattr(mavutil, 'mavlink_connection', Mock(return_value=connection))
    monkeypatch.setattr(mavutil, 'set_dialect', lambda name: None)
    monkeypatch.setattr(mavutil.mavlink, 'WIRE_PROTOCOL_VERSION', '2.0')
    if fault:
        with pytest.raises(RuntimeError):
            ns['_init_transport'](obj)
    else:
        ns['_init_transport'](obj)
    connection.mav.param_set_send.assert_not_called()
    connection.mav.hil_gps_send.assert_not_called()
    connection.mav.odometry_send.assert_not_called()


def test_manual_seed_waits_for_px4_agreement_and_never_forces_reset(bridge):
    bridge.phase = 'spoof'
    bridge.heading_source = 'manual'
    feed(bridge, 10., yaw=0.)
    bridge.external_yaw.store(10., (0,0,0), math.pi/2, (0,0,0,1), True)
    bridge.navigation_timing.trusted_capture_s = 10.
    bridge._verify_external_yaw_startup(0.)
    assert not bridge.external_yaw.startup_verified
    bridge._external_yaw_tick()
    assert bridge._mav.mav.odometry_send.call_count == 1
    assert bridge._mav.mav.odometry_send.call_args.kwargs['reset_counter'] == 0
    for i in range(1, 8):
        feed(bridge, 10+i*.05, yaw=0.)
        bridge.external_yaw.store(bridge.clock.now, (0,0,0), math.pi/2, (0,0,0,1), True)
        bridge._verify_external_yaw_startup(0.)
    assert not bridge.external_yaw.startup_verified
    for i in range(8, 15):
        feed(bridge, 10+i*.05, yaw=math.pi/2)
        bridge.external_yaw.store(bridge.clock.now, (0,0,0), math.pi/2, (0,0,0,1), True)
        bridge._verify_external_yaw_startup(0.)
    assert bridge.external_yaw.startup_verified
    bridge._mav.mav.param_set_send.assert_not_called()
    bridge._mav.mav.command_long_send.assert_not_called()
    bridge.px4_armed = True
    clear_sends(bridge)
    bridge._external_yaw_tick()
    bridge._mav.mav.odometry_send.assert_not_called()


def test_transport_failure_and_missing_sync_stop_gps_too(bridge):
    feed(bridge, 10.)
    bridge._odom_callback(vio_message(10., 0.))
    bridge._mav.mav.odometry_send.side_effect = OSError('link failed')
    outputs(bridge)
    assert bridge.continuity_recovering
    bridge._publish_fix.assert_not_called()
    assert bridge.external_yaw.last_output is None  # failed write is not a sent heading


def test_runtime_init_enabled_without_ros_hardware(monkeypatch, tmp_path):
    # Exercise the real ROS constructor: parameter declarations, imports,
    # subscriptions and timer binding, but never open a serial device.
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
    monkeypatch.setenv('ROS_DOMAIN_ID', '123')
    import rclpy
    from vio_px4_bridge.vio_px4_gps_bridge import Phase, VioPx4GpsBridge
    monkeypatch.setattr(VioPx4GpsBridge, '_init_transport', lambda self: None)
    monkeypatch.setattr(VioPx4GpsBridge, '_send_status_text', lambda *a: None)
    status_path = tmp_path / 'navigation-status.json'
    rclpy.init(args=['--ros-args', '-p', 'external_yaw_enabled:=true',
                    '-p', f'status_file:={status_path}'])
    try:
        node = VioPx4GpsBridge()
        assert isinstance(node.external_yaw, ExternalYaw)
        node._write_status_file()
        status = json.loads(status_path.read_text())
        assert status['external_yaw_heading_deg'] is None
        assert status['hil_gps_lat_deg'] is None
        assert status['hil_gps_eph_m'] is None
        assert status['external_yaw_output_active'] is False
        assert status['external_yaw_sigma_deg'] == pytest.approx(1.)
        now = time.monotonic()
        node.phase = Phase.LIVE
        node.heading_offset_rad = 0.
        node.last_px4_heartbeat_time = now
        synchronize(node.external_yaw, now)
        node.external_yaw.variance_floor = .04
        node.external_yaw.store(now, (0, 0, 0), math.radians(156), (0, 0, 0, 1), True)
        node.navigation_timing.trusted_capture_s = now
        node.external_yaw.mark_sent(now)  # simulate a successful write; no serial connection
        node._mav = Mock()
        node.have_vio = True
        node.latest_fix_ros_stamp_s = node.get_clock().now().nanoseconds * 1e-9 - .03
        node._publish_fix(dict(lat=40.4135896, lon=-79.9462941, alt=200.,
                               vn=0., ve=0., vd=0., eph=.126, epv=5.))
        node._write_status_file()
        status = json.loads(status_path.read_text())
        assert status['external_yaw_heading_deg'] == pytest.approx(156.)
        assert status['external_yaw_variance_rad2'] == .04
        assert status['external_yaw_sigma_deg'] == pytest.approx(math.degrees(.2))
        assert status['external_yaw_output_active'] is True
        assert status['hil_gps_lat_deg'] == pytest.approx(40.4135896)
        assert status['hil_gps_lon_deg'] == pytest.approx(-79.9462941)
        assert status['hil_gps_eph_m'] == .13
        assert status['hil_gps_sample_age_s'] >= .03
        assert status['pose_gate_reason'] is None
        node.continuity.pending_quarantine = 'position_jump'
        node.continuity_recovering = True
        node._write_status_file()
        status = json.loads(status_path.read_text())
        assert status['external_yaw_heading_deg'] == pytest.approx(156.)
        assert status['external_yaw_output_active'] is False
        assert status['hil_gps_lat_deg'] == pytest.approx(40.4135896)
        assert status['hil_gps_eph_m'] == .13
        assert status['pose_gate_reason'] == 'position_jump'
        node.destroy_node()
    finally:
        rclpy.shutdown()


@pytest.mark.parametrize('enabled', ['false', 'true'])
@pytest.mark.parametrize('legacy_yaw', ['1.0', '-180.0'])
def test_runtime_rejects_second_mounting_rotation_before_opening_transport(monkeypatch, enabled, legacy_yaw):
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
    monkeypatch.setenv('ROS_DOMAIN_ID', '123')
    import rclpy
    from vio_px4_bridge.vio_px4_gps_bridge import VioPx4GpsBridge
    transport = Mock()
    monkeypatch.setattr(VioPx4GpsBridge, '_init_transport', transport)
    rclpy.init(args=['--ros-args', '-p', 'external_yaw_enabled:='+enabled,
                    '-p', 'child_to_body_yaw_deg:='+legacy_yaw])
    node = VioPx4GpsBridge.__new__(VioPx4GpsBridge)
    try:
        with pytest.raises(ValueError, match='camera mounting belongs to cuVSLAM extrinsics'):
            node.__init__()
        transport.assert_not_called()
    finally:
        node.destroy_node()
        rclpy.shutdown()
