"""Synthetic, hardware-free checks for the experimental 3D recovery branch."""
import math
import random
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from vio_px4_bridge import frame_rotation as rot
from vio_px4_bridge.local_continuity import LocalPoseContinuity
from vio_px4_bridge.recovery_frame import ClockSync, RecoveryFrame, FramePair
from test_px4_recovery import bridge  # production-method harness, no ROS/FCU


def q(r=0, p=0, y=0):
    return rot.from_rpy(*(math.radians(x) for x in (r, p, y)))


def add(a, b):
    return tuple(x+y for x, y in zip(a, b))


def sync(clock, now):
    sent = now-.002
    token = clock.request(sent)
    assert clock.observe(int((sent+now)/2*1e9), token, now)


def message(stamp, position, orientation, velocity=(0, 0, 0)):
    # Convert NED/FRD body pose back to ROS local ENU/FLU for the real callback.
    enu_ned = (0., math.sqrt(.5), math.sqrt(.5), 0.)
    ros_q = rot.multiply(rot.multiply(enu_ned, orientation), (0., 1., 0., 0.))
    w, x, y, z = ros_q
    n, e, d = position
    vn, ve, vd = velocity
    sec = int(stamp)
    return NS(header=NS(stamp=NS(sec=sec, nanosec=round((stamp-sec)*1e9)), frame_id='odom'),
              child_frame_id='drone_link', pose=NS(pose=NS(
                  position=NS(x=e, y=n, z=-d), orientation=NS(w=w, x=x, y=y, z=z))),
              twist=NS(twist=NS(linear=NS(x=ve, y=vn, z=-vd), angular=NS(x=0., y=0., z=0.))))


@pytest.fixture
def dynamic(bridge):
    b = bridge
    class PhaseString(str):
        @property
        def value(self):
            return str(self)
    b.phase = PhaseString('live')
    b.dynamic_repositioning_enabled = True
    b.frame_reference = RecoveryFrame(samples=2)
    b.continuity = LocalPoseContinuity(confirmation_samples=3, recovery_samples=3,
                                      full_attitude=True)
    b.inertial_active = False
    b.continuity_recovering = False
    b.heading_offset_rad = 0.
    b.dynamic_pair = b.dynamic_recovery_orientation = None
    b.dynamic_last_match = None
    b.dynamic_tilt_key = b.dynamic_tilt_error = None
    b.dynamic_tilt_count = 0
    b.tilt_disagreement_rad = math.radians(10)
    b.tilt_disagreement_samples = 3
    b.last_timesync_request = -math.inf
    b.get_clock = lambda: NS(now=lambda: NS(nanoseconds=round((1000+b.clock.now)*1e9)))
    for i in range(5):
        sync(b.frame_reference.clock, 9.5+i*.1)
    b.recovery_velocity_agreement_samples = 2
    b.initial_px4_xy = (100., 200.)  # unrelated PX4 local origin
    return b


def feed(b, t, raw_position=(0., 0., 0.), raw_q=rot.IDENTITY,
         raw_velocity=(0., 0., 0.), fcu_q=rot.IDENTITY,
         px4_position=(0., 0.), px4_velocity=(0., 0.), variance=.01, reset=0,
         synchronized=True):
    b.clock.now = t
    if synchronized:
        sync(b.frame_reference.clock, t)
    b._integrate_px4_yaw((t, t, rot.yaw(fcu_q)))
    b._update_px4_odometry((t, t, *add(px4_position, b.initial_px4_xy),
                           *px4_velocity, variance, variance, reset))
    b._record_px4_frame_attitude(t, t, fcu_q)
    # Camera is 50 ms behind the latest PX4 telemetry.
    b._odom_callback(message(1000+t-.05, raw_position, raw_q, raw_velocity))
    b._tick()


def prime(b):
    for i in range(12):
        feed(b, 10+i*.05)
    assert not b.continuity_recovering


@pytest.mark.parametrize('pitch', [-90, -89.999, -40, 0, 40, 89.999, 90])
@pytest.mark.parametrize('yaw', [-180, -179.999, 0, 179.999, 180])
def test_rigid_frame_recovery_preserves_anchor_and_all_future_axes(pitch, yaw):
    rng = random.Random(23)
    c = q(38.5, pitch, yaw)
    actual_q = q(2, -3, 156)
    raw_q = rot.multiply(rot.inverse(c), actual_q)
    gate = LocalPoseContinuity(full_attitude=True, confirmation_samples=2,
                               recovery_samples=1, max_speed_mps=1e6,
                               max_acceleration_mps2=1e6)
    anchor, raw_anchor = (10., -20., 4.), (100., -72., 9.)
    gate.update(anchor, rot.yaw(actual_q), (0,0,0), 0, 1, 'odom', orientation=actual_q)
    gate.quarantine('external_reset')
    for t in (2, 3, 4):
        out = gate.update(raw_anchor, rot.yaw(raw_q), (0,0,0), 0, t, 'odom',
                          orientation=raw_q, recovery_displacement=(0,0,0),
                          recovery_orientation=actual_q)
    assert not out.recovering
    assert out.position == pytest.approx(anchor)
    for i in range(100):
        delta = tuple(rng.uniform(-10,10) for _ in range(3))
        raw = add(raw_anchor, rot.rotate(rot.inverse(c), delta))
        out = gate.update(raw, rot.yaw(raw_q), (0,0,0), 0, 5+i, 'odom', orientation=raw_q)
        assert out.position == pytest.approx(add(anchor, delta), abs=1e-10)


def test_full_rotation_reset_is_detected_even_without_position_or_yaw_jump():
    gate = LocalPoseContinuity(full_attitude=True)
    gate.update((0,0,0), 0, (0,0,0), 0, 1, 'odom', orientation=q())
    result = gate.update((0,0,0), 0, (0,0,0), 0, 1.01, 'odom', orientation=q(40))
    assert result.reason == 'attitude_jump' and result.recovering


def test_clock_rejects_unrequested_delayed_and_duplicate_responses():
    c = ClockSync()
    assert not c.observe(1000, 123, 10.)
    token = c.request(10.)
    assert not c.observe(10_000_000_000, token, 10.1)
    for i in range(5): sync(c, 11+i*.1)
    assert c.valid(11.4)
    assert c.boot_time(1001., 1001.1, 11.4) == pytest.approx(11.3)
    assert not c.valid(15.)
    token = c.request(11.5)
    assert c.observe(1_000_000_000, token, 11.502)
    assert c.generation == 1 and not c.valid(11.502)
    assert not c.observe(1_000_000_000, token, 11.503)


def test_recovery_candidate_is_not_refit_to_hide_drift_or_duplicate_evidence():
    f = RecoveryFrame(samples=2)
    pair = FramePair(1., q(), 1., (1.,1.,0.,0.,0.,0.,.01,.01,0), 1.)
    assert not f.check(q(40,-20,120), q(), pair)
    for _ in range(100):
        assert not f.check(q(40,-20,120), q(), pair)
    assert f.good_samples == 0
    pair = FramePair(1.1, q(), 1.1, pair.motion, 1.1)
    assert not f.check(q(60,-20,120), q(), pair)
    assert f.reason == '3d_attitude_disagreement'
    assert f.good_samples == 0


def test_stationary_tilted_frame_recovers_without_turning(dynamic):
    b = dynamic
    prime(b)
    before = b.continuity.last_output_position
    bad_q = q(-38.5,-21.9,153)
    for i in range(25):
        feed(b, 10.6+i*.05, (100., -40., 5.), bad_q)
        if i < 5:
            assert b.continuity_recovering
            assert b._set_live_output_state.call_args.args[0] is False
    assert not b.continuity_recovering
    assert b.continuity.last_output_position == pytest.approx(before)
    assert rot.distance(b.continuity.last_output_orientation, q()) < 1e-6


def test_moving_recovery_time_matches_position_without_double_count(dynamic):
    b = dynamic
    prime(b)
    c = q(40,-25,100)
    for i in range(45):
        t = 10.6+i*.05
        # Both sensors describe northward motion; their arbitrary origins differ.
        # PX4 is at time t, the camera is at t-.05.
        delta = (max(0., t-.05-10.6), 0., 0.)
        raw = add((100., -70., 5.), rot.rotate(rot.inverse(c), delta))
        feed(b, t, raw, rot.inverse(c), rot.rotate(rot.inverse(c), (1.,0.,0.)),
             px4_position=(max(0.,t-10.6), 0.), px4_velocity=(1.,0.))
    assert not b.continuity_recovering
    assert b.continuity.last_output_position[:2] == pytest.approx((t-.05-10.6, 0.), abs=1e-6)


def test_level_check_blocks_until_actual_fcu_tilt_returns_to_normal(dynamic):
    b = dynamic
    prime(b)
    for i in range(20):
        feed(b, 10.6+i*.05, (100.,0.,0.), q(45), fcu_q=q(15))
        assert b.continuity_recovering
    assert b.frame_reference.reason == 'fcu_not_nearly_level'
    for i in range(25):
        feed(b, 11.6+i*.05, (100.,0.,0.), q(45), fcu_q=q(2,-3))
    assert not b.continuity_recovering
    assert rot.distance(b.continuity.last_output_orientation, q(2,-3)) < 1e-6


def test_new_candidate_cannot_resume_without_fresh_time_reference(dynamic):
    b = dynamic
    prime(b)
    b.frame_reference.clock.estimates.clear()
    for i in range(30):
        feed(b, 10.6+i*.05, (100.,0.,0.), q(40), synchronized=False)
        assert b.continuity_recovering
        assert b._set_live_output_state.call_args.args[0] is False
    assert b.frame_reference.reason == 'waiting_for_time_sync'
    for i in range(30): feed(b, 12.1+i*.05, (100.,0.,0.), q(40))
    assert not b.continuity_recovering


def test_long_loss_can_recover_with_frozen_position_and_fresh_attitude(dynamic):
    b = dynamic
    prime(b)
    for i in range(601):
        feed(b, 10.6+i*.05, (100.+i%2*20.,0.,0.), q(40+i%2*20), variance=16.)
        assert b.continuity_recovering
    for i in range(30): feed(b, 40.65+i*.05, (100.,0.,0.), q(40), variance=16.)
    assert not b.continuity_recovering
    assert b.continuity.last_output_position == pytest.approx((0.,0.,0.))
    assert b.frozen_position_uncertainty_m > 0  # no fabricated precision


def test_genuine_small_body_tilt_is_not_a_world_frame_error(dynamic):
    b = dynamic
    prime(b)
    for i in range(80):
        t = 10.6+i*.05
        current, camera = q(2*math.sin(t),3*math.cos(t)), q(2*math.sin(t-.05),3*math.cos(t-.05))
        feed(b, t, raw_q=camera, fcu_q=current)
        assert not b.continuity_recovering


def test_slow_tilt_drift_triggers_without_overspeed(dynamic):
    b = dynamic
    prime(b)
    for i in range(30):
        feed(b, 10.6+i*.05, raw_q=q(i*.5))
        if b.continuity_recovering:
            break
    assert b.continuity_recovering
    assert b._set_live_output_state.call_args.args[0] is False


def test_fast_camera_and_slower_px4_can_recover_without_extrapolation(dynamic):
    b = dynamic
    prime(b)
    was_quarantined = False
    for i in range(180):
        t = 10.6+i/90
        b.clock.now = t
        sync(b.frame_reference.clock, t)
        # Independent 20 Hz telemetry and 90 Hz camera; image latency 20 ms.
        if i == 0 or int((t-10.6)*20) != int((t-10.6-1/90)*20):
            b.frame_reference.observe_attitude(t, t, q())
            b._integrate_px4_yaw((t, t, 0.))
            b._update_px4_odometry((t,t,100.,200.,0.,0.,.01,.01,0))
        b._odom_callback(message(1000+t-.02, (100.,-30.,5.), q(40,-20,150)))
        b._tick()
        was_quarantined |= b.continuity_recovering
    assert was_quarantined
    assert not b.continuity_recovering
    assert b.continuity.last_output_position == pytest.approx((0.,0.,0.))


def test_complete_camera_blackout_stops_gps_and_recovers_fresh(dynamic):
    b = dynamic
    prime(b)
    for i in range(80):
        t = 10.6+i*.05
        b.clock.now = t
        sync(b.frame_reference.clock, t)
        b.frame_reference.observe_attitude(t, t, q())
        b._integrate_px4_yaw((t, t, 0.))
        b._update_px4_odometry((t,t,100.,200.,0.,0.,.01,.01,0))
        b._tick()  # no camera callbacks at all
        if t > 10.55+b.vio_timeout_s:
            assert b.continuity_recovering
            assert b._set_live_output_state.call_args.args[0] is False
    for i in range(30):
        feed(b, 14.6+i*.05, (100.,-30.,5.), q(40,-20,150))
    assert not b.continuity_recovering
    assert b.continuity.last_output_position == pytest.approx((0.,0.,0.))


def test_fcu_reset_requires_new_attitude_evidence_but_not_permanent_lockout(dynamic):
    b = dynamic
    prime(b)
    for i in range(5):
        feed(b, 10.6+i*.05, (100.,0.,0.), q(40))
    feed(b, 10.85, (100.,0.,0.), q(40), reset=1)
    assert b.continuity_recovering
    assert not b.inertial_position_safe
    assert b.frame_reference.good_samples == 0
    for i in range(30):
        feed(b, 10.9+i*.05, (100.,0.,0.), q(40), reset=1)
    assert not b.continuity_recovering
    assert b.continuity.last_output_position == pytest.approx((0.,0.,0.))


def test_startup_between_fcu_packets_uses_recent_checked_pair_not_stale_evidence(dynamic):
    b = dynamic
    prime(b)
    b.heading_source = 'manual'
    b.camera_readiness.reason.return_value = None
    assert b._handoff_block_reason() is None
    b.clock.now += .02
    b._odom_callback(message(1000+b.clock.now-.005, (0.,0.,0.), q()))
    assert b.dynamic_pair is None  # the upper FCU bracket has not arrived yet
    assert b._handoff_block_reason() is None  # previous checked pair still fresh
    b.clock.now += .3
    assert b._handoff_block_reason() is not None


def test_missing_pair_cannot_authorize_output_or_preserve_proof_indefinitely():
    f = RecoveryFrame(samples=1)
    pair = FramePair(1., q(), 1., (10.,1.,0.,0.,0.,0.,.01,.01,0), 1.)
    for i in range(5): sync(f.clock, 9.6+i*.1)
    f.observe_attitude(10., 1., q())
    f.observe_motion(pair.motion)
    assert not f.check(q(40), q(), pair)
    f.reason = 'waiting_for_time_matched_px4'
    assert f.waiting_for_next_pair(10.1)
    assert not f.waiting_for_next_pair(10.3)
    assert not f.check(q(40), q(), None)
    assert f.candidate is None and f.good_samples == 0


@pytest.mark.parametrize('offset', [-179, -90, 65, 179])
def test_recovery_respects_initial_earth_heading_and_body_twist(dynamic, offset):
    b = dynamic
    b.heading_offset_rad = math.radians(offset)
    for i in range(12):
        feed(b, 10+i*.05, fcu_q=q(y=offset))
    c = q(40,-25,100)
    for i in range(45):
        t = 10.6+i*.05
        distance = max(0., t-.05-10.6)
        px4_xy = rot.rotate(q(y=offset), (max(0.,t-10.6),0.,0.))[:2]
        px4_v = rot.rotate(q(y=offset), (1.,0.,0.))[:2]
        raw = add((100., -70., 5.), rot.rotate(rot.inverse(c), (distance,0.,0.)))
        # raw velocity below is in the raw world; first test that path.
        feed(b, t, raw, rot.inverse(c), rot.rotate(rot.inverse(c), (1.,0.,0.)),
             fcu_q=q(y=offset), px4_position=px4_xy, px4_velocity=px4_v)
    assert not b.continuity_recovering
    assert b.continuity.last_output_position == pytest.approx((distance,0.,0.), abs=1e-6)
    before = b.continuity.last_output_position
    # Same physical velocity, now expressed in drone_link as real ROS twist.
    b.twist_is_body_frame = True
    b.clock.now += .05
    body_msg = message(1000+b.clock.now-.05, raw, rot.inverse(c))
    body_msg.twist.twist.linear = NS(x=1., y=0., z=0.)
    b._odom_callback(body_msg)
    assert b.continuity.last_output_position == pytest.approx(before)
    assert b.continuity._transform_velocity(b.continuity.last_raw_velocity) == pytest.approx((1.,0.,0.), abs=1e-6)


def test_ros_basis_conversion_against_known_north_east_and_roll_pitch():
    # ROS identity points east, so its NED heading is +90 degrees.
    assert rot.distance(rot.ros_to_ned_frd((0.,0.,0.,1.)), q(y=90)) < 1e-7
    # ROS +90 degree yaw points north. Body roll keeps its sign; pitch flips.
    for r, p, y in ((0,0,90), (20,10,45), (-15,-10,-170)):
        w,x,j,z = q(r,p,y)
        actual = rot.ros_to_ned_frd((x,j,z,w))
        assert rot.distance(actual, q(r,-p,90-y)) < 1e-7
    assert rot.rotate(q(y=90), (1.,0.,0.)) == pytest.approx((0.,1.,0.))
    assert rot.rotate(q(p=90), (1.,0.,0.)) == pytest.approx((0.,0.,-1.))


def test_quaternion_sign_and_yaw_wrap_do_not_trigger_false_resets():
    gate = LocalPoseContinuity(full_attitude=True)
    first = q(2,-3,179)
    gate.update((0,0,0), rot.yaw(first), (0,0,0), 0, 1, 'odom', orientation=first)
    second = tuple(-v for v in q(2,-3,-179))
    out = gate.update((0,0,0), rot.yaw(second), (0,0,0), 0, 1.05, 'odom', orientation=second)
    assert not out.recovering


@pytest.mark.parametrize('failure', ['body_frame', 'nan_position', 'zero_orientation', 'old_capture_time'])
def test_invalid_input_stops_output_before_another_good_sample(dynamic, failure):
    b = dynamic
    prime(b)
    bad = message(1000+b.clock.now, (0.,0.,0.), q())
    if failure == 'body_frame': bad.child_frame_id = 'camera_link'
    if failure == 'nan_position': bad.pose.pose.position.x = math.nan
    if failure == 'zero_orientation': bad.pose.pose.orientation = NS(x=0.,y=0.,z=0.,w=0.)
    if failure == 'old_capture_time': bad.header.stamp.sec -= 2
    b._odom_callback(bad)
    b._tick()
    assert b.continuity_recovering
    assert b._set_live_output_state.call_args.args[0] is False


@pytest.mark.parametrize('component', [1, 191])
def test_mavlink_dispatch_uses_fcu_odometry_quaternion_at_sample_time(dynamic, component):
    b = dynamic
    prime(b)
    b.frame_reference.clear_history('test')
    b.clock.now = 11.
    odom = NS(get_type=lambda: 'ODOMETRY', get_srcSystem=lambda: 1,
              get_srcComponent=lambda: component, time_usec=10_980_000,
              frame_id=1, child_frame_id=1, estimator_type=8,
              x=100.,y=200.,vx=0.,vy=0.,vz=0.,q=q(2,-3,156),
              pose_covariance=[.01]*21, reset_counter=0, quality=0)
    b._mav = NS(mav=Mock(), recv_match=Mock(side_effect=[odom,None]))
    b.poll_real()
    if component == 1:
        received, stamp, orientation = b.frame_reference.attitudes[-1]
        assert received == 11. and stamp == pytest.approx(10.98)
        assert rot.distance(orientation, q(2,-3,156)) < 1e-7
        assert b.frame_reference.motions[-1][1] == stamp
    else:
        assert not b.frame_reference.attitudes and not b.frame_reference.motions


def test_timesync_clock_restart_discards_old_proof_and_stops_gps(dynamic):
    b = dynamic
    prime(b)
    b.clock.now = 11.
    token = b.frame_reference.clock.request(10.998)
    reply = NS(get_type=lambda: 'TIMESYNC', get_srcSystem=lambda: 1,
               get_srcComponent=lambda: 1, tc1=100_000_000, ts1=token)
    b._mav = NS(mav=Mock(), recv_match=Mock(side_effect=[reply,None]))
    b.poll_real()
    assert b.continuity_recovering
    assert not b.frame_reference.clock.valid(11.)
    assert not b.frame_reference.attitudes and not b.frame_reference.motions
    assert b.dynamic_last_match is None
    assert b._set_live_output_state.call_args.args[0] is False
