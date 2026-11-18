"""Incremental heading alignment: synthetic tests, not flight accuracy claims."""
import math

import pytest

from vio_px4_bridge import frame_rotation as rot
from vio_px4_bridge.local_continuity import LocalPoseContinuity
from test_px4_recovery import bridge
from test_dynamic_repositioning import dynamic, feed, prime, q, message, sync


def advance(gate, p, orientation, stamp, heading=0., velocity=(0.,0.,0.)):
    return gate.update(p, rot.yaw(orientation), velocity, 0., stamp, 'odom|drone_link',
                       orientation=orientation, heading_reference=heading, require_heading=True)


def test_heading_change_at_stationary_offset_does_not_rotate_previous_position():
    g = LocalPoseContinuity(full_attitude=True, max_speed_mps=20)
    advance(g, (50.,30.,0.), q(), 1.)
    advance(g, (60.,30.,0.), q(), 2.)
    for i in range(1, 101):
        out = advance(g, (60.,30.,0.), q(y=i*.1), 2+i*.05)
        assert not out.recovering
        assert out.position == pytest.approx((60.,30.,0.), abs=1e-10)
    # Future forward movement is rotated back by ten degrees; old ten metres
    # are not retroactively rotated about the raw origin or home.
    step = rot.rotate(q(y=10), (1.,0.,0.))
    p = tuple(a+b for a,b in zip((60.,30.,0.), step))
    out = advance(g, p, q(y=10), 8.)
    assert out.position == pytest.approx((61.,30.,0.), abs=1e-10)


@pytest.mark.parametrize('initial_heading', [-179., -90., 0., 67., 179.])
def test_gradual_vo_yaw_drift_is_not_used_as_absolute_heading(initial_heading):
    g = LocalPoseContinuity(full_attitude=True)
    raw = [0.,0.,0.]
    heading = math.radians(initial_heading)
    advance(g, raw, q(y=initial_heading), 1., heading)
    for i in range(1, 2001):
        error = i*.01  # 20 degrees of accumulated VO yaw error over 20 seconds
        step = rot.rotate(q(y=initial_heading+error-.005), (.01,0.,0.))
        raw = [raw[k]+step[k] for k in range(3)]
        out = advance(g, raw, q(y=initial_heading+error), 1+i*.01, heading)
        assert not out.recovering
    expected = rot.rotate(q(y=initial_heading), (20.,0.,0.))
    assert out.position == pytest.approx(expected, abs=1e-8)
    assert math.dist(raw, expected) > 3.  # unchanged raw VO remains curved


def test_real_turns_are_preserved_and_yaw_wrap_is_not_a_reset():
    g = LocalPoseContinuity(full_attitude=True)
    radius = 2.
    advance(g, (radius,0.,0.), q(y=90), 1., math.pi/2)
    for i in range(1, 2001):
        angle = math.tau*i/2000
        p = (radius*math.cos(angle), radius*math.sin(angle), 0.)
        orientation = rot.from_rpy(0.,0.,angle+math.pi/2)
        out = advance(g, p, orientation, 1+i*.02, rot.yaw(orientation))
        assert not out.recovering
        assert out.position == pytest.approx(p, abs=1e-9)
    assert out.position == pytest.approx((radius,0.,0.), abs=1e-9)


def test_unpaired_samples_hold_output_without_losing_or_double_counting_motion():
    g = LocalPoseContinuity(full_attitude=True)
    advance(g, (0.,0.,0.), q(), 1.)
    for i in (1,2):
        out = advance(g, (i*.1,0.,0.), q(), 1+i*.1, None)
        assert out.waiting_for_heading and not out.recovering
        assert out.position == (0.,0.,0.)
        assert g.last_output_timestamp == 1.
        assert g.last_timestamp == 1+i*.1  # physical gates still see every sample
    out = advance(g, (.3,0.,0.), q(), 1.3)
    assert not out.waiting_for_heading and not out.recovering
    assert out.position == pytest.approx((.3,0.,0.))


def test_corrupt_sample_between_matches_is_not_skipped():
    g = LocalPoseContinuity(full_attitude=True)
    advance(g, (0.,0.,0.), q(), 1.)
    bad = advance(g, (100.,0.,0.), q(), 1.01, None)
    assert bad.recovering and bad.reason == 'position_jump'
    returned = advance(g, (0.,0.,0.), q(), 1.02)
    assert returned.recovering  # no shortcut even though it returned to old pose


def test_fcu_reset_in_healthy_tracking_stops_gps_and_preserves_position(dynamic):
    b = dynamic
    prime(b)
    before = b.continuity.last_output_position
    feed(b, 10.6, fcu_q=q(y=10), reset=1)
    assert b.continuity_recovering
    assert b.continuity.last_output_position == before
    assert b._set_live_output_state.call_args.args[0] is False
    assert not b.inertial_position_safe
    for i in range(1,35):
        feed(b, 10.6+i*.05, fcu_q=q(y=10), reset=1)
    assert not b.continuity_recovering
    assert b.continuity.last_output_position == pytest.approx(before)
    assert math.degrees(b.continuity.last_output_yaw) == pytest.approx(10.)
    assert b.frozen_position_uncertainty_m > 0.


def test_impossible_fcu_heading_step_stops_gps_even_without_reset_counter(dynamic):
    b = dynamic
    prime(b)
    feed(b, 10.6, fcu_q=q(y=90))  # 90 degrees / .05s exceeds aircraft rate limit
    assert b.continuity_recovering
    assert b._set_live_output_state.call_args.args[0] is False
    assert b.inertial_failure_reason == 'px4_attitude_rate_limit'


def test_loss_of_fcu_reference_does_not_fall_back_to_vo_yaw(dynamic):
    b = dynamic
    prime(b)
    for i in range(1,15):
        b.clock.now = 10.55+i*.05
        b._odom_callback(message(1000+b.clock.now-.02, (0.,0.,0.), q()))
        b._tick()  # camera continues but there is no FCU telemetry
    assert b.continuity_recovering
    assert b._set_live_output_state.call_args.args[0] is False
    for i in range(30): feed(b, 11.3+i*.05)
    assert not b.continuity_recovering


def test_bridge_continuous_alignment_with_fixed_earth_basis(dynamic):
    b = dynamic
    b.heading_offset_rad = math.radians(65)
    b.continuity.max_acceleration_mps2 = 50.
    for i in range(12): feed(b, 10+i*.05, fcu_q=q(y=65))
    raw = [0.,0.,0.]
    for i in range(1,401):
        error = i*.05
        step = rot.rotate(q(y=error-.025), (.05,0.,0.))
        raw = [raw[k]+step[k] for k in range(3)]
        feed(b, 10.55+i*.05, raw, q(y=error),
             rot.rotate(q(y=error), (1.,0.,0.)), fcu_q=q(y=65))
        assert not b.continuity_recovering
    assert b.heading_offset_rad == math.radians(65)  # never rotate the whole map
    assert b.continuity.last_output_position == pytest.approx((20.,0.,0.), abs=1e-8)
    assert b._set_live_output_state.call_args.args[0] is True


def test_quarantine_boundary_uses_last_published_not_last_unpaired_camera_time(dynamic):
    b = dynamic
    prime(b)
    trusted = b.continuity.last_output_timestamp
    b.clock.now = 10.57
    b._odom_callback(message(1010.565, (.01,0.,0.), q()))
    assert b.dynamic_pair is None
    assert b.continuity.last_timestamp > trusted
    assert b.continuity.last_output_timestamp == trusted
    b._dynamic_quarantine('test_loss')
    assert b.inertial_start_odometry[1] == pytest.approx(trusted-1000)


def test_fast_camera_slow_fcu_keeps_all_displacement_when_only_some_samples_match(dynamic):
    b = dynamic
    prime(b)
    b.continuity.max_acceleration_mps2 = 100.
    for i in range(181):
        t = 10.6+i/90
        b.clock.now = t
        sync(b.frame_reference.clock, t)
        if i == 0 or int((t-10.6)*20) != int((t-10.6-1/90)*20):
            b._integrate_px4_yaw((t,t,0.))
            b._update_px4_odometry((t,t,100.,200.,0.,0.,.01,.01,0))
            b._record_px4_frame_attitude(t,t,q())
        distance = max(0., t-.02-10.6)
        b._odom_callback(message(1000+t-.02, (distance,0.,0.), q(), (1.,0.,0.)))
        b._tick()
        assert not b.continuity_recovering
    published_distance = b.continuity.last_output_timestamp-1000-10.6
    assert b.continuity.last_output_position == pytest.approx((published_distance,0.,0.))
