"""Shared production callbacks: GPS-only must be as guarded as GPS plus yaw."""
import math
from types import SimpleNamespace

import pytest

from test_external_yaw import (bridge, legacy_bridge, feed, outputs, clear_sends,
                               silence, synchronize, vio_message)
from vio_px4_bridge.external_yaw import NavigationTiming
from vio_px4_bridge.local_continuity import LocalPoseContinuity


@pytest.fixture(params=[False, True], ids=['gps_only', 'gps_and_yaw'])
def navigation(request, bridge):
    if not request.param:
        bridge.external_yaw = None
        bridge.navigation_timing = NavigationTiming()
    return bridge


@pytest.mark.parametrize('fault', ['position', 'yaw', 'nan', 'angular_nan', 'frame',
                                  'world_frame', 'zero_q', 'nonunit_q', 'old',
                                  'future', 'duplicate', 'regression'])
def test_all_bad_camera_inputs_latch_shared_quarantine(navigation, fault):
    b = navigation
    feed(b, 10.)
    b._odom_callback(vio_message(10., 0.))
    outputs(b)
    assert b._publish_fix.call_count == 1
    clear_sends(b)
    feed(b, 10.05)
    msg = vio_message(10.05, 100. if fault == 'position' else 0.,
                      yaw=2. if fault == 'yaw' else 0.)
    if fault == 'nan': msg.pose.pose.position.x = math.nan
    if fault == 'angular_nan': msg.twist.twist.angular.x = math.nan
    if fault == 'frame': msg.child_frame_id = 'camera_link'
    if fault == 'world_frame': msg.header.frame_id = ''
    if fault == 'zero_q':
        msg.pose.pose.orientation = SimpleNamespace(w=0., x=0., y=0., z=0.)
    if fault == 'nonunit_q': msg.pose.pose.orientation.w *= 2
    if fault in ('old', 'future', 'duplicate', 'regression'):
        stamp = {'old': 1., 'future': 11., 'duplicate': 10., 'regression': 9.99}[fault]
        msg.header.stamp = vio_message(stamp, 0.).header.stamp
    b._odom_callback(msg)
    outputs(b)
    assert b.continuity_recovering
    assert b.navigation_timing.trusted_capture_s == 10.
    silence(b)


@pytest.mark.parametrize('duration', [2., 30.])
def test_blackout_keeps_movement_since_last_trusted_capture(navigation, duration):
    b = navigation
    feed(b, 10., vn=1.)
    b._odom_callback(vio_message(10., 0., vn=1.))
    outputs(b)
    for i in range(1, round(duration/.05)+1):
        t = 10+i*.05
        feed(b, t, north=t-10, vn=1.)
        outputs(b)
    assert b.continuity_recovering and b.inertial_active
    assert b.navigation_timing.boundary.north == pytest.approx(0.)
    assert b.inertial_start_time == 10.
    clear_sends(b)
    for i in range(1, 40):
        t = 10+duration+i*.05
        feed(b, t, north=t-10, vn=1.)
        b._odom_callback(vio_message(t, 100+i*.05, vn=1.))
        outputs(b)
        if not b.continuity_recovering:
            break
        silence(b)
    assert not b.continuity_recovering
    assert b.inertial_position_safe
    # Previously GPS-only recovered near 0.3 instead of ~2.3 after a 2s loss.
    assert b.continuity.last_output_position[0] == pytest.approx(t-10)
    assert b._publish_fix.call_count == 1
    assert b._mav.mav.odometry_send.call_count == int(b.external_yaw is not None)
    assert b._select_accuracy()[0] >= .5


def test_callback_detects_blackout_even_when_watchdog_has_not_run(navigation):
    b = navigation
    feed(b, 10.)
    b._odom_callback(vio_message(10., 0.))
    feed(b, 10.3)
    b._odom_callback(vio_message(10.3, 0.))  # below the legacy 1s epoch-gap limit
    assert b.continuity_recovering
    assert b.navigation_timing.trusted_capture_s == 10.


def test_reused_fcu_evidence_cannot_hide_new_camera_turn(navigation):
    b = navigation
    b.continuity = LocalPoseContinuity(confirmation_samples=3, recovery_samples=65)
    feed(b, 10.)
    b._odom_callback(vio_message(10., 0.))
    feed(b, 10.01)
    b._odom_callback(vio_message(10.01, 100., yaw=2.))
    for i in range(2, 51):
        t = 10+i*.01
        if i % 5 == 0:
            feed(b, t)
        b.clock.now = t
        b._odom_callback(vio_message(t, 100., yaw=2.))
    assert b.recovery_yaw_agreement.good_samples >= 2
    clear_sends(b)
    # No new ODOMETRY evidence after 10.5; still within the .25s age limit.
    # Camera turns at 300deg/s, below its configured 360deg/s rate limit.
    for i in range(51, 70):
        t = 10+i*.01
        b.clock.now = t
        synchronize(b.navigation_timing, t)
        b._odom_callback(vio_message(t, 100., yaw=2+math.radians(3*(i-50))))
        outputs(b)
        assert b.continuity_recovering
        silence(b)
    assert b.recovery_yaw_agreement.good_samples == 0


def test_clock_reset_stops_both_even_with_heading_disabled(navigation):
    b = navigation
    feed(b, 10.)
    b._odom_callback(vio_message(10., 0.))
    outputs(b)
    clear_sends(b)
    b.navigation_timing.clock.reset()
    outputs(b)
    silence(b)
    assert b.continuity_recovering
    assert b.navigation_timing.generation == b.navigation_timing.clock.generation


@pytest.mark.parametrize('initial_reset,new_reset', [(0, 1), (255, 0)])
def test_delayed_camera_cannot_recover_using_heading_before_fcu_reset(
        navigation, initial_reset, new_reset):
    """190ms-old camera data must wait for a match in the NEW FCU epoch.

    Two-sample agreement windows are permitted settings. Previously those
    windows passed using old buffered FCU poses after its yaw reset by 90deg.
    """
    b = navigation
    b.continuity = LocalPoseContinuity(confirmation_samples=3, recovery_samples=10)
    feed(b, 10., reset=initial_reset)
    b._odom_callback(vio_message(10., 0.))
    b._invalidate_navigation('review_blackout')
    for i in range(1, 20):
        feed(b, 10+i*.05, reset=initial_reset)
    clear_sends(b)
    for i in range(98, 161):
        now = round(10+i*.01, 2)
        if i % 5 == 0:
            feed(b, now, reset=new_reset if now >= 11. else initial_reset,
                 yaw=math.pi/2 if now >= 11. else 0.)
        else:
            b.clock.now = now
            synchronize(b.navigation_timing, now)
            b.last_px4_heartbeat_time = now
        capture = round(now-.19, 2)
        b._odom_callback(vio_message(capture, 100.))
        outputs(b)
        if capture < 11.:
            assert b.continuity_recovering, 'recovered using pre-reset heading'
            silence(b)
        if not b.continuity_recovering:
            break
    assert not b.continuity_recovering, 'fresh post-reset data must allow recovery'
    assert b.navigation_timing.recovery_pose.reset_counter == new_reset
    assert b.navigation_timing.recovery_pose.boot_s >= b.recovery_reset_stamp
    assert b.continuity.last_output_yaw == pytest.approx(math.pi/2)
    assert b._publish_fix.call_count == 1
    assert b._mav.mav.odometry_send.call_count == int(b.external_yaw is not None)


def test_reset_timestamp_rejects_old_match_even_if_counter_matches(navigation):
    b = navigation
    feed(b, 10.)
    b._odom_callback(vio_message(10., 0.))
    b._invalidate_navigation('review_blackout')
    for i in range(1, 20):
        feed(b, 10+i*.05)
    feed(b, 11., reset=1, yaw=.4)
    feed(b, 11.05, reset=0, yaw=.8)
    b.external_sample_time = 10.90
    # This buffered sample and the latest one happen to have the same counter,
    # but there have been two resets between them. Require post-reset TIME too.
    assert b._matched_recovery_target((0., 0., 0.), 0.) is None
    assert b.recovery_yaw_agreement_state == 'waiting_for_post_reset_fcu'
    assert b.navigation_timing.recovery_pose is None
    assert b.recovery_velocity_agreement_count == 0
    assert b.recovery_yaw_agreement.good_samples == 0


def test_first_rc_high_still_triggers_gps_only_startup(bridge):
    b = bridge
    b.external_yaw = None
    b.navigation_timing = NavigationTiming()  # no warmup requirement for GPS-only spoof
    b.phase = 'waiting'
    b.rc_trigger_enabled = True
    b.rc_channel = 6
    b.rc_low_pwm_max, b.rc_high_pwm_min = 1200, 1800
    b.last_rc_position = None
    b.spoof_failure_reason = None
    b.spoof_duration_s = 15.
    b.startup_heading = SimpleNamespace(reset=lambda: None)
    msg = SimpleNamespace(get_type=lambda: 'RC_CHANNELS', get_srcSystem=lambda: 1,
                          get_srcComponent=lambda: 1, chan6_raw=1900)
    b._mav.recv_match.side_effect = [msg, None]
    b._poll_mavlink()
    assert b.phase == 'spoof'
    assert b.start_time == 10.
