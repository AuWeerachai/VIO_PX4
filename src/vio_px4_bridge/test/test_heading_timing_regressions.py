"""Regression cases from the stationary Jetson test on 19 November 2026."""
import math

import pytest

from vio_px4_bridge import frame_alignment as f
from vio_px4_bridge.px4_heading_reference import Px4HeadingReference
from test_full3d_recovery import message
from test_px4_heading_reference import synced
from test_px4_recovery import bridge
from test_stabilized_bridge import stabilized, prime, telemetry


def test_known_bracket_outside_tolerance_is_not_a_future_packet_wait():
    r = Px4HeadingReference()
    synced(r)
    r.observe(10., 5., f.IDENTITY, 0)
    r.observe(10.07, 5.07, f.IDENTITY, 0)
    assert r.match(1010.035, 1010.075, 10.075) is None
    assert r.reason == 'heading_pair_outside_tolerance'
    # More recent telemetry cannot repair this already-known old bracket.
    r.observe(10.12, 5.12, f.IDENTITY, 0)
    assert r.match(1010.035, 1010.12, 10.12) is None
    assert r.reason == 'heading_pair_outside_tolerance'


def test_frame_before_retained_history_is_not_a_future_packet_wait():
    r = Px4HeadingReference()
    synced(r)
    r.observe(10., 5., f.IDENTITY, 0)
    r.observe(10.05, 5.05, f.IDENTITY, 0)
    assert r.match(1009.98, 1010.06, 10.06) is None
    assert r.reason == 'camera_before_heading_history'


@pytest.mark.parametrize('jump', ['none', 'position', 'heading'])
def test_unpairable_head_cannot_block_or_hide_raw_jump(stabilized, jump):
    b = stabilized
    prime(b)
    b.clock.now = 10.055
    b._odom_callback(message(1010.035, (100 if jump == 'position' else .003, 0, 0),
                             f.from_rpy(0, 0, math.pi) if jump == 'heading' else f.IDENTITY))
    assert len(b.pending_odometry) == 1  # upper bracket not here yet
    telemetry(b, 10.07)
    b._drain_pending_odometry()
    assert not b.pending_odometry
    assert b.continuity_recovering == (jump != 'none')
    assert b.pose_jump_count == int(jump != 'none')
    assert b.last_vio_time == pytest.approx(9.98)  # unmatched sample is NOT output
    b.clock.now = 10.08
    b._odom_callback(message(1010.06, (.003, 0, 0), f.IDENTITY))
    b._tick()
    if jump != 'none':
        b._publish_fix.assert_not_called()
        assert b.continuity_recovering
    else:
        assert b._publish_fix.called
        assert b.last_vio_time == pytest.approx(10.06)
        assert b.latest_fix['lat'] == b.home_lat


def test_unmatched_startup_sample_cannot_replace_last_matched_output(stabilized):
    b = stabilized
    prime(b)
    telemetry(b, 10.07)
    b.phase = 'spoof'
    b.spoof_until_vio = False
    previous_fix = dict(b.latest_fix)
    b.clock.now = 10.075
    b._odom_callback(message(1010.035, (.01, 0, 0), f.IDENTITY))
    assert not b.pending_odometry
    assert b.latest_fix == previous_fix
    assert b.last_vio_time == pytest.approx(9.98)
    assert b.last_heading_output_time == pytest.approx(9.98)


@pytest.mark.parametrize('moving', [False, True])
def test_uneven_px4_capture_spacing_keeps_fresh_camera_flowing(stabilized, moving):
    b = stabilized
    prime(b)
    speed = .1 if moving else 0.
    events = []
    for i in range(1, 2701):
        capture = 10+i/90
        latency = .03+.01*math.sin(i*.19)
        events.append((capture+latency, 'camera', capture))
    capture = 10.
    # Average 20 Hz, but real capture spacing is not exactly 50 ms.
    gaps = (.07, .03, .08, .02, .05)
    for i in range(610):
        capture += gaps[i % len(gaps)]
        events.append((capture+.008, 'px4', capture))
    for i in range(1, 303):
        events.append((10+i/10, 'gps', None))
    last_sync = 10.
    last_camera = None
    for now, kind, capture in sorted(events):
        b.clock.now = now
        if now-last_sync > .5:
            synced(b.heading_reference, now)
            last_sync = now
        if kind == 'px4':
            telemetry(b, now, capture=capture, sync=False)
            b._drain_pending_odometry()
        elif kind == 'camera':
            last_camera = capture
            b._odom_callback(message(1000+capture, (speed*(capture-10), 0, 0), f.IDENTITY,
                                     world_velocity=(speed, 0, 0), body_twist=False))
        else:
            b._tick()
        assert not b.continuity_recovering
        assert b.pose_jump_count == 0
        assert len(b.pending_odometry) < 12
        assert b.frozen_position_uncertainty_m == 0
    assert b.last_vio_time > last_camera-.03
    assert b.continuity.last_output_position[0] == pytest.approx(
        max(0, speed*(b.last_vio_time-10)-.01), abs=1e-8)


def test_recovery_replays_cached_motion_since_last_output_without_fake_gap(stabilized):
    b = stabilized
    prime(b)
    for i in range(1, 7):
        t = 10+i*.05
        telemetry(b, t, north=.1*(t-10), vn=.1)
    # Output is old, but all intervening PX4 packets are present and fresh.
    b._heading_quarantine('time_matched_camera_output_stale')
    assert b.continuity_recovering
    assert b.inertial_position_safe
    assert b.inertial_last_odometry[1] == pytest.approx(5.30)
    assert b.inertial_displacement_ned == pytest.approx((.03, 0, 0))
    telemetry(b, 10.35, north=.035, vn=.1)
    assert b.inertial_position_safe
    assert b.inertial_displacement_ned == pytest.approx((.035, 0, 0))
    b._tick()
    b._publish_fix.assert_not_called()  # replay alone cannot authorize recovery


@pytest.mark.parametrize('violation,expected', [
    ({'variance': 16.}, 'position_uncertainty'),
    ({'vn': 100.}, 'velocity_limit'),
    ({'vn': 1.}, 'acceleration_limit'),
    ({'north': 100.}, 'position_step_limit'),
])
def test_cached_motion_is_checked_not_just_skipped_to_latest(stabilized, violation, expected):
    b = stabilized
    prime(b)
    b.continuity.max_acceleration_mps2 = 5.
    for i in range(1, 6):
        telemetry(b, 10+i*.05, **(violation if i == 2 else {}))
    b._heading_quarantine('time_matched_camera_output_stale')
    assert not b.inertial_position_safe
    assert b.inertial_failure_reason.startswith(expected)
    b._tick()
    b._publish_fix.assert_not_called()


def test_real_px4_gap_still_rejects_position_recovery(stabilized):
    b = stabilized
    prime(b)
    telemetry(b, 10.3)  # missing telemetry, not merely a late camera match
    b._tick()
    assert b.continuity_recovering
    assert not b.inertial_position_safe
    b._publish_fix.assert_not_called()
