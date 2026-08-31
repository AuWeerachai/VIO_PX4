import math

from vio_px4_bridge.local_continuity import LocalPoseContinuity


def update(tracker, x, y=0.0, yaw=0.0, t=0.0, vx=0.0, frame="odom|base"):
    return tracker.update((x, y, 0.0), yaw, (vx, 0.0, 0.0), 0.0, t, frame)


def test_normal_motion_passes_through():
    tracker = LocalPoseContinuity()
    first = update(tracker, 0.0, t=1.0, vx=1.0)
    second = update(tracker, 1.0, t=2.0, vx=1.0)
    assert first.position == (0.0, 0.0, 0.0)
    assert second.position == (1.0, 0.0, 0.0)
    assert not second.reanchored


def test_confirmed_position_jump_preserves_continuity():
    tracker = LocalPoseContinuity(confirmation_samples=3, recovery_samples=2)
    update(tracker, 10.0, t=1.0, vx=1.0)
    before = update(tracker, 11.0, t=2.0, vx=1.0)
    suspect = update(tracker, 100.0, t=3.0, vx=1.0)
    assert suspect.recovering and not suspect.reanchored
    assert suspect.position == before.position
    assert update(tracker, 101.0, t=4.0, vx=1.0).recovering
    jumped = update(tracker, 102.0, t=5.0, vx=1.0)
    assert jumped.reanchored
    assert jumped.reason == "position_jump"
    assert jumped.position == (13.0, 0.0, 0.0)
    assert jumped.recovering
    # New-frame movement accumulated during confirmation is retained.
    next_pose = update(tracker, 103.0, t=6.0, vx=1.0)
    assert next_pose.position == (14.0, 0.0, 0.0)


def test_isolated_outlier_is_discarded_without_reanchor():
    tracker = LocalPoseContinuity()
    update(tracker, 0.0, t=1.0, vx=1.0)
    before = update(tracker, 1.0, t=2.0, vx=1.0)
    bad = update(tracker, 50.0, t=3.0, vx=1.0)
    assert bad.recovering and bad.position == before.position
    recovered = update(tracker, 3.0, t=4.0, vx=1.0)
    assert not recovered.recovering
    assert not recovered.reanchored
    assert recovered.epoch == 0
    assert recovered.position == (3.0, 0.0, 0.0)


def test_yaw_jump_preserves_position_and_heading():
    tracker = LocalPoseContinuity(
        max_yaw_rate_rad_s=math.radians(10),
        confirmation_samples=2,
    )
    before = update(tracker, 2.0, y=3.0, yaw=0.1, t=1.0)
    assert update(tracker, 2.0, y=3.0, yaw=1.5, t=2.0).recovering
    jumped = update(tracker, 2.0, y=3.0, yaw=1.5, t=3.0)
    assert jumped.reason == "yaw_jump"
    assert math.isclose(jumped.yaw, before.yaw)
    assert all(math.isclose(a, b) for a, b in zip(jumped.position, before.position))


def test_frame_change_reanchors():
    tracker = LocalPoseContinuity(confirmation_samples=2)
    before = update(tracker, 4.0, t=1.0)
    assert update(tracker, 0.0, t=2.0, frame="new_odom|base").recovering
    changed = update(tracker, 0.0, t=3.0, frame="new_odom|base")
    assert changed.reason == "frame_changed"
    assert changed.position == before.position


def test_recovery_gate_clears_after_stable_samples():
    tracker = LocalPoseContinuity(max_speed_mps=5.0,
                                  confirmation_samples=2, recovery_samples=2)
    update(tracker, 0.0, t=1.0)
    assert update(tracker, 10.0, t=1.1).recovering
    assert update(tracker, 10.0, t=1.2).recovering
    assert update(tracker, 10.0, t=1.3).recovering
    assert not update(tracker, 10.0, t=1.4).recovering


def test_physical_speed_gate_quarantines_sample():
    tracker = LocalPoseContinuity(max_speed_mps=5.0)
    update(tracker, 0.0, t=1.0)
    result = update(tracker, 0.0, t=2.0, vx=10.0)
    assert result.recovering
    assert result.reason == "speed_limit"
    assert result.event == "quarantine_started"
    assert "speed_m_s=10.000" in result.detail
    assert "limit_m_s=5.000" in result.detail


def test_pose_derived_speed_detects_jump_with_low_reported_velocity():
    tracker = LocalPoseContinuity(
        max_speed_mps=5.0,
        max_acceleration_mps2=1000.0,
    )
    update(tracker, 0.0, t=1.0, vx=0.0)
    # The pose implies 10 m/s even though cuVSLAM reports zero velocity.
    result = update(tracker, 10.0, t=2.0, vx=0.0)
    assert result.recovering
    assert result.reason == "position_jump"
    assert result.position == (0.0, 0.0, 0.0)
    assert result.event == "quarantine_started"
    assert "pose_implied_speed_m_s=10.000" in result.detail


def test_pose_rate_gate_is_independent_of_message_frequency():
    for hz in (30.0, 90.0):
        dt = 1.0 / hz
        tracker = LocalPoseContinuity(
            max_speed_mps=10.0, max_acceleration_mps2=1000.0
        )
        update(tracker, 0.0, t=1.0, vx=9.0)
        accepted = update(tracker, 9.0 * dt, t=1.0 + dt, vx=9.0)
        assert not accepted.recovering

        tracker = LocalPoseContinuity(
            max_speed_mps=10.0, max_acceleration_mps2=1000.0
        )
        update(tracker, 0.0, t=1.0, vx=0.0)
        rejected = update(tracker, 11.0 * dt, t=1.0 + dt, vx=0.0)
        assert rejected.recovering
        assert rejected.reason == "position_jump"
