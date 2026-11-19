"""Rigid-coordinate reset tests: real callback, gates, and GPS output timer.

No ROS process, camera, serial port, or flight controller is started. The test
trajectory has known ground truth; arbitrary full-RPY resets affect only the
synthetic camera's coordinate frame, not the simulated aircraft.
"""
import math
import random
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vio_px4_bridge import frame_alignment as f
from vio_px4_bridge.geo_utils import ned_offset_to_lla
from vio_px4_bridge.local_continuity import LocalPoseContinuity
from test_px4_recovery import bridge, telemetry


def matrix(r, p, y):
    """Independent Rz Ry Rx reference (not using the production quaternions)."""
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return ((cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr),
            (sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr), (-sp, cp*sr, cp*cr))


def mv(m, v):
    return tuple(sum(a*b for a, b in zip(row, v)) for row in m)


def add(a, b):
    return tuple(x+y for x, y in zip(a, b))


def test_quaternions_match_independent_rotation_matrices():
    rng = random.Random(20261118)
    for _ in range(250):
        angles = tuple(rng.uniform(-math.pi, math.pi) for _ in range(3))
        v = tuple(rng.uniform(-10, 10) for _ in range(3))
        q = f.from_rpy(*angles)
        assert f.rotate(q, v) == pytest.approx(mv(matrix(*angles), v), abs=1e-12)
        assert f.rotate(f.inverse(q), f.rotate(q, v)) == pytest.approx(v, abs=1e-12)
        assert f.angle_between(q, tuple(-x for x in q)) == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize('angles', [(0, 0, 0), (.1, -.2, .3), (2.3, 1.57, -3.14)])
def test_enu_flu_to_ned_frd_transforms_both_world_and_body(angles):
    q = f.ros_to_ned_frd(f.from_rpy(*angles))
    for body in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        body_flu = (body[0], -body[1], -body[2])
        enu = mv(matrix(*angles), body_flu)
        expected = (enu[1], enu[0], -enu[2])
        assert f.rotate(q, body) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize('bad', [(0, 0, 0, 0), (float('nan'), 0, 0, 1), (1, 2, 3)])
def test_invalid_orientation_is_rejected(bad):
    with pytest.raises(ValueError):
        f.normalize(bad)


def message(t, raw_position, raw_orientation, world_velocity=(0, 0, 0), body_twist=True):
    # Inverse convention conversion: both fixed rotations are self-inverse.
    q = f.ros_to_ned_frd(raw_orientation)
    n, e, d = raw_position
    vector = lambda x, y, z: SimpleNamespace(x=x, y=y, z=z)
    if body_twist:
        body = f.rotate(f.inverse(raw_orientation), world_velocity)
        linear = (body[0], -body[1], -body[2])
    else:
        linear = (world_velocity[1], world_velocity[0], -world_velocity[2])
    return SimpleNamespace(
        header=SimpleNamespace(frame_id='odom', stamp=SimpleNamespace(
            sec=int(t), nanosec=round((t-int(t))*1e9))), child_frame_id='drone_link',
        pose=SimpleNamespace(pose=SimpleNamespace(position=vector(e, n, -d),
            orientation=SimpleNamespace(w=q[0], x=q[1], y=q[2], z=q[3]))),
        twist=SimpleNamespace(twist=SimpleNamespace(linear=vector(*linear), angular=vector(0, 0, 0))))


def setup_level_run(b, body_rpy=(0., 0., 1.1), offset=.7, body_twist=True):
    b.continuity = LocalPoseContinuity(confirmation_samples=3, recovery_samples=4,
                                     max_speed_mps=15, max_acceleration_mps2=60)
    b.inertial_active = b.continuity_recovering = False
    b.heading_offset_rad = offset
    b.twist_is_body_frame = body_twist
    b.body_rpy = body_rpy
    b.physical_q_local = f.multiply(f.from_rpy(0, 0, -offset), f.from_rpy(*body_rpy))
    b.local_anchor = (1.2, -3.4, .7)
    telemetry(b, 10., 0, roll=body_rpy[0], pitch=body_rpy[1], yaw=body_rpy[2])
    b._odom_callback(message(10., b.local_anchor, b.physical_q_local, body_twist=body_twist))
    return dict(b.latest_fix)


def stationary_reset(b, jump, raw_offset, start_time, local_position=None, reset_counter=0):
    """Aircraft stays still; only its VO coordinate system changes."""
    p = b.local_anchor if local_position is None else local_position
    raw_p = add(raw_offset, f.rotate(jump, p))
    raw_q = f.multiply(jump, b.physical_q_local)
    b._publish_fix.reset_mock()
    before = dict(b.latest_fix)
    for i in range(24):
        t = start_time+i*.05
        telemetry(b, t, reset_counter, roll=b.body_rpy[0], pitch=b.body_rpy[1], yaw=b.body_rpy[2])
        b._odom_callback(message(t, raw_p, raw_q, body_twist=b.twist_is_body_frame))
        b._tick()
        if i == 0:
            assert b.continuity_recovering
        if b.continuity_recovering:
            b._publish_fix.assert_not_called()
            assert b.latest_fix == before
        else:
            assert b._publish_fix.call_count == 1
            return t, raw_p, raw_q
    pytest.fail('stable full-3D epoch did not recover')


@pytest.mark.parametrize('jump_rpy', [
    (133.35, 13.09, -51.42), (0, 90, 0), (180, 0, 0),
    (-60, 80, 179.9), (12, -17, -179.9), (0, 0, 120),
])
@pytest.mark.parametrize('physical_rpy', [(0, 0, 1.1), (.06, -.09, math.radians(179))])
@pytest.mark.parametrize('body_twist', [True, False])
def test_stationary_reset_then_rectangle_returns_home(bridge, jump_rpy, physical_rpy, body_twist):
    b = bridge
    initial = setup_level_run(b, physical_rpy, body_twist=body_twist)
    jump = f.from_rpy(*(math.radians(x) for x in jump_rpy))
    shift = (900, -150, -800)
    t, raw_p, raw_q = stationary_reset(b, jump, shift, 10.05)
    assert (b.latest_fix['lat'], b.latest_fix['lon']) == (initial['lat'], initial['lon'])
    assert b.continuity.last_output_position == pytest.approx(b.local_anchor, abs=1e-10)
    assert f.angle_between(b.continuity.last_output_orientation, b.physical_q_local) < 1e-10
    fixed_rotation = b.continuity.rotation_q
    # Test a known 1m x 1m square in earth N/E, not raw VO axes. Large raw Z
    # changes on a horizontal leg must become horizontal after the correction.
    north = east = 0.
    for vn, ve in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
        for _ in range(20):
            t += .05
            north += vn*.05
            east += ve*.05
            local_delta = f.rotate(f.from_rpy(0, 0, -b.heading_offset_rad), (north, east, 0))
            local_v = f.rotate(f.from_rpy(0, 0, -b.heading_offset_rad), (vn, ve, 0))
            p = add(shift, f.rotate(jump, add(b.local_anchor, local_delta)))
            v = f.rotate(jump, local_v)
            # Later PX4 attitude must NOT continuously rotate accepted motion.
            telemetry(b, t, 0, yaw=physical_rpy[2]+.2)
            b._odom_callback(message(t, p, raw_q, v, body_twist))
            assert not b.continuity_recovering
            assert b.continuity.rotation_q == fixed_rotation
            expected = ned_offset_to_lla(b.home_lat, b.home_lon, b.home_alt, north, east, 0)
            assert b.latest_fix['lat'] == pytest.approx(expected[0], abs=1e-11, rel=0)
            assert b.latest_fix['lon'] == pytest.approx(expected[1], abs=1e-11, rel=0)
            assert b.continuity.last_output_position[2] == pytest.approx(b.local_anchor[2], abs=1e-10)
    assert b.latest_fix['lat'] == pytest.approx(initial['lat'], abs=1e-11, rel=0)
    assert b.latest_fix['lon'] == pytest.approx(initial['lon'], abs=1e-11, rel=0)


def test_roll_only_jump_at_same_position_is_quarantined_and_corrected(bridge):
    b = bridge
    initial = setup_level_run(b, offset=0)
    jump = f.from_rpy(math.pi, 0, 0)
    raw_q = f.multiply(jump, b.physical_q_local)
    for i in range(16):
        t = 10.05+i*.05
        telemetry(b, t, 0, yaw=b.body_rpy[2])
        b._odom_callback(message(t, b.local_anchor, raw_q))
        if i == 0:
            assert b.continuity_recovering
            assert b.last_pose_jump_reason == 'attitude_jump'
    assert not b.continuity_recovering
    assert b.latest_fix['lat'] == initial['lat']
    assert f.angle_between(b.continuity.last_output_orientation, b.physical_q_local) < 1e-10


def test_quaternion_sign_changes_do_not_create_false_quarantine(bridge):
    b = bridge
    setup_level_run(b)
    for i in range(1, 40):
        q = tuple((-1 if i % 2 else 1)*x for x in b.physical_q_local)
        b._odom_callback(message(10+i*.01, b.local_anchor, q))
        assert not b.continuity_recovering
        assert b.pose_jump_count == 0


@pytest.mark.parametrize('bad', ['missing', 'stale', 'wrong_timestamp'])
def test_full_attitude_required_not_yaw_only_and_recovers_when_restored(bridge, bad):
    b = bridge
    setup_level_run(b)
    jump = f.from_rpy(2.3, .4, -1.2)
    p, q = (100, 200, 300), f.multiply(jump, b.physical_q_local)
    for i in range(1, 40):
        t = 10+i*.05
        telemetry(b, t, 0, yaw=b.body_rpy[2])
        if bad == 'missing':
            b.latest_px4_attitude = None
        elif bad == 'stale':
            b.latest_px4_attitude = (t-1, t, f.from_rpy(*b.body_rpy))
        else:
            b.latest_px4_attitude = (t, t-.1, f.from_rpy(*b.body_rpy))
        b._odom_callback(message(t, p, q))
        b._tick()
        assert b.continuity_recovering
        b._publish_fix.assert_not_called()
    for i in range(40, 64):
        t = 10+i*.05
        telemetry(b, t, 0, yaw=b.body_rpy[2])
        b._odom_callback(message(t, p, q))
        if not b.continuity_recovering:
            break
    assert not b.continuity_recovering


def test_recovery_agreement_rotation_not_refitted_on_every_sample(bridge):
    b = bridge
    setup_level_run(b, offset=0)
    b.recovery_yaw_agreement.max_residual_rad = math.radians(3)
    jump = f.from_rpy(2.3, .4, -1.2)
    q = f.multiply(jump, b.physical_q_local)
    telemetry(b, 10.05, 0, yaw=b.body_rpy[2])
    b._odom_callback(message(10.05, (100, 200, 300), q))
    fixed = None
    for i in range(2, 35):
        t = 10+i*.05
        telemetry(b, t, 0, yaw=b.body_rpy[2])
        # False camera turning: 80deg/s is under the motion gate, but disagrees
        # with a stationary FCU. Re-fitting alignment each sample would hide it.
        false_q = f.multiply(jump, f.from_rpy(0, 0, b.body_rpy[2]+math.radians(4*i)))
        b._odom_callback(message(t, (100, 200, 300), false_q))
        b._tick()
        assert b.continuity_recovering
        b._publish_fix.assert_not_called()
        if b.continuity.suspect_samples == [] and b.recovery_frame_rotation is not None:
            if fixed is None:
                fixed = b.recovery_frame_rotation
            assert b.recovery_frame_rotation == fixed
    assert b.recovery_yaw_agreement.good_samples == 0


def test_invalid_or_reordered_attitude_cannot_replace_full_reference(bridge):
    b = bridge
    setup_level_run(b)
    jump = f.from_rpy(2, 1, -2)
    telemetry(b, 10.05, 0, yaw=1.1)
    b._odom_callback(message(10.05, (100, 200, 300), f.multiply(jump, b.physical_q_local)))
    telemetry(b, 10.1, 0, roll=.1, pitch=.2, yaw=1.1)
    newest = b.latest_px4_attitude
    telemetry(b, 10.09, 0, roll=1, pitch=1, yaw=1.5)
    assert b.latest_px4_attitude == newest
    invalid = SimpleNamespace(get_type=lambda: 'ATTITUDE', get_srcSystem=lambda: 1,
        get_srcComponent=lambda: 1, time_boot_ms=10150, roll=float('nan'), pitch=0., yaw=1.1)
    b._mav = SimpleNamespace(recv_match=Mock(side_effect=[invalid, None]))
    b.poll_real()
    assert b.latest_px4_attitude is None


def test_small_real_tilts_after_handoff_are_preserved_not_flattened(bridge):
    b = bridge
    setup_level_run(b)
    jump = f.from_rpy(2.3, -.7, 1.5)
    shift = (100, 200, 300)
    t, raw_p, _ = stationary_reset(b, jump, shift, 10.05)
    correction = b.continuity.rotation_q
    desired = f.from_rpy(.05, -.08, b.body_rpy[2]-.1)
    desired_local = f.multiply(f.from_rpy(0, 0, -b.heading_offset_rad), desired)
    b._odom_callback(message(t+.05, raw_p, f.multiply(jump, desired_local)))
    assert not b.continuity_recovering
    assert b.continuity.rotation_q == correction
    assert f.angle_between(b.continuity.last_output_orientation, desired_local) < 1e-10


def test_raw_euler_singularity_does_not_trigger_false_quarantine(bridge):
    b = bridge
    setup_level_run(b, body_rpy=(0, -.01, 0), offset=0)
    jump = f.from_rpy(0, math.pi/2, 0)
    t, p, old_q = stationary_reset(b, jump, (100, 200, 300), 10.05)
    new_true_q = f.from_rpy(0, .01, 0)
    raw_q = f.multiply(jump, new_true_q)
    assert abs(f.yaw(raw_q) - f.yaw(old_q)) > 3  # Euler yaw flipped ~180deg.
    assert f.angle_between(raw_q, old_q) == pytest.approx(.02)
    b._odom_callback(message(t+.05, p, raw_q))
    assert not b.continuity_recovering
    assert f.angle_between(b.continuity.last_output_orientation, new_true_q) < 1e-10


@pytest.mark.parametrize('body_twist', [True, False])
def test_moving_recovery_uses_full_rotation_for_velocity_and_px4_displacement(bridge, body_twist):
    b = bridge
    setup_level_run(b, body_rpy=(.07, -.09, 1.1), body_twist=body_twist)
    jump = f.from_rpy(math.radians(133), math.radians(90), -2.)
    shift = (200, -400, 300)
    raw_q = f.multiply(jump, b.physical_q_local)
    inv_heading = f.from_rpy(0, 0, -b.heading_offset_rad)
    telemetry(b, 10.05, 0, roll=b.body_rpy[0], pitch=b.body_rpy[1], yaw=b.body_rpy[2])
    b._odom_callback(message(10.05, add(shift, f.rotate(jump, b.local_anchor)), raw_q,
                            body_twist=body_twist))
    assert b.continuity_recovering
    velocity_earth = (.5, .25, 0.)
    raw_velocity = f.rotate(jump, f.rotate(inv_heading, velocity_earth))
    for i in range(1, 30):
        t = 10.05+i*.05
        delta = tuple(v*(t-10.05) for v in velocity_earth)
        telemetry(b, t, 0, roll=b.body_rpy[0], pitch=b.body_rpy[1], yaw=b.body_rpy[2],
                  north=delta[0], east=delta[1], vn=.5, ve=.25)
        local = add(b.local_anchor, f.rotate(inv_heading, delta))
        raw_p = add(shift, f.rotate(jump, local))
        b._odom_callback(message(t, raw_p, raw_q, raw_velocity, body_twist))
        b._tick()
        if b.continuity_recovering:
            b._publish_fix.assert_not_called()
        else:
            break
    assert not b.continuity_recovering
    assert b.inertial_position_safe
    assert b.frozen_position_uncertainty_m == 0
    expected = ned_offset_to_lla(b.home_lat, b.home_lon, b.home_alt, *delta)
    assert b.latest_fix['lat'] == pytest.approx(expected[0], abs=1e-11, rel=0)
    assert b.latest_fix['lon'] == pytest.approx(expected[1], abs=1e-11, rel=0)


def test_frozen_position_still_recovers_full_frame_after_px4_reset(bridge):
    b = bridge
    original = setup_level_run(b, body_rpy=(.04, .03, -2.))
    raw_q = f.multiply(f.from_rpy(2.3, -.9, .5), b.physical_q_local)
    for i in range(1, 25):
        t = 10+i*.05
        telemetry(b, t, 0 if i == 1 else 1,
                  roll=.04, pitch=.03, yaw=-2.)
        b._odom_callback(message(t, (100, -200, 300), raw_q))
        b._tick()
        if not b.continuity_recovering:
            break
        b._publish_fix.assert_not_called()
    assert not b.continuity_recovering
    assert not b.inertial_position_safe
    assert b.latest_fix['lat'] == original['lat']
    assert b.latest_fix['lon'] == original['lon']
    assert f.angle_between(b.continuity.last_output_orientation, b.physical_q_local) < 1e-10
    assert b.frozen_position_uncertainty_m > 0  # Do not hide unknown global uncertainty.


def test_two_different_rpy_resets_do_not_rotate_previous_positions(bridge):
    b = bridge
    original = setup_level_run(b)
    local = b.local_anchor
    t = 10.
    for jump, shift, direction in [(f.from_rpy(2.3, .4, -.5), (100, 200, 300), 1),
                                    (f.from_rpy(-1.5, -1.2, 2.1), (-400, 900, -800), -1)]:
        before = dict(b.latest_fix)
        t, _, raw_q = stationary_reset(b, jump, shift, t+.05, local)
        assert b.latest_fix['lat'] == before['lat']
        assert b.latest_fix['lon'] == before['lon']
        delta = f.rotate(f.from_rpy(0, 0, -b.heading_offset_rad), (.05*direction, 0, 0))
        velocity = tuple(v/.05 for v in delta)
        for _ in range(20):
            t += .05
            local = add(local, delta)
            telemetry(b, t, 0, yaw=b.body_rpy[2])
            b._odom_callback(message(t, add(shift, f.rotate(jump, local)), raw_q,
                                    f.rotate(jump, velocity)))
            assert not b.continuity_recovering
        t += .05
        telemetry(b, t, 0, yaw=b.body_rpy[2])
        b._odom_callback(message(t, add(shift, f.rotate(jump, local)), raw_q))
    assert b.latest_fix['lat'] == pytest.approx(original['lat'], abs=1e-11, rel=0)
    assert b.latest_fix['lon'] == pytest.approx(original['lon'], abs=1e-11, rel=0)


@pytest.mark.parametrize('velocity_ok,yaw_ok', [(False, False), (False, True), (True, False), (True, True)])
def test_agreement_checks_stay_independent_and_use_correct_rotations(bridge, velocity_ok, yaw_ok):
    b = bridge
    b.heading_offset_rad = .2
    b.latest_px4_attitude = (10., 10., f.from_rpy(.1, -.1, .4))
    target_q = f.multiply(f.from_rpy(0, 0, -.2), b.latest_px4_attitude[2])
    raw_q = f.from_rpy(2.3, .4, -1.2)
    baseline_rotation = f.from_rpy(-.2, .3, .4)
    b.recovery_frame_rotation = baseline_rotation
    b._recovery_velocity_agrees = Mock(return_value=velocity_ok)
    b._recovery_yaw_agrees = Mock(return_value=yaw_ok)
    result = b._validated_recovery_target((1, 2, 3), f.yaw(raw_q), raw_q)
    # Neither check short-circuits the other. Velocity uses the FINAL handoff
    # rotation; yaw changes use the FIXED baseline, not a per-sample fit.
    b._recovery_velocity_agrees.assert_called_once()
    b._recovery_yaw_agrees.assert_called_once()
    used_rotation = b._recovery_velocity_agrees.call_args.args[2]
    assert f.angle_between(used_rotation, f.alignment(raw_q, target_q)) < 1e-10
    assert b._recovery_yaw_agrees.call_args.args[0] == pytest.approx(
        f.yaw(f.multiply(baseline_rotation, raw_q)))
    assert (result is not None) == (velocity_ok and yaw_ok)
    assert (b.recovery_target_orientation is not None) == (velocity_ok and yaw_ok)
