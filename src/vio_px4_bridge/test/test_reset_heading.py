import math

import pytest

from vio_px4_bridge.reset_heading import ResetHeadingEvidence, multiply


def qz(angle):
    return (math.cos(angle/2), 0., 0., math.sin(angle/2))


def test_stationary_long_loss_with_quaternion_sign_flips():
    guard = ResetHeadingEvidence(math.radians(2), .25, .05)
    for i in range(1001):
        sign = 1 if i % 2 else -1
        assert guard.observe((i*.05, i*.05, (sign, 0, 0, 0), (0, 0, 0)))
    assert guard.verifies(45., 50., .25)
    assert guard.error_sum == 0.


def test_tilted_real_body_rotation_not_a_heading_reset():
    # A 32-degree pitched vehicle rotates around its body Z, not world Z.
    pitch = math.radians(32)
    initial = (math.cos(pitch/2), 0., math.sin(pitch/2), 0.)
    guard = ResetHeadingEvidence(math.radians(2), .25, .05)
    for i in range(501):
        t = i*.05
        q = multiply(initial, qz(.5*t))
        assert guard.observe((t, t, q, (0, 0, .5)))
    assert guard.error_sum < 1e-10


@pytest.mark.parametrize('jump_deg', [3, 30, 90, 180])
def test_unexplained_rotation_latches_even_if_later_stable(jump_deg):
    guard = ResetHeadingEvidence(math.radians(2), .25, .05)
    guard.observe((0, 0, qz(0), (0, 0, 0)))
    assert not guard.observe((.05, .05, qz(math.radians(jump_deg)), (0, 0, 0)))
    assert guard.failure == 'unexplained_rotation_budget_exceeded'
    assert not guard.observe((.10, .10, qz(math.radians(jump_deg)), (0, 0, 0)))


def test_small_unexplained_steps_cannot_renew_error_budget():
    guard = ResetHeadingEvidence(math.radians(2), .25, .05)
    for i in range(11):
        guard.observe((i*.05, i*.05, qz(math.radians(i*.3)), (0, 0, 0)))
    assert guard.failure == 'unexplained_rotation_budget_exceeded'


@pytest.mark.parametrize('sample', [
    (.3, .3, qz(0), (0, 0, 0)),
    (.05, -.1, qz(0), (0, 0, 0)),
    (.05, .05, (0, 0, 0, 0), (0, 0, 0)),
    (.05, .05, qz(0), (float('nan'), 0, 0)),
])
def test_missing_or_invalid_rotation_evidence_is_not_authority(sample):
    guard = ResetHeadingEvidence(math.radians(2), .25, .05)
    guard.observe((0, 0, qz(0), (0, 0, 0)))
    assert not guard.observe(sample)
    assert not guard.verifies(.05, .05, .25)


def test_reordered_packet_does_not_move_baseline_or_count_as_evidence():
    guard = ResetHeadingEvidence(math.radians(2), .25, .05)
    guard.observe((1., 1., qz(0), (0, 0, 0)))
    assert not guard.observe((1.01, .99, qz(2), (0, 0, 0)))
    assert guard.failure is None and guard.intervals == 0
    assert guard.observe((1.05, 1.05, qz(0), (0, 0, 0)))
    assert guard.verifies(1.05, 1.05, .25)
    assert not guard.verifies(1.05, 2., .25)
