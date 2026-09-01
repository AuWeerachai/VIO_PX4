import math

from vio_px4_bridge.recovery_yaw import RecoveryYawAgreement


def deg(value):
    return math.radians(value)


def test_arbitrary_new_epoch_yaw_is_ignored_when_changes_agree():
    gate = RecoveryYawAgreement(deg(5.0), required_samples=3)
    assert not gate.update(deg(-80), deg(20), 1.0).agreed
    assert not gate.update(deg(-78), deg(22), 2.0).agreed
    assert not gate.update(deg(-75), deg(25), 3.0).agreed
    result = gate.update(deg(-70), deg(30), 4.0)
    assert result.agreed
    assert result.event == "agreement_confirmed"
    assert math.isclose(result.residual_rad, 0.0, abs_tol=1e-9)


def test_disagreement_restarts_window_without_tainting_recovery():
    gate = RecoveryYawAgreement(deg(5.0), required_samples=2)
    gate.update(deg(40), deg(0), 1.0)
    bad = gate.update(deg(70), deg(2), 2.0)
    assert not bad.agreed
    assert bad.event == "disagreement_reset"
    assert bad.good_samples == 0
    assert not gate.update(deg(72), deg(4), 3.0).agreed
    recovered = gate.update(deg(75), deg(7), 4.0)
    assert recovered.agreed


def test_duplicate_px4_attitude_sample_is_not_counted_twice():
    gate = RecoveryYawAgreement(deg(5.0), required_samples=2)
    gate.update(deg(0), deg(0), 1.0)
    first = gate.update(deg(1), deg(1), 2.0)
    duplicate = gate.update(deg(1.5), deg(1), 2.0)
    assert first.good_samples == 1
    assert duplicate.good_samples == 1
    assert not duplicate.evaluated
    assert not duplicate.agreed


def test_wrapped_yaw_changes_agree_across_pi_boundary():
    gate = RecoveryYawAgreement(deg(3.0), required_samples=1)
    gate.update(deg(179), deg(10), 1.0)
    result = gate.update(deg(-179), deg(12), 2.0)
    assert result.agreed
    assert math.isclose(result.residual_rad, 0.0, abs_tol=1e-9)
