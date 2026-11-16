"""Compact operator status keeps real output state and recovery warnings visible."""
import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def render_status(tmp_path, capsys):
    source = Path(__file__).parents[3] / 'scripts/vio-launch/vio_launch.py'
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'show_navigation_status')
    namespace = dict(Config=SimpleNamespace, json=json,
                     math=math,
                     time=SimpleNamespace(time=lambda: 100.))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    cfg = SimpleNamespace(state_dir=tmp_path, recovery_yaw_agreement_samples=5,
                          recovery_velocity_agreement_samples=3)

    def render(status, *, raw=False):
        if status is not None:
            (tmp_path / 'navigation-status.json').write_text(json.dumps(status))
        namespace['show_navigation_status'](cfg)
        output = capsys.readouterr().out
        # Content assertions ignore padding; alignment tests inspect raw columns.
        return output if raw else '\n'.join(' '.join(line.split()) for line in output.splitlines())

    return render


@pytest.mark.parametrize('position_safe', [True, False])
@pytest.mark.parametrize('heading_safe', [True, False])
def test_separate_recovery_status_and_honest_age(render_status, position_safe, heading_safe):
    status = dict(updated_unix_s=100., navigation='gps_output_stopped',
                  px4_connected=True, pose_gate_quarantine=True, vio_age_s=47.8848,
                  inertial_recovery_active=True, recovery_position_safe=position_safe,
                  inherited_position_uncertainty_m=1.2,
                  recovery_heading_safe=heading_safe,
                  inertial_recovery_failure='position_uncertainty',
                  recovery_heading_failure=None if heading_safe else 'yaw_message_gap',
                  recovery_velocity_agreement_count=2,
                  recovery_yaw_agreement_count=4 if heading_safe else 0,
                  recovery_yaw_agreement_state='collecting' if heading_safe
                  else 'blocked_heading_unverified')
    output = render_status(status)
    if position_safe:
        assert 'Recovery: position following PX4' in output
        assert 'Position frozen:' not in output
    else:
        assert 'Recovery: position frozen' in output
        assert 'Position frozen: position_uncertainty' in output
    assert 'Position uncertainty: ≥1.20 m (from PX4)' in output
    assert 'Last trusted VO: 47.88 s ago' in output
    assert 'Latency:' not in output
    assert 'Recovery checks: velocity 2/3' in output
    assert 'restart/re-alignment required' not in output
    if heading_safe:
        assert '| heading ready' in output
        assert '| yaw 4/5' in output
    else:
        assert '| heading waiting for FCU' in output
        assert '| yaw 0/5' in output
        assert 'yaw_message_gap' in output


@pytest.mark.parametrize('mode', ['sending', 'paused', 'not_sent', 'legacy', 'disabled', 'stale_file'])
def test_heading_status_labels_values_covariance_and_pauses(render_status, mode):
    status = dict(updated_unix_s=100., navigation='live_vio', px4_connected=True,
                  vio_fresh=True, vio_age_s=.02,
                  hil_gps_output_active=True, external_yaw_enabled=True,
                  external_yaw_clock_ready=True, external_yaw_status='ready',
                  external_yaw_sigma_deg=5.,  # display the actual sent variance, not this setting
                  external_yaw_heading_deg=156., external_yaw_variance_rad2=math.radians(1)**2,
                  external_yaw_sample_age_s=.03, external_yaw_output_active=True,
                  hil_gps_lat_deg=40.4135896, hil_gps_lon_deg=-79.9462941,
                  hil_gps_eph_m=.1, hil_gps_sample_age_s=.03)
    if mode == 'paused':
        status.update(external_yaw_output_active=False, pose_gate_quarantine=True,
                      hil_gps_output_active=False, output_block_reason='continuity_quarantine',
                      external_yaw_status='continuity_quarantine', external_yaw_sample_age_s=.35,
                      pose_gate_reason='position_jump', hil_gps_sample_age_s=.35)
    elif mode == 'not_sent':
        status.update(external_yaw_heading_deg=None, external_yaw_variance_rad2=None,
                      external_yaw_output_active=False, external_yaw_sample_age_s=None)
    elif mode == 'legacy':
        del status['external_yaw_heading_deg']
        del status['external_yaw_variance_rad2']
    elif mode == 'disabled':
        status['external_yaw_enabled'] = False
    elif mode == 'stale_file':
        status['updated_unix_s'] = 97.
    output = render_status(status)
    assert '1 sigma' not in output
    assert 'heading std' not in output
    if mode in ('sending', 'paused'):
        assert '156.0°' in output
        assert 'yaw std 1.00 °' in output
        assert 'rad²' not in output
        if mode == 'sending':
            assert 'VO heading: healthy | 156.0°' in output and 'age 30 ms' in output
            assert 'Output: GPS ON | Heading ON' in output
            assert len([line for line in output.splitlines() if line.startswith('- ')]) == 3
            assert 'HIL_GPS: healthy | 40.4135896, -79.9462941 | hAcc 0.10 m | age 30 ms' in output
            assert 'Blocked:' not in output
            assert 'clock' not in output
            assert 'last sent' not in output
        else:
            assert 'VO heading: pose-jump detected | last sent 156.0°' in output
            assert 'HIL_GPS: pose-jump detected | last sent 40.4135896, -79.9462941' in output
            assert 'age 350 ms' in output
            assert 'VO heading: healthy' not in output
            assert 'Output: GPS OFF | Heading OFF' in output
            assert output.count('continuity_quarantine') == 1
    elif mode == 'not_sent':
        assert 'VO heading: blocked | not sent yet' in output
    elif mode == 'legacy':
        assert 'restart the bridge after updating' in output
    elif mode == 'disabled':
        assert 'Output: GPS ON | Heading disabled' in output
        assert 'HIL_GPS: healthy' in output and 'age 30 ms' in output
        assert 'Last trusted VO:' not in output
    else:
        assert 'Bridge: not reporting' in output
        assert 'Output: unknown (stale status)' in output
        assert 'GPS ON' not in output
    if mode not in ('sending', 'paused'):
        assert '156.0' not in output
        assert 'yaw std ' not in output
    assert 'VIO:' not in output
    assert 'Pose gate:' not in output
    assert next(line for line in output.splitlines() if line.startswith('- ')).startswith('- Output:')


def test_compact_spoof_view_preserves_countdown_and_failed_checks(render_status):
    output = render_status(dict(
        updated_unix_s=100., navigation='spoof_active', px4_connected=True,
        spoof_remaining_s=10.25, hil_gps_output_active=False,
        external_yaw_enabled=True, external_yaw_output_active=False,
        external_yaw_heading_deg=None, external_yaw_clock_ready=False,
        external_yaw_status='clock_not_ready', handoff_check_reason='heading alignment incomplete'))
    assert 'HIL_GPS: spoofing (10.2 s left)' in output
    assert 'Output: GPS ON (spoof) | Heading OFF' in output
    assert 'Startup check: heading alignment incomplete' in output
    assert 'heading clock warming/stale' in output
    assert 'VO heading: blocked | not sent yet' in output
    assert 'GPS output stopped' not in output


def test_waiting_for_rc_is_not_labelled_a_navigation_failure(render_status):
    output = render_status(dict(
        updated_unix_s=100., navigation='waiting_for_rc', px4_connected=True,
        vio_fresh=True, external_yaw_enabled=False))
    assert 'HIL_GPS: waiting for RC trigger' in output
    assert 'Output: GPS OFF | Heading disabled' in output
    assert 'Blocked:' not in output


def test_compact_view_keeps_position_uncertainty_warning_after_recovery(render_status):
    output = render_status(dict(
        updated_unix_s=100., navigation='live_vio', px4_connected=True,
        vio_fresh=True, hil_gps_output_active=True, frozen_position_uncertainty_m=4.,
        inherited_position_uncertainty_m=4.))
    assert 'HIL_GPS: healthy' in output
    assert 'Position warning: movement discarded; verify/reset origin.' in output
    assert 'Position uncertainty: ≥4.00 m' in output


def test_missing_status_does_not_claim_output_or_fcu_fallback(render_status):
    output = render_status(None)
    assert 'Bridge: not running (no status received)' in output
    assert 'GPS ON' not in output
    assert 'failsafe' not in output


def test_output_states_are_reported_individually_not_inferred_from_vio_health(render_status):
    output = render_status(dict(
        updated_unix_s=100., navigation='live_vio', px4_connected=True,
        vio_fresh=True, hil_gps_output_active=True, external_yaw_enabled=True,
        external_yaw_output_active=False, external_yaw_heading_deg=None,
        external_yaw_status='clock_not_ready', external_yaw_clock_ready=False))
    assert 'HIL_GPS: healthy' in output
    assert 'Output: GPS ON | Heading OFF' in output
    assert 'Blocked: clock_not_ready; heading clock warming/stale' in output


@pytest.mark.parametrize('position_frozen', [True, False])
def test_legacy_recovery_snapshot_still_shows_its_safety_state(render_status, position_frozen):
    output = render_status(dict(
        updated_unix_s=100., navigation='gps_output_stopped', pose_gate_quarantine=True,
        inertial_recovery_active=True, inertial_recovery_safe=True,
        inertial_fallback_to_frozen_pose=position_frozen,
        inertial_recovery_failure='duration_limit' if position_frozen else None))
    assert ('position frozen' if position_frozen else 'position following PX4') in output
    assert 'heading ready' in output
    assert 'Recovery checks: velocity 0/3 | yaw 0/5' in output


@pytest.mark.parametrize('reason, label', [
    ('position_jump', 'pose-jump detected'), ('yaw_jump', 'pose-jump detected'),
    ('speed_limit', 'pose-jump detected'), ('acceleration_limit', 'pose-jump detected'),
    ('yaw_rate_limit', 'pose-jump detected'), ('tracking_gap', 'stale VO'),
    ('heading_sample_stale', 'stale VO'), ('invalid_odometry', 'quarantined'),
    ('ROS_clock_jump', 'quarantined'), (None, 'quarantined'),
])
def test_cutoff_cause_is_not_always_called_a_pose_jump(render_status, reason, label):
    output = render_status(dict(
        updated_unix_s=100., navigation='gps_output_stopped', px4_connected=True,
        pose_gate_quarantine=True, pose_gate_reason=reason, vio_fresh=False,
        hil_gps_output_active=False, hil_gps_lat_deg=40., hil_gps_lon_deg=-79.,
        hil_gps_eph_m=.13, hil_gps_sample_age_s=2.,
        external_yaw_enabled=True, external_yaw_output_active=False,
        external_yaw_clock_ready=True, external_yaw_status='continuity_quarantine',
        external_yaw_heading_deg=156., external_yaw_variance_rad2=.04,
        external_yaw_sample_age_s=2., output_block_reason='continuity_quarantine'))
    assert f'HIL_GPS: {label} | last sent 40.0000000, -79.0000000 | hAcc 0.13 m | age 2000 ms' in output
    assert f'VO heading: {label} | last sent 156.0° | yaw std 11.46 ° | age 2000 ms' in output
    assert 'Output: GPS OFF | Heading OFF' in output


def test_gps_has_no_synthetic_sample_values_before_first_send(render_status):
    output = render_status(dict(
        updated_unix_s=100., navigation='waiting_for_rc', px4_connected=True,
        hil_gps_lat_deg=None, hil_gps_lon_deg=None, hil_gps_eph_m=None))
    assert 'HIL_GPS: waiting for RC trigger | not sent yet' in output
    assert 'hAcc' not in output


def test_unknown_gps_capture_age_is_not_reported_as_zero(render_status):
    output = render_status(dict(
        updated_unix_s=100., navigation='live_vio', px4_connected=True,
        vio_fresh=True, hil_gps_output_active=True,
        hil_gps_lat_deg=40., hil_gps_lon_deg=-79., hil_gps_eph_m=.1,
        hil_gps_sample_age_s=None, hil_gps_send_age_s=0.))
    assert 'hAcc 0.10 m | age unavailable' in output
    assert 'age 0 ms' not in output


@pytest.mark.parametrize('mode', ['healthy', 'paused', 'mixed', 'spoof', 'not_sent', 'unknown_age'])
def test_gps_and_heading_columns_align_without_truncating_values(render_status, mode):
    status = dict(updated_unix_s=100., navigation='live_vio', px4_connected=True,
                  vio_fresh=True, hil_gps_output_active=True,
                  hil_gps_lat_deg=-89.1234567, hil_gps_lon_deg=179.7654321,
                  hil_gps_eph_m=123.45, hil_gps_sample_age_s=.123,
                  external_yaw_enabled=True, external_yaw_output_active=True,
                  external_yaw_clock_ready=True, external_yaw_status='ready',
                  external_yaw_heading_deg=156., external_yaw_variance_rad2=math.radians(1.)**2,
                  external_yaw_sample_age_s=.03)
    if mode == 'paused':
        status.update(hil_gps_output_active=False, external_yaw_output_active=False,
                      pose_gate_quarantine=True, pose_gate_reason='position_jump')
    elif mode == 'mixed':
        status['external_yaw_output_active'] = False
    elif mode == 'spoof':
        status.update(navigation='spoof_active', spoof_remaining_s=10.,
                      external_yaw_output_active=False)
    elif mode == 'not_sent':
        status.update(hil_gps_lat_deg=None, external_yaw_heading_deg=None)
    elif mode == 'unknown_age':
        status.update(hil_gps_sample_age_s=None, external_yaw_sample_age_s=None)
    output = render_status(status, raw=True)
    gps_line, yaw_line = [line for line in output.splitlines()
                          if line.startswith(('- HIL_GPS:', '- VO heading:'))]
    gps_bars = [i for i, char in enumerate(gps_line) if char == '|']
    yaw_bars = [i for i, char in enumerate(yaw_line) if char == '|']
    assert gps_bars == yaw_bars
    assert len(gps_bars) == (1 if mode == 'not_sent' else 3)
    if mode != 'not_sent':
        assert '-89.1234567, 179.7654321' in gps_line
        assert 'hAcc 123.45 m' in ' '.join(gps_line.split())
        assert '156.0°' in yaw_line
        assert 'yaw std 1.00 °' in ' '.join(yaw_line.split())
        gps_uncertainty, yaw_uncertainty = gps_line.split('|')[2], yaw_line.split('|')[2]
        assert gps_uncertainty.index('.') == yaw_uncertainty.index('.')
        assert gps_uncertainty.index(' m') == yaw_uncertainty.index(' °')
    if mode == 'paused':
        assert 'pose-jump detected' in gps_line and 'pose-jump detected' in yaw_line
        assert 'last sent' in gps_line and 'last sent' in yaw_line
    if mode == 'unknown_age':
        assert 'age unavailable' in gps_line and 'age unavailable' in yaw_line


@pytest.mark.parametrize('sigma_deg', [0., .1, 1., 18., 60.])
def test_heading_std_converts_actual_sent_variance_not_configured_noise(render_status, sigma_deg):
    status = dict(updated_unix_s=100., navigation='live_vio', px4_connected=True,
                  external_yaw_enabled=True, external_yaw_output_active=True,
                  external_yaw_heading_deg=156., external_yaw_variance_rad2=math.radians(sigma_deg)**2,
                  external_yaw_sigma_deg=5., external_yaw_sample_age_s=.03,
                  hil_gps_output_active=True, hil_gps_lat_deg=40., hil_gps_lon_deg=-79.,
                  hil_gps_eph_m=.1, hil_gps_sample_age_s=.03)
    output = render_status(status)
    assert f'yaw std {sigma_deg:.2f} °' in output
    assert 'yaw std 5.00 °' not in output
    assert status['external_yaw_variance_rad2'] == math.radians(sigma_deg)**2


@pytest.mark.parametrize('variance', [-.1, math.nan, math.inf])
def test_invalid_heading_variance_is_not_rendered_as_perfect_accuracy(render_status, variance):
    output = render_status(dict(
        updated_unix_s=100., navigation='gps_output_stopped', external_yaw_enabled=True,
        external_yaw_output_active=False, external_yaw_heading_deg=156.,
        external_yaw_variance_rad2=variance))
    assert 'yaw std unavailable' in output
    assert 'yaw std 0.00' not in output
