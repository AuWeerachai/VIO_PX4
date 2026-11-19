"""Status must distinguish a frozen position from a lost heading reference."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('heading_safe', [True, False])
def test_separate_recovery_status_and_honest_age(tmp_path, capsys, heading_safe):
    source = Path(__file__).parents[3] / 'scripts/vio-launch/vio_launch.py'
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'show_navigation_status')
    namespace = dict(Config=SimpleNamespace, json=json,
                     time=SimpleNamespace(time=lambda: 100.))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    cfg = SimpleNamespace(state_dir=tmp_path, recovery_yaw_agreement_samples=5,
                          recovery_velocity_agreement_samples=3)
    status = dict(updated_unix_s=100., navigation='gps_output_stopped',
                  px4_connected=True, pose_gate_quarantine=True, vio_age_s=47.8848,
                  inertial_recovery_active=True, recovery_position_safe=False,
                  inherited_position_uncertainty_m=1.2,
                  frozen_position_uncertainty_m=5.0,
                  recovery_heading_safe=heading_safe,
                  inertial_recovery_failure='position_uncertainty',
                  recovery_heading_failure=None if heading_safe else 'yaw_message_gap',
                  recovery_yaw_agreement_state='collecting' if heading_safe
                  else 'blocked_heading_unverified')
    (tmp_path / 'navigation-status.json').write_text(json.dumps(status))
    namespace['show_navigation_status'](cfg)
    output = capsys.readouterr().out
    assert 'Pose-jump episodes: unavailable (running bridge has no counter)' in output
    assert 'Position recovery: frozen' in output
    assert 'position_uncertainty' in output
    assert 'Position uncertainty inherited from PX4: at least 1.20 m' in output
    assert 'verify/reset origin' not in output
    assert 'Global position uncertainty:' not in output
    assert 'Last accepted VIO sample: 47.88 s ago' in output
    assert 'Latency:' not in output
    if heading_safe:
        assert 'Heading recovery: using current FCU heading' in output
        assert 'restart/re-alignment required' not in output
    else:
        assert 'Heading recovery: waiting for fresh FCU heading' in output
        assert 'restart/re-alignment required' not in output
        assert 'yaw_message_gap' in output


@pytest.mark.parametrize('reason,label', [
    ('speed_limit', 'speed limit'),
    ('acceleration_limit', 'acceleration limit'),
    ('yaw_rate_limit', 'yaw-rate limit'),
    ('position_jump', 'position change exceeds speed limit'),
    ('yaw_jump', 'heading change exceeds yaw-rate limit'),
    ('attitude_jump', '3D attitude change exceeds rotation-rate limit'),
    ('tracking_gap', 'tracking gap'),
    ('timestamp_regression', 'duplicate/backward timestamp'),
    ('frame_changed', 'coordinate frame changed'),
    ('new_reason', 'new_reason'),
    (None, 'none'),
])
def test_status_keeps_latest_episode_trigger_after_recovery(tmp_path, capsys, reason, label):
    source = Path(__file__).parents[3] / 'scripts/vio-launch/vio_launch.py'
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'show_navigation_status')
    namespace = dict(Config=SimpleNamespace, json=json,
                     time=SimpleNamespace(time=lambda: 100.))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    status = dict(updated_unix_s=100., navigation='live_vio', px4_connected=True,
                  vio_fresh=True, pose_gate_quarantine=False, hil_gps_output_active=True,
                  pose_jump_count=2 if reason else 0, last_pose_jump_reason=reason,
                  last_pose_jump_detail='measured=61 limit=60' if reason else None,
                  last_pose_jump_stage='startup')
    (tmp_path / 'navigation-status.json').write_text(json.dumps(status))
    namespace['show_navigation_status'](SimpleNamespace(state_dir=tmp_path))
    output = capsys.readouterr().out
    assert 'Pose-jump gate status: healthy' in output
    assert f"Pose-jump episodes: {status['pose_jump_count']} since Path A start" in output
    assert f'Latest episode trigger: {label}' in output
    if reason:
        assert 'during startup' in output
        assert 'measured=61 limit=60' in output
    else:
        assert 'during startup' not in output
