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
                  recovery_heading_safe=heading_safe,
                  inertial_recovery_failure='position_uncertainty',
                  recovery_heading_failure=None if heading_safe else 'yaw_message_gap',
                  recovery_yaw_agreement_state='collecting' if heading_safe
                  else 'blocked_heading_unverified')
    (tmp_path / 'navigation-status.json').write_text(json.dumps(status))
    namespace['show_navigation_status'](cfg)
    output = capsys.readouterr().out
    assert 'Position recovery: frozen' in output
    assert 'position_uncertainty' in output
    assert 'Last accepted VIO sample: 47.88 s ago' in output
    assert 'Latency:' not in output
    if heading_safe:
        assert 'Heading recovery: tracking' in output
        assert 'restart/re-alignment required' not in output
    else:
        assert 'Heading recovery: blocked' in output
        assert 'yaw_message_gap' in output
