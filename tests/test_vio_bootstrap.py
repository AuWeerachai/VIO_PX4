import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT/'bootstrap/jetson-config/isaac_ros_overrides/scripts'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cli = load('vio_launch_bootstrap_test', ROOT/'scripts/vio-launch/vio_launch.py')
fixes = load('vio_imu_fixes_test', SCRIPTS/'apply_vio_imu_fixes.py')


def test_default_and_explicit_modes(tmp_path):
    cfg = cli.Config(isaac_ros_dir=tmp_path)
    assert cli.selected_camera_mode(cfg) == 'vio'
    assert cli.camera_mode_label(cfg) == 'VIO (stereo + IMU)'
    mode = tmp_path/'src/isaac_ros_common/scripts/vio_imu_mode'
    mode.parent.mkdir(parents=True)
    mode.write_text('vo\n')
    assert cli.camera_mode_label(cfg) == 'VO (stereo only)'
    mode.write_text('garbage')
    with pytest.raises(RuntimeError):
        cli.selected_camera_mode(cfg)


def test_menu_reports_selected_mode(tmp_path):
    cfg = cli.Config(isaac_ros_dir=tmp_path)
    with patch.object(cli, 'select_menu', return_value='detach') as menu, patch.object(cli, '_clear'):
        assert cli.interactive_main(cfg) == 0
    assert 'Selected camera mode: VIO (stereo + IMU)' in menu.call_args.kwargs['subtitle']


def test_readiness_failure_blocks_reuse(tmp_path):
    cfg = cli.Config(isaac_ros_dir=tmp_path)
    with patch.object(cli.shutil, 'which', return_value='/bin/mock'), \
         patch.object(cli, 'tmux_session_exists', return_value=True), \
         patch.object(cli, 'isaac_container_running', return_value=True), \
         patch.object(cli, 'require_ros_message'), \
         patch.object(cli, 'require_camera_ready', side_effect=RuntimeError('wrong mode')), \
         patch.object(cli, 'start_visible_isaac_session') as start:
        with pytest.raises(RuntimeError, match='wrong mode'):
            cli.cmd_vio(cfg)
        start.assert_not_called()


def test_shutdown_waits_for_camera_before_killing_terminal():
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1, stdout=b'')
    with patch.object(cli, 'stop_rviz_session'), \
         patch.object(cli, 'tmux_session_exists', return_value=True), \
         patch.object(cli, 'isaac_container_running', return_value=True), \
         patch.object(cli.subprocess, 'run', side_effect=run):
        cli.stop_visible_isaac_sessions()
    assert calls[0][:2] == ['tmux', 'send-keys']
    assert calls[1][:2] == ['docker', 'exec']
    assert calls[2][:2] == ['tmux', 'kill-session']


def test_start_failure_saves_terminal_before_cleanup(tmp_path):
    cfg = cli.Config(state_dir=tmp_path)
    with patch.object(cli.shutil, 'which', return_value='/bin/mock'), \
         patch.object(cli, 'tmux_session_exists', return_value=False), \
         patch.object(cli, 'isaac_container_running', return_value=True), \
         patch.object(cli, 'start_visible_isaac_session'), \
         patch.object(cli, 'tmux_command'), \
         patch.object(cli.time, 'sleep'), \
         patch.object(cli, 'require_ros_message', side_effect=RuntimeError('no odometry')), \
         patch.object(cli, 'tmux_recent_output', return_value='VIO IMU FAILED: example failure'), \
         patch.object(cli, 'stop_visible_isaac_sessions') as stop:
        def assert_log_saved():
            assert (cfg.log_dir/'camera-startup-failure.log').exists()
        stop.side_effect = assert_log_saved
        with pytest.raises(RuntimeError, match='Recent camera output'):
            cli.cmd_vio(cfg)
    assert 'example failure' in (cfg.log_dir/'camera-startup-failure.log').read_text()


def test_readiness_runs_as_camera_user_and_propagates_failure(tmp_path):
    with patch.object(cli.subprocess, 'run', return_value=SimpleNamespace(returncode=1, stdout='IMU missing', stderr='')) as run:
        with pytest.raises(RuntimeError, match='IMU missing'):
            cli.require_camera_ready(cli.Config(isaac_ros_dir=tmp_path))
    assert run.call_args.args[0][:4] == ['docker', 'exec', '-u', 'admin']
    assert 'vio --timeout 12' in run.call_args.args[0][-1]


def test_path_a_defers_health_but_path_b_does_not(tmp_path):
    cfg = cli.Config(isaac_ros_dir=tmp_path)
    with patch.object(cli, 'cmd_vio') as camera, \
         patch.object(cli, 'find_proc', return_value=None), \
         patch.object(cli, 'cmd_gps'), patch.object(cli, 'cmd_ev_bridge'):
        cli.cmd_run(cfg, 'a')
        camera.assert_called_with(cfg, defer_health=True)
        cli.cmd_run(cfg, 'b')
        camera.assert_called_with(cfg, defer_health=False)


def test_failed_spoof_stops_gps_before_camera_and_preserves_evidence(tmp_path):
    cfg = cli.Config(state_dir=tmp_path)
    events = []
    with patch.object(cli, 'stop_managed', side_effect=lambda cfg, name: events.append(name)), \
         patch.object(cli, 'tmux_recent_output', return_value='camera diagnostics'), \
         patch.object(cli, 'stop_visible_isaac_sessions', side_effect=lambda: events.append('camera')):
        cli.stop_failed_spoof(cfg, 'IMU timestamps unstable')
    assert events == ['gps-bridge', 'camera']
    assert 'IMU timestamps unstable' in (cfg.log_dir/'camera-spoof-failure.log').read_text()
    assert 'camera diagnostics' in (cfg.log_dir/'camera-spoof-failure.log').read_text()


def test_started_only_probe_does_not_request_full_data_validation(tmp_path):
    with patch.object(cli.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='started', stderr='')) as run:
        cli.require_camera_ready(cli.Config(isaac_ros_dir=tmp_path), started_only=True)
    assert '--started-only' in run.call_args.args[0][-1]


def test_camera_launcher_only_configures_imu():
    script = (SCRIPTS/'start_vio.sh').read_text()
    assert 'stage_realsense_imu.py --configure-only' in script


def test_source_fixes_idempotent_and_keep_other_edits(tmp_path):
    impl = tmp_path/'src/impl/visual_slam_impl.cpp'
    launch = tmp_path/'launch/isaac_ros_visual_slam_core.launch.py'
    impl.parent.mkdir(parents=True)
    launch.parent.mkdir(parents=True)
    impl.write_text('// unrelated user edit\nCUVSLAM_RegisterImuMeasurement(\n      cuvslam_handle, latest_ts, &imu_measurement);\n')
    launch.write_text("                    ('/visual_slam/image_0', 'left/image_rect_mono'),\n")
    changed = fixes.patched_sources(tmp_path)
    for path, text in changed.items():
        path.write_text(text)
    assert '// unrelated user edit' in impl.read_text()
    assert 'imu_ts' in impl.read_text()
    assert fixes.patched_sources(tmp_path) == changed
    impl.write_text('unknown source')
    with pytest.raises(RuntimeError, match='Unknown'):
        fixes.patched_sources(tmp_path)


def test_builder_cache_does_not_require_ros_or_network(tmp_path):
    import os
    pins = dict(line.split('=', 1) for line in (SCRIPTS/'vio_imu_release.env').read_text().splitlines() if line and not line.startswith('#'))
    root = tmp_path/'.vio-imu'/pins['VIO_IMU_OVERLAY_ID']
    for relative in ['sdk/lib/librealsense2.so.2.56.5', 'ros_install/local_setup.bash',
                     'ros_install/realsense2_camera/lib/librealsense2_camera.so']:
        p = root/relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    (root/'READY').write_text(' '.join(pins[k] for k in ['VIO_IMU_SDK_COMMIT', 'VIO_IMU_ROS_COMMIT', 'VIO_IMU_OVERLAY_ID'])+'\n')
    result = subprocess.run(['bash', str(SCRIPTS/'build_vio_imu_overlay.sh')],
                            env={**os.environ, 'ISAAC_ROS_WS': str(tmp_path)},
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    assert 'cached' in result.stdout
