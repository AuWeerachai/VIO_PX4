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
selection = load('vio_camera_selection_test', SCRIPTS/'camera_selection.py')
environment = load('vo_environment_test', SCRIPTS/'check_vo_environment.py')


@pytest.fixture(autouse=True)
def isolate_installed_camera_helpers(request):
    # Launcher unit tests mock the external installation. The deployment tests
    # below exercise the real hash guard against complete/mixed installations.
    if request.node.name.startswith('test_camera_deployment_'):
        yield
    else:
        with patch.object(cli, 'require_camera_deployment'):
            yield


def test_camera_identity_rejects_wrong_serial_and_wrong_model():
    selection.verify_identity('333422300181', '333422300181', 'Intel_RealSense_D456')
    for serial, model in [('123', 'Intel_RealSense_D456'),
                          ('333422300181', 'Intel_RealSense_D435')]:
        with pytest.raises(RuntimeError, match='Wrong VIO camera'):
            selection.verify_identity('333422300181', serial, model)


@pytest.mark.parametrize('serial', ['', 'auto', '123; touch /tmp/oops', 333422300181])
def test_camera_selection_refuses_unconfigured_or_invalid_serial(serial):
    with pytest.raises(ValueError):
        selection.normalize_serial(serial)


def test_camera_selection_only_changes_device_selectors():
    base = dict(serial_no='', device_type='', usb_port_id='2-1',
                enable_infra1=True, enable_infra2=True, enable_gyro=False,
                depth_module=dict(profile='640x360x90', emitter_enabled=0))
    actual = selection.selected_config(base, '333422300181')
    assert actual == {**base, 'serial_no': '_333422300181',
                      'device_type': 'd456', 'usb_port_id': ''}
    assert base['serial_no'] == ''


def test_saved_camera_serial_survives_config_round_trip(tmp_path):
    cfg = cli.apply_dict(cli.Config(), {'camera_serial': '333422300181'})
    with patch.object(cli, 'user_config_path', return_value=tmp_path/'config.json'):
        cli.save_config(cfg)
        assert cli.load_config().camera_serial == '333422300181'


def test_unconfigured_camera_cannot_launch_or_reuse():
    with patch.object(cli, 'start_visible_isaac_session') as start:
        with pytest.raises(RuntimeError, match='rear D456 SDK serial'):
            cli.cmd_vio(cli.Config())
    start.assert_not_called()


def test_stale_cli_cannot_save_configuration(tmp_path):
    with patch.object(cli, '_LOADED_CLI_HASH', b'outdated'), \
         patch.object(cli, 'user_config_path', return_value=tmp_path/'config.json'), \
         patch.object(cli, 'atomic_write_json') as write:
        with pytest.raises(cli.StaleLauncherError, match='updated while this menu was open'):
            cli.save_config(cli.Config())
    write.assert_not_called()
    assert not (tmp_path/'config.json').exists()


def test_stale_cli_cannot_start_managed_process(tmp_path):
    with patch.object(cli, '_LOADED_CLI_HASH', b'outdated'), \
         patch.object(cli.subprocess, 'Popen') as start:
        with pytest.raises(cli.StaleLauncherError):
            cli.start_managed(cli.Config(state_dir=tmp_path), 'gps-bridge', 'false')
    start.assert_not_called()
    assert not (tmp_path/'state.json').exists()


def test_stale_cli_cannot_launch_a_path():
    with patch.object(cli, '_LOADED_CLI_HASH', b'outdated'), \
         patch.object(cli, 'cmd_vio') as camera, patch.object(cli, 'cmd_gps') as gps:
        with pytest.raises(cli.StaleLauncherError):
            cli.cmd_run(cli.Config(), 'a')
    camera.assert_not_called()
    gps.assert_not_called()


@pytest.mark.parametrize('direct_gps', [False, True])
def test_duplicate_path_a_is_rejected_without_camera_or_process_changes(direct_gps, capsys):
    cfg = cli.Config(camera_serial='333422300181')
    with patch.object(cli, 'find_proc', side_effect=lambda cfg, name: {'pid': 123} if name == 'gps-bridge' else None), \
         patch.object(cli, 'cmd_vio') as camera, \
         patch.object(cli, 'require_camera_ready') as readiness, \
         patch.object(cli, 'serial_users') as serial, \
         patch.object(cli, 'start_managed') as start, \
         patch.object(cli, 'stop_managed') as stop:
        with pytest.raises(RuntimeError, match=r'Path A is already running \(GPS bridge pid=123\)'):
            if direct_gps:
                cli.cmd_gps(cfg)
            else:
                cli.cmd_run(cfg, 'a')
    camera.assert_not_called()
    readiness.assert_not_called()
    serial.assert_not_called()
    start.assert_not_called()
    stop.assert_not_called()
    assert 'Starting Path A' not in capsys.readouterr().out


@pytest.mark.parametrize('direct_gps', [False, True])
def test_stale_cli_warning_precedes_duplicate_path_a_check(direct_gps):
    with patch.object(cli, '_LOADED_CLI_HASH', b'outdated'), \
         patch.object(cli, 'find_proc', return_value={'pid': 123}) as process, \
         patch.object(cli, 'cmd_vio') as camera, \
         patch.object(cli, 'require_camera_ready') as readiness:
        with pytest.raises(cli.StaleLauncherError, match='updated while this menu was open'):
            if direct_gps:
                cli.cmd_gps(cli.Config())
            else:
                cli.cmd_run(cli.Config(), 'a')
    process.assert_not_called()
    camera.assert_not_called()
    readiness.assert_not_called()


def test_bridge_started_during_camera_readiness_is_reported_not_silently_reused():
    active = False
    def camera_ready(*args, **kwargs):
        nonlocal active
        active = True
    def process(cfg, name):
        return {'pid': 456} if active and name == 'gps-bridge' else None
    with patch.object(cli, 'find_proc', side_effect=process), \
         patch.object(cli, 'cmd_vio', side_effect=camera_ready) as camera, \
         patch.object(cli, 'require_camera_ready') as readiness, \
         patch.object(cli, 'start_managed') as start:
        with pytest.raises(RuntimeError, match=r'Path A is already running \(GPS bridge pid=456\)'):
            cli.cmd_run(cli.Config(), 'a')
    camera.assert_called_once()
    readiness.assert_not_called()
    start.assert_not_called()


def test_update_while_menu_is_waiting_is_detected_before_dispatch():
    choices = iter(['run-a', 'detach'])
    def select(*args, **kwargs):
        cli._LOADED_CLI_HASH = b'updated during prompt'
        return next(choices)
    with patch.object(cli, '_LOADED_CLI_HASH', cli._LOADED_CLI_HASH), \
         patch.object(cli, 'select_menu', side_effect=select), \
         patch.object(cli, '_clear'), patch.object(cli, 'pause'), \
         patch.object(cli, 'cmd_run') as run:
        assert cli.interactive_main(cli.Config()) == 0
    run.assert_not_called()


@pytest.mark.parametrize('active', ['ev-bridge', 'mavros'])
def test_legacy_publisher_is_rejected_before_touching_camera(active):
    with patch.object(cli, 'find_proc', side_effect=lambda cfg, name: {'pid': 123} if name == active else None), \
         patch.object(cli, 'cmd_vio') as camera:
        with pytest.raises(RuntimeError, match='already running'):
            cli.cmd_run(cli.Config(), 'a')
    camera.assert_not_called()


@pytest.mark.parametrize('existing_session', [False, True])
def test_camera_is_not_restarted_under_active_gps(existing_session):
    cfg = cli.Config(camera_serial='333422300181')
    with patch.object(cli.shutil, 'which', return_value='/bin/mock'), \
         patch.object(cli, 'tmux_session_exists', return_value=existing_session), \
         patch.object(cli, 'isaac_container_running', return_value=existing_session), \
         patch.object(cli, 'require_camera_ready', side_effect=RuntimeError('Wrong VIO camera')), \
         patch.object(cli, 'find_proc', side_effect=lambda cfg, name: {'pid': 123} if name == 'gps-bridge' else None), \
         patch.object(cli, 'start_visible_isaac_session') as start, \
         patch.object(cli, 'stop_visible_isaac_sessions') as stop:
        with pytest.raises(RuntimeError, match='Cannot restart the camera'):
            cli.cmd_vio(cfg, defer_health=True)
    start.assert_not_called()
    stop.assert_not_called()


def test_direct_gps_command_cannot_bypass_camera_identity():
    with patch.object(cli, 'find_proc', return_value=None), \
         patch.object(cli, 'require_camera_ready', side_effect=RuntimeError('Wrong VIO camera')), \
         patch.object(cli, 'start_managed') as start, \
         patch.object(cli, 'serial_users') as port_check:
        with pytest.raises(RuntimeError, match='Wrong VIO camera'):
            cli.cmd_gps(cli.Config(camera_serial='333422300181'))
    start.assert_not_called()
    port_check.assert_not_called()


def test_old_launcher_missing_serial_cannot_open_any_camera(tmp_path):
    import os
    (tmp_path/'src/isaac_ros_common').mkdir(parents=True)
    env = {**os.environ, 'ISAAC_ROS_WS': str(tmp_path), 'VIO_MODE': 'vo',
           'AMENT_PREFIX_PATH': '', 'LD_LIBRARY_PATH': '', 'CMAKE_PREFIX_PATH': ''}
    env.pop('VIO_CAMERA_SERIAL', None)
    result = subprocess.run(['bash', str(SCRIPTS/'start_vio.sh')],
                            env=env, capture_output=True, text=True, timeout=5)
    assert result.returncode == 2
    assert 'Leave the old CLI and reopen ~/vio-launch' in result.stderr
    assert 'No camera was opened' in result.stderr


def test_heading_defaults_migrate_saved_compass_without_changing_extrinsics():
    assert cli.Config().heading_source == 'px4'
    cfg = cli.apply_dict(cli.Config(), dict(heading_source='compass',
                         mag_declination_deg=35., camera_pitch_deg=32., camera_yaw_deg=180.))
    assert cfg.heading_source == 'px4'
    assert (cfg.camera_pitch_deg, cfg.camera_yaw_deg) == (32., 180.)
    assert 'PX4' in cli.heading_label(cfg)
    assert '35' not in cli.heading_label(cfg)
    assert cli.apply_dict(cfg, {'heading_source': 'manual'}).heading_source == 'manual'


def test_heading_menu_no_longer_prompts_for_bridge_declination():
    cfg = cli.Config()
    with patch.object(cli, '_clear'), patch.object(cli, 'pause'), \
         patch.object(cli, 'prompt_line', side_effect=['px4', '0']) as prompt, \
         patch.object(cli, 'save_config') as save:
        changed = cli.configure_heading(cfg)
    assert changed.heading_source == 'px4'
    assert prompt.call_count == 2
    assert all('declination' not in call.args[0].lower() for call in prompt.call_args_list)
    save.assert_called_once_with(changed)


def test_default_and_explicit_modes(tmp_path):
    cfg = cli.Config(isaac_ros_dir=tmp_path)
    assert cli.selected_camera_mode(cfg) == 'vo'
    assert cli.camera_mode_label(cfg) == 'VO (stereo only)'
    mode = tmp_path/'src/isaac_ros_common/scripts/vio_imu_mode'
    mode.parent.mkdir(parents=True)
    mode.write_text('vo\n')
    assert cli.camera_mode_label(cfg) == 'VO (stereo only)'
    mode.write_text('vio\n')
    assert cli.camera_mode_label(cfg) == 'VO (stereo only)'
    mode.write_text('garbage')
    assert cli.selected_camera_mode(cfg) == 'vo'


def test_menu_reports_selected_mode(tmp_path):
    cfg = cli.Config(isaac_ros_dir=tmp_path)
    with patch.object(cli, 'select_menu', return_value='detach') as menu, patch.object(cli, '_clear'):
        assert cli.interactive_main(cfg) == 0
    assert 'Selected camera mode: VO (stereo only)' in menu.call_args.kwargs['subtitle']


def test_readiness_failure_blocks_reuse(tmp_path):
    cfg = cli.Config(isaac_ros_dir=tmp_path, camera_serial='333422300181')
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
    cfg = cli.Config(state_dir=tmp_path, camera_serial='333422300181')
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
    assert 'vo --timeout 12' in run.call_args.args[0][-1]


def test_path_a_defers_health_until_spoof_without_changing_launch_order(tmp_path):
    cfg = cli.Config(isaac_ros_dir=tmp_path)
    events = []
    with patch.object(cli, 'cmd_vio', side_effect=lambda *a, **kw: events.append('camera')) as camera, \
         patch.object(cli, 'find_proc', return_value=None), \
         patch.object(cli, 'cmd_gps', side_effect=lambda *a: events.append('gps')) as gps:
        cli.cmd_run(cfg, 'a')
        camera.assert_called_once_with(cfg, defer_health=True)
        gps.assert_called_once_with(cfg)
    assert events == ['camera', 'gps']


@pytest.mark.parametrize('path', ['b', 'B', 'ev', 'vision', 'external-vision'])
def test_removed_path_is_rejected_without_starting_camera_or_gps(path):
    with patch.object(cli, 'cmd_vio') as camera, patch.object(cli, 'cmd_gps') as gps:
        with pytest.raises(RuntimeError, match='Only Path A'):
            cli.cmd_run(cli.Config(), path)
    camera.assert_not_called()
    gps.assert_not_called()


@pytest.mark.parametrize('args', [['ev-bridge'], ['run', 'b'], ['run', 'ev']])
def test_parser_rejects_removed_commands_before_loading_config(args):
    with patch.object(cli, 'load_config') as load_config:
        with pytest.raises(SystemExit) as stopped:
            cli.main(args)
    assert stopped.value.code == 2
    load_config.assert_not_called()


@pytest.mark.parametrize('path', ['a', 'gps', 'hil', 'internship', 'A', 'GPS'])
def test_gps_aliases_still_work(path):
    args = cli.build_parser().parse_args(['run', path])
    assert cli.normalize_path(args.path) == 'a'


def test_menu_has_only_path_a_and_status_is_option_two():
    with patch.object(cli, 'select_menu', return_value='detach') as menu, \
         patch.object(cli, '_clear'):
        assert cli.interactive_main(cli.Config()) == 0
    choices = [item.value for item in menu.call_args.args[1] if not item.disabled]
    assert choices[:2] == ['run-a', 'nav-status']
    assert 'run-b' not in choices


def test_obsolete_mavros_config_is_ignored_without_losing_camera_settings(tmp_path):
    data = dict(mavros_dir='/old/mavros', camera_serial='333422300181',
                camera_pitch_deg=30.5, camera_yaw_deg=180., home_alt=200.)
    cfg = cli.apply_dict(cli.Config(), data)
    assert not hasattr(cfg, 'mavros_dir')
    with patch.object(cli, 'user_config_path', return_value=tmp_path/'config.json'):
        cli.save_config(cfg)
        saved = cli.load_saved_overrides()
    assert 'mavros_dir' not in saved
    assert all(saved[key] == value for key, value in data.items() if key != 'mavros_dir')


def test_package_and_bootstrap_no_longer_install_path_b():
    import runpy
    import xml.etree.ElementTree as ET
    package = ROOT/'src/vio_px4_bridge'
    assert not (package/'vio_px4_bridge/mavros_ev_bridge.py').exists()
    dependencies = ET.parse(package/'package.xml').getroot()
    assert not any('mavros' in (child.text or '') for child in dependencies)
    with patch('setuptools.setup') as setup:
        runpy.run_path(str(package/'setup.py'))
    assert setup.call_args.kwargs['entry_points']['console_scripts'] == [
        'vio_px4_gps_bridge = vio_px4_bridge.vio_px4_gps_bridge:main']
    assert 'mavros' not in (ROOT/'bootstrap/bootstrap_jetson.sh').read_text().lower()
    assert not (ROOT/'bootstrap/jetson-config/mavros_release.env').exists()
    assert not (ROOT/'bootstrap/jetson-config/mavros_overrides/proven_jetson.patch').exists()


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
    assert '--timeout 12' in run.call_args.args[0][-1]


def test_identity_discovery_timeout_retries_once_without_restarting_camera(tmp_path):
    pending = SimpleNamespace(returncode=1, stdout='Camera readiness failed: VO not ready: D456 identity verified=False, fusion=False', stderr='')
    success = SimpleNamespace(returncode=0, stdout='VO node started; D456 serial verified', stderr='')
    with patch.object(cli.subprocess, 'run', side_effect=[pending, success]) as run, \
         patch.object(cli, 'stop_visible_isaac_sessions') as stop:
        cli.require_camera_ready(cli.Config(isaac_ros_dir=tmp_path), started_only=True)
    assert run.call_count == 2
    stop.assert_not_called()


def test_persistent_identity_timeout_still_blocks_gps(tmp_path):
    pending = SimpleNamespace(returncode=1, stdout='VO not ready: D456 identity verified=False', stderr='')
    with patch.object(cli.subprocess, 'run', return_value=pending) as run:
        with pytest.raises(RuntimeError, match='identity verified=False'):
            cli.require_camera_ready(cli.Config(isaac_ros_dir=tmp_path), started_only=True)
    assert run.call_count == 2


@pytest.mark.parametrize('error', ['Wrong VIO camera: expected D456 serial 333422300181',
                                  'Running camera mode differs from selected mode',
                                  'Camera identity service returned no response'])
def test_real_identity_or_mode_failure_is_not_retried(tmp_path, error):
    with patch.object(cli.subprocess, 'run', return_value=SimpleNamespace(returncode=1, stdout=error, stderr='')) as run:
        with pytest.raises(RuntimeError, match=error):
            cli.require_camera_ready(cli.Config(isaac_ros_dir=tmp_path), started_only=True)
    assert run.call_count == 1


def test_camera_launcher_restores_vo_without_loading_imu_overlay():
    script = (SCRIPTS/'start_vio.sh').read_text()
    assert 'enable_imu_fusion:=False' in script
    assert 'stage_realsense_imu.py' not in script
    assert 'build_vio_imu_overlay.sh' not in script
    assert 'ros_install/local_setup.bash' not in script
    assert 'CAMERA_PITCH_RAD' in script and 'CAMERA_YAW_RAD' in script


def test_vo_shell_launch_preserves_extrinsics_and_ignores_old_mode_file(tmp_path):
    import os
    import shutil
    workspace = tmp_path/'workspace'
    common = workspace/'src/isaac_ros_common'
    (common/'scripts').mkdir(parents=True)
    (common/'scripts/vio_imu_mode').write_text('vio\n')
    shutil.copy(SCRIPTS/'camera_selection.py', common/'scripts/camera_selection.py')
    # This argument test mocks ROS; environment checks are tested separately.
    (common/'scripts/vo_environment.sh').write_text('true\n')
    (common/'scripts/check_vo_environment.py').write_text('pass\n')
    base = workspace/'share/isaac_ros_realsense/config/realsense_stereo.yaml'
    base.parent.mkdir(parents=True)
    base.write_text('enable_infra1: true\nenable_infra2: true\nenable_gyro: false\n')
    bin_dir = tmp_path/'bin'
    bin_dir.mkdir()
    recorder = bin_dir/'ros2'
    recorder.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$VO_TEST_CALLS"\n'
                        'if [ "$1" = pkg ]; then printf "%s\\n" "$ISAAC_ROS_WS"; fi\n')
    recorder.chmod(0o755)
    calls = tmp_path/'calls'
    env = {**os.environ, 'ISAAC_ROS_WS': str(workspace),
           'PATH': str(bin_dir)+':'+os.environ['PATH'], 'VO_TEST_CALLS': str(calls),
           'AMENT_PREFIX_PATH': '', 'LD_LIBRARY_PATH': '', 'CMAKE_PREFIX_PATH': '',
           'CAMERA_X_M': '-0.15', 'CAMERA_Y_M': '0', 'CAMERA_Z_M': '-0.165',
           'CAMERA_ROLL_RAD': '-0.02059', 'CAMERA_PITCH_RAD': '0.5585',
           'CAMERA_YAW_RAD': '3.14159'}
    env['VIO_CAMERA_SERIAL'] = '333422300181'
    env.pop('VIO_MODE', None)
    result = subprocess.run(['bash', str(SCRIPTS/'start_vio.sh')],
                            env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    launched = calls.read_text()
    assert 'enable_imu_fusion:=False' in launched
    assert '--x -0.15 --y 0 --z -0.165' in launched
    assert '--yaw 3.14159 --pitch 0.5585 --roll -0.02059' in launched
    assert 'base_frame:=drone_link' in launched
    assert 'realsense_config_file:=/tmp/vio-realsense-' in launched
    assert '333422300181' in result.stdout


@pytest.mark.parametrize('overrides', [
    {'VIO_MODE': 'vio'},
    {'VIO_MODE': 'vo', 'AMENT_PREFIX_PATH': '/workspaces/isaac_ros-dev/.vio-imu/test/ros_install'},
])
def test_vo_shell_refuses_imu_request_or_sourced_overlay(tmp_path, overrides):
    import os
    (tmp_path/'src/isaac_ros_common').mkdir(parents=True)
    result = subprocess.run(['bash', str(SCRIPTS/'start_vio.sh')],
                            env={**os.environ, 'ISAAC_ROS_WS': str(tmp_path), **overrides},
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 2


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
