import importlib.util
import json
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


@pytest.mark.parametrize('path,active', [('a', 'ev-bridge'), ('a', 'mavros')])
def test_conflicting_path_is_rejected_before_touching_camera(path, active):
    with patch.object(cli, 'find_proc', side_effect=lambda cfg, name: {'pid': 123} if name == active else None), \
         patch.object(cli, 'cmd_vio') as camera:
        with pytest.raises(RuntimeError, match='already running'):
            cli.cmd_run(cli.Config(), path)
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


@pytest.mark.parametrize('command', ['cmd_gps'])
def test_direct_bridge_commands_cannot_bypass_camera_identity(command):
    with patch.object(cli, 'find_proc', return_value=None), \
         patch.object(cli, 'require_camera_ready', side_effect=RuntimeError('Wrong VIO camera')), \
         patch.object(cli, 'start_managed') as start, \
         patch.object(cli, 'serial_users') as port_check:
        with pytest.raises(RuntimeError, match='Wrong VIO camera'):
            getattr(cli, command)(cli.Config(camera_serial='333422300181'))
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


def test_heading_menu_has_initial_reference_but_no_second_mounting_correction(capsys):
    cfg = cli.Config()
    with patch.object(cli, '_clear'), patch.object(cli, 'pause'), \
         patch.object(cli, 'prompt_line', side_effect=['off', 'px4']) as prompt, \
         patch.object(cli, 'save_config') as save:
        changed = cli.configure_heading(cfg)
    assert changed.heading_source == 'px4'
    assert prompt.call_count == 2
    labels = [call.args[0] for call in prompt.call_args_list]
    assert labels[1] == 'Initial heading reference (px4/manual)'
    assert all('declination' not in label.lower() and 'child-to-body' not in label.lower()
               for label in labels)
    assert "drone_link's forward direction" in capsys.readouterr().out
    save.assert_called_once_with(changed)


@pytest.mark.parametrize('enabled', [False, True])
def test_legacy_mounting_correction_is_not_loaded_or_saved(tmp_path, enabled):
    path = tmp_path/'config.json'
    saved = dict(child_to_body_yaw_deg=180., external_yaw_enabled=enabled,
                 camera_x_m=-.26, camera_y_m=.01, camera_z_m=-.165,
                 camera_roll_deg=.35, camera_pitch_deg=30.5, camera_yaw_deg=180.)
    path.write_text(json.dumps(saved))
    with patch.object(cli, 'user_config_path', return_value=path):
        cfg = cli.load_config()
        assert not hasattr(cfg, 'child_to_body_yaw_deg')
        for key, value in saved.items():
            if key != 'child_to_body_yaw_deg':
                assert getattr(cfg, key) == value
        cli.save_config(cfg)
    assert 'child_to_body_yaw_deg' not in json.loads(path.read_text())


def test_external_heading_settings_round_trip(tmp_path):
    with patch.object(cli, 'user_config_path', return_value=tmp_path/'config.json'):
        cfg = cli.apply_dict(cli.Config(), dict(external_yaw_enabled=True,
                            external_yaw_noise_deg=8., external_yaw_rate_hz=40.,
                            external_yaw_max_age_s=.15, external_yaw_startup_tolerance_deg=5.))
        cli.save_config(cfg)
        reloaded = cli.load_config()
        for field in ('external_yaw_enabled', 'external_yaw_noise_deg',
                      'external_yaw_rate_hz', 'external_yaw_max_age_s',
                      'external_yaw_startup_tolerance_deg'):
            assert getattr(cfg, field) == getattr(reloaded, field)
    assert cli.Config().external_yaw_enabled is False


def test_heading_default_is_one_degree_without_overwriting_existing_config():
    assert cli.Config().external_yaw_noise_deg == 1.
    assert cli.apply_dict(cli.Config(), {'external_yaw_noise_deg': 5.}).external_yaw_noise_deg == 5.
    assert cli.Config().external_yaw_max_age_s == .2


@pytest.mark.parametrize('source', ['px4', 'manual'])
def test_heading_menu_enables_yaw_without_touching_physical_gates(source):
    answers = ['on', source] + (['156'] if source == 'manual' else [])
    answers += ['5', '30', '.2', '10']
    cfg = cli.Config()
    with patch.object(cli, '_clear'), patch.object(cli, 'pause'), \
         patch.object(cli, 'prompt_line', side_effect=answers), patch.object(cli, 'save_config'):
        result = cli.configure_heading(cfg)
    assert result.external_yaw_enabled
    assert result.heading_source == source
    assert result.gate_max_speed_m_s == cfg.gate_max_speed_m_s
    assert result.gate_max_acceleration_m_s2 == cfg.gate_max_acceleration_m_s2
    assert result.camera_pitch_deg == cfg.camera_pitch_deg
    assert result.camera_yaw_deg == cfg.camera_yaw_deg
    assert result.manual_heading_deg == (156. if source == 'manual' else cfg.manual_heading_deg)


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


def test_path_a_defers_health_and_removed_path_b_never_touches_camera(tmp_path):
    cfg = cli.Config(isaac_ros_dir=tmp_path)
    with patch.object(cli, 'cmd_vio') as camera, \
         patch.object(cli, 'find_proc', return_value=None), \
         patch.object(cli, 'cmd_gps'):
        cli.cmd_run(cfg, 'a')
        camera.assert_called_with(cfg, defer_health=True)
        camera.reset_mock()
        for name in ('b', 'ev', 'vision', 'external-vision'):
            with pytest.raises(RuntimeError, match='Path B was removed'):
                cli.cmd_run(cfg, name)
        camera.assert_not_called()


def test_path_b_has_no_runtime_or_bootstrap_entry_points():
    assert not hasattr(cli, 'cmd_ev_bridge')
    assert not hasattr(cli.Config(), 'mavros_dir')
    assert 'mavros_ev_bridge' not in (ROOT/'src/vio_px4_bridge/setup.py').read_text()
    assert '<exec_depend>mavros</exec_depend>' not in (ROOT/'src/vio_px4_bridge/package.xml').read_text()
    assert 'MAVROS' not in (ROOT/'bootstrap/bootstrap_jetson.sh').read_text()
    assert not (ROOT/'src/vio_px4_bridge/vio_px4_bridge/mavros_ev_bridge.py').exists()
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(['ev-bridge'])


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
    # This shell/argument test does not have ROS installed. The actual ABI guard
    # is tested separately below and against the Jetson image during deployment.
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


def test_vo_environment_rejects_mixed_prefixes_and_versions(tmp_path):
    base = tmp_path/'base'
    for package in ('realsense2_camera', 'realsense2_camera_msgs'):
        manifest = base/'share'/package/'package.xml'
        manifest.parent.mkdir(parents=True)
        manifest.write_text('<package><version>4.51.1</version></package>')
    assert environment.validate_packages(base, base, base) == '4.51.1'
    with pytest.raises(RuntimeError, match='matched base-image'):
        environment.validate_packages(base, tmp_path/'overlay', base)
    (base/'share/realsense2_camera_msgs/package.xml').write_text(
        '<package><version>4.57.6</version></package>')
    with pytest.raises(RuntimeError, match='versions differ'):
        environment.validate_packages(base, base, base)


def test_vo_environment_checks_native_linkage_not_only_package_names():
    environment.validate_linkage(
        'librealsense2_camera_msgs__rosidl_typesupport_cpp.so => '
        '/opt/ros/humble/lib/librealsense2_camera_msgs__rosidl_typesupport_cpp.so (0xabcd)')
    with pytest.raises(RuntimeError, match='Incompatible RealSense message library'):
        environment.validate_linkage(
            'librealsense2_camera_msgs__rosidl_typesupport_cpp.so => '
            '/workspaces/isaac_ros-dev/install/realsense2_camera_msgs/lib/typesupport.so (0xabcd)')
    with pytest.raises(RuntimeError, match='Missing camera dependency'):
        environment.validate_linkage('librealsense2.so.2.55 => not found')


def test_vo_environment_filters_entire_workspace_and_preserves_cuda_paths(tmp_path):
    import os
    workspace = tmp_path/'isaac'
    overlay = workspace/'install/isaac_ros_visual_slam/share/isaac_ros_visual_slam/package.bash'
    overlay.parent.mkdir(parents=True)
    overlay.write_text('export VSLAM_OVERLAY_LOADED=yes\n')
    # Mock only the absent system ROS installation; execute the real filtering
    # and package-local overlay sourcing code, starting with contaminated paths.
    command = r'''
set -eu
source() {
  if [[ "$1" == /opt/ros/humble/setup.bash ]]; then
    export BASE_ROS_LOADED=yes
  else
    builtin source "$@"
  fi
}
builtin source "$1"
[[ "$BASE_ROS_LOADED:$VSLAM_OVERLAY_LOADED" == yes:yes ]]
[[ "$LD_LIBRARY_PATH" == /usr/local/cuda/lib64:/opt/ros/humble/lib ]]
[[ "$AMENT_PREFIX_PATH" == /opt/ros/humble ]]
[[ "$PYTHONPATH" == /opt/ros/humble/python ]]
[[ $- == *u* ]]
'''
    contaminated = str(workspace/'install/realsense2_camera_msgs')
    env = {**os.environ, 'ISAAC_ROS_WS': str(workspace),
           'LD_LIBRARY_PATH': f'{contaminated}/lib:/usr/local/cuda/lib64:/opt/ros/humble/lib',
           'AMENT_PREFIX_PATH': f'{workspace}/install:{contaminated}:/opt/ros/humble',
           'PYTHONPATH': f'{workspace}/.vio-imu/old/python:/opt/ros/humble/python'}
    result = subprocess.run(['bash', '-c', command, 'test', str(SCRIPTS/'vo_environment.sh')],
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_no_full_workspace_overlay_in_camera_entrypoints():
    import inspect
    for function in (cli.cmd_vio, cli.start_rviz_session, cli.require_camera_ready):
        assert '${ISAAC_ROS_WS}/install/setup.bash' not in inspect.getsource(function)
    assert 'vo_environment.sh' in inspect.getsource(cli.cmd_vio)
    assert 'vo_environment.sh' in inspect.getsource(cli.require_camera_ready)


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
