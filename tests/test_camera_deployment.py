import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'bootstrap/jetson-config/isaac_ros_overrides/scripts'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cli = load('camera_deployment_cli_test', ROOT / 'scripts/vio-launch/vio_launch.py')
verify = load('verify_camera_deployment_test', ROOT / 'bootstrap/verify_camera_deployment.py')
environment = load('camera_deployment_environment_test', SCRIPTS / 'check_vo_environment.py')
MANIFEST = json.loads((ROOT / 'scripts/vio-launch/camera_release.json').read_text())


@pytest.fixture
def deployment(tmp_path):
    workspace = tmp_path / 'isaac'
    scripts = workspace / 'src/isaac_ros_common/scripts'
    scripts.mkdir(parents=True)
    for name, digest in MANIFEST.items():
        source = SCRIPTS.parent / name if name.startswith('.') else SCRIPTS / name
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
        shutil.copy(source, scripts / name)
    return cli.Config(isaac_ros_dir=workspace, state_dir=tmp_path / 'state',
                      camera_serial='333422300181'), scripts


def test_complete_deployment_matches_source_manifest_and_copies(deployment):
    cfg, _ = deployment
    assert len(MANIFEST) == 7
    cli.require_camera_deployment(cfg)
    assert verify.verify(ROOT, cfg.isaac_ros_dir) == 7


@pytest.mark.parametrize('name', sorted(MANIFEST))
@pytest.mark.parametrize('fault', ['missing', 'modified'])
def test_mixed_deployment_blocks_camera_and_identity_request(deployment, name, fault):
    cfg, scripts = deployment
    path = scripts / name
    if fault == 'missing':
        path.unlink()
    else:
        path.write_text(path.read_text() + '\n# wrong deployment\n')
    with patch.object(cli.subprocess, 'run') as run, \
         patch.object(cli, 'start_visible_isaac_session') as start:
        with pytest.raises(RuntimeError, match='Camera deployment mismatch'):
            cli.cmd_vio(cfg)
        with pytest.raises(RuntimeError, match='Camera deployment mismatch'):
            cli.require_camera_ready(cfg, started_only=True)
    start.assert_not_called()
    run.assert_not_called()


def test_missing_manifest_fails_closed(deployment, tmp_path):
    cfg, _ = deployment
    with patch.object(cli, '_CLI_FILE', tmp_path / 'vio_launch.py'):
        with pytest.raises(RuntimeError, match='Camera deployment mismatch'):
            cli.require_camera_deployment(cfg)


def test_bootstrap_verifier_reports_mismatch(deployment):
    cfg, scripts = deployment
    (scripts / 'check_camera_ready.py').write_text('# newer incompatible helper\n')
    result = subprocess.run([
        sys.executable, str(ROOT / 'bootstrap/verify_camera_deployment.py'),
        '--repo', str(ROOT), '--isaac-ws', str(cfg.isaac_ros_dir),
    ], text=True, capture_output=True)
    assert result.returncode == 1
    assert 'check_camera_ready.py' in result.stderr


def test_no_full_workspace_overlay_in_camera_entrypoints():
    for function in (cli.cmd_vio, cli.start_rviz_session, cli.require_camera_ready):
        source = inspect.getsource(function)
        assert '${ISAAC_ROS_WS}/install/setup.bash' not in source
        assert 'vo_environment.sh' in source
    bootstrap = (ROOT / 'bootstrap/bootstrap_jetson.sh').read_text()
    assert 'vo_environment.sh check_vo_environment.py' in bootstrap
    assert 'verify_camera_deployment.py' in bootstrap


def test_vo_environment_rejects_mixed_prefixes_and_versions(tmp_path):
    base = tmp_path / 'base'
    for package in ('realsense2_camera', 'realsense2_camera_msgs'):
        manifest = base / 'share' / package / 'package.xml'
        manifest.parent.mkdir(parents=True)
        manifest.write_text('<package><version>4.51.1</version></package>')
    assert environment.validate_packages(base, base, base) == '4.51.1'
    with pytest.raises(RuntimeError, match='matched base-image'):
        environment.validate_packages(base, tmp_path / 'overlay', base)
    (base / 'share/realsense2_camera_msgs/package.xml').write_text(
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
    workspace = tmp_path / 'isaac'
    overlay = workspace / 'install/isaac_ros_visual_slam/share/isaac_ros_visual_slam/package.bash'
    overlay.parent.mkdir(parents=True)
    overlay.write_text('export VSLAM_OVERLAY_LOADED=yes\n')
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
    contaminated = str(workspace / 'install/realsense2_camera_msgs')
    env = {**os.environ, 'ISAAC_ROS_WS': str(workspace),
           'LD_LIBRARY_PATH': f'{contaminated}/lib:/usr/local/cuda/lib64:/opt/ros/humble/lib',
           'AMENT_PREFIX_PATH': f'{workspace}/install:{contaminated}:/opt/ros/humble',
           'PYTHONPATH': f'{workspace}/.vio-imu/old/python:/opt/ros/humble/python'}
    result = subprocess.run(['bash', '-c', command, 'test', str(SCRIPTS / 'vo_environment.sh')],
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
