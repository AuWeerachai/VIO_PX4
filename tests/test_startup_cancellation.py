"""Cancel slow startup using a second CLI; no ROS, Docker or serial required."""
from concurrent.futures import ThreadPoolExecutor
import errno
import subprocess
import threading
import time
from unittest.mock import patch

import pytest

from test_vio_bootstrap import cli
from test_managed_processes import no_login_popen


@pytest.mark.parametrize('probe_fails', [False, True])
def test_stop_all_during_camera_probe_prevents_late_restart(tmp_path, probe_fails):
    cfg = cli.Config(state_dir=tmp_path/'state', vio_px4_dir=tmp_path/'repo',
                     camera_serial='333422300181', rviz_enabled=True)
    entered, release, stopped = threading.Event(), threading.Event(), threading.Event()

    def slow_probe(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        if probe_fails:
            raise RuntimeError('probe returned after Stop all')

    with patch.object(cli.shutil, 'which', return_value='/bin/mock'), \
         patch.object(cli, 'tmux_session_exists', return_value=False), \
         patch.object(cli, 'isaac_container_running', side_effect=lambda: not stopped.is_set()), \
         patch.object(cli, 'start_visible_isaac_session') as camera_start, \
         patch.object(cli, 'tmux_command') as command, \
         patch.object(cli, 'require_camera_ready', side_effect=slow_probe), \
         patch.object(cli, 'stop_visible_isaac_sessions', side_effect=stopped.set) as camera_stop, \
         patch.object(cli, 'start_rviz_session') as rviz, \
         patch.object(cli, 'cmd_gps') as gps, \
         ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(cli.cmd_run, cfg, 'a')
        try:
            assert entered.wait(8)
            # Status and Stop are available, but a duplicate starter is not.
            assert cli.prune_state(cfg) == []
            with pytest.raises(cli.ProcessOperationBusy, match='startup is in progress'):
                cli.cmd_vio(cfg)
            cli.cmd_shutdown(cfg)
            assert stopped.is_set()
        finally:
            release.set()
        with pytest.raises(cli.StartupCancelled):
            pending.result(timeout=5)
        camera_start.assert_called_once()
        command.assert_called_once()
        camera_stop.assert_called_once()  # no second cleanup from the old starter
        gps.assert_not_called()
        rviz.assert_not_called()
    assert not (cfg.log_dir/'camera-startup-failure.log').exists()
    # The cancelled startup releases its lease; an intentional restart works.
    with cli.startup_operation(cfg):
        cli.check_startup_current(cfg)


def test_stop_during_docker_wait_never_sends_launch_command(tmp_path):
    cfg = cli.Config(state_dir=tmp_path/'state', vio_px4_dir=tmp_path/'repo',
                     camera_serial='333422300181')
    entered, release = threading.Event(), threading.Event()

    def slow_docker_check():
        entered.set()
        assert release.wait(10)
        return False

    with patch.object(cli.shutil, 'which', return_value='/bin/mock'), \
         patch.object(cli, 'tmux_session_exists', return_value=False), \
         patch.object(cli, 'isaac_container_running', side_effect=slow_docker_check), \
         patch.object(cli, 'start_visible_isaac_session'), \
         patch.object(cli, 'tmux_command') as command, \
         patch.object(cli, 'stop_visible_isaac_sessions') as stop, \
         ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(cli.cmd_vio, cfg, True)
        try:
            assert entered.wait(5)
            cli.cmd_stop(cfg)
            stop.assert_called_once()
        finally:
            release.set()
        with pytest.raises(cli.StartupCancelled):
            pending.result(timeout=5)
        command.assert_not_called()


def test_stop_during_managed_readiness_kills_registered_group(tmp_path):
    cfg = cli.Config(state_dir=tmp_path/'state', vio_px4_dir=tmp_path/'repo')
    registered = threading.Event()
    popen, register = subprocess.Popen, cli.register_proc
    record = None

    def notify_registered(*args):
        result = register(*args)
        registered.set()
        return result

    def slow_start():
        with cli.startup_operation(cfg):
            return cli.start_managed(cfg, 'gps-bridge', 'sleep 60', startup_check_s=30)

    with patch.object(cli.subprocess, 'Popen', side_effect=lambda args, **kw:
                      no_login_popen(popen, args, **kw)), \
         patch.object(cli, 'register_proc', side_effect=notify_registered), \
         patch.object(cli, 'stop_visible_isaac_sessions'), \
         ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(slow_start)
        try:
            assert registered.wait(5)
            # The registration critical section may still be unwinding.
            deadline = time.monotonic()+2
            while True:
                try:
                    record = cli.read_state(cfg)['procs'][0]
                    cli.cmd_stop(cfg)
                    break
                except cli.ProcessOperationBusy:
                    assert time.monotonic() < deadline
                    time.sleep(.01)
            with pytest.raises(cli.StartupCancelled):
                pending.result(timeout=5)
            assert not cli.managed_alive(record)
            assert cli.read_state(cfg)['procs'] == []
        finally:
            if record is not None:
                cli.terminate_managed(record)


def test_cancellation_does_not_need_free_disk_space(tmp_path):
    cfg = cli.Config(state_dir=tmp_path/'state', vio_px4_dir=tmp_path/'repo')
    entered, release = threading.Event(), threading.Event()

    def start():
        with cli.startup_operation(cfg):
            entered.set()
            assert release.wait(10)
            cli.check_startup_current(cfg)

    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(start)
        try:
            assert entered.wait(5)
            with patch.object(cli, 'atomic_write_json', side_effect=OSError(errno.ENOSPC, 'full')), \
                 patch.object(cli, 'stop_visible_isaac_sessions') as stop:
                cli.cmd_stop(cfg)
                stop.assert_called_once()
        finally:
            release.set()
        with pytest.raises(cli.StartupCancelled):
            pending.result(timeout=5)
