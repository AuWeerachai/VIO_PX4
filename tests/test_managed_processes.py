"""Use disposable private process groups; never touch ROS, Docker or serial."""
import multiprocessing
import os
from pathlib import Path
import select
import shlex
import signal
import subprocess
import sys
import time
from unittest.mock import patch

import pytest

from test_vio_bootstrap import cli


def _paused_launcher(state_dir, entered, release, result):
    """Separate CLI process paused after checking that no bridge is running."""
    cfg = cli.Config(state_dir=Path(state_dir))
    find = cli.find_proc
    popen = subprocess.Popen

    def paused_find(config, name):
        record = find(config, name)
        entered.set()
        if not release.wait(10):
            raise RuntimeError('test did not release the paused launcher')
        return record

    try:
        with patch.object(cli, 'find_proc', side_effect=paused_find), \
             patch.object(cli.subprocess, 'Popen', side_effect=lambda args, **kw:
                          popen(['bash', '--noprofile', '--norc', '-c', args[-1]], **kw)):
            record = cli.start_managed(cfg, 'gps-bridge', 'sleep 60', .05)
        result.put(('ok', record))
    except Exception as exc:
        result.put(('error', str(exc)))


def test_two_cli_processes_cannot_start_or_overwrite_the_same_bridge(tmp_path):
    context = multiprocessing.get_context('spawn')
    entered, release, result = context.Event(), context.Event(), context.Queue()
    worker = context.Process(target=_paused_launcher,
                             args=(str(tmp_path), entered, release, result))
    cfg = cli.Config(state_dir=tmp_path)
    popen = subprocess.Popen
    extra = None
    worker.start()
    try:
        assert entered.wait(10), 'launcher did not reach the critical section'
        with patch.object(cli.subprocess, 'Popen', side_effect=lambda args, **kw:
                          popen(['bash', '--noprofile', '--norc', '-c', args[-1]], **kw)):
            with pytest.raises(RuntimeError, match='Another vio-launch'):
                extra = cli.start_managed(cfg, 'gps-bridge', 'sleep 60', .05)
        # A status refresh/prune or stop must not overwrite the same transaction.
        for operation in (cli.prune_state, cli.stop_managed):
            with pytest.raises(RuntimeError, match='Another vio-launch'):
                operation(cfg)
    finally:
        release.set()
        worker.join(timeout=10)
        outcome, record = result.get(timeout=5)
        try:
            if outcome == 'ok':
                assert cli.read_state(cfg)['procs'] == [record]
                with pytest.raises(RuntimeError, match='already running'):
                    cli.start_managed(cfg, 'gps-bridge', 'sleep 60', .05)
                assert cli.stop_managed(cfg) == ['gps-bridge']
                assert not cli.managed_alive(record)
                assert cli.read_state(cfg)['procs'] == []
            else:
                pytest.fail(record)
        finally:
            if outcome == 'ok':
                cli.terminate_managed(record)
            if extra is not None:
                cli.terminate_managed(extra)
            result.close()


@pytest.fixture
def owned_group(tmp_path):
    child = ("import os,signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
             "print(os.getpid(), flush=True); time.sleep(60)")
    parent = f"import subprocess,sys,time; subprocess.Popen([sys.executable, '-c', {child!r}]); time.sleep(60)"
    process = subprocess.Popen([sys.executable, '-c', parent], start_new_session=True,
                               stdout=subprocess.PIPE, text=True)
    try:
        assert select.select([process.stdout], [], [], 5)[0], 'child did not become ready'
        child_pid = int(process.stdout.readline())
        record = dict(name='test-owned', pid=process.pid,
                      startTicks=cli.process_start_ticks(process.pid),
                      bootId=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
        cfg = cli.Config(state_dir=tmp_path)
        cli.register_proc(cfg, record)
        yield cfg, record, process, child_pid
    finally:
        # Only this test's validated private group is a cleanup target.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
        process.stdout.close()


def test_stop_kills_term_ignoring_child_even_after_leader_exits(owned_group):
    cfg, record, parent, child = owned_group
    assert child in cli.managed_group(record)
    assert cli.stop_managed(cfg) == ['test-owned']
    parent.wait(timeout=5)
    assert cli.process_start_ticks(child) is None  # absent or zombie, not publishing
    assert cli.read_state(cfg)['procs'] == []


def test_prune_keeps_orphan_group_for_later_stop(owned_group):
    cfg, record, parent, child = owned_group
    os.kill(parent.pid, signal.SIGTERM)
    parent.wait(timeout=5)
    assert not cli.managed_leader_alive(record)
    assert cli.managed_alive(record)
    assert cli.prune_state(cfg) == [record]
    cli.stop_managed(cfg)
    assert cli.process_start_ticks(child) is None


def test_pid_reuse_and_previous_boot_are_never_signalled(owned_group):
    cfg, record, parent, child = owned_group
    for stale in (dict(record, startTicks=record['startTicks']-1),
                  dict(record, bootId='previous-boot')):
        with patch.object(cli.os, 'killpg') as kill:
            assert not cli.managed_alive(stale)
            cli.terminate_managed(stale)
            kill.assert_not_called()
    assert cli.managed_alive(record)


def test_failed_stop_keeps_record_and_does_not_claim_success(owned_group):
    cfg, record, parent, child = owned_group
    with patch.object(cli.os, 'killpg', side_effect=PermissionError('denied')):
        with pytest.raises(PermissionError):
            cli.stop_managed(cfg)
    assert cli.read_state(cfg)['procs'] == [record]
    assert cli.managed_alive(record)


def test_old_startup_watcher_cannot_stop_replacement_or_its_camera(owned_group):
    cfg, record, parent, child = owned_group
    # Same name but different process identity (also covers PID reuse).
    previous = dict(record, startTicks=record['startTicks']-1)
    with patch.object(cli, 'terminate_managed') as stop, \
         patch.object(cli, 'stop_visible_isaac_sessions') as camera:
        with pytest.raises(RuntimeError, match='replaced by another launch'):
            cli.stop_managed(cfg, record['name'], expected_proc=previous)
        with patch.object(cli, 'find_proc', return_value=record):
            with pytest.raises(RuntimeError, match='no longer managed by this launch'):
                cli.stop_failed_spoof(cfg, 'stale failure', expected_proc=previous)
        stop.assert_not_called()
        camera.assert_not_called()
    assert cli.read_state(cfg)['procs'] == [record]
    assert cli.managed_alive(record)


def test_registration_cannot_replace_live_ownership(owned_group):
    cfg, record, parent, child = owned_group
    with pytest.raises(RuntimeError, match='refusing to overwrite ownership'):
        cli.register_proc(cfg, record)
    assert cli.read_state(cfg)['procs'] == [record]


def test_operation_lock_is_reentrant_and_released_after_exception(tmp_path):
    cfg = cli.Config(state_dir=tmp_path)
    with pytest.raises(RuntimeError, match='test exception'):
        with cli.process_operation(cfg):
            cli.write_state(cfg, {'procs': []})
            assert cli.read_state(cfg) == {'procs': []}
            raise RuntimeError('test exception')
    # Open a new descriptor, as another CLI would. A leaked lock would block it.
    with (cfg.state_dir/'process-manager.lock').open('a') as lock:
        cli.fcntl.flock(lock.fileno(), cli.fcntl.LOCK_EX | cli.fcntl.LOCK_NB)
        cli.fcntl.flock(lock.fileno(), cli.fcntl.LOCK_UN)
    assert cli.read_state(cfg) == {'procs': []}


def test_proc_inspection_failure_is_not_pruned_as_dead(owned_group):
    cfg, record, parent, child = owned_group
    with patch.object(cli, 'process_identity', side_effect=PermissionError('denied')):
        with pytest.raises(PermissionError):
            cli.prune_state(cfg)
    assert cli.read_state(cfg)['procs'] == [record]


@pytest.mark.parametrize('payload', ['', '{', '[]', '{}', '{"procs": {}}',
                                    '{"procs": [null]}', '{"procs": [{}]}'])
def test_corrupt_state_never_means_no_running_bridge(owned_group, payload):
    cfg, record, parent, child = owned_group
    read = Path.read_text

    def corrupt_state(path, *args, **kwargs):
        return payload if path == cfg.state_file else read(path, *args, **kwargs)

    with patch.object(Path, 'read_text', new=corrupt_state), \
         patch.object(cli.subprocess, 'Popen') as start, \
         patch.object(cli, 'stop_visible_isaac_sessions') as camera:
        for operation in (cli.read_state, cli.require_no_publishers_for_camera_restart,
                          cli.cmd_stop):
            with pytest.raises(RuntimeError, match='Cannot verify managed process state'):
                operation(cfg)
        with pytest.raises(RuntimeError, match='Cannot verify managed process state'):
            cli.start_managed(cfg, 'gps-bridge', 'sleep 60', .05)
        start.assert_not_called()
        camera.assert_not_called()
    assert cli.read_state(cfg)['procs'] == [record]
    assert cli.managed_alive(record)


@pytest.mark.parametrize('error', [PermissionError('denied'), OSError('I/O error'),
                                  UnicodeError('invalid encoding')])
def test_unreadable_state_blocks_camera_restart_and_stop(owned_group, error):
    cfg, record, parent, child = owned_group
    read = Path.read_text

    def unreadable_state(path, *args, **kwargs):
        if path == cfg.state_file:
            raise error
        return read(path, *args, **kwargs)

    with patch.object(Path, 'read_text', new=unreadable_state), \
         patch.object(cli, 'stop_visible_isaac_sessions') as camera:
        with pytest.raises(RuntimeError, match='Cannot verify managed process state'):
            cli.require_no_publishers_for_camera_restart(cfg)
        with pytest.raises(RuntimeError, match='Cannot verify managed process state'):
            cli.cmd_stop(cfg)
        camera.assert_not_called()
    assert cli.read_state(cfg)['procs'] == [record]
    assert cli.managed_alive(record)


@pytest.mark.parametrize('change', [dict(pid=True), dict(pid='123'), dict(pid=0),
                                    dict(startTicks=None), dict(startTicks=-1),
                                    dict(name=''), dict(bootId=[]), 'duplicate'])
def test_invalid_ownership_records_fail_closed(owned_group, change):
    cfg, record, parent, child = owned_group
    records = [record, record] if change == 'duplicate' else [{**record, **change}]
    with patch.object(cli.json, 'loads', return_value={'procs': records}):
        with pytest.raises(RuntimeError, match='Cannot verify managed process state'):
            cli.prune_state(cfg)
    assert cli.read_state(cfg)['procs'] == [record]
    assert cli.managed_alive(record)


def test_stop_all_does_not_touch_camera_if_gps_group_cannot_stop(tmp_path):
    with patch.object(cli, 'stop_managed', side_effect=RuntimeError('still running')), \
         patch.object(cli, 'stop_visible_isaac_sessions') as camera:
        with pytest.raises(RuntimeError, match='still running'):
            cli.cmd_stop(cli.Config(state_dir=tmp_path))
    camera.assert_not_called()


def test_start_registers_ownership_and_zombie_is_not_healthy(tmp_path):
    cfg = cli.Config(state_dir=tmp_path)
    popen = subprocess.Popen
    # Do not source the user's ROS login profile in a disposable process test.
    with patch.object(cli.subprocess, 'Popen', side_effect=lambda args, **kw:
                      popen(['bash', '--noprofile', '--norc', '-c', args[-1]], **kw)):
        with pytest.raises(RuntimeError, match='failed to start'):
            cli.start_managed(cfg, 'test-short-lived', 'exit 0', startup_check_s=.2)
    assert cli.read_state(cfg)['procs'] == []


def test_startup_failure_cleans_up_a_surviving_child(tmp_path):
    cfg = cli.Config(state_dir=tmp_path)
    child = ("import os,signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
             "print(os.getpid(),flush=True); time.sleep(60)")
    parent = (f"import subprocess,sys,time; subprocess.Popen([sys.executable, '-c', {child!r}]); "
              "time.sleep(.1)")
    command = f'{shlex.quote(sys.executable)} -c {shlex.quote(parent)}'
    popen = subprocess.Popen
    with patch.object(cli.subprocess, 'Popen', side_effect=lambda args, **kw:
                      popen(['bash', '--noprofile', '--norc', '-c', args[-1]], **kw)):
        with pytest.raises(RuntimeError, match='failed to start'):
            cli.start_managed(cfg, 'test-orphan', command, startup_check_s=.5)
    pids = [int(line) for line in (cfg.log_dir/'test-orphan.log').read_text().splitlines()
            if line.isdecimal()]
    assert len(pids) == 1
    assert cli.process_start_ticks(pids[0]) is None
    assert cli.read_state(cfg)['procs'] == []
