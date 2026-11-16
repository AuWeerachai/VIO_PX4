"""Use disposable private process groups; never touch ROS, Docker or serial."""
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


def test_proc_inspection_failure_is_not_pruned_as_dead(owned_group):
    cfg, record, parent, child = owned_group
    with patch.object(cli, 'process_identity', side_effect=PermissionError('denied')):
        with pytest.raises(PermissionError):
            cli.prune_state(cfg)
    assert cli.read_state(cfg)['procs'] == [record]


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
