"""Bound stdout/stderr during a live run, independent of the operator CLI."""
import multiprocessing
from pathlib import Path
import shlex
import subprocess
import sys
import time
from unittest.mock import patch

import pytest

from test_vio_bootstrap import cli, load, ROOT
from test_managed_processes import no_login_popen

runner = load('vio_managed_logging_test', ROOT/'scripts/vio-launch/managed_process.py')


def test_rotation_bounds_even_a_single_oversized_line(tmp_path):
    log = runner.BoundedLog(tmp_path/'test.log', limit=1024)
    data = bytes(range(256))*19
    try:
        log.write(data)
    finally:
        log.close()
    assert log.path.stat().st_size == len(data) % 1024
    assert log.backup.stat().st_size == 1024
    assert log.backup.read_bytes()+log.path.read_bytes() == data[3*1024:]


def test_oversized_existing_files_are_trimmed_to_latest_bytes(tmp_path):
    path = tmp_path/'test.log'
    path.write_bytes(b'a'*2048+b'current')
    path.with_suffix('.log.1').write_bytes(b'b'*4096+b'previous')
    log = runner.BoundedLog(path, limit=1024)
    log.close()
    assert path.stat().st_size == 1024
    assert log.backup.stat().st_size == 1024
    assert path.read_bytes().endswith(b'current')
    assert log.backup.read_bytes().endswith(b'previous')


def _detached_writer(state_dir, result):
    cfg = cli.Config(state_dir=Path(state_dir))
    code = ('import sys,time; time.sleep(.3); '
            'sys.stdout.buffer.write(b"x"*(12*1024*1024)); '
            'sys.stderr.write("STDERR-END\\n"); sys.stdout.flush(); '
            'print("LIVE-END",flush=True); time.sleep(60)')
    command = f'{shlex.quote(sys.executable)} -c {shlex.quote(code)}'
    popen = subprocess.Popen
    with patch.object(cli.subprocess, 'Popen', side_effect=lambda args, **kw:
                      no_login_popen(popen, args, **kw)):
        result.put(cli.start_managed(cfg, 'writer', command, startup_check_s=.1))


def test_live_logging_stays_bounded_after_cli_exits(tmp_path):
    context = multiprocessing.get_context('spawn')
    result = context.Queue()
    launcher = context.Process(target=_detached_writer, args=(str(tmp_path), result))
    record = None
    cfg = cli.Config(state_dir=tmp_path)
    launcher.start()
    try:
        record = result.get(timeout=10)
        launcher.join(timeout=5)
        assert launcher.exitcode == 0
        path = Path(record['logFile'])
        deadline = time.monotonic()+8
        # The startup header quotes the command, including its marker. Wait
        # for the actual final output line, not that quoted source text.
        while not path.exists() or not path.read_bytes().endswith(b'LIVE-END\n'):
            assert time.monotonic() < deadline
            time.sleep(.05)
        assert cli.managed_alive(record), 'writer must still be running'
        assert path.stat().st_size <= runner.LOG_BYTES
        assert path.with_suffix('.log.1').stat().st_size == runner.LOG_BYTES
        assert b'STDERR-END' in path.read_bytes()
        cli.stop_managed(cfg)
        assert not cli.managed_alive(record)
    finally:
        if record is not None:
            cli.terminate_managed(record)
        launcher.join(timeout=5)
        result.close()


def test_leaving_cli_closes_ssh_style_output_pipes(tmp_path):
    code = f'''
import sys, subprocess
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, {str(ROOT/'scripts/vio-launch')!r})
import vio_launch as cli
popen = subprocess.Popen
def spawn(args, **kwargs):
    return popen([*args[:-3], 'bash', '--noprofile', '--norc', '-c', args[-1]], **kwargs)
with patch.object(cli.subprocess, 'Popen', side_effect=spawn):
    cli.start_managed(cli.Config(state_dir=Path({str(tmp_path)!r})), 'writer', 'sleep 60', .1)
print('CLI_EXITED', flush=True)
'''
    launcher = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
    cfg = cli.Config(state_dir=tmp_path)
    try:
        stdout, stderr = launcher.communicate(timeout=5)
        assert launcher.returncode == 0, stderr
        assert stdout.strip() == 'CLI_EXITED'
        assert len(cli.prune_state(cfg)) == 1  # publisher survives, SSH pipes do not
    finally:
        cli.stop_managed(cfg)
        launcher.communicate(timeout=5)


def test_disk_failure_does_not_block_or_kill_running_command(tmp_path):
    marker = tmp_path/'child-finished'
    code = (f'import sys; from pathlib import Path; '
            f'sys.stdout.buffer.write(b"x"*(8*1024*1024)); sys.stdout.flush(); '
            f'Path({str(marker)!r}).touch()')
    # An isolated private session simulates ENOSPC after the initial header.
    supervisor_code = f'''
import errno, importlib.util, sys
spec = importlib.util.spec_from_file_location('runner', {runner.__file__!r})
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
original = runner.BoundedLog.write
calls = 0
def full_disk(self, data):
    global calls
    calls += 1
    if calls > 1:
        raise OSError(errno.ENOSPC, 'simulated full disk')
    return original(self, data)
runner.BoundedLog.write = full_disk
raise SystemExit(runner.supervise({str(tmp_path/'test.log')!r}, 'test', [sys.executable, '-c', {code!r}]))
'''
    process = subprocess.Popen([sys.executable, '-c', supervisor_code],
                               start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    record = dict(name='test', pid=process.pid, startTicks=cli.process_start_ticks(process.pid))
    try:
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, stderr
        assert marker.exists()
        assert stderr.count(b'logging disabled:') == 1
    finally:
        cli.terminate_managed(record)
        process.wait(timeout=5)


def test_supervisor_refuses_a_shared_process_group(tmp_path):
    with pytest.raises(RuntimeError, match='private session'):
        runner.supervise(tmp_path/'test.log', 'unsafe', ['true'])
