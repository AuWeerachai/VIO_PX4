#!/usr/bin/env python3
"""Private process-group supervisor with bounded stdout/stderr retention.

This process outlives the operator CLI. Children share its private group so the
existing TERM/KILL ownership checks stop the logger and all command descendants.
"""
import argparse
import os
from pathlib import Path
import select
import shlex
import signal
import subprocess
import sys
import time

LOG_BYTES = 5 * 1024 * 1024


class BoundedLog:
    """Keep the latest two files, each at most limit bytes (also for long lines)."""
    def __init__(self, path, limit=LOG_BYTES):
        if limit <= 0:
            raise ValueError('log limit must be positive')
        self.path = Path(path)
        self.backup = self.path.with_suffix(self.path.suffix + '.1')
        self.limit = limit
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Old, unbounded logs must not turn the retained backup into a large file.
        for existing in (self.path, self.backup):
            try:
                with existing.open('rb') as stream:
                    if os.fstat(stream.fileno()).st_size <= limit:
                        continue
                    stream.seek(-limit, os.SEEK_END)
                    tail = stream.read(limit)
                temporary = existing.with_name(f'.{existing.name}.bounded-{os.getpid()}')
                try:
                    temporary.write_bytes(tail)
                    os.replace(temporary, existing)
                finally:
                    temporary.unlink(missing_ok=True)
            except FileNotFoundError:
                pass
        self.stream = self.path.open('ab', buffering=0)
        self.size = os.fstat(self.stream.fileno()).st_size

    def write(self, data):
        remaining = memoryview(data)
        while remaining:
            if self.size >= self.limit:
                self.stream.close()
                os.replace(self.path, self.backup)
                self.stream = self.path.open('ab', buffering=0)
                self.size = 0
            count = self.stream.write(remaining[:self.limit-self.size])
            if not count:
                raise OSError('log write made no progress')
            self.size += count
            remaining = remaining[count:]

    def close(self):
        self.stream.close()


def group_has_children():
    """Ignore zombies; only this supervisor's already-verified private group."""
    pid = os.getpid()
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == pid:
            continue
        try:
            fields = (entry/'stat').read_text().rsplit(')', 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if fields[0] != 'Z' and int(fields[2]) == pid:
            return True
    return False


def supervise(log_path, name, command):
    if os.getpgrp() != os.getpid() or os.getsid(0) != os.getpid():
        raise RuntimeError('Managed command requires its own private session')
    log = BoundedLog(log_path)
    header = (f"\n===== {time.strftime('%Y-%m-%dT%H:%M:%S')} start {name} =====\n"
              f"$ {shlex.join(command)}\n")
    log.write(header.encode())
    requested_signal = None

    def requested(signum, _frame):
        nonlocal requested_signal
        requested_signal = requested_signal or signum

    signal.signal(signal.SIGTERM, requested)
    signal.signal(signal.SIGINT, requested)
    command_process = None
    try:
        command_process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        pipe = command_process.stdout
        os.set_blocking(pipe.fileno(), False)
        pipe_open = True
        stopping_at = None
        logging_failed = False
        while True:
            result = command_process.poll()
            if (requested_signal is not None or result is not None) and stopping_at is None:
                # A dead ros2/bash leader must not leave a publisher or pipe-holding
                # child behind. Ignore our own forwarded signal, drain its last logs.
                stopping_at = time.monotonic()
                os.killpg(os.getpid(), requested_signal or signal.SIGTERM)
            if pipe_open:
                readable, _, _ = select.select([pipe], [], [], .05)
                if readable:
                    data = os.read(pipe.fileno(), 65536)
                    if not data:
                        pipe.close()
                        pipe_open = False
                    elif not logging_failed:
                        try:
                            log.write(data)
                        except OSError as exc:
                            # Disk failure must not block navigation on a full pipe
                            # or kill it with SIGPIPE. Keep draining/discarding output.
                            logging_failed = True
                            try:
                                print(f'vio-launch logging disabled: {exc}', file=sys.stderr, flush=True)
                            except OSError:
                                pass
            else:
                time.sleep(.02)
            if stopping_at is not None:
                children = group_has_children()
                if result is not None and not children and not pipe_open:
                    return result if result >= 0 else 128-result
                if children and (result is not None or time.monotonic()-stopping_at >= 1.0):
                    # If the actual command has exited, there is no healthy
                    # publisher to preserve. Do not report its logger as alive
                    # for an extra grace period while orphan children linger.
                    # Includes this supervisor; a private session was checked above.
                    os.killpg(os.getpid(), signal.SIGKILL)
    finally:
        log.close()
        if command_process is not None and group_has_children():
            os.killpg(os.getpid(), signal.SIGKILL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('missing command')
    return supervise(args.log, args.name, command)


if __name__ == '__main__':
    raise SystemExit(main())
