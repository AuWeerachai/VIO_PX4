"""Regress Ctrl+C finalization using an isolated localhost ROS domain, no hardware."""
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import tempfile
import time


def main():
    folder = Path(tempfile.mkdtemp(prefix='extrinsic-interrupt-test-'))/'capture'
    env = dict(os.environ, ROS_DOMAIN_ID='198', ROS_LOCALHOST_ONLY='1',
               EXTRINSIC_CALIBRATION_TEST_DATA='1', OPENBLAS_NUM_THREADS='1')
    import extrinsic_calibration
    proc = subprocess.Popen([sys.executable, '-s', '-u', extrinsic_calibration.__file__,
                             'capture', '--height-m', '.47', '--grid-frame', 'map', '--seconds', '30',
                             '--telemetry-port', '14794', '--output', str(folder)],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            start_new_session=True)
    buffer, interrupted = b'', False
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)
    deadline = time.monotonic()+15
    try:
        while proc.poll() is None and time.monotonic() < deadline:
            for key, _ in selector.select(.2):
                buffer += os.read(key.fileobj.fileno(), 8192)
            if b'RECORDING.' in buffer and not interrupted:
                os.kill(proc.pid, signal.SIGINT)
                interrupted = True
        if proc.poll() is None:
            raise AssertionError('observer did not finish after Ctrl+C')
        output, _ = proc.communicate(timeout=2)
        buffer += output
        print(buffer.decode(errors='replace'))
        meta = json.loads((folder/'capture.json').read_text())
        assert interrupted and proc.returncode == 2
        assert meta['capture_complete'] is False
        assert meta['capture_problems']
        assert 'stop_utc' in meta and 'vo_poses' in meta['diagnostics']
        assert b'Traceback' not in buffer
        print('CTRL+C REGRESSION PASS:', folder)
    finally:
        selector.close()
        if proc.poll() is None:
            proc.kill()  # This isolated synthetic-test observer only.
            proc.wait(timeout=3)


if __name__ == '__main__':
    main()
