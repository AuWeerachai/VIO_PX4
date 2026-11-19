"""Operator PC receiver for requests explicitly made by STOP -> Y on Jetson.

Uses outbound SSH only. No listener, root privileges, or flight-stack changes.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess
import time
import uuid

from extrinsic_calibration import load_frames, finalize
from floor_geometry import Limits
from local_workflow import open_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('host')
    parser.add_argument('--control-path', required=True)
    args = parser.parse_args()
    options = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', '-o', 'ControlPath='+args.control_path]
    def ssh(command):
        return subprocess.run(['ssh', *options, args.host, command], capture_output=True, text=True, timeout=15)
    root = Path.home()/'Desktop/Camera_Calibration_Results'
    done = set()
    while True:
        try:
            found = ssh('mkdir -p "$HOME/.local/state/extrinsic_calibration"; '
                        'touch "$HOME/.local/state/extrinsic_calibration/desktop_receiver.json"; '
                        'find "$HOME/.local/state/extrinsic_calibration" -mindepth 2 -maxdepth 2 '
                        '-name desktop_transfer_request.json -print')
            if found.returncode:
                time.sleep(3)
                continue
            for request in found.stdout.splitlines():
                if request in done:
                    continue
                source = str(Path(request).parent)
                receipt = source+'/desktop_transfer_result.json'
                if ssh('test -f '+shlex.quote(receipt)).returncode == 0:
                    done.add(request)
                    continue
                run = Path(source).name
                try:
                    label = datetime.strptime(run[:15], '%Y%m%d-%H%M%S').strftime('%Y-%m-%d_%H-%M-%S')
                except ValueError:
                    label = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                destination = root/(label+'_'+uuid.uuid4().hex[:6])
                destination.mkdir(parents=True, exist_ok=False)
                try:
                    subprocess.run(['scp', *options, '-r', args.host+':'+source,
                                    str(destination/'capture')], check=True, timeout=120)
                    meta, frames = load_frames(destination/'capture')
                    output = destination/'plots'
                    output.mkdir()
                    rc = finalize(output, meta, frames, Limits(**meta['thresholds']),
                                  meta['yaw_deg_fixed'], source_folder=destination/'capture')
                    picture = output/'calibration_before_after.png'
                    if not picture.exists():
                        raise RuntimeError('No comparison plot produced; see downloaded report.')
                    message = 'Transferred and plotted on your computer: '+str(picture)
                    if rc:
                        message += '\nFit is provisional; see the report.'
                    open_summary(output)
                except Exception as exc:
                    message = 'Transfer/plot failed: '+str(exc)+'. Original recording preserved.'
                payload = json.dumps({'message': message})
                ssh('printf %s '+shlex.quote(payload)+' > '+shlex.quote(receipt))
                print(message, flush=True)
                done.add(request)
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(str(exc), flush=True)
        time.sleep(2)


if __name__ == '__main__':
    main()
