"""Copy a finished capture to the operator PC; analyze only on that PC."""
from datetime import datetime
import os
from pathlib import Path
import re
import shlex
import subprocess
import uuid
import json
import time

from floor_geometry import CalibrationError


def transfer_and_plot(folder, host):
    folder = Path(folder).resolve()
    if not (folder/'capture.json').is_file():
        raise CalibrationError(f'No capture.json in {folder}')
    if not re.fullmatch(r'(?:[A-Za-z0-9_][A-Za-z0-9_.-]*@)?[A-Za-z0-9_][A-Za-z0-9_.-]*', host):
        raise CalibrationError('Use your computer SSH alias or username@IP.')
    run = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'_'+uuid.uuid4().hex[:6]
    relative = 'Desktop/Camera_Calibration_Results/'+run
    options = ['-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=10',
               '-o', 'ServerAliveCountMax=3']
    # The PC must already have this tool and SSH access. Never install/enable services.
    locate = ('calibration_tool="$HOME/extrinsic_calibration/extrinsic_calibration.py"; '
              'if [ ! -f "$calibration_tool" ]; then calibration_tool="$HOME/Desktop/extrinsic_calibration/extrinsic_calibration.py"; fi; ')
    setup = (locate+'test -f "$calibration_tool" && '
             'mkdir -p "$HOME/Desktop/Camera_Calibration_Results" && '
             'mkdir "$HOME/'+relative+'"')
    if subprocess.run(['ssh', *options, host, setup], check=False).returncode:
        raise CalibrationError('Cannot reach the computer or its calibration tool. Recording remains on Jetson.')
    if subprocess.run(['scp', *options, '-r', str(folder), host+':'+relative+'/capture'], check=False).returncode:
        raise CalibrationError('Transfer failed. Original recording remains on Jetson; plotting was not started.')
    print('Transfer complete. Generating plots on your computer...', flush=True)
    command = (locate+'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/python3 -s '
               '"$calibration_tool" analyze '
               '"$HOME/'+relative+'/capture" --output "$HOME/'+relative+'/plots" --open')
    if subprocess.run(['ssh', *options, host, command], check=False).returncode:
        raise CalibrationError(f'Plotting did not finish successfully. Transferred data: {host}:~/{relative}/capture')
    print(f'Saved on your computer: ~/{relative}/plots/calibration_before_after.png')


def offer_transfer(folder):
    try:
        answer = input('Transfer recording to your computer and generate plots? [Y/n]: ').strip().lower()
        while answer not in ('', 'y', 'yes', 'n', 'no'):
            answer = input('Enter Y to transfer and plot, or N to keep the recording here: ').strip().lower()
        if answer in ('n', 'no'):
            print(f'Recording kept here: {folder}')
            return
        heartbeat = Path.home()/'.local/state/extrinsic_calibration/desktop_receiver.json'
        if heartbeat.is_file() and time.time()-heartbeat.stat().st_mtime < 20:
            request = Path(folder)/'desktop_transfer_request.json'
            request.write_text(json.dumps({'requested': time.time()})+'\n')
            print('Transfer requested. Your computer is downloading and plotting this recording.', flush=True)
            receipt = Path(folder)/'desktop_transfer_result.json'
            for _ in range(180):
                if receipt.is_file():
                    result = json.loads(receipt.read_text())
                    print(result['message'], flush=True)
                    return
                time.sleep(1)
            print('Still waiting for the computer. Recording and transfer request are preserved.')
            return
        # SSH_CONNECTION gives the client IP, never its login username. Ask rather than guess.
        client = os.environ.get('SSH_CONNECTION', '').split()
        if client:
            print(f'Your SSH client address is {client[0]}; use your COMPUTER login username, not teamc.')
        host = input('Your computer SSH target (username@IP or SSH alias): ').strip()
        transfer_and_plot(folder, host)
    except (EOFError, KeyboardInterrupt):
        print(f'\nTransfer cancelled. Recording saved: {folder}')
    except (CalibrationError, OSError) as exc:
        print(f'Transfer/plot failed: {exc}')
        print('Retry without recording again: ./extrinsic_calibration transfer '+shlex.quote(str(folder))+' username@computer-IP')
