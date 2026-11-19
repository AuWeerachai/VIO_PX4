"""Workstation orchestration: passive remote capture, local copy, local plots.

No FCU access, service management, configuration writes or analysis on the Jetson.
SSH credentials are handled by the user's SSH client, never stored by this tool.
"""
from datetime import datetime
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
import uuid

from floor_geometry import CalibrationError, Limits, matrix_rpy


def open_summary(folder):
    path = Path(folder)/'calibration_before_after.png'
    if not path.is_file():
        path = Path(folder)/'current_floor_view.png'
    if not path.is_file():
        print(f'No summary image; see {folder}/report.txt')
        return False
    if not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        print(f'No local graphical display. Saved image: {path}')
        return False
    opener = shutil.which('xdg-open')
    if not opener:
        print(f'No image opener installed. Saved image: {path}')
        return False
    try:
        subprocess.Popen([opener, str(path)], stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
    except OSError as exc:
        print(f'Could not open image ({exc}); it is saved at {path}')
        return False
    print(f'Opening local summary: {path}')
    return True


def validate_remote_args(args):
    if not re.fullmatch(r'(?:[A-Za-z0-9_][A-Za-z0-9_.-]*@)?[A-Za-z0-9_][A-Za-z0-9_.-]*', args.host):
        raise CalibrationError('use an SSH alias or user@hostname/IP, not shell options or a URL')
    for name in ('height_m', 'height_tolerance_m'):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise CalibrationError(f'{name} must be positive and finite')
    if (args.height_m is not None and args.height_tolerance_m is not None
            and args.height_tolerance_m >= args.height_m):
        raise CalibrationError('height allowance must be smaller than height')


def record_from_jetson(args):
    from extrinsic_calibration import finalize, load_frames
    validate_remote_args(args)
    if not shutil.which('ssh') or not shutil.which('scp'):
        raise CalibrationError('ssh and scp are required on this computer')
    run_id = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'_'+uuid.uuid4().hex[:6]
    destination = Path(args.output_root).expanduser().resolve()/run_id
    destination.mkdir(parents=True, exist_ok=False)
    remote_relative = '.local/state/extrinsic_calibration/'+run_id
    command = ('exec "$HOME"/extrinsic_calibration/extrinsic_calibration capture --no-transfer-prompt '
               '--output "$HOME"/'+shlex.quote(remote_relative))
    for name in ('height_m', 'height_tolerance_m', 'grid_frame'):
        value = getattr(args, name)
        if value is not None:
            command += ' --'+name.replace('_', '-')+' '+shlex.quote(str(value))
    print(f'Recording on {args.host}; plots will be created on THIS computer.', flush=True)
    print(f'Desktop folder: {destination}', flush=True)
    print('Use START / STOP as usual. STOP downloads the capture and opens the local summary.', flush=True)
    # Private, temporary multiplexing avoids a second password prompt for the copy.
    # Closing this socket cannot close another operator's SSH connection or VO.
    with tempfile.TemporaryDirectory(prefix='extrinsic-ssh-') as tmp:
        options = ['-o', 'ControlPath='+str(Path(tmp)/'control'), '-o', 'ControlMaster=auto',
                   '-o', 'ControlPersist=60', '-o', 'ConnectTimeout=10',
                   '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=3']
        try:
            # Inherited terminal preserves prompts and START/STOP input; no remote shell background jobs.
            result = subprocess.run(['ssh', *options, '-t', args.host, command], check=False)
            if result.returncode not in (0, 2):
                raise CalibrationError(f'remote recording did not finish normally (SSH exit {result.returncode}); '
                                       f'raw data, if any, remains on Jetson at ~/{remote_relative}')
            print('Recording stopped. Downloading data; analysis runs locally only.', flush=True)
            copy = subprocess.run(['scp', *options, '-r', args.host+':'+remote_relative,
                                   str(destination/'capture')], check=False)
            if copy.returncode:
                raise CalibrationError(f'copy failed; remote raw data is preserved at ~/{remote_relative}')
        finally:
            try:
                subprocess.run(['ssh', *options, '-O', 'exit', args.host], stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               check=False, timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass
    source, output = destination/'capture', destination/'plots'
    meta, frames = load_frames(source)
    if len(frames):
        active = matrix_rpy(frames[0]['r_bc'])
        print(f'Captured active extrinsics: roll {active[0]:+.3f}°, pitch {active[1]:+.3f}°, '
              f'yaw {active[2]:+.3f}°. Fitting locally...', flush=True)
    output.mkdir(exist_ok=False)
    # Even a complete but inconsistent fit gets diagnostic plots, never an invented recommendation.
    rc = finalize(output, meta, frames, Limits(**meta['thresholds']), meta['yaw_deg_fixed'], source_folder=source)
    print(f'LOCAL RESULTS: {destination}', flush=True)
    if not args.no_open:
        open_summary(output)
    return rc
