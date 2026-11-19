"""Offline orientation plots. No ROS or flight-controller access."""
import csv
import json
from pathlib import Path

import numpy as np

from floor_geometry import CalibrationError, matrix_rpy, quaternion_matrix, rpy_matrix


def save_pose_plot(output, source, frames, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    times, rpy = [], []
    path = Path(source)/'vo_pose.csv'
    if path.exists():
        with path.open() as stream:
            previous = -float('inf')
            for row in csv.DictReader(stream):
                try:
                    t = float(row['stamp'])
                    if t-previous < .05:
                        continue  # Plot at most 20 Hz; raw file remains full rate.
                    r = quaternion_matrix(*(float(row[k]) for k in ('qx', 'qy', 'qz', 'qw')))
                    angles = matrix_rpy(r)
                    times.append(t)
                    rpy.append(angles)
                    previous = t
                except (ValueError, KeyError, CalibrationError):
                    continue
    if not times:
        return None
    origin = times[0]
    ft, grid = [], []
    for frame in frames:
        try:
            grid.append(matrix_rpy(frame['r_gb']))
            ft.append(float(frame['stamp']))
        except CalibrationError:
            continue
    pt, px4 = [], []
    ppath = Path(source)/'px4.jsonl'
    if ppath.exists():
        with ppath.open() as stream:
            for line in stream:
                packet = json.loads(line)
                if packet['type'] != 'ATTITUDE':
                    continue
                d = packet['data']
                r_ned_frd = rpy_matrix(*np.degrees([d['roll'], d['pitch'], d['yaw']]))
                # NED->ENU on left, FLU->FRD on right. Never compare raw conventions.
                ned_to_enu = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
                try:
                    px4.append(matrix_rpy(ned_to_enu @ r_ned_frd @ np.diag([1, -1, -1])))
                    pt.append(packet['received_unix_s'])
                except CalibrationError:
                    continue
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for t, angles, name, style in [(times, rpy, 'VO body in odom (raw)', '-'),
                                  (ft, grid, 'VO body in RViz grid frame', ':'),
                                  (pt, px4, 'PX4 body in ENU (receive-time)', '--')]:
        if not len(t):
            continue
        a = np.array(angles)
        # Relative yaw changes only: VO/map and north have different yaw origins.
        a[:, 2] = np.degrees(np.unwrap(np.radians(a[:, 2])))
        a[:, 2] -= a[0, 2]
        for i, ax in enumerate(axes):
            ax.plot(np.array(t)-origin, a[:, i], style, label=name, linewidth=1.)
    for i, ax in enumerate(axes):
        ax.set_ylabel(('Roll (deg)', 'Pitch (deg)', 'Yaw change (deg)')[i])
        ax.grid(alpha=.25)
        for jump in report.get('raw_vo_discontinuity_times', []):
            ax.axvline(jump-origin, color='#ba3c3c', alpha=.3, linewidth=.8)
    axes[-1].set_xlabel('Seconds from first recorded VO pose')
    axes[0].legend(loc='best', fontsize=8)
    heading = 'Body orientation history — real tilt and VO reference kept separate'
    if report.get('synthetic_test'):
        heading = 'SYNTHETIC TEST — '+heading
    fig.suptitle(heading, fontsize=12)
    fig.text(.5, .025, 'Red lines: abrupt raw VO steps/gaps, not a diagnosis of cause. Yaw origins removed separately.\n'
             'PX4 uses receive-time approximation. VO roll/pitch are relative to their named frames, not independent gravity.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .95))
    fig.savefig(Path(output)/'orientation_history.png', dpi=160)
    plt.close(fig)
    return 'orientation_history.png'
