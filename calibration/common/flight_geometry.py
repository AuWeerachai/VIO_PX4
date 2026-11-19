"""Offline flight calibration with independent PX4 tilt and explicit rejection reasons."""
from dataclasses import asdict, replace
import json
import math
from pathlib import Path

import numpy as np

from floor_geometry import (CalibrationError, angle_deg, fit_floor, matrix_rpy,
                            rotation_angle_deg, rpy_matrix, suggest_extrinsic, unit, level_basis,
                            fit_absolute_mount)
from passive_telemetry import validate_packet


def px4_up_flu(roll, pitch, yaw=0.):
    """PX4 FRD->NED Euler angles (radians) to physical up in body FLU.

    Yaw cancels; magnetic north accuracy is irrelevant to this gravity vector.
    """
    r = rpy_matrix(*np.degrees([roll, pitch, yaw]))
    return np.diag([1., -1., -1.]) @ r.T @ np.array([0., 0., -1.])


class FlightTelemetry:
    """Match using host receive time, NOT claiming sensor-clock synchronization.

    Use only low roll/pitch rate and stable tilt windows to bound sensitivity to
    unknown transport delay. Preserve original PX4 boot time for independent audit.
    """
    def __init__(self, packets, max_pair_age=.15, half_window=.2):
        self.max_pair_age, self.half_window = max_pair_age, half_window
        self.att = [p for p in packets if p['type'] == 'ATTITUDE']
        self.odom = [p for p in packets if p['type'] == 'ODOMETRY']
        self.at = np.array([p['received_unix_s'] for p in self.att])
        self.ot = np.array([p['received_unix_s'] for p in self.odom])
        offsets = [p['received_unix_s']-p['received_monotonic_s'] for p in packets]
        self.clock_bad = bool(offsets and np.ptp(offsets) > .02)
        for stream, stamps in ((self.att, self.at), (self.odom, self.ot)):
            if len(stamps) > 1 and np.any(np.diff(stamps) <= 0):
                self.clock_bad = True
        self.baseline = None

    def anchor_height(self, start_time):
        o, _ = self.nearest(self.odom, self.ot, start_time)
        self.baseline = (o['data']['z'], o['data']['reset_counter'], o['data']['boot_s'])

    @classmethod
    def load(cls, folder):
        path = Path(folder)/'px4.jsonl'
        if not path.exists():
            return cls([])
        packets = []
        with path.open() as stream:
            for line in stream:
                try:
                    packets.append(validate_packet(line.encode()))
                except (ValueError, KeyError, TypeError) as exc:
                    raise CalibrationError('invalid saved PX4 telemetry; cannot calibrate') from exc
        return cls(packets)

    def nearest(self, stream, stamps, t):
        if len(stamps) == 0:
            raise CalibrationError('no passive PX4 telemetry; bridge diagnostic copy required')
        index = int(np.searchsorted(stamps, t))
        index = min((max(0, index-1), min(len(stamps)-1, index)), key=lambda i: abs(stamps[i]-t))
        if abs(stamps[index]-t) > self.max_pair_age:
            raise CalibrationError('PX4 telemetry too far from cloud time')
        return stream[index], abs(float(stamps[index]-t))

    def pair(self, t, initial_height):
        if self.clock_bad:
            raise CalibrationError('host clock stepped or diagnostic times regressed')
        a, age = self.nearest(self.att, self.at, t)
        o, odom_age = self.nearest(self.odom, self.ot, t)
        lo = np.searchsorted(self.at, t-self.half_window)
        hi = np.searchsorted(self.at, t+self.half_window, side='right')
        window = self.att[lo:hi]
        if (len(window) < 5 or window[0]['received_unix_s'] > t-.1
                or window[-1]['received_unix_s'] < t+.1):
            raise CalibrationError('PX4 tilt window lacks samples before and after cloud')
        keys = {(p['system_id'], p['component_id']) for p in window+[o]}
        if len(keys) != 1:
            raise CalibrationError('PX4 identity changed')
        if any(p['phase'] != 'live' or p['quarantined'] for p in window+[o]):
            raise CalibrationError('bridge not live/healthy around this sample')
        broad = self.att[np.searchsorted(self.at, t-.5):np.searchsorted(self.at, t+.5, side='right')]
        if any(p['quarantined'] for p in broad):
            raise CalibrationError('sample too close to bridge quarantine')
        boot = np.array([p['data']['boot_s'] for p in window])
        if np.any(np.diff(boot) <= 0) or np.max(np.diff(boot)) > .15:
            raise CalibrationError('PX4 timestamp discontinuity or telemetry gap')
        up = px4_up_flu(a['data']['roll'], a['data']['pitch'])
        spread = max(angle_deg(up, px4_up_flu(p['data']['roll'], p['data']['pitch'])) for p in window)
        tilt_rate = max(math.hypot(p['data']['rollspeed'], p['data']['pitchspeed']) for p in window)
        if spread > 1. or math.degrees(tilt_rate) > 5.:
            raise CalibrationError('tilt changing too fast for receive-time matching')
        if angle_deg(up, [0, 0, 1]) > 20:
            raise CalibrationError('body tilt exceeds 20 deg calibration envelope')
        # The height filter follows relative FCU z, never uses VO z as truth.
        # Anchor at START, whose height the operator supplied, not a later fit.
        if self.baseline is None:
            raise CalibrationError('no PX4 height sample at START; cannot identify floor reliably')
        z0, reset, boot0 = self.baseline
        if o['data']['reset_counter'] != reset or o['data']['boot_s'] < boot0:
            raise CalibrationError('PX4 reset: original floor-height reference no longer usable')
        height = initial_height-(o['data']['z']-z0)
        return dict(up=up, height=height, age=age, odom_age=odom_age,
                    tilt_spread_deg=spread, tilt_rate_deg_s=math.degrees(tilt_rate))


def pose_jump_times(folder):
    """Mark abrupt raw VO steps, NOT claims about the cause of tracking failure."""
    import csv
    from floor_geometry import quaternion_matrix
    path = Path(folder)/'vo_pose.csv'
    if not path.exists():
        return [], ['no raw VO pose recording']
    previous, jumps, issues = None, [], []
    with path.open() as stream:
        for row in csv.DictReader(stream):
            try:
                stamp = float(row['stamp'])
                p = np.array([float(row[k]) for k in ('x', 'y', 'z')])
                r = quaternion_matrix(*(float(row[k]) for k in ('qx', 'qy', 'qz', 'qw')))
                if not np.isfinite(p).all() or not math.isfinite(stamp):
                    raise ValueError('nonfinite pose')
                if previous is not None:
                    dt = stamp-previous[0]
                    # Deliberately conservative data-selection guards, not flight gates.
                    if (dt <= 0 or dt > .25 or np.linalg.norm(p-previous[1]) > max(.15, 5*dt)
                            or rotation_angle_deg(r @ previous[2].T) > max(5., 90*dt)):
                        jumps.append(stamp)
                previous = stamp, p, r
            except (ValueError, KeyError, CalibrationError):
                issues.append('malformed raw VO pose')
    if previous is None:
        issues.append('no usable raw VO poses')
    return jumps, list(dict.fromkeys(issues))


def analyze_flight(frames, limits, telemetry, folder, fixed_yaw_deg=180.):
    limits.validate()
    jumps, errors = pose_jump_times(folder)
    rows, good, best, previous_stamp = [], [], None, -math.inf
    camera_normals, independent_up = [], []
    report = dict(accepted=False, reasons=[], captured_frames=len(frames), good_frames=0,
                  thresholds=asdict(limits), per_frame=rows, yaw_deg_fixed=fixed_yaw_deg,
                  yaw_is_estimated=False, translation_is_estimated=False,
                  mode='flight', raw_vo_discontinuity_times=jumps,
                  assumption='horizontal floor; PX4 roll/pitch and entered initial body height are valid',
                  timing='host receive-time approximation; stable-tilt windows only; not hardware clock synchronization',
                  caveat='Floor reconstruction bias and mounting error cannot be separated by this fit alone.')
    if not frames:
        report['reasons'] = ['no captured clouds']
        return report, []
    first = frames[0]
    r_bc = first['r_bc']
    try:
        meta = json.loads((Path(folder)/'capture.json').read_text())
        telemetry.anchor_height(meta['start_unix_s'])
    except (CalibrationError, KeyError, OSError, ValueError) as exc:
        errors.append('initial PX4 height reference unavailable: '+str(exc))
    for i, f in enumerate(frames):
        t = float(f['stamp'])
        row = dict(stamp=t, frame_index=i, accepted=False)
        rows.append(row)
        try:
            if t <= previous_stamp:
                errors.append('cloud time regressed')
                raise CalibrationError('cloud time regressed')
            previous_stamp = t
            if (rotation_angle_deg(f['r_bc'] @ r_bc.T) > .01
                    or np.linalg.norm(f['t_bc']-first['t_bc']) > 1e-5):
                errors.append('camera extrinsics changed during capture')
                raise CalibrationError('camera extrinsics changed')
            paired = telemetry.pair(t, limits.height_m)
            if any(abs(t-j) < .5 for j in jumps):
                raise CalibrationError('near an abrupt raw VO step or tracking gap')
            up = paired['up']
            # Rotate into a gravity-aligned frame only for plane selection.
            # Physical flight tilt must not be mistaken for a mounting error.
            level = level_basis(up)
            fit = fit_floor(f['points_body'] @ level.T,
                            replace(limits, height_m=paired['height']), seed=i)
            fit['normal'] = level.T @ fit['normal']
            for key in ('points', 'roi'):
                fit[key] = fit[key] @ level
            rpy, correction = suggest_extrinsic(fit['normal'], r_bc, fixed_yaw_deg, up)
            if rotation_angle_deg(correction) > limits.max_correction_deg:
                raise CalibrationError('suggested mounting change exceeds allowed calibration range')
            fit['frame_index'] = i
            row.update(accepted=True, roll_deg=float(rpy[0]), pitch_deg=float(rpy[1]),
                       plane_rms_m=fit['rms'], floor_height_m=fit['height_m'],
                       px4_up_body=up.tolist(), pair_age_s=paired['age'],
                       tilt_spread_deg=paired['tilt_spread_deg'], count=fit['count'])
            camera_normal = r_bc.T @ fit['normal']
            row['floor_normal_camera'] = camera_normal.tolist()
            row['floor_normal_grid'] = (f['r_gb'] @ fit['normal']).tolist()
            row['floor_normal_odom'] = (f['r_ob'] @ fit['normal']).tolist()
            camera_normals.append(camera_normal)
            independent_up.append(up)
            good.append((rpy, correction))
            if best is None or fit['rms'] < best['rms']:
                best = fit  # Keep one full cloud fit, not an unbounded flight's clouds.
        except CalibrationError as exc:
            row['reason'] = str(exc)
    accepted_rows = [row for row in rows if row['accepted']]
    if len(good) < limits.min_frames:
        errors.append(f'only {len(good)} accepted frames; need {limits.min_frames}')
    if len(good)/len(frames) < .25:
        errors.append('less than 25% of snapshots usable; collect more slow, stable views')
    if len(accepted_rows) < 2 or accepted_rows[-1]['stamp']-accepted_rows[0]['stamp'] < limits.min_duration_s:
        errors.append('insufficient duration of accepted flight data')
    report['good_frames'] = len(good)
    if good:
        angles = np.array([entry[0] for entry in good])
        # Wrap about the active mounting before taking medians.
        old = matrix_rpy(r_bc)
        angles[:, :2] = old[:2]+(angles[:, :2]-old[:2]+180) % 360-180
        rpy, residuals = fit_absolute_mount(camera_normals, independent_up, fixed_yaw_deg)
        correction = rpy_matrix(*rpy) @ r_bc.T
        spread = max(rotation_angle_deg(c @ correction.T) for _, c in good)
        if spread > limits.max_normal_spread_deg:
            errors.append(f'mounting estimates inconsistent across flight ({spread:.2f} deg maximum deviation)')
        report.update(current_rpy_deg=old.tolist(), candidate_rpy_deg=rpy.tolist(), unchanged_xyz_m=first['t_bc'].tolist(),
                      mounting_max_deviation_deg=spread,
                      repeatability_std_roll_pitch_deg=np.std(angles[:, :2], axis=0).tolist(),
                      mean_floor_rms_m=float(np.mean([r['plane_rms_m'] for r in accepted_rows])),
                      absolute_fit=dict(method='camera-frame normals to independent PX4 up; no mount prior',
                                        samples=len(camera_normals),
                                        median_error_deg=float(np.median(residuals)),
                                        p95_error_deg=float(np.percentile(residuals, 95)),
                                        max_error_deg=float(max(residuals)),
                                        hardware_verified=False))
        if not errors:
            report.update(accepted=True, suggested_rpy_deg=rpy.tolist(), correction_rotation=correction.tolist())
    report['reasons'] = list(dict.fromkeys(errors))
    return report, [best] if best is not None else []
