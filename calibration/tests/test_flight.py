"""Flight geometry and receive-only safety tests, all synthetic."""
import ast
import csv
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from floor_geometry import CalibrationError, Limits, rpy_matrix, suggest_extrinsic
from flight_geometry import FlightTelemetry, analyze_flight, px4_up_flu
from passive_telemetry import PassiveTelemetry, validate_packet


def packet(t, kind='ATTITUDE', roll=0., pitch=0., yaw=0., **extra):
    data = dict(boot_s=t-50, roll=math.radians(roll), pitch=math.radians(pitch),
                yaw=math.radians(yaw), rollspeed=0., pitchspeed=0., yawspeed=0.)
    if kind == 'ODOMETRY':
        data = dict(boot_s=t-50, z=-1., reset_counter=0, frame_id=1)
    data.update(extra)
    return dict(schema=1, source='vio_px4_bridge_diagnostic', type=kind,
                received_unix_s=t, received_monotonic_s=t-20, system_id=1, component_id=1,
                phase='live', quarantined=False, data=data)


def flight_scene(t, i, actual=(1.2, 32, 180), old=(.35, 36, 180)):
    rng = np.random.default_rng(int(round(i*1000)))
    roll, pitch = 3*math.sin(i/10), 2*math.cos(i/12)
    up = px4_up_flu(math.radians(roll), math.radians(pitch))
    x = np.array([1., 0., 0.])-up*up[0]
    x /= np.linalg.norm(x)
    level = np.stack((x, np.cross(up, x), up))
    floor = np.column_stack((rng.uniform(-2.2, -.2, 600), rng.uniform(-1.2, 1.2, 600),
                             -1.+rng.normal(0, .002, 600))) @ level
    trans = np.array([-.26, 0, -.165])
    r_old, r_true = rpy_matrix(*old), rpy_matrix(*actual)
    measured = (floor-trans) @ r_true @ r_old.T+trans
    return dict(stamp=np.array(t), points_body=measured, r_bc=r_old, t_bc=trans,
                r_ob=rpy_matrix(roll, -pitch, i*2), t_ob=np.array([i*.2, .1*math.sin(i), 0]),
                r_gb=rpy_matrix(roll+2, -pitch+1, i*2), t_gb=np.array([i*.2, 0., .1])), roll, pitch


def fixture(folder):
    frames, packets = [], []
    for i in range(16):
        t = 101+i*.6
        f, roll, pitch = flight_scene(t, i)
        frames.append(f)
        for t1 in np.arange(t-.25, t+.26, .05):
            packets += [packet(float(t1), roll=roll, pitch=pitch), packet(float(t1), 'ODOMETRY')]
    # Start height reference exists exactly at START.
    start = float(frames[0]['stamp'])-.1
    (folder/'capture.json').write_text(json.dumps(dict(start_unix_s=start)))
    with (folder/'vo_pose.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['stamp', 'received_unix_s', 'frame_id', 'child_frame_id',
                         'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
        for t in np.arange(100.5, 111., .02):
            writer.writerow([t, t+.01, 'odom', 'drone_link', t*.1, 0, 0, 0, 0, 0, 1])
    return frames, packets


class FlightTests(unittest.TestCase):
    def test_flight_tilt_not_mount_error_and_yaw_cancels(self):
        old, true = rpy_matrix(.35, 40, 180), rpy_matrix(1.2, 32, 180)
        for roll in (-8, 0, 9):
            for pitch in (-7, 0, 6):
                up = px4_up_flu(*np.radians([roll, pitch]))
                for yaw in (0, 90, 179, -179):
                    np.testing.assert_allclose(up, px4_up_flu(*np.radians([roll, pitch, yaw])), atol=1e-12)
                n = old @ true.T @ up
                angles, correction = suggest_extrinsic(n, old, 180, up)
                np.testing.assert_allclose(angles, [1.2, 32, 180], atol=1e-8)
                np.testing.assert_allclose(correction @ n, up, atol=1e-10)

    def test_flight_varying_tilt_translation_recovers_mount(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames, packets = fixture(Path(tmp))
            result, fits = analyze_flight(frames, Limits(1.), FlightTelemetry(packets), tmp)
            self.assertTrue(result['accepted'], result['reasons'])
            np.testing.assert_allclose(result['suggested_rpy_deg'], [1.2, 32, 180], atol=.04)
            self.assertEqual(len(fits), 1)  # Bounded geometry memory.

    def test_missing_telemetry_never_recommends(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames, _ = fixture(Path(tmp))
            result, _ = analyze_flight(frames, Limits(1.), FlightTelemetry([]), tmp)
            self.assertFalse(result['accepted'])
            self.assertNotIn('suggested_rpy_deg', result)

    def test_raw_odom_epoch_rotation_does_not_become_mount_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames, packets = fixture(Path(tmp))
            for i, frame in enumerate(frames):
                if i > 7:
                    # Frame was already expressed in body at exact cloud time;
                    # a world/map reorientation must not change the mount fit.
                    frame['r_ob'] = rpy_matrix(90, 35, 120) @ frame['r_ob']
                    frame['r_gb'] = rpy_matrix(-60, 20, -100) @ frame['r_gb']
            result, _ = analyze_flight(frames, Limits(1.), FlightTelemetry(packets), tmp)
            self.assertTrue(result['accepted'], result['reasons'])
            np.testing.assert_allclose(result['suggested_rpy_deg'], [1.2, 32, 180], atol=.04)

    def test_changed_mount_and_inconsistent_surface_do_not_recommend(self):
        for kind in ('mount', 'surface'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                frames, packets = fixture(Path(tmp))
                if kind == 'mount':
                    frames[8]['r_bc'] = rpy_matrix(.35, 37, 180)
                else:
                    for i in range(8, 16):
                        frames[i], _, _ = flight_scene(float(frames[i]['stamp']), i, actual=(1.2, 35, 180))
                result, _ = analyze_flight(frames, Limits(1.), FlightTelemetry(packets), tmp)
                self.assertFalse(result['accepted'])
                self.assertNotIn('suggested_rpy_deg', result)

    def test_quaternion_sign_and_yaw_wrap_not_mistaken_for_step(self):
        from flight_geometry import pose_jump_times
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'vo_pose.csv'
            with path.open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['stamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
                for i in range(30):
                    yaw = math.radians(179+i*.1)
                    sign = (-1)**i
                    writer.writerow([100+i*.02, 0, 0, 0, 0, 0, sign*math.sin(yaw/2), sign*math.cos(yaw/2)])
            jumps, errors = pose_jump_times(tmp)
            self.assertEqual(jumps, [])
            self.assertEqual(errors, [])

    def test_pairing_gaps_fast_tilt_quarantine_and_reset(self):
        base = [p for t in np.arange(100.7, 101.31, .05) for p in (packet(float(t)), packet(float(t), 'ODOMETRY'))]
        for defect in ('gap', 'fast', 'quarantine', 'reset', 'clock'):
            p = json.loads(json.dumps(base))
            tl = FlightTelemetry(p)
            tl.anchor_height(100.75)
            if defect == 'gap':
                tl = FlightTelemetry([])
            elif defect == 'fast':
                for x in tl.att:
                    x['data']['rollspeed'] = math.radians(20)
            elif defect == 'quarantine':
                tl.att[5]['quarantined'] = True
            elif defect == 'reset':
                for x in tl.odom:
                    x['data']['reset_counter'] += 1
            else:
                tl.clock_bad = True
            with self.subTest(defect=defect), self.assertRaises(CalibrationError):
                tl.pair(101., 1.)

    def test_relative_height_uses_px4_down_not_vo_z(self):
        packets = [p for t in np.arange(100.5, 102., .05) for p in (packet(float(t)), packet(float(t), 'ODOMETRY', z=-1.-(t-100.5)))]
        tl = FlightTelemetry(packets)
        tl.anchor_height(100.5)
        self.assertAlmostEqual(tl.pair(101., 1.)['height'], 1.5, places=8)

    def test_rx_only_and_fail_closed(self):
        with patch('passive_telemetry.socket.socket') as sock:
            receiver = PassiveTelemetry()
            sock.return_value.bind.assert_called_once_with(('127.0.0.1', 14691))
            receiver.close()
            sock.return_value.send.assert_not_called()
            sock.return_value.sendto.assert_not_called()
        p = packet(101.)
        self.assertEqual(validate_packet(json.dumps(p).encode())['type'], 'ATTITUDE')
        p['data']['roll'] = float('nan')
        with self.assertRaises(ValueError):
            validate_packet(json.dumps(p).encode())
        # Structurally prevent accidental future command/parameter/serial paths.
        import passive_telemetry
        source = Path(passive_telemetry.__file__).read_text()
        calls = [n.func.attr for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        self.assertFalse(set(calls) & {'send', 'sendto', 'sendall', 'mavlink_connection', 'Serial'})


if __name__ == '__main__':
    unittest.main()
