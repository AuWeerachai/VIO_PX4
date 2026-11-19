"""Synthetic, known-ground-truth checks. No flight controller/camera is contacted."""
from dataclasses import replace
import math
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from floor_geometry import (CalibrationError, Limits, analyze_frames, fit_floor,
                            matrix_rpy, read_xyz, rpy_matrix, suggest_extrinsic)


def scene(old=(.35, 40, 180), actual=(1.2, 32, 180), height=.47, seed=5, noise=.002):
    rng = np.random.default_rng(seed)
    floor = np.column_stack((rng.uniform(-2.5, -.2, 550), rng.uniform(-1.1, 1.1, 550),
                             -height+rng.normal(0, noise, 550)))
    # A rear wall and a few outliers must not be used as the floor.
    wall = np.column_stack((np.full(250, -2.7), rng.uniform(-1.3, 1.3, 250),
                            rng.uniform(-height, 1.4, 250)))
    physical = np.vstack((floor, wall))
    t = np.array([-.26, 0, -.165])
    old_r, true_r = rpy_matrix(*old), rpy_matrix(*actual)
    p = (physical-t) @ true_r @ old_r.T+t
    return dict(stamp=np.array(10+seed*.5), points_body=p, r_bc=old_r, t_bc=t,
                r_ob=np.eye(3), t_ob=np.zeros(3), r_gb=rpy_matrix(4, -3, 17), t_gb=np.array([.2, -.1, .3]))


class GeometryTests(unittest.TestCase):
    def test_stereo_projection_to_mount_fit_and_fresh_level_geometry(self):
        # Independent forward image model: known rear 30-degree mount, stereo
        # pixels -> triangulation -> old body transform -> recovered CLI angles.
        rng = np.random.default_rng(12)
        ground = np.column_stack((rng.uniform(-2.5, -.6, 600),
                                  rng.uniform(-1., 1., 600), np.full(600, -1.)))
        true = rpy_matrix(.5, 30., 180.)
        old = rpy_matrix(.35, 30.5, 180.)
        t = np.array([-.26, 0., -.165])
        link_from_optical = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
        optical = (ground-t) @ true @ link_from_optical
        self.assertTrue(np.all(optical[:, 2] > 0.))
        fx, fy, cx, cy, baseline = 400., 405., 320., 180., .095
        ul = fx*optical[:, 0]/optical[:, 2]+cx
        ur = fx*(optical[:, 0]-baseline)/optical[:, 2]+cx
        v = fy*optical[:, 1]/optical[:, 2]+cy
        z = fx*baseline/(ul-ur)
        reconstructed = np.column_stack(((ul-cx)*z/fx, (v-cy)*z/fy, z)) @ link_from_optical.T
        old_body = reconstructed @ old.T+t
        fit = fit_floor(old_body, Limits(1.))
        values, _ = suggest_extrinsic(fit['normal'], old)
        np.testing.assert_allclose(values, [.5, 30., 180.], atol=1e-9)
        new_body = reconstructed @ rpy_matrix(*values).T+t
        np.testing.assert_allclose(new_body, ground, atol=1e-10)
        self.assertLess(np.ptp(new_body[:, 2]), 1e-10)

    def test_known_roll_pitch_front_and_rear_and_nadir_boundary(self):
        for yaw in [0, 180, -180, 80]:
            for roll, pitch in [(0, 32), (1.2, 30.5), (-4, 40), (3, -25), (2, 80)]:
                with self.subTest(yaw=yaw, roll=roll, pitch=pitch):
                    old = rpy_matrix(2, pitch+3, yaw)
                    true = rpy_matrix(roll, pitch, yaw)
                    n = old @ true.T @ [0, 0, 1]
                    suggested, correction = suggest_extrinsic(n, old, yaw)
                    np.testing.assert_allclose(suggested, [roll, pitch, yaw], atol=1e-9)
                    np.testing.assert_allclose(correction @ n, [0, 0, 1], atol=1e-12)

    def test_rear_pitch_sign_40_to_32(self):
        old, true = rpy_matrix(0, 40, 180), rpy_matrix(0, 32, 180)
        normal = old @ true.T @ [0, 0, 1]
        self.assertGreater(-normal[0]/normal[2], 0)  # floor rises toward drone front
        result, _ = suggest_extrinsic(normal, old)
        self.assertAlmostEqual(result[1], 32)

    def test_nadir_does_not_fabricate_roll(self):
        with self.assertRaisesRegex(CalibrationError, "near-nadir"):
            suggest_extrinsic([0, 0, 1], rpy_matrix(0, 90, 180))

    def test_invalid_rotations_and_normals(self):
        for normal in ([float('nan'), 0, 1], [0, 0, 0]):
            with self.assertRaises(CalibrationError):
                suggest_extrinsic(normal, np.eye(3))
        with self.assertRaises(CalibrationError):
            suggest_extrinsic([0, 0, 1], np.diag([1, 1, -1]))

    def test_ground_with_wall_returns_correct_mount(self):
        f = scene()
        fit = fit_floor(f['points_body'], Limits(.47))
        suggested, _ = suggest_extrinsic(fit['normal'], f['r_bc'])
        np.testing.assert_allclose(suggested, [1.2, 32, 180], atol=.08)
        self.assertGreater(fit['count'], 400)

    def test_wall_ceiling_table_and_narrow_strip_rejected(self):
        rng = np.random.default_rng(40)
        xy = rng.uniform(-1, 1, (600, 2))
        cases = [np.column_stack((np.ones(600), xy)),
                 np.column_stack((xy, np.ones(600))),
                 np.column_stack((xy, np.full(600, -.18))),
                 np.column_stack((xy[:, 0], xy[:, 1]*.02, np.full(600, -.47)))]
        for points in cases:
            with self.subTest(case=points[0]):
                with self.assertRaises(CalibrationError):
                    fit_floor(points, Limits(.47))

    def test_two_plausible_planes_rejected(self):
        rng = np.random.default_rng(3)
        points = np.vstack([np.column_stack((rng.uniform(-1, 1, (600, 2)), np.full(600, z)))
                            for z in (-.9, -1.1)])
        with self.assertRaisesRegex(CalibrationError, 'ambiguous'):
            fit_floor(points, Limits(1, height_tolerance_m=.2))

    def test_repeated_points_not_counted_as_features(self):
        p = np.repeat([[0, 0, -.47], [1, 0, -.47], [0, 1, -.47]], 1000, axis=0)
        with self.assertRaises(CalibrationError):
            fit_floor(p, Limits(.47))

    def test_series_consistent_and_yaw_not_fitted(self):
        frames = [scene(seed=i) for i in range(13)]
        report, _ = analyze_frames(frames, Limits(.47))
        self.assertTrue(report['accepted'], report['reasons'])
        np.testing.assert_allclose(report['suggested_rpy_deg'], [1.2, 32, 180], atol=.06)
        self.assertFalse(report['yaw_is_estimated'])
        self.assertFalse(report['translation_is_estimated'])
        np.testing.assert_allclose(report['unchanged_xyz_m'], frames[0]['t_bc'])

    def test_pose_reset_motion_static_tf_change_and_timestamp_reject(self):
        for defect in ('motion', 'rotation', 'extrinsic', 'timestamp'):
            frames = [scene(seed=i) for i in range(3)]
            if defect == 'motion':
                frames[1]['t_ob'] = np.array([.2, 0, 0])
            elif defect == 'rotation':
                frames[1]['r_ob'] = rpy_matrix(0, 90, 0)
            elif defect == 'extrinsic':
                frames[1]['r_bc'] = rpy_matrix(0, 30, 180)
            else:
                frames[1]['stamp'] = frames[0]['stamp']
            report, _ = analyze_frames(frames, Limits(.47, min_frames=3, min_duration_s=.5))
            self.assertFalse(report['accepted'])
            self.assertNotIn('suggested_rpy_deg', report)

    def test_large_correction_rejected(self):
        frames = [scene(old=(0, 48, 180), seed=i) for i in range(3)]
        report, _ = analyze_frames(frames, Limits(.47, min_frames=3, min_duration_s=.5))
        self.assertFalse(report['accepted'])

    def test_normal_changes_rejected(self):
        frames = [scene(old=(0, 34, 180), actual=(0, 30+i*2, 180), seed=i) for i in range(3)]
        report, _ = analyze_frames(frames, Limits(.47, min_frames=3, min_duration_s=.5))
        self.assertFalse(report['accepted'])
        self.assertTrue(any('changes over time' in x for x in report['reasons']))

    def test_rotation_fit_invariant_to_map_and_odom_orientation(self):
        frames = [scene(seed=i) for i in range(3)]
        for f in frames:
            f['r_ob'] = rpy_matrix(40, -30, 100)
            f['r_gb'] = rpy_matrix(-15, 60, -20)
        report, _ = analyze_frames(frames, Limits(.47, min_frames=3, min_duration_s=.5))
        self.assertTrue(report['accepted'], report['reasons'])
        np.testing.assert_allclose(report['suggested_rpy_deg'], [1.2, 32, 180], atol=.08)


class DecoderTests(unittest.TestCase):
    def cloud(self, big=False):
        order = '>' if big else '<'
        data = b''
        for row in range(2):
            for col in range(2):
                data += struct.pack(order+'ffff', row, col, -1., 99.)
            data += b'rowpadding!!'
        fields = [SimpleNamespace(name=n, offset=i*4, datatype=7, count=1) for i, n in enumerate('xyz')]
        return SimpleNamespace(width=2, height=2, fields=fields, point_step=16,
                               row_step=44, data=data, is_bigendian=big)

    def test_padded_endianness(self):
        for big in (False, True):
            np.testing.assert_allclose(read_xyz(self.cloud(big)), [[0, 0, -1], [0, 1, -1], [1, 0, -1], [1, 1, -1]])

    def test_bad_payload_rejected(self):
        msg = self.cloud()
        msg.data = msg.data[:-1]
        with self.assertRaises(CalibrationError):
            read_xyz(msg)
        msg = self.cloud()
        msg.fields[0].count = 2
        with self.assertRaises(CalibrationError):
            read_xyz(msg)


class ReportTests(unittest.TestCase):
    def test_report_and_three_view_images(self):
        from extrinsic_calibration import finalize
        frames = [scene(seed=i) for i in range(3)]
        meta = dict(grid_frame='map', body_frame='drone_link', camera_frame='camera_link',
                    odom_frame='odom', topic='/synthetic', level_confirmed=True, synthetic_test=True)
        with tempfile.TemporaryDirectory() as path:
            folder = Path(path)
            rc = finalize(folder, meta, frames, Limits(.47, min_frames=3, min_duration_s=.5))
            self.assertEqual(rc, 0)
            for name in ('current_floor_view.png', 'predicted_floor_view.png'):
                self.assertGreater((folder/name).stat().st_size, 10000)
            self.assertIn('Yaw:   +180.00', (folder/'report.txt').read_text())


if __name__ == '__main__':
    unittest.main()
