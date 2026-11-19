"""Local presentation and remote orchestration tests; no hardware or SSH access."""
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from extrinsic_calibration import make_limits, parser, recommendation_label, plane_side_section
from floor_geometry import CalibrationError
from local_workflow import open_summary, record_from_jetson, validate_remote_args


class PresentationTests(unittest.TestCase):
    def test_remote_default_is_named_desktop_results_folder(self):
        args = parser().parse_args(['remote', 'teamc@jetson'])
        self.assertEqual(args.output_root, Path.home()/'Desktop/Camera_Calibration_Results')

    def test_thirty_degree_mount_not_changed_by_thirteen_degree_vo_grid_tilt(self):
        from extrinsic_calibration import attach_pictured_plane_fit
        from floor_geometry import rpy_matrix
        from flight_geometry import px4_up_flu
        old = rpy_matrix(.35, 30.5, 180.)
        up = px4_up_flu(*np.radians([2., -3.]))
        report = dict(mode='flight', yaw_deg_fixed=180., per_frame=[dict(
            frame_index=0, accepted=True, px4_up_body=up.tolist())])
        for true_pitch in [27., 30., 32., 35.]:
            actual = rpy_matrix(.5, true_pitch, 180.)
            normal = old @ actual.T @ up
            fit = dict(normal=normal, frame_index=0, count=100, rms=.002)
            for roll, pitch, yaw in [(0., 0., 0.), (13., 0., 90.), (-7., 11., -97.), (50., -30., 180.)]:
                frame = dict(stamp=100., r_bc=old, t_bc=np.zeros(3), r_gb=rpy_matrix(roll, pitch, yaw))
                attach_pictured_plane_fit(report, fit, frame)
                np.testing.assert_allclose(report['pictured_plane_fit']['rpy_deg'], [.5, true_pitch, 180.], atol=1e-8)
                self.assertNotEqual(report['pictured_plane_fit']['target'], 'rviz_grid_normal')

    def test_mount_fit_has_no_fallback_to_grid_when_px4_missing(self):
        from extrinsic_calibration import attach_pictured_plane_fit
        report = dict(mode='flight', yaw_deg_fixed=180., per_frame=[])
        with self.assertRaisesRegex(CalibrationError, 'matched PX4'):
            attach_pictured_plane_fit(report, dict(frame_index=0), {})
        self.assertNotIn('pictured_plane_fit', report)

    def test_recommendation_is_three_decimal_degrees(self):
        label, _ = recommendation_label(dict(accepted=True, suggested_rpy_deg=[1.23456, 30.5, 180]))
        self.assertIn('RECOMMENDED EXTRINSICS', label)
        self.assertIn('+1.235°', label)
        self.assertIn('+30.500°', label)

    def test_bad_fit_is_never_presented_as_recommended(self):
        label, _ = recommendation_label(dict(accepted=False, candidate_rpy_deg=[1.2, 32., 180.]))
        self.assertIn('PROVISIONAL FIT', label)
        self.assertIn('NOT RECOMMENDED; do not apply', label)
        label, _ = recommendation_label(dict(accepted=False))
        self.assertIn('UNAVAILABLE', label)

    def test_approximate_height_default_and_explicit_measurement(self):
        args = parser().parse_args(['capture', '--height-m', '1.47'])
        self.assertEqual(make_limits(args).height_tolerance_m, .5)
        args = parser().parse_args(['capture', '--height-m', '.47'])
        self.assertAlmostEqual(make_limits(args).height_tolerance_m, .235)
        args = parser().parse_args(['capture', '--height-m', '1.47', '--height-tolerance-m', '.12'])
        self.assertEqual(make_limits(args).height_tolerance_m, .12)
        args = parser().parse_args(['capture', '--stationary', '--height-m', '.47'])
        self.assertEqual(make_limits(args).height_tolerance_m, .12)

    def test_open_only_local_summary_and_headless_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path/'current_floor_view.png').touch()
            with patch.dict(os.environ, {}, clear=True), patch('local_workflow.subprocess.Popen') as popen:
                self.assertFalse(open_summary(path))
                popen.assert_not_called()
            with patch.dict(os.environ, {'DISPLAY': ':0'}), patch('local_workflow.shutil.which', return_value='/usr/bin/xdg-open'), patch('local_workflow.subprocess.Popen') as popen:
                self.assertTrue(open_summary(path))
                self.assertEqual(popen.call_args.args[0], ['/usr/bin/xdg-open', str(path/'current_floor_view.png')])
                (path/'calibration_before_after.png').touch()
                self.assertTrue(open_summary(path))
                self.assertEqual(popen.call_args.args[0], ['/usr/bin/xdg-open', str(path/'calibration_before_after.png')])

    def test_three_panels_and_recommendation_really_reach_figure(self):
        from extrinsic_calibration import save_plots
        from floor_geometry import Limits, analyze_frames
        from test_calibration import scene
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.figure import Figure
        frames = [scene(seed=i) for i in range(3)]
        report, fits = analyze_frames(frames, Limits(.47, min_frames=3, min_duration_s=.5))
        observed = []
        def inspect(fig, *args, **kwargs):
            observed.append(([a.name for a in fig.axes], [t.get_text() for t in fig.texts],
                             [a.xaxis_inverted() for a in fig.axes],
                             [a.get_title() for a in fig.axes]))
            # Exact screen-coordinate projection: Y view must discard Y,
            # not project XYZ through an isometric camera.
            fitted = min(fits, key=lambda item: item['rms'])
            raw = frames[fitted['frame_index']]
            centre = np.mean(fitted['points'] @ raw['r_gb'].T+raw['t_gb'], axis=0)
            expected = raw['points_body'] @ raw['r_gb'].T+raw['t_gb']-centre
            actual_y = np.asarray(fig.axes[2].collections[0].get_offsets())
            actual_x = np.asarray(fig.axes[1].collections[0].get_offsets())
            # Synthetic fixture points all pass the 6 m/finite display filter.
            if len(observed) == 1:
                np.testing.assert_allclose(actual_y, expected[:, [0, 2]])
                np.testing.assert_allclose(actual_x, expected[:, [1, 2]])
        with tempfile.TemporaryDirectory() as tmp, patch.object(Figure, 'savefig', inspect):
            save_plots(Path(tmp), report, fits, frames, 'map')
        self.assertEqual(len(observed), 3)
        self.assertEqual(observed[0][0], ['3d', 'rectilinear', 'rectilinear'])
        self.assertEqual(observed[0][2][1:], [False, True])
        self.assertIn('+Y (X–Z)', observed[0][3][2])
        self.assertTrue(any('Active:' in t for t in observed[0][1]))
        self.assertIn('CLI Roll:', observed[1][3][1])
        self.assertIn('CLI Pitch:', observed[1][3][2])
        self.assertIn('=', observed[1][3][1])
        self.assertIn('=', observed[1][3][2])
        self.assertFalse(any('tilt' in title.lower() for item in observed for title in item[3]))

    def test_side_plane_sections_and_hidden_dimension(self):
        # Plane z = .2*x + .3*y - 1. Oblique in both dimensions.
        h = np.array([-2., 0., 3.])
        np.testing.assert_allclose(plane_side_section([-.2, -.3, 1], 1, 0, 2., h), .3*h-.6)
        np.testing.assert_allclose(plane_side_section([-.2, -.3, 1], 1, 1, 2., h), .2*h-.4)
        self.assertIsNone(plane_side_section([1, 0, 0], 0, 1, 2., h))

    def test_plot_does_not_replace_absolute_recording_fit_with_single_snapshot(self):
        from extrinsic_calibration import attach_pictured_plane_fit
        from flight_geometry import analyze_flight, FlightTelemetry
        from floor_geometry import Limits, rpy_matrix
        from test_flight import fixture
        with tempfile.TemporaryDirectory() as tmp:
            frames, packets = fixture(Path(tmp))
            result, fits = analyze_flight(frames, Limits(1.), FlightTelemetry(packets), tmp)
        result['accepted'] = False
        expected = list(np.round(result['candidate_rpy_deg'], 3))
        fits[0]['normal'] = rpy_matrix(.5, 0., 0.) @ fits[0]['normal']
        attach_pictured_plane_fit(result, fits[0], frames[fits[0]['frame_index']])
        pictured = result['pictured_plane_fit']
        self.assertEqual(pictured['target'], 'matched_px4_tilt')
        self.assertGreater(pictured['after_tilt_to_physical_level_deg'], .2)
        np.testing.assert_allclose(pictured['rpy_deg'], expected)
        np.testing.assert_allclose(pictured['rpy_deg'], [1.2, 32., 180.], atol=.06)
        self.assertFalse(result['accepted'])  # Drawing a preview does not certify calibration.
        self.assertEqual(list(np.round(result['candidate_rpy_deg'], 3)), expected)

    def test_rotated_cloud_and_plane_are_parallel_to_grid_without_flattening_points(self):
        from extrinsic_calibration import attach_pictured_plane_fit
        from calibration_plots import plot_scene
        from floor_geometry import Limits, analyze_frames, level_basis
        from test_calibration import scene
        frames = [scene(seed=i) for i in range(3)]
        result, fits = analyze_frames(frames, Limits(.47, min_frames=3, min_duration_s=.5))
        fit = min(fits, key=lambda f: f['rms']); frame = frames[fit['frame_index']]
        attach_pictured_plane_fit(result, fit, frame)
        cases, grid, _, _, _ = plot_scene(result, fit, frame)
        self.assertLess(np.ptp(cases[1]['surface'][:, :, 2]), .0001)
        self.assertLess(np.ptp(grid[:, :, 2]), 1e-10)
        np.testing.assert_allclose(cases[1]['normal'], [0, 0, 1], atol=2e-5)
        np.testing.assert_allclose(cases[0]['odom_rotation'],
                                   level_basis(result['pictured_plane_fit']['target_up_body']) @ frame['r_ob'].T)
        np.testing.assert_array_equal(cases[0]['odom_rotation'], cases[1]['odom_rotation'])
        # Genuine rigid rotation: retain cloud shape/noise, do not simply set Z=0.
        before, after = cases[0]['ground'], cases[1]['ground']
        np.testing.assert_allclose(np.linalg.norm(before-before[0], axis=1),
                                   np.linalg.norm(after-after[0], axis=1), atol=1e-10)
        self.assertGreater(float(np.std(after[:, 2])), .0001)
        self.assertAlmostEqual(float(np.std(after[:, 2])), fit['rms'], places=6)
        # The raw RViz view must not be silently replaced by a horizontal one.
        raw_cases, _, _, _, _ = plot_scene(result, fit, frame, reference='recorded')
        np.testing.assert_allclose(raw_cases[0]['normal'], frame['r_gb'] @ fit['normal'])
        self.assertGreater(np.ptp(raw_cases[1]['surface'][:, :, 2]), .01)

    def test_inconsistent_capture_still_draws_plane_and_rotation_preview(self):
        from extrinsic_calibration import finalize
        from floor_geometry import Limits
        from test_calibration import scene
        frames = [scene(actual=(i*2., 32., 180.), seed=i) for i in range(3)]
        meta = dict(grid_frame='map', body_frame='drone_link', camera_frame='camera_link',
                    odom_frame='odom', topic='/synthetic', level_confirmed=True, synthetic_test=True)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            rc = finalize(folder, meta, frames, Limits(.47, min_frames=3, min_duration_s=.5))
            self.assertEqual(rc, 2)
            result = json.loads((folder/'report.json').read_text())
            self.assertFalse(result['accepted'])
            self.assertIn('pictured_plane_fit', result)
            for name in ('current_floor_view.png', 'predicted_floor_view.png', 'calibration_before_after.png'):
                self.assertGreater((folder/name).stat().st_size, 10000)

    def test_cli_file_report_and_plot_use_the_same_rounded_mount(self):
        from extrinsic_calibration import finalize
        from floor_geometry import Limits, rpy_matrix
        from test_calibration import scene
        frames = [scene(actual=(-.73149, 33.82451, 180.), seed=i) for i in range(3)]
        meta = dict(grid_frame='map', body_frame='drone_link', camera_frame='camera_link',
                    odom_frame='odom', topic='/synthetic', level_confirmed=True, synthetic_test=True)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            finalize(folder, meta, frames, Limits(.47, min_frames=3, min_duration_s=.5))
            result = json.loads((folder/'report.json').read_text())
            cli = result['cli_values']
            values = [cli['roll_deg'], cli['pitch_deg'], cli['yaw_deg']]
            self.assertEqual(values, result['pictured_plane_fit']['rpy_deg'])
            self.assertEqual(values, list(np.round(values, 3)))
            np.testing.assert_allclose(result['pictured_plane_fit']['correction_rotation'],
                                       rpy_matrix(*values) @ frames[0]['r_bc'].T, atol=1e-12)
            text = (folder/'recommended_cli_settings.txt').read_text()
            for name, value in zip(('Roll:', 'Pitch:', 'Yaw:'), values):
                row = next(line for line in text.splitlines() if line.startswith(name))
                self.assertEqual(float(row.split()[1]), value)
            self.assertFalse(cli['applied'])


class WorkflowTests(unittest.TestCase):
    def args(self, root, host='teamc@jetson.example'):
        return SimpleNamespace(host=host, output_root=root, no_open=False,
                               height_m=1.47, height_tolerance_m=None, grid_frame='map')

    def test_host_option_shell_injection_and_bad_height_rejected(self):
        for host in ('-oProxyCommand=bad', 'teamc@jetson;command', 'ssh://jetson', 'a b'):
            with self.subTest(host=host), self.assertRaises(CalibrationError):
                validate_remote_args(self.args('/tmp/not-created', host))
        args = self.args('/tmp/not-created')
        args.height_m = float('nan')
        with self.assertRaises(CalibrationError):
            validate_remote_args(args)

    def test_stop_then_copy_then_local_analysis_then_view(self):
        calls, order = [], []
        with tempfile.TemporaryDirectory() as tmp:
            def run(cmd, **kwargs):
                calls.append(cmd)
                if cmd[0] == 'scp':
                    order.append('copy')
                    dest = Path(cmd[-1]); dest.mkdir()
                    (dest/'frames').mkdir()
                    (dest/'capture.json').write_text(json.dumps(dict(schema=1, thresholds=dict(height_m=1.47), yaw_deg_fixed=180)))
                elif '-O' not in cmd:
                    order.append('capture')
                return SimpleNamespace(returncode=0)
            def analyze(output, meta, frames, limits, yaw, source_folder):
                order.append('local_analysis')
                self.assertTrue((source_folder/'capture.json').is_file())
                self.assertEqual(output.parent, source_folder.parent)
                return 0
            with patch('local_workflow.shutil.which', side_effect=lambda x: '/usr/bin/'+x), patch('local_workflow.subprocess.run', side_effect=run), patch('extrinsic_calibration.finalize', side_effect=analyze), patch('local_workflow.open_summary', side_effect=lambda p: order.append('view')):
                self.assertEqual(record_from_jetson(self.args(tmp)), 0)
        self.assertEqual(order, ['capture', 'copy', 'local_analysis', 'view'])
        remote = next(c for c in calls if c[0] == 'ssh' and '-t' in c)
        self.assertIn('extrinsic_calibration capture', remote[-1])
        self.assertNotIn('analyze', remote[-1])
        self.assertNotIn('password', ' '.join(remote))
        self.assertTrue(any('-O' in c for c in calls))

    def test_failed_ssh_or_copy_never_analyzes(self):
        for failure in ('ssh', 'scp'):
            with tempfile.TemporaryDirectory() as tmp:
                def run(cmd, **kwargs):
                    return SimpleNamespace(returncode=1 if cmd[0] == failure and '-O' not in cmd else 0)
                with patch('local_workflow.shutil.which', return_value='/usr/bin/tool'), patch('local_workflow.subprocess.run', side_effect=run), patch('extrinsic_calibration.finalize') as analyze:
                    with self.assertRaises(CalibrationError):
                        record_from_jetson(self.args(tmp))
                    analyze.assert_not_called()


if __name__ == '__main__':
    unittest.main()
