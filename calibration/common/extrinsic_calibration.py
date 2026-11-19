#!/usr/bin/env python3
"""Standalone passive floor calibration: capture, analyze, and plot. Never applies settings."""
import argparse
from collections import deque
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

# Avoid competing with cuVSLAM for many CPU threads on the Jetson.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

from floor_geometry import (CalibrationError, Limits, analyze_frames, angle_deg,
                            quaternion_matrix, read_xyz, rpy_matrix, suggest_extrinsic,
                            angle_change_lines)

VERSION = "1.4.1"


def recommendation_label(report):
    """Never label an inconsistent/partial fit as a recommended setting."""
    if report.get("accepted") and "suggested_rpy_deg" in report:
        r, p, y = report["suggested_rpy_deg"]
        return f"RECOMMENDED EXTRINSICS: Roll {r:+.3f}°   |   Pitch {p:+.3f}°   |   Yaw {y:+.3f}° (fixed)", "#146344"
    if "candidate_rpy_deg" in report:
        r, p, _ = report["candidate_rpy_deg"]
        return f"PROVISIONAL FIT: Roll {r:+.3f}°   |   Pitch {p:+.3f}°   —   NOT RECOMMENDED; do not apply", "#9b3c16"
    return "ROLL / PITCH UNAVAILABLE — no usable floor and reference pair", "#9b3c16"


def attach_pictured_plane_fit(report, fit, frame):
    """Camera-to-body mounting fit, independent of arbitrary VO world attitude.

    Use the independently paired PX4 tilt (flight) or the confirmed level-body
    assumption (stationary). Never use RViz +Z to infer the mounting angle.
    """
    if report.get("mode") == "flight":
        row = next((row for row in report['per_frame']
                    if row['frame_index'] == fit['frame_index'] and row['accepted']), None)
        if row is None or 'px4_up_body' not in row:
            raise CalibrationError('mount fit requires the matched PX4 attitude; no grid fallback')
        up = np.asarray(row['px4_up_body'], dtype=float)
        reference = 'matched_px4_tilt'
    else:
        up = np.array([0., 0., 1.])
        reference = 'independently_level_body'
    # The displayed snapshot must not replace the all-frame absolute fit.
    if report.get('absolute_fit'):
        solution = np.asarray(report['candidate_rpy_deg'], dtype=float)
    else:
        solution, _ = suggest_extrinsic(fit["normal"], frame["r_bc"], report["yaw_deg_fixed"], up)
    # Plot exactly the rounded absolute solution printed for the operator.
    rpy = np.round(solution, 3)
    correction = rpy_matrix(*rpy) @ frame['r_bc'].T
    after_normal = correction @ fit["normal"]
    report["pictured_plane_fit"] = dict(
        frame_index=fit["frame_index"], stamp=float(frame["stamp"]),
        selection="display snapshot; angles use all paired frames" if report.get('absolute_fit') else "lowest RMS qualified floor snapshot", points=int(fit["count"]),
        plane_rms_m=float(fit["rms"]), rpy_deg=rpy.tolist(),
        solver_rpy_deg=solution.tolist(),
        correction_rotation=correction.tolist(), normal_body=fit["normal"].tolist(),
        target=reference, target_up_body=up.tolist(),
        before_tilt_to_physical_level_deg=angle_deg(fit['normal'], up),
        after_tilt_to_physical_level_deg=angle_deg(after_normal, up),
        before_tilt_to_grid_deg=angle_deg(frame["r_gb"] @ fit["normal"], [0, 0, 1]),
        after_tilt_to_grid_deg=angle_deg(frame["r_gb"] @ after_normal, [0, 0, 1]),
        scope="absolute camera-to-body fit using independent body tilt; yaw and XYZ fixed; not applied",
        preview="same camera-relative points with current/proposed mounting, in an independently level-start reference; not a cuVSLAM replay or a measured post-change result")
    report['cli_values'] = dict(roll_deg=float(rpy[0]), pitch_deg=float(rpy[1]),
                               yaw_deg=float(rpy[2]), xyz_m=np.asarray(frame['t_bc']).tolist(),
                               units='metres and ROS FLU degrees; absolute values, not increments',
                               source='all paired camera-frame floor normals and independent body tilt' if report.get('absolute_fit') else 'pictured floor patch and independent body tilt',
                               status='candidate_for_manual_validation' if report.get('accepted') else 'provisional_candidate',
                               applied=False)


def plane_side_section(normal_odom, offset_odom, view_axis, depth, horizontal):
    """True side-view section, not a perspective view of a rectangular mesh.

    +X view: hold X=depth, show Y/Z. +Y view: hold Y=depth, show X/Z;
    its screen X axis is reversed separately (positive world X points left).
    """
    if view_axis not in (0, 1):
        raise ValueError("side view must be X or Y")
    n = np.asarray(normal_odom, dtype=float)
    if abs(n[2]) < 1e-8:
        return None  # A vertical plane cannot be represented as Z(horizontal).
    horizontal_axis = 1-view_axis
    return -(n[horizontal_axis]*np.asarray(horizontal)+n[view_axis]*depth+offset_odom)/n[2]


def new_directory(path=None):
    if path is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = Path.home()/".local/state/extrinsic_calibration"/(stamp+"-"+uuid.uuid4().hex[:6])
    path = Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n", encoding="utf-8")


class FrameFiles:
    """Lazy disk-backed snapshots; flight length does not consume cloud RAM."""
    def __init__(self, folder):
        self.paths = sorted((Path(folder)/"frames").glob("*.npz"))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with np.load(self.paths[index], allow_pickle=False) as saved:
            return {key: saved[key].copy() for key in saved.files}


def load_frames(folder):
    folder = Path(folder).expanduser()
    meta = json.loads((folder/"capture.json").read_text())
    if meta.get("schema") != 1:
        raise CalibrationError("unsupported capture format")
    return meta, FrameFiles(folder)


def tf_arrays(tf):
    t, q = tf.transform.translation, tf.transform.rotation
    translation = np.array([t.x, t.y, t.z])
    if not np.isfinite(translation).all():
        raise CalibrationError("invalid TF translation")
    return quaternion_matrix(q.x, q.y, q.z, q.w), translation


def save_plots(folder, report, fits, frames, grid_frame):
    from calibration_plots import save_plane_plots
    if fits:
        pictured = min(fits, key=lambda item: item["rms"])
        attach_pictured_plane_fit(report, pictured, frames[pictured["frame_index"]])
    return save_plane_plots(folder, report, fits, frames, grid_frame)


def finalize(folder, meta, frames, limits, yaw_deg=180.0, capture_problems=(), source_folder=None):
    flight = meta.get("mode") == "flight"
    if flight:
        from flight_geometry import FlightTelemetry, analyze_flight
        report, fits = analyze_flight(frames, limits, FlightTelemetry.load(source_folder), source_folder, yaw_deg)
    else:
        report, fits = analyze_frames(frames, limits, yaw_deg)
    report.update(tool_version=VERSION, grid_frame=meta["grid_frame"],
                  body_frame=meta["body_frame"], odom_frame=meta["odom_frame"],
                  camera_frame=meta["camera_frame"], topic=meta["topic"],
                  level_confirmed=meta.get("level_confirmed", False),
                  synthetic_test=meta.get("synthetic_test", False),
                  created_unix_s=time.time(), capture_start_unix_s=meta.get('start_unix_s'),
                  hardware_verified=False)
    problems = list(capture_problems) + meta.get("capture_problems", [])
    if flight and not meta.get("capture_complete"):
        problems.append("recording was not completed with STOP; partial data is diagnostic only")
    if not flight and not report["level_confirmed"]:
        problems.append("level drone and horizontal floor were not confirmed")
    if problems:
        report["accepted"] = False
        report["reasons"] = list(dict.fromkeys(report["reasons"]+problems))
        report.pop("suggested_rpy_deg", None)
        report.pop("correction_rotation", None)
    if fits:
        representative = min(fits, key=lambda x: x["rms"])
        frame = frames[representative["frame_index"]]
        n = representative["normal"]
        report["representative_frame_index"] = representative["frame_index"]
        report["floor_normal_in_grid"] = (frame["r_gb"] @ n).tolist()
        report["floor_normal_in_odom"] = (frame["r_ob"] @ n).tolist()
        report["floor_tilt_to_grid_deg"] = angle_deg(frame["r_gb"] @ n, [0, 0, 1])
        report["floor_tilt_to_odom_deg"] = angle_deg(frame["r_ob"] @ n, [0, 0, 1])
        attach_pictured_plane_fit(report, representative, frame)
    write_json(folder/"report.json", report)  # Preserve evidence even if plotting fails.
    figures = save_plots(folder, report, fits, frames, meta["grid_frame"]) if frames else []
    if flight:
        from flight_plots import save_pose_plot
        plot = save_pose_plot(folder, source_folder, frames, report)
        if plot:
            figures.append(plot)
    report["plots"] = figures
    write_json(folder/"report.json", report)
    lines = ["EXTRINSIC CALIBRATION", "", f"Grid frame: {meta['grid_frame']} (operator confirmed)",
             f"Good floor frames: {report.get('good_frames', 0)}/{len(frames)}"]
    if report.get('synthetic_test'):
        lines[0:0] = ["SYNTHETIC TEST DATA — NOT YOUR CAMERA. DO NOT APPLY THESE VALUES.", ""]
    if "measured_tilt_body_deg" in report:
        lines.extend([f"Measured floor tilt relative to level body: {report['measured_tilt_body_deg']:.3f} deg",
                      f"Floor tilt relative to RViz grid: {report['floor_tilt_to_grid_deg']:.3f} deg",
                      f"Floor tilt relative to odom XY: {report['floor_tilt_to_odom_deg']:.3f} deg",
                      f"Floor fit RMS: {report['mean_floor_rms_m']*1000:.3f} mm",
                      f"Maximum frame-to-frame normal deviation: {report['normal_max_deviation_deg']:.3f} deg"])
    if flight and fits:
        lines.extend([f"Floor tilt relative to RViz grid: {report['floor_tilt_to_grid_deg']:.3f} deg",
                      f"Floor fit RMS: {report['mean_floor_rms_m']*1000:.3f} mm",
                      f"Mounting estimate maximum deviation: {report['mounting_max_deviation_deg']:.3f} deg",
                      "PX4 accounts for real body tilt. Fast-tilting and poorly matched samples are excluded.",
                      "Time matching uses host receive time; it is not hardware synchronization."])
    if "pictured_plane_fit" in report:
        pictured = report["pictured_plane_fit"]
        r, p, y = pictured["rpy_deg"]
        lines.extend(["", "ABSOLUTE VO EXTRINSICS (candidate; nothing applied)",
                      f"Roll: {r:+.3f} deg; Pitch: {p:+.3f} deg; Yaw: {y:+.3f} deg (fixed)",
                      f"Frame: {pictured['frame_index']}; points: {pictured['points']}; RMS: {pictured['plane_rms_m']*1000:.3f} mm",
                      f"Mounting reference: {pictured['target']}; yaw and XYZ are unchanged.",
                      "current_floor_view.png retains the actual RViz cloud/grid. The comparison models a fresh level start,",
                      "using independent body tilt, not a floor-flattening display rotation. Verify after restarting VO."])
    if 'cli_values' in report:
        candidate = report['cli_values']
        roll, pitch, yaw = [candidate[k] for k in ('roll_deg', 'pitch_deg', 'yaw_deg')]
        x, y, z = candidate['xyz_m']
        values = ['VO EXTRINSIC CANDIDATE — absolute ROS FLU values',
                  f'Roll:  {roll:+.3f} deg', f'Pitch: {pitch:+.3f} deg',
                  f'Yaw:   {yaw:+.3f} deg (fixed; not estimated)',
                  f'XYZ:   ({x:+.3f}, {y:+.3f}, {z:+.3f}) m (unchanged)',
                  f"Quality: {candidate['status']}",
                  'Same values as the plot. Nothing applied.',
                  'After entering values manually, restart VO with the body level and verify the new floor.']
        values[1:1] = ['CLI '+line for line in angle_change_lines(report['current_rpy_deg'], [roll, pitch, yaw])] + ['']
        (folder/'recommended_cli_settings.txt').write_text('\n'.join(values)+'\n', encoding='utf-8')
        lines.extend(['', *values])
    if not report['accepted']:
        lines.extend(['', 'Across-recording checks (the pictured fit remains available for review):']
                     + ['- '+x for x in report['reasons']])
    if report.get('repeatability_std_roll_pitch_deg') is not None:
        rs, ps = report['repeatability_std_roll_pitch_deg']
        lines.extend([f"Estimate repeatability std: roll {rs:.3f} deg; pitch {ps:.3f} deg (not absolute accuracy)"])
    lines.extend(["", "Assumptions: "+report.get('assumption', '')+".",
                  "A fit alone cannot prove these assumptions. Three decimals are display precision only.",
                  "Stereo reconstruction bias can look like mounting error. Repeatability is NOT absolute accuracy.",
                  "A single floor cannot determine yaw or XYZ. No floor flattening is applied to flight data.",
                  "", "Reports: "+str(folder)])
    text = "\n".join(lines)+"\n"
    (folder/"report.txt").write_text(text, encoding="utf-8")
    if 'cli_values' in report:
        print('\n'.join(values))
        print(f"Floor fit: {report['pictured_plane_fit']['points']} points, "
              f"RMS {report['pictured_plane_fit']['plane_rms_m']*1000:.3f} mm.")
        if not report['accepted']:
            print('Recording consistency needs review; details are in report.txt.')
        print(f'Results: {folder}')
    else:
        print(text)
    return 0 if report["accepted"] else 2


def capture(args):
    if args.height_m is None:
        args.height_m = float(input("Height from Cube Orange to the physical ground plane (m): "))
    if args.grid_frame is None:
        args.grid_frame = input("RViz Global Options > Fixed Frame [map]: ").strip() or "map"
    if not args.level_confirmed:
        print("GROUND ONLY. Keep the disarmed drone stationary and independently level over a known flat floor.")
        print("Measure height at the Cube/drone_link origin, not at the camera. Do not cover the camera.")
        if input("Confirm those conditions and the RViz frame above by typing LEVEL: ").strip() != "LEVEL":
            raise CalibrationError("capture cancelled; nothing changed")
    limits = make_limits(args)
    limits.validate()
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from rclpy.time import Time
        from rclpy.duration import Duration
        from sensor_msgs.msg import PointCloud2
        from tf2_ros import Buffer, TransformListener
        from tf2_py import TransformException
    except ImportError as exc:
        raise CalibrationError("ROS 2 unavailable. Run source /opt/ros/humble/setup.bash first.") from exc
    folder = new_directory(args.output)
    (folder/"frames").mkdir()
    meta = dict(schema=1, version=VERSION, start_utc=datetime.now(timezone.utc).isoformat(),
                level_confirmed=True, grid_frame=args.grid_frame, body_frame=args.body_frame,
                camera_frame=args.camera_frame, odom_frame=args.odom_frame, topic=args.topic,
                thresholds=asdict(limits), yaw_deg_fixed=args.yaw_deg,
                capture_problems=[], diagnostics={},
                synthetic_test=os.environ.get("EXTRINSIC_CALIBRATION_TEST_DATA") == "1")
    frames, clouds, fingerprints = [], deque(maxlen=40), set()
    rclpy.init()
    node = Node("extrinsic_calibration_observer")
    buffer = Buffer(cache_time=Duration(seconds=30))
    listener = TransformListener(buffer, node)
    subscription = node.create_subscription(PointCloud2, args.topic, clouds.append, qos_profile_sensor_data)
    start = time.monotonic()
    last_sample = start-args.interval
    last_stamp, last_print = 0, start
    diagnostics = dict(received=0, missing_tf=0, stale_cloud=0, malformed_cloud=0)
    print(f"Reading {args.topic}; grid={args.grid_frame}; yaw fixed={args.yaw_deg:g} deg.")
    print(f"Recording for {args.seconds:g} s. No camera, bridge or PX4 settings will be changed.")
    print(f"Output: {folder}", flush=True)

    def lookup(target, source, stamp):
        if target == source:
            return np.eye(3), np.zeros(3)
        return tf_arrays(buffer.lookup_transform(target, source, stamp))

    try:
        while time.monotonic()-start < args.seconds:
            rclpy.spin_once(node, timeout_sec=.05)
            now = time.monotonic()
            if node.count_publishers(args.topic) > 1:
                meta["capture_problems"].append("multiple point-cloud publishers on the selected topic")
                break
            if now-last_sample < args.interval:
                continue
            # Try newest usable cloud, then slightly older ones. TF queries use
            # the exact cloud stamp, NEVER latest-transform/time-zero fallback.
            for msg in reversed(list(clouds)):
                stamp_ns = msg.header.stamp.sec*1_000_000_000+msg.header.stamp.nanosec
                if stamp_ns <= last_stamp or stamp_ns <= 0:
                    continue
                age = (node.get_clock().now().nanoseconds-stamp_ns)*1e-9
                if not -.02 <= age <= args.max_cloud_age:
                    diagnostics["stale_cloud"] += 1
                    continue
                stamp = Time.from_msg(msg.header.stamp)
                try:
                    r_bc, t_bc = lookup(args.body_frame, args.camera_frame, stamp)
                    r_bs, t_bs = lookup(args.body_frame, msg.header.frame_id, stamp)
                    r_ob, t_ob = lookup(args.odom_frame, args.body_frame, stamp)
                    r_gb, t_gb = lookup(args.grid_frame, args.body_frame, stamp)
                    points = read_xyz(msg)
                except TransformException:
                    diagnostics["missing_tf"] += 1
                    continue
                except CalibrationError:
                    diagnostics["malformed_cloud"] += 1
                    continue
                if not len(points):
                    continue
                if len(points) > 12000:
                    points = points[np.linspace(0, len(points)-1, 12000, dtype=int)]
                fingerprints.add(hashlib.sha256(points.tobytes()).hexdigest())
                body = points @ r_bs.T+t_bs
                frame = dict(stamp=np.array(stamp_ns*1e-9), points_body=body,
                             r_bc=r_bc, t_bc=t_bc, r_ob=r_ob, t_ob=t_ob, r_gb=r_gb, t_gb=t_gb)
                np.savez_compressed(folder/"frames"/f"{len(frames):04d}.npz", **frame)
                frames.append(frame)
                last_stamp, last_sample = stamp_ns, now
                diagnostics["received"] += 1
                break
            if now-last_print >= 2:
                print(f"{now-start:.0f}/{args.seconds:g} s: {len(frames)} timestamp-matched snapshots", flush=True)
                last_print = now
    except KeyboardInterrupt:
        meta["capture_problems"].append("capture interrupted")
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    if len(fingerprints) < 3:
        meta["capture_problems"].append("fewer than 3 changing cloud contents; data may be frozen")
    if not frames or time.monotonic()-last_sample > max(1.5, 3*args.interval):
        meta["capture_problems"].append("cloud or timestamp-matched TF stopped before capture ended")
    expected = args.seconds/args.interval
    if len(frames) < .6*expected:
        meta["capture_problems"].append("insufficient capture coverage; clouds/TF were missing for too long")
    meta["diagnostics"] = diagnostics
    meta["unique_cloud_contents"] = len(fingerprints)
    meta["captured_frames"] = len(frames)
    write_json(folder/"capture.json", meta)
    print("Stationary capture complete. Fitting the floor and generating the three-view plots...", flush=True)
    return finalize(folder, meta, frames, limits, args.yaw_deg)


def make_limits(args):
    values = {key: getattr(args, key) for key in ("height_m", "height_tolerance_m", "distance_threshold_m",
              "min_points", "min_span_m", "max_range_m", "max_correction_deg")}
    if values["height_tolerance_m"] is None:
        # An approximate Cube-to-ground height is a surface-search aid.
        values["height_tolerance_m"] = min(.12 if args.stationary else .5, args.height_m*.5)
    return Limits(**values)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--version", action="version", version=VERSION)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("capture", help="passively record the live cuVSLAM cloud + timestamp-matched TF")
    c.add_argument("--height-m", type=float, help="height from Cube Orange to the physical ground plane at START, metres")
    c.add_argument("--grid-frame", help="RViz's actual Fixed Frame, normally map or odom; confirm in GUI")
    c.add_argument("--body-frame", default="drone_link")
    c.add_argument("--camera-frame", default="camera_link")
    c.add_argument("--odom-frame", default="odom")
    c.add_argument("--topic", default="/visual_slam/vis/observations_cloud")
    c.add_argument("--seconds", type=float, help="explicit timed recording; default waits for START and STOP")
    c.add_argument("--interval", type=float, default=.5)
    c.add_argument("--max-cloud-age", type=float, default=1.)
    c.add_argument("--level-confirmed", action="store_true", help="explicitly affirm disarmed, stationary, level body and level floor")
    c.add_argument("--stationary", action="store_true", help="legacy level-ground calibration instead of PX4-assisted flight recording")
    c.add_argument("--pose-topic", default="/visual_slam/tracking/odometry")
    c.add_argument("--telemetry-port", type=int, default=14691, help="receive-only localhost diagnostic port; never a serial port")
    c.add_argument("--min-free-mb", type=float, default=1024., help="stop recorder only if disk free space drops below this")
    c.add_argument("--yaw-deg", type=float, default=180., help="fixed mounting yaw; never estimated from the floor")
    c.add_argument("--output", type=Path, help="new results directory; existing directories are never overwritten")
    c.add_argument("--no-transfer-prompt", action="store_true", help="capture only; the PC remote wrapper handles transfer")
    c.add_argument("--height-tolerance-m", type=float,
                   help="floor search +/- metres; flight default min(0.5 m, half the entered height)")
    c.add_argument("--distance-threshold-m", type=float, default=.025)
    c.add_argument("--min-points", type=int, default=50)
    c.add_argument("--min-span-m", type=float, default=.35)
    c.add_argument("--max-range-m", type=float, default=6.)
    c.add_argument("--max-correction-deg", type=float, default=10.)
    a = sub.add_parser("analyze", help="regenerate report/plots from a saved capture, without ROS")
    a.add_argument("capture_directory", type=Path)
    a.add_argument("--output", type=Path, help="new report directory")
    a.add_argument("--height-tolerance-m", type=float, help="override an old capture's height search allowance; raw capture is unchanged")
    a.add_argument("--open", action="store_true", dest="open_plot", help="open the summary on this computer's desktop")
    r = sub.add_parser("remote", help="run on this PC: record on Jetson; STOP downloads, plots and opens results locally")
    r.add_argument("host", help="SSH target, e.g. teamc@JETSON_IP; normal SSH authentication")
    r.add_argument("--output-root", type=Path, default=Path.home()/"Desktop/Camera_Calibration_Results")
    r.add_argument("--no-open", action="store_true", help="save to Desktop without opening an image viewer")
    r.add_argument("--height-m", type=float, help="height from Cube Orange to the physical ground plane at START, metres; otherwise prompt on Jetson")
    r.add_argument("--height-tolerance-m", type=float, help="explicit floor-height search allowance")
    r.add_argument("--grid-frame", help="optional RViz fixed frame; otherwise prompt on Jetson")
    t = sub.add_parser("transfer", help="on Jetson: transfer a saved recording to your PC and plot there")
    t.add_argument("capture_directory", type=Path)
    t.add_argument("host", help="your COMPUTER username@IP or SSH alias")
    return p


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = parser().parse_args(argv or ["capture"])
    try:
        if args.command == "transfer":
            from transfer_results import transfer_and_plot
            transfer_and_plot(args.capture_directory, args.host)
            return 0
        if args.command == "remote":
            from local_workflow import record_from_jetson
            return record_from_jetson(args)
        if args.command == "capture":
            if (not all(math.isfinite(x) and x > 0 for x in (args.interval, args.max_cloud_age, args.min_free_mb))
                    or (args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0))
                    or not math.isfinite(args.yaw_deg)):
                raise CalibrationError("invalid capture duration, interval, age, or yaw")
            if args.stationary:
                args.seconds = args.seconds if args.seconds is not None else 20.
                return capture(args)
            if args.yaw_deg != 180.:
                raise CalibrationError("flight tool fixes mounting yaw at 180 degrees; yaw is not estimated")
            if args.max_cloud_age < .3 or not 1024 <= args.telemetry_port <= 65535:
                raise CalibrationError("max cloud age must allow 0.3 s and diagnostic port must be 1024..65535")
            from flight_capture import capture_flight
            return capture_flight(args)
        meta, frames = load_frames(args.capture_directory)
        folder = new_directory(args.output)
        limits = Limits(**meta["thresholds"])
        if args.height_tolerance_m is not None:
            limits = replace(limits, height_tolerance_m=args.height_tolerance_m)
        result = finalize(folder, meta, frames, limits, meta["yaw_deg_fixed"], source_folder=args.capture_directory)
        if args.open_plot:
            from local_workflow import open_summary
            open_summary(folder)
        return result
    except (CalibrationError, OSError, ValueError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
