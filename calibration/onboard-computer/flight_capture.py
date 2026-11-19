"""Lightweight flight recorder. Operator controls START/STOP; analysis is offline."""
from collections import deque
from dataclasses import asdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import queue
import resource
import shutil
import sys
import threading
import time

import numpy as np

from floor_geometry import CalibrationError, read_xyz, matrix_rpy
from passive_telemetry import PassiveTelemetry


def command_reader(commands):
    # Runs on a daemon thread, so a waiting input() can never delay STOP/shutdown.
    while True:
        line = sys.stdin.readline()
        if not line:
            commands.put('EOF')
            return
        command = line.strip().upper()
        commands.put(command)
        if command in ('STOP', 'DONE'):
            return  # Release stdin for the post-recording transfer prompt.


def capture_flight(args):
    from extrinsic_calibration import VERSION, make_limits, new_directory, tf_arrays, write_json
    if args.height_m is None:
        args.height_m = float(input('Height from Cube Orange to the physical ground plane at START (m): '))
    if args.grid_frame is None:
        args.grid_frame = input('RViz Global Options > Fixed Frame [map]: ').strip() or 'map'
    limits = make_limits(args)
    limits.validate()
    print(f'Floor-height search: {limits.height_m:.3f} +/- {limits.height_tolerance_m:.3f} m; '
          'this is a surface-selection aid, not a measured angle.')
    print('READ ONLY: no settings, commands, serial access, or automatic calibration changes.')
    print('Record over a known horizontal floor. PX4 tilt is tracked; yaw stays fixed at 180 degrees.')
    print('START begins recording. STOP (or DONE) saves, then offers transfer and plotting on your computer.')
    print('Use a second operator; do not operate this terminal while piloting.')
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from rclpy.duration import Duration
    from sensor_msgs.msg import PointCloud2
    from nav_msgs.msg import Odometry
    from tf2_ros import Buffer, TransformListener
    from tf2_py import TransformException

    telemetry = PassiveTelemetry(args.telemetry_port)
    folder = new_directory(args.output)
    (folder/'frames').mkdir()
    meta = dict(schema=1, version=VERSION, mode='flight', created_utc=datetime.now(timezone.utc).isoformat(),
                grid_frame=args.grid_frame, body_frame=args.body_frame, camera_frame=args.camera_frame,
                odom_frame=args.odom_frame, topic=args.topic, pose_topic=args.pose_topic,
                thresholds=asdict(limits), yaw_deg_fixed=args.yaw_deg, capture_problems=[], diagnostics={},
                capture_complete=False, synthetic_test=os.environ.get('EXTRINSIC_CALIBRATION_TEST_DATA') == '1',
                timing='PX4 host receive-time approximation; original boot timestamps retained',
                height_reference='operator supplied height at START, followed using relative PX4 local z')
    write_json(folder/'capture.json', meta)
    px4_file = (folder/'px4.jsonl').open('w', buffering=65536)
    vo_file = (folder/'vo_pose.csv').open('w', newline='', buffering=65536)
    pose_writer = csv.writer(vo_file)
    pose_writer.writerow(['stamp', 'received_unix_s', 'frame_id', 'child_frame_id',
                          'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
    commands = queue.Queue()
    if args.seconds is None:
        threading.Thread(target=command_reader, args=(commands,), daemon=True).start()
    else:
        print(f'Explicit timed capture: automatically starts, stops after {args.seconds:.3f} s.')
        commands.put('START')
    try:
        os.nice(10)  # This recorder only. VO/bridge scheduling is untouched.
    except OSError:
        pass
    rclpy.init()
    node = Node('extrinsic_calibration_observer')
    buffer = Buffer(cache_time=Duration(seconds=10))
    listener = TransformListener(buffer, node)
    clouds = deque(maxlen=8)
    last_cloud_received = [0.]
    recording, begin, start_ns = False, None, None
    last_stamp, last_sample, last_print, last_flush = 0, 0., time.monotonic(), time.monotonic()
    frame_count, unique, count_pose, age_sum, age_sq = 0, set(), 0, 0., 0.
    diagnostics = dict(missing_tf=0, stale_cloud=0, malformed_cloud=0, wrong_pose_frames=0)

    def on_pose(msg):
        nonlocal count_pose, age_sum, age_sq
        if not recording:
            return
        stamp = msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        if stamp < start_ns*1e-9:
            return
        received = time.time()
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        pose_writer.writerow([stamp, received, msg.header.frame_id, msg.child_frame_id,
                              p.x, p.y, p.z, q.x, q.y, q.z, q.w])
        count_pose += 1
        age = received-stamp
        age_sum += age
        age_sq += age*age
        if msg.header.frame_id != args.odom_frame or msg.child_frame_id != args.body_frame:
            diagnostics['wrong_pose_frames'] += 1

    def on_cloud(msg):
        now = time.monotonic()
        if now-last_cloud_received[0] >= .08:
            clouds.append(msg)
            last_cloud_received[0] = now

    sub_cloud = node.create_subscription(PointCloud2, args.topic, on_cloud, qos_profile_sensor_data)
    sub_pose = node.create_subscription(Odometry, args.pose_topic, on_pose, qos_profile_sensor_data)

    def lookup(target, source, stamp):
        if target == source:
            return np.eye(3), np.zeros(3)
        return tf_arrays(buffer.lookup_transform(target, source, stamp))

    print(f'Output: {folder}', flush=True)
    if args.seconds is None:
        print('READY. Type START once, then press Enter. STATUS shows counters; STOP ends the session.', flush=True)
    try:
        while True:
            rclpy.spin_once(node, timeout_sec=.02)
            now = time.monotonic()
            telemetry.drain(px4_file if recording else None)
            while not commands.empty():
                command = commands.get_nowait()
                if command == 'START' and not recording:
                    recording, begin = True, now
                    start_ns = node.get_clock().now().nanoseconds
                    meta.update(start_unix_s=start_ns*1e-9, start_utc=datetime.now(timezone.utc).isoformat())
                    write_json(folder/'capture.json', meta)
                    clouds.clear()
                    print('RECORDING STARTED. Type STATUS for counters; STOP then Enter to finish.', flush=True)
                elif command == 'START':
                    print('Already recording. Type STATUS for counters or STOP to finish.', flush=True)
                elif command == 'STATUS':
                    tele = 'receiving' if telemetry.last_received and now-telemetry.last_received < 1 else 'MISSING'
                    elapsed = now-begin if recording else 0
                    print(f'{"RECORDING" if recording else "NOT RECORDING — enter START"} | '
                          f'{elapsed:.1f} s | {frame_count} clouds | {count_pose} VO poses | PX4 {tele}', flush=True)
                elif command in ('STOP', 'DONE'):
                    meta['capture_complete'] = recording
                    return_code = 0 if recording else 2
                    break
                elif command == 'EOF':
                    meta['capture_problems'].append('input closed before explicit STOP')
                    return_code = 2
                    break
                elif command:
                    print(f'Unrecognized command {command!r}. Type START once and press Enter, '
                          'or STATUS / STOP.', flush=True)
            else:
                return_code = None
            if return_code is not None:
                break
            if recording and args.seconds is not None and now-begin >= args.seconds:
                meta['capture_complete'] = True
                return_code = 0
                break
            if now-last_flush >= 1:
                px4_file.flush()
                vo_file.flush()
                last_flush = now
                if recording and (node.count_publishers(args.topic) > 1 or node.count_publishers(args.pose_topic) > 1):
                    meta['capture_problems'].append('multiple cloud/VO pose publishers')
                    return_code = 2
                    break
                if shutil.disk_usage(folder).free < args.min_free_mb*1024**2:
                    meta['capture_problems'].append('low disk space; recorder stopped, flight stack unchanged')
                    return_code = 2
                    break
            if recording and now-last_sample >= args.interval:
                for msg in reversed(list(clouds)):
                    ns = msg.header.stamp.sec*1_000_000_000+msg.header.stamp.nanosec
                    age = (node.get_clock().now().nanoseconds-ns)*1e-9
                    # Wait for future PX4 window/TF samples, but never use latest TF.
                    if ns <= max(last_stamp, start_ns) or age < .25:
                        continue
                    if age > args.max_cloud_age:
                        diagnostics['stale_cloud'] += 1
                        continue
                    stamp = Time.from_msg(msg.header.stamp)
                    try:
                        r_bc, t_bc = lookup(args.body_frame, args.camera_frame, stamp)
                        r_bs, t_bs = lookup(args.body_frame, msg.header.frame_id, stamp)
                        r_ob, t_ob = lookup(args.odom_frame, args.body_frame, stamp)
                        r_gb, t_gb = lookup(args.grid_frame, args.body_frame, stamp)
                        points = read_xyz(msg)
                    except TransformException:
                        diagnostics['missing_tf'] += 1
                        continue
                    except CalibrationError:
                        diagnostics['malformed_cloud'] += 1
                        continue
                    if not len(points):
                        continue
                    if len(points) > 12000:
                        points = points[np.linspace(0, len(points)-1, 12000, dtype=int)]
                    if len(unique) < 100:
                        unique.add(hashlib.sha256(points.tobytes()).hexdigest())
                    if frame_count == 0:
                        active_rpy = matrix_rpy(r_bc)
                        meta['active_extrinsics'] = dict(
                            source='timestamp-matched live TF: '+args.body_frame+' <- '+args.camera_frame,
                            stamp_unix_s=ns*1e-9, xyz_m=t_bc.tolist(), rpy_deg=active_rpy.tolist())
                        print(f'Active extrinsics read from VO: roll {active_rpy[0]:+.3f}°, '
                              f'pitch {active_rpy[1]:+.3f}°, yaw {active_rpy[2]:+.3f}°; '
                              f'XYZ ({t_bc[0]:+.3f}, {t_bc[1]:+.3f}, {t_bc[2]:+.3f}) m.', flush=True)
                        write_json(folder/'capture.json', meta)
                    np.savez_compressed(folder/'frames'/f'{frame_count:08d}.npz',
                                        stamp=np.array(ns*1e-9), points_body=points @ r_bs.T+t_bs,
                                        r_bc=r_bc, t_bc=t_bc, r_ob=r_ob, t_ob=t_ob, r_gb=r_gb, t_gb=t_gb)
                    frame_count += 1
                    last_stamp, last_sample = ns, now
                    break
            # Do not write periodic status through the operator's partially typed
            # command. Timed captures have no command input; interactive ones use STATUS.
            if args.seconds is not None and now-last_print >= 3:
                tele = 'receiving' if telemetry.last_received and now-telemetry.last_received < 1 else 'MISSING'
                elapsed = now-begin if recording else 0
                print(f'{"RECORDING" if recording else "READY — type START"} | {elapsed:.1f} s | '
                      f'{frame_count} clouds | {count_pose} VO poses | PX4 {tele}', flush=True)
                last_print = now
    except KeyboardInterrupt:
        meta['capture_problems'].append('interrupted; partial data saved')
        return_code = 2
    except Exception as exc:
        meta['capture_problems'].append(f'recording error: {exc}')
        return_code = 2
    finally:
        duration = time.monotonic()-begin if begin is not None else 0
        px4_file.close()
        vo_file.close()
        telemetry.close()
        node.destroy_node()
        # Ctrl+C may already have shut down ROS's default context. The idempotent
        # call must not abort saving capture.json and the recording summary.
        rclpy.try_shutdown()
        mean_age = age_sum/count_pose if count_pose else 0.
        diagnostics.update(vo_poses=count_pose, mean_pose_age_s=mean_age,
                           std_pose_age_s=math.sqrt(max(0., age_sq/max(count_pose, 1)-mean_age**2)),
                           mean_received_pose_rate_hz=count_pose/max(duration, 1e-9),
                           peak_recorder_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                           telemetry_counts=dict(telemetry.counts))
        if diagnostics['wrong_pose_frames']:
            meta['capture_problems'].append('VO odometry frame IDs do not match configured body/odom')
        if len(unique) < 3:
            meta['capture_problems'].append('cloud did not change enough to rule out frozen data')
        meta.update(stop_utc=datetime.now(timezone.utc).isoformat(), elapsed_s=duration,
                    diagnostics=diagnostics, captured_frames=frame_count)
        write_json(folder/'capture.json', meta)
    if begin is None:
        print(f'NOT RECORDED: START was never accepted. Empty session metadata: {folder}', flush=True)
        print('Run the recorder again, type START once, and press Enter. Wait for RECORDING STARTED.')
        return return_code
    print(f'SAVED: {folder} ({frame_count} clouds, {count_pose} VO poses)', flush=True)
    if meta['capture_complete'] and not getattr(args, 'no_transfer_prompt', False) and sys.stdin.isatty():
        from transfer_results import offer_transfer
        offer_transfer(folder)
    else:
        print('Recording saved; no analysis was run on this recorder.')
    return return_code
