"""Synthetic START/STOP ROS + actual one-way bridge UDP copy integration test.

Only localhost DDS domain 197/test port; no serial/camera/FCU. Flight stack never started.
"""
import os
os.environ['ROS_DOMAIN_ID'] = '197'
os.environ['ROS_LOCALHOST_ONLY'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['EXTRINSIC_CALIBRATION_TEST_DATA'] = '1'
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from floor_geometry import matrix_rpy, rpy_matrix
from ros_smoke_test import quaternion, transform
from test_flight import flight_scene


def main():
    package = Path(os.environ.get('VIO_CALIBRATION_BRIDGE_SOURCE',
        str(Path(__file__).resolve().parents[2]/'src/vio_px4_bridge')))
    sys.path.insert(0, str(package))
    from vio_px4_bridge.diagnostic_telemetry import DiagnosticTelemetry
    mirror = DiagnosticTelemetry(port=14792)
    root = Path(tempfile.mkdtemp(prefix='extrinsic-flight-smoke-'))
    output, result = root/'capture', root/'analysis'
    import extrinsic_calibration
    script = extrinsic_calibration.__file__
    command = [sys.executable, '-s', script, 'capture', '--height-m', '1.', '--grid-frame', 'flight_map',
               '--body-frame', 'flight_body', '--camera-frame', 'flight_camera', '--odom-frame', 'flight_odom',
               '--topic', '/extrinsic_flight_test/cloud', '--pose-topic', '/extrinsic_flight_test/pose',
               '--telemetry-port', '14792', '--output', str(output)]
    rclpy.init()
    node = Node('synthetic_flight_calibration_test')
    cloud_pub = node.create_publisher(PointCloud2, '/extrinsic_flight_test/cloud', qos_profile_sensor_data)
    pose_pub = node.create_publisher(Odometry, '/extrinsic_flight_test/pose', qos_profile_sensor_data)
    static, dynamic = StaticTransformBroadcaster(node), TransformBroadcaster(node)
    stamp = node.get_clock().now().to_msg()
    static.sendTransform([transform('flight_odom', 'flight_map', [.1, .2, .3], quaternion(4, -3, 17), stamp),
                          transform('flight_body', 'flight_camera', [-.26, 0, -.165], quaternion(.35, 36, 180), stamp)])
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    begin, count, started, stopped = time.monotonic(), 0, False, False
    try:
        while proc.poll() is None and time.monotonic()-begin < 25:
            elapsed = time.monotonic()-begin
            stamp = node.get_clock().now().to_msg()
            now = stamp.sec+stamp.nanosec*1e-9
            frame, roll, pitch = flight_scene(now, count*.05)
            q = quaternion(*matrix_rpy(frame['r_ob']))
            dynamic.sendTransform(transform('flight_odom', 'flight_body', frame['t_ob'], q, stamp))
            pose = Odometry()
            pose.header.frame_id, pose.child_frame_id, pose.header.stamp = 'flight_odom', 'flight_body', stamp
            pose.pose.pose.position.x, pose.pose.pose.position.y, pose.pose.pose.position.z = map(float, frame['t_ob'])
            pose.pose.pose.orientation.x, pose.pose.pose.orientation.y, pose.pose.pose.orientation.z, pose.pose.pose.orientation.w = map(float, q)
            pose_pub.publish(pose)
            p = (frame['points_body'] @ frame['r_ob'].T+frame['t_ob']).astype('<f4')
            cloud = PointCloud2()
            cloud.header.frame_id, cloud.header.stamp = 'flight_odom', stamp
            cloud.height, cloud.width, cloud.point_step, cloud.row_step = 1, len(p), 12, len(p)*12
            cloud.fields = [PointField(name=n, offset=j*4, datatype=PointField.FLOAT32, count=1) for j, n in enumerate('xyz')]
            cloud.is_dense, cloud.data = True, p.tobytes()
            cloud_pub.publish(cloud)
            common = dict(get_srcSystem=lambda: 1, get_srcComponent=lambda: 1)
            attitude = SimpleNamespace(**common, get_type=lambda: 'ATTITUDE', time_boot_ms=int(elapsed*1000),
                roll=np.radians(roll), pitch=np.radians(pitch), yaw=np.radians(90-count*.1),
                rollspeed=.005, pitchspeed=.005, yawspeed=.02)
            odom = SimpleNamespace(**common, get_type=lambda: 'ODOMETRY', time_usec=int(elapsed*1e6),
                                   z=-1., frame_id=1, reset_counter=0)
            mirror.publish(attitude, 'live', False)
            mirror.publish(odom, 'live', False)
            if elapsed > 2. and not started:
                proc.stdin.write('startstart\nstart\nstatus\n')
                proc.stdin.flush()
                started = True
            if elapsed > 13. and not stopped:
                proc.stdin.write('STOP\n')
                proc.stdin.flush()
                stopped = True
            rclpy.spin_once(node, timeout_sec=.025)
            count += 1
        if proc.poll() is None:
            proc.terminate()
            raise RuntimeError('recorder failed to stop')
        stdout, _ = proc.communicate(timeout=5)
        print(stdout)
        assert proc.returncode == 0, f'capture failed: {root}'
        assert "Unrecognized command 'STARTSTART'" in stdout
        assert 'RECORDING STARTED.' in stdout
        assert stdout.count('READY. Type START once') == 1
        assert 'READY — type START |' not in stdout
        assert not (output/'report.json').exists(), 'must never analyze during recording'
        meta = json.loads((output/'capture.json').read_text())
        assert meta['capture_complete'] and meta['elapsed_s'] > 8
        assert meta['captured_frames'] >= 10
        assert meta['diagnostics']['vo_poses'] > 100
        assert meta['diagnostics']['telemetry_counts']['ATTITUDE'] > 100
        subprocess.run([sys.executable, '-s', script, 'analyze', str(output), '--output', str(result)], check=True, timeout=30)
        report = json.loads((result/'report.json').read_text())
        np.testing.assert_allclose(report['suggested_rpy_deg'], [1.2, 32, 180], atol=.15)
        for name in ('current_floor_view.png', 'predicted_floor_view.png', 'orientation_history.png'):
            assert (result/name).is_file(), name
        print(f'FLIGHT RECORDING TEST PASS — SYNTHETIC DATA ONLY: {root}')
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)
        mirror.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
