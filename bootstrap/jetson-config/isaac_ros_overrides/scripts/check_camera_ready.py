#!/usr/bin/env python3
"""Read-only readiness check in the container, run as the camera user (admin)."""
import argparse
import math
import time
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.srv import GetParameters
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from realsense2_camera_msgs.srv import DeviceInfo
from camera_selection import normalize_serial, verify_identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['vio', 'vo'])
    parser.add_argument('--timeout', type=float, default=12.0)
    parser.add_argument('--started-only', action='store_true',
                        help='Check selected mode and node availability, not data health')
    parser.add_argument('--serial', required=True, help='Expected rear D456 SDK serial')
    args = parser.parse_args()
    serial = normalize_serial(args.serial)
    rclpy.init()
    node = rclpy.create_node('vio_launch_camera_readiness')
    imu = []
    odom = []
    def got_imu(m):
        values = (m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z,
                  m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z)
        if all(math.isfinite(v) for v in values):
            imu.append((time.monotonic(), m.header.stamp.sec + m.header.stamp.nanosec * 1e-9,
                        math.sqrt(sum(v*v for v in values[:3])), m.header.frame_id))
    def got_odom(m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        if m.child_frame_id == 'drone_link' and all(math.isfinite(v) for v in (p.x,p.y,p.z,q.x,q.y,q.z,q.w)):
            odom.append(time.monotonic())
    subscriptions = [node.create_subscription(Imu, '/imu', got_imu, qos_profile_sensor_data),
                     node.create_subscription(Odometry, '/visual_slam/tracking/odometry', got_odom, qos_profile_sensor_data)]
    client = node.create_client(GetParameters, '/visual_slam_node/get_parameters')
    identity_client = node.create_client(DeviceInfo, '/device_info')
    deadline = time.monotonic() + args.timeout
    future = None
    fusion = None
    identity_future = None
    identity_verified = False
    try:
        while time.monotonic() < deadline:
            if identity_future is None and identity_client.service_is_ready():
                identity_future = identity_client.call_async(DeviceInfo.Request())
            if identity_future is not None and identity_future.done() and not identity_verified:
                info = identity_future.result()
                if info is None:
                    raise RuntimeError('Camera identity service returned no response')
                verify_identity(serial, info.serial_number, info.device_name)
                identity_verified = True
            if future is None and client.service_is_ready():
                req = GetParameters.Request()
                req.names = ['enable_imu_fusion']
                future = client.call_async(req)
            rclpy.spin_once(node, timeout_sec=0.05)
            if future is not None and future.done():
                result = future.result()
                if result and result.values and result.values[0].type == 1:
                    fusion = result.values[0].bool_value
                if fusion is not None and fusion != (args.mode == 'vio'):
                    raise RuntimeError('Running camera mode differs from selected mode. Stop all processes, then restart.')
                if args.started_only and fusion is not None and identity_verified:
                    print(f'{args.mode.upper()} node started; D456 serial {serial} verified; data health not yet validated.')
                    return 0
            now = time.monotonic()
            fresh_odom = odom and now - odom[-1] < 0.5
            fresh_imu = (len(imu) >= 300 and now - imu[-1][0] < 0.25
                         and imu[-1][0]-imu[0][0] >= 2.0)
            recent = imu[-300:]
            ordered_imu = (fresh_imu
                           and all(0 < b[1]-a[1] < 0.1 for a,b in zip(recent, recent[1:]))
                           and all(s[3] == 'camera_imu_optical_frame' for s in recent)
                           and 7 < sum(s[2] for s in recent)/len(recent) < 12)
            if identity_verified and fusion is not None and fresh_odom and (args.mode == 'vo' or ordered_imu):
                print(f'{args.mode.upper()} ready: D456 serial {serial}; body odometry fresh; IMU fusion={fusion}.')
                return 0
        raise RuntimeError(f'{args.mode.upper()} not ready: D456 identity verified={identity_verified}, fusion={fusion}, odometry samples={len(odom)}, IMU samples={len(imu)}')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'Camera readiness failed: {exc}')
        raise SystemExit(1)
