"""Synthetic ROS integration test, isolated from real cameras and PX4.

Run after sourcing ROS. Uses localhost-only DDS and a separate domain. The
capture must recover known +1.2/+32/+180 extrinsics from an incorrect mount.
"""
import os
os.environ["ROS_DOMAIN_ID"] = "197"
os.environ["ROS_LOCALHOST_ONLY"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["EXTRINSIC_CALIBRATION_TEST_DATA"] = "1"
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from test_calibration import scene


def quaternion(r, p, y):
    r, p, y = np.radians([r, p, y])/2
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return (sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy)


def transform(parent, child, t, q, stamp):
    tf = TransformStamped()
    tf.header.frame_id, tf.child_frame_id, tf.header.stamp = parent, child, stamp
    tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = map(float, t)
    tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w = map(float, q)
    return tf


def main():
    root = Path(tempfile.mkdtemp(prefix="extrinsic-calibration-ros-test-"))
    output = root/"capture"
    rclpy.init()
    node = Node("synthetic_floor_test")
    pub = node.create_publisher(PointCloud2, "/extrinsic_calibration_test/cloud", qos_profile_sensor_data)
    static = StaticTransformBroadcaster(node)
    dynamic = TransformBroadcaster(node)
    f = scene()
    rg, tg = f["r_gb"], f["t_gb"]
    q = quaternion(4, -3, 17)
    stamp = node.get_clock().now().to_msg()
    static.sendTransform([
        transform("test_odom", "test_map", -rg.T@tg, (-q[0], -q[1], -q[2], q[3]), stamp),
        transform("test_body", "test_camera", f["t_bc"], quaternion(.35, 40, 180), stamp)])
    import extrinsic_calibration
    command = [sys.executable, "-s", extrinsic_calibration.__file__,
               "capture", "--height-m", ".47", "--grid-frame", "test_map",
               "--body-frame", "test_body", "--camera-frame", "test_camera",
               "--odom-frame", "test_odom", "--topic", "/extrinsic_calibration_test/cloud",
               "--seconds", "9", "--stationary", "--level-confirmed", "--output", str(output)]
    process = subprocess.Popen(command)
    begin, i = time.monotonic(), 0
    try:
        while process.poll() is None and time.monotonic()-begin < 35:
            stamp = node.get_clock().now().to_msg()
            dynamic.sendTransform(transform("test_odom", "test_body", [0, 0, 0], [0, 0, 0, 1], stamp))
            f = scene(seed=i)
            p = (f["points_body"] @ rg.T+tg).astype("<f4")
            msg = PointCloud2()
            msg.header.frame_id, msg.header.stamp = "test_map", stamp
            msg.height, msg.width, msg.point_step, msg.row_step = 1, len(p), 12, len(p)*12
            msg.is_dense = True
            msg.fields = [PointField(name=name, offset=j*4, datatype=PointField.FLOAT32, count=1)
                          for j, name in enumerate("xyz")]
            msg.data = p.tobytes()
            pub.publish(msg)
            rclpy.spin_once(node, timeout_sec=.05)
            i += 1
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
            raise RuntimeError("capture test timed out")
        if process.returncode:
            raise RuntimeError(f"capture exited {process.returncode}; evidence: {root}")
        report = json.loads((output/"report.json").read_text())
        np.testing.assert_allclose(report["suggested_rpy_deg"], [1.2, 32, 180], atol=.12)
        assert report["grid_frame"] == "test_map"
        assert (output/"current_floor_view.png").is_file() and (output/"predicted_floor_view.png").is_file()
        print(f"ROS integration PASS; SYNTHETIC DATA ONLY: {output}")
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
