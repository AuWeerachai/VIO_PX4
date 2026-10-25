#!/usr/bin/env python3
"""Enable motion after stereo startup and require a fresh, healthy IMU window."""
from collections import deque
import math
import time


class ImuStartupWindow:
    """Startup-only gate; window restarts never extend the overall deadline."""

    MAX_GAP_S = 0.1
    WINDOW_S = 2.0
    MIN_SAMPLES = 300
    TIMEOUT_S = 25.0

    def __init__(self, now):
        self.deadline = now + self.TIMEOUT_S
        self.samples = deque(maxlen=2048)
        self.resets = 0
        self.last_gap = "none"

    def add(self, stamp, values, frame, received):
        if not all(math.isfinite(v) for v in (stamp, received, *values)):
            raise RuntimeError("Non-finite IMU data")
        if frame != "camera_imu_optical_frame":
            raise RuntimeError(f"Unexpected combined IMU frame: {frame}")
        reason = None
        if self.samples:
            previous = self.samples[-1]
            sensor_gap = stamp - previous[0]
            arrival_gap = received - previous[2]
            if sensor_gap <= 0:
                raise RuntimeError("IMU timestamps duplicate or regress")
            if arrival_gap < 0:
                raise RuntimeError("IMU receipt clock regressed")
            if max(sensor_gap, arrival_gap) > self.MAX_GAP_S:
                reason = (f"sensor gap={sensor_gap:.3f}s, "
                          f"arrival gap={arrival_gap:.3f}s")
                self.samples.clear()
                self.resets += 1
                self.last_gap = reason
        norm = math.sqrt(sum(v * v for v in values[:3]))
        self.samples.append((stamp, norm, received))
        return reason

    def ready(self, now, last_stereo):
        if now >= self.deadline:
            raise RuntimeError(
                f"No healthy 2-second IMU window within {self.TIMEOUT_S:g}s; "
                f"samples in current window={len(self.samples)}, "
                f"gap resets={self.resets}, last gap: {self.last_gap}"
            )
        if len(self.samples) < self.MIN_SAMPLES:
            return False
        first, last = self.samples[0], self.samples[-1]
        # A burst of queued data is not two seconds of healthy live delivery.
        if min(last[0] - first[0], last[2] - first[2]) < self.WINDOW_S:
            return False
        if now - last[2] > self.MAX_GAP_S:
            return False
        if last_stereo is None or now - last_stereo > 1.0:
            return False
        mean_gravity = sum(s[1] for s in self.samples) / len(self.samples)
        if not 7 < mean_gravity < 12:
            raise RuntimeError(
                "Accelerometer magnitude not plausible for stationary bench startup"
            )
        return True


def main():
    # Keep the startup-window logic testable without ROS installed.
    import rclpy
    from rclpy.parameter import Parameter
    from rcl_interfaces.srv import SetParameters
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Imu, CameraInfo

    rclpy.init()
    node = rclpy.create_node("vio_staged_imu_startup")
    window = None
    last_stereo = None

    def imu(message):
        # Ignore samples collected during parameter changes/sensor restarts.
        if window is None:
            return
        values = [message.linear_acceleration.x, message.linear_acceleration.y,
                  message.linear_acceleration.z, message.angular_velocity.x,
                  message.angular_velocity.y, message.angular_velocity.z]
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        reason = window.add(stamp, values, message.header.frame_id, time.monotonic())
        if reason:
            print(f"VIO IMU: restarting healthy startup window ({reason}); "
                  "GPS startup remains blocked.", flush=True)

    def stereo(_message):
        nonlocal last_stereo
        last_stereo = time.monotonic()

    try:
        client = node.create_client(SetParameters, "/realsense2_camera/set_parameters")
        subscriptions = [
            node.create_subscription(Imu, "/imu", imu, qos_profile_sensor_data),
            node.create_subscription(CameraInfo, "/left/camera_info_rect", stereo,
                                     qos_profile_sensor_data),
        ]
        deadline = time.monotonic() + 30
        while last_stereo is None and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if last_stereo is None:
            raise RuntimeError("No stereo CameraInfo within 30 seconds")
        if not client.wait_for_service(timeout_sec=10):
            raise RuntimeError("RealSense parameter service unavailable")

        def set_param(name, value):
            request = SetParameters.Request()
            request.parameters = [Parameter(name, value=value).to_parameter_msg()]
            future = client.call_async(request)
            rclpy.spin_until_future_complete(node, future, timeout_sec=5)
            if not future.done() or future.result() is None:
                raise RuntimeError(f"{name}: timeout")
            results = future.result().results
            if len(results) != 1 or not results[0].successful:
                raise RuntimeError(f"{name}: {future.result()}")
            print(f"VIO IMU: {name}={value}", flush=True)

        for name, value in [("unite_imu_method", 2), ("gyro_fps", 200),
                            ("accel_fps", 200), ("enable_gyro", True),
                            ("enable_accel", True)]:
            set_param(name, value)
        window = ImuStartupWindow(time.monotonic())
        print("VIO IMU: waiting for 2 continuous healthy seconds after configuration "
              "(at least 300 samples; gap limit 0.1s; timeout 25s).", flush=True)
        while not window.ready(time.monotonic(), last_stereo):
            rclpy.spin_once(node, timeout_sec=0.05)
        count = len(window.samples)
        duration = window.samples[-1][0] - window.samples[0][0]
        print(f"VIO IMU READY: {count} samples, {(count - 1) / duration:.1f} Hz; "
              f"timestamps monotonic; startup gap resets={window.resets}.", flush=True)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"VIO IMU FAILED: {exc}. Do not use this run for flight.", flush=True)
        raise SystemExit(1)
