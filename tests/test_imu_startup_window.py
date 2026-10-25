"""Pure-Python regression tests for the staged IMU startup gate."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = (Path(__file__).resolve().parents[1] /
          'bootstrap/jetson-config/isaac_ros_overrides/scripts/stage_realsense_imu.py')
spec = importlib.util.spec_from_file_location('imu_startup_window_test', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Window = module.ImuStartupWindow
FRAME = 'camera_imu_optical_frame'
VALUES = [0.0, 0.0, 9.81, 0.0, 0.0, 0.0]


def feed(window, start=0.0, count=402, receipt_step=0.005):
    for i in range(count):
        window.add(start + i * 0.005, VALUES, FRAME, start + i * receipt_step)
    return start + (count - 1) * receipt_step


def test_normal_200hz_startup():
    window = Window(0)
    now = feed(window)
    assert window.ready(now, now)
    assert window.resets == 0


def test_reported_295ms_gap_requires_entirely_new_window():
    window = Window(0)
    # These pre-gap samples must not contaminate the gravity average either.
    for i in range(50):
        window.add(i * .005, [0, 0, 50, 0, 0, 0], FRAME, i * .005)
    after_gap = .245 + .295
    reason = window.add(after_gap, VALUES, FRAME, after_gap)
    assert 'sensor gap=0.295s' in reason
    assert len(window.samples) == 1
    assert window.resets == 1
    assert not window.ready(after_gap, after_gap)
    now = feed(window, after_gap + .005)
    assert window.ready(now, now)


def test_continuing_gaps_cannot_extend_25_second_deadline():
    window = Window(0)
    for i in range(125):
        t = i * .2
        window.add(t, VALUES, FRAME, t)
        assert not window.ready(t, t)
    assert window.deadline == 25
    with pytest.raises(RuntimeError, match='within 25s'):
        window.ready(25, 25)


def test_absent_imu_times_out():
    window = Window(0)
    assert not window.ready(24.9, 24.9)
    with pytest.raises(RuntimeError, match='samples in current window=0'):
        window.ready(25, 25)


@pytest.mark.parametrize('stamp', [1.0, 0.5])
def test_duplicate_or_regressing_timestamp_fails(stamp):
    window = Window(0)
    window.add(1, VALUES, FRAME, 1)
    with pytest.raises(RuntimeError, match='duplicate or regress'):
        window.add(stamp, VALUES, FRAME, 1.005)


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_nonfinite_values_fail(bad):
    window = Window(0)
    with pytest.raises(RuntimeError, match='Non-finite'):
        window.add(0, [bad, *VALUES[1:]], FRAME, 0)


def test_wrong_frame_fails():
    with pytest.raises(RuntimeError, match='Unexpected combined IMU frame'):
        Window(0).add(0, VALUES, 'camera_link', 0)


def test_delivery_gap_resets_even_with_good_sensor_timestamps():
    window = Window(0)
    window.add(0, VALUES, FRAME, 0)
    assert window.add(.005, VALUES, FRAME, .295)
    assert window.resets == 1
    assert len(window.samples) == 1


def test_queued_burst_is_not_a_live_two_second_window():
    window = Window(0)
    now = feed(window, receipt_step=.0001)
    assert not window.ready(now, now)


def test_stale_imu_or_stereo_cannot_pass():
    window = Window(0)
    now = feed(window)
    assert not window.ready(now + .2, now + .2)
    assert not window.ready(now, now - 1.1)
    assert not window.ready(now, None)


def test_count_alone_is_not_enough():
    window = Window(0)
    now = feed(window, count=300)
    assert not window.ready(now, now)


def test_nonphysical_gravity_fails():
    window = Window(0)
    for i in range(402):
        window.add(i * .005, [0] * 6, FRAME, i * .005)
    with pytest.raises(RuntimeError, match='magnitude not plausible'):
        window.ready(2.005, 2.005)
