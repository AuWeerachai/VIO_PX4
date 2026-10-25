#!/usr/bin/env python3
"""Idempotently apply two audited fixes, refusing unknown source layouts."""
from pathlib import Path
import sys
import re


def patched_sources(package):
    impl = package / 'src/impl/visual_slam_impl.cpp'
    launch = package / 'launch/isaac_ros_visual_slam_core.launch.py'
    text = impl.read_text()
    pattern = r'CUVSLAM_RegisterImuMeasurement\(\s*cuvslam_handle,\s*(?:latest_ts|imu_ts),\s*&imu_measurement\)'
    if len(list(re.finditer(pattern, text))) != 1:
        raise RuntimeError(f'Unknown IMU registration layout: {impl}')
    text = re.sub(pattern, lambda m: m.group().replace('latest_ts', 'imu_ts'), text)
    launch_text = launch.read_text()
    remap = "                    ('visual_slam/imu', '/imu'),\n"
    anchor = "                    ('/visual_slam/image_0', 'left/image_rect_mono'),\n"
    if remap not in launch_text:
        if launch_text.count(anchor) != 1:
            raise RuntimeError(f'Unknown cuVSLAM remapping layout: {launch}')
        launch_text = launch_text.replace(anchor, remap + anchor)
    return {impl: text, launch: launch_text}


if __name__ == '__main__':
    changes = patched_sources(Path(sys.argv[1]))
    # Validate both before writing either. Retain unrelated local edits.
    for path, text in changes.items():
        if path.read_text() != text:
            path.write_text(text)
    print('cuVSLAM: actual IMU timestamps and /imu remapping verified.')
