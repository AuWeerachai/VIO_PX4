#!/usr/bin/env python3
"""Fail before opening hardware/services if VO mixes RealSense installations."""
from pathlib import Path
import os
import subprocess
import xml.etree.ElementTree as ET


def validate_packages(driver_prefix, messages_prefix, base=Path('/opt/ros/humble')):
    base = Path(base).resolve()
    prefixes = [Path(driver_prefix).resolve(), Path(messages_prefix).resolve()]
    if any(prefix != base for prefix in prefixes):
        raise RuntimeError(f'RealSense must use the matched base-image driver and messages; '
                           f'got driver={prefixes[0]}, messages={prefixes[1]}')
    versions = [ET.parse(prefix / 'share' / package / 'package.xml').getroot().findtext('version')
                for prefix, package in zip(prefixes, ('realsense2_camera', 'realsense2_camera_msgs'))]
    if not versions[0] or versions[0] != versions[1]:
        raise RuntimeError(f'RealSense driver/message versions differ: {versions}')
    return versions[0]


def validate_linkage(output, base=Path('/opt/ros/humble')):
    libraries = []
    for line in output.splitlines():
        if 'not found' in line:
            raise RuntimeError(f'Missing camera dependency: {line.strip()}')
        if 'librealsense2_camera_msgs' in line and '=>' in line:
            path = Path(line.split('=>', 1)[1].strip().split()[0]).resolve()
            if not path.is_relative_to(Path(base).resolve()):
                raise RuntimeError(f'Incompatible RealSense message library selected: {path}')
            libraries.append(path)
    if not libraries:
        raise RuntimeError('Could not verify RealSense camera message linkage')


def validate():
    from ament_index_python.packages import get_package_prefix
    workspace = Path(os.environ['ISAAC_ROS_WS'])
    custom_source = workspace / 'src/isaac_ros_common/isaac_ros_visual_slam/isaac_ros_visual_slam/package.xml'
    if custom_source.is_file():
        expected = workspace / 'install/isaac_ros_visual_slam'
        actual = Path(get_package_prefix('isaac_ros_visual_slam'))
        if actual.resolve() != expected.resolve():
            raise RuntimeError(f'Customized cuVSLAM overlay not active: expected {expected}, got {actual}')
    driver = Path(get_package_prefix('realsense2_camera'))
    messages = Path(get_package_prefix('realsense2_camera_msgs'))
    version = validate_packages(driver, messages)
    linkage = subprocess.run(['ldd', str(driver / 'lib/librealsense2_camera.so')],
                             text=True, capture_output=True, timeout=5, check=True)
    validate_linkage(linkage.stdout)
    import realsense2_camera_msgs.srv._device_info as info
    if not Path(info.__file__).resolve().is_relative_to(messages.resolve()):
        raise RuntimeError(f'Camera identity Python types came from another overlay: {info.__file__}')
    # This is the six-field service in the pinned 4.51.1 image. The experimental
    # seven-field response cannot be serialized by the older driver safely.
    fields = tuple(info.DeviceInfo.Response.get_fields_and_field_types())
    if fields != ('device_name', 'serial_number', 'firmware_version',
                  'usb_type_descriptor', 'firmware_update_id', 'sensors'):
        raise RuntimeError(f'Unexpected camera identity interface: {fields}')
    return version


if __name__ == '__main__':
    try:
        print(f'VO environment verified: RealSense driver/messages {validate()} (base image).')
    except Exception as exc:
        raise SystemExit(f'VIO setup failed: {exc}. No camera/service was opened.')
