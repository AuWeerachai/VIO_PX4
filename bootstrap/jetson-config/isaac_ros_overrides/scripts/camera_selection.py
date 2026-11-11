#!/usr/bin/env python3
"""Pin the VIO RealSense by SDK serial without changing its stream settings."""
import argparse
from pathlib import Path
import re


def normalize_serial(value):
    if not isinstance(value, str):
        raise ValueError('Camera serial must be a string, not a number.')
    serial = value.strip().removeprefix('_')
    if not re.fullmatch(r'[0-9]+', serial):
        raise ValueError('Configure the rear D456 SDK serial; automatic camera selection is disabled.')
    return serial


def verify_identity(expected_serial, actual_serial, device_name):
    expected = normalize_serial(expected_serial)
    if actual_serial != expected or 'd456' not in device_name.lower():
        raise RuntimeError(
            f'Wrong VIO camera: expected D456 serial {expected}; '
            f'got {device_name!r} serial {actual_serial!r}. Stop and restart VIO.')


def selected_config(base, serial):
    if not isinstance(base, dict) or 'enable_infra1' not in base or 'enable_infra2' not in base:
        raise ValueError('Expected the pinned Isaac ROS flat RealSense stereo configuration.')
    # The underscore prevents ROS/YAML from treating the SDK serial as an integer.
    return {**base, 'serial_no': '_' + normalize_serial(serial),
            'device_type': 'd456', 'usb_port_id': ''}


def main():
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('--serial', required=True)
    parser.add_argument('--base', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = selected_config(yaml.safe_load(Path(args.base).read_text()), args.serial)
    Path(args.output).write_text(yaml.safe_dump(result, sort_keys=False))
    print(f"VIO camera locked to rear D456, SDK serial {normalize_serial(args.serial)}")


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        raise SystemExit(f'VIO setup failed: {exc}')
