#!/usr/bin/env python3
"""Read-only check that source, release manifest and installed helpers agree."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def verify(repo, isaac_ws):
    launcher = repo / 'scripts/vio-launch/vio_launch.py'
    spec = importlib.util.spec_from_file_location('verify_deployment_cli', launcher)
    cli = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = cli
    spec.loader.exec_module(cli)
    manifest = json.loads(launcher.with_name('camera_release.json').read_text())
    overrides = repo / 'bootstrap/jetson-config/isaac_ros_overrides'
    for name, digest in manifest.items():
        if Path(name).name != name:
            raise RuntimeError('Invalid camera manifest filename')
        source = overrides / name if name.startswith('.') else overrides / 'scripts' / name
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Source does not match the camera release manifest: {source}')
    cli.require_camera_deployment(cli.Config(isaac_ros_dir=isaac_ws))
    return len(manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--isaac-ws', type=Path, default=Path.home() / 'workspaces/isaac_ros-dev')
    args = parser.parse_args()
    try:
        count = verify(args.repo.resolve(), args.isaac_ws.resolve())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'Deployment verification failed: {exc}', file=sys.stderr)
        return 1
    print(f'Camera deployment verified: {count} source/manifest/installed files match.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
