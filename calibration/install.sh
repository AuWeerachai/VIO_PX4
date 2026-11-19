#!/usr/bin/env bash
set -euo pipefail
role=${1:-}
case "$role" in local|onboard-computer) ;; *) echo 'Usage: bash calibration/install.sh local|onboard-computer [destination]'; exit 2;; esac
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
destination=${2:-"$HOME/extrinsic_calibration"}
mkdir -p -- "$destination"
# Keep one deployable flat runtime; source ownership remains separated by role.
# Both roles need the saved-data readers. No ROS, firmware or flight settings are edited.
for area in common local onboard-computer; do
    cp -- "$source_dir/$area/"*.py "$destination/"
done
cp -- "$source_dir/extrinsic_calibration" "$source_dir/local/start_receiver" "$destination/"
cp -- "$source_dir/README.md" "$destination/README.md"
chmod +x -- "$destination/extrinsic_calibration" "$destination/start_receiver"
/usr/bin/python3 -s -m py_compile "$destination/"*.py
echo "Installed $role tool: $destination"
echo 'This does not install dependencies, enable services, modify the bridge, or start recording.'
