#!/usr/bin/env bash
# Run on the Jetson, while disarmed, with the actual FC adapter selected.
# This changes host USB buffering only, not baud rate or PX4 configuration.
set -euo pipefail

if [[ $# != 1 || "$1" == --help ]]; then
  echo 'Usage: bash bootstrap/configure_fcu_serial_latency.sh /dev/ttyUSB0'
  echo 'Ground only. Pin 1 ms receive latency to this FTDI adapter serial.'
  exit 0
fi
device="$(readlink -f -- "$1")"
[[ -c "$device" && "$device" == /dev/ttyUSB* ]] || {
  echo 'Select the connected FC USB serial device, not a camera or UDP link.' >&2; exit 1;
}
tty_name="${device##*/}"
driver="$(readlink -f "/sys/class/tty/$tty_name/device/driver")"
latency="/sys/bus/usb-serial/devices/$tty_name/latency_timer"
[[ "${driver##*/}" == ftdi_sio && -f "$latency" ]] || {
  echo 'This helper only supports an FTDI device with latency_timer.' >&2; exit 1;
}
serial=""
while IFS='=' read -r key value; do
  [[ "$key" == ID_SERIAL_SHORT ]] && serial="$value"
done < <(udevadm info --query=property --name="$device")
[[ "$serial" =~ ^[a-zA-Z0-9_-]{1,64}$ ]] || {
  echo 'Cannot uniquely identify the FTDI adapter; no changes made.' >&2; exit 1;
}

rule="/etc/udev/rules.d/99-vio-fcu-ftdi-$serial.rules"
rule_file="$(mktemp)"
trap 'rm -f -- "$rule_file"' EXIT
printf '%s\n' '# VIO FC adapter only: remove this rule to restore the driver default on reconnect.' \
  "ACTION==\"add|change\", SUBSYSTEM==\"usb-serial\", DRIVER==\"ftdi_sio\", ATTRS{serial}==\"$serial\", ATTR{latency_timer}=\"1\"" > "$rule_file"
if [[ -e "$rule" ]] && ! cmp -s "$rule_file" "$rule"; then
  echo "Existing rule differs; preserve/review it before replacement: $rule" >&2
  exit 1
fi
echo "FC adapter: $device; FTDI serial $serial; current latency $(<"$latency") ms"
sudo install -m 0644 "$rule_file" "$rule"
sudo udevadm control --reload-rules
# Apply to this existing adapter without resetting it or triggering other devices.
printf '1\n' | sudo tee "$latency" >/dev/null
[[ "$(<"$latency")" == 1 ]]
echo "Verified 1 ms latency. Persistent rule: $rule"
