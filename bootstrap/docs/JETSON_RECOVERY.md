# Fresh Jetson infrastructure and recovery checklist

Updated 2026-11-16 against the bootstrap scripts and repository `fcc1222`.
The recorded deployed baseline was `0606a1f`; the newer optional heading output
is not yet deployed/flight-validated. This guide is not evidence of a completed
clean-machine rebuild. Do not replace pinned dependencies with latest releases.

**New disk:** follow sections 2, 3, 5 and 6. **Existing working NVMe:** follow
sections 1 and 4 first; do not reinstall just because the Jetson module changed.
The bootstrap prepares the project workspaces, not the entire operating system.

## 1. Preserve the existing installation first

The inspected Jetson used NVMe for its root filesystem. A healthy NVMe contains
the installed OS, workspaces, Docker storage (unless separately mounted), user
configuration and saved recordings. The module's QSPI/UEFI boot firmware is
separate. Moving the disk therefore preserves much more than cloning this repo,
but does not guarantee that a replacement module can boot it immediately.

- Fully power off before touching the NVMe or reseating connectors.
- Prefer a complete disk image/clone before repairs, filesystem checks that
  write, firmware flashing, or first boot on replacement hardware.
- Use the same Orin Nano module/carrier combination where possible. Match the
  replacement's boot firmware to the installed L4T release and carrier BSP.
- Do not run SDK Manager installation/flashing against the original SSD merely
  to test it: installation can overwrite the data we are trying to recover.
- Preserve PX4's SD-card logs separately. FC firmware/parameters are not stored
  on the Jetson NVMe. A crash can also leave the last recording unfinalized.
- If device-bound disk encryption or secure boot was enabled, collect its
  recovery material before migration; do not assume this disk is portable.

### Boot failure observed on 2026-11-15

The post-crash monitor photo showed `UEFI Interactive Shell v2.2`, not Ubuntu.
Its displayed mappings were `MemoryMapped(...)` and `Fv(...)`; no obvious NVMe
partition was listed. Ethernet carrier and the power LED were present, but SSH
and AnyDesk were unavailable. This establishes that Linux had not booted on
that attempt, **not** that the NVMe or module is conclusively damaged.

At `Shell>`, `exit` returns to the firmware menu. Photograph Boot Manager and
check for the NVMe/Ubuntu boot entry before modifying anything. If storage is
absent, power off before checking seating, then inspect/image the SSD on a
separate machine with a compatible NVMe enclosure. Never initialize or format
an unrecognized disk. Boot order, boot firmware, disk detection and filesystem
integrity are separate checks. This is the historical failure observation, not
a diagnosis of the replacement Jetson's current condition.

## 2. Required infrastructure

| Layer | Current baseline / requirement |
|---|---|
| Hardware | Jetson Orin Nano, compatible carrier, adequate power/cooling and NVMe |
| Host OS/BSP | Ubuntu **22.04**, ARM64, compatible JetPack 6 / **L4T 36.4.x**; see the version distinction below |
| GPU/container | Matching NVIDIA host drivers, Docker, NVIDIA Container Runtime and Docker access for the operator |
| Host ROS | ROS 2 **Humble** at `/opt/ros/humble/setup.bash`; colcon and Python runtime dependencies |
| Isaac ROS | **release-3.2**, pinned Common and Visual SLAM commits in `../jetson-config/isaac_ros_release.env` |
| cuVSLAM assets | NGC Visual SLAM asset **3.1.0**, including the interface-spec JSON |
| Camera driver | Base-image librealsense **2.55.1**, NVIDIA wrapper **release/4.51.1-isaac**; experimental IMU overlay is inactive |
| Container | `aarch64.ros2_humble.realsense` image key; `run_dev.sh`, mounted workspace, project-owned Docker/source/launch patches |
| MAVROS | Separate host workspace, commit `c655e6343ec81687d51e7185bcba7a651e361fcd` plus our patch; only Path B uses it at runtime |
| Camera | Rear D456 on USB 3, both infrared streams, explicit SDK serial and measured body-to-camera extrinsics |
| Bridge | This repo, built `vio_px4_bridge`, ROS Python/messages, `pymavlink`, serial permissions |
| FC link | Inspected link `/dev/ttyUSB0:921600`; verify the device on replacement hardware; Path A owns it exclusively |
| Launcher UI | `tmux`, `gnome-terminal`, working desktop/display for visible camera terminal; RViz is optional |
| PX4 | Preserve the actual aircraft firmware and parameter export; compatible HIL_GPS receiver, correct heading/height/reference configuration |

### Platform version: do not confuse the image with the installed system

Earlier deployment notes recorded **L4T 36.4.7**. NVIDIA's official
**JetPack 6.2.1** image ships **L4T 36.4.4**, not 36.4.7. Keep the recorded value
as a historical observation to reconcile with the actual disk/package inventory,
not as the name of an official image to download. The script's 36.4-family check
does not validate every BSP/package combination. Match the module, carrier and
boot firmware before flashing. [NVIDIA JetPack 6.2.1](https://developer.nvidia.com/embedded/jetpack-sdk-621)

NVIDIA's Isaac ROS 3.2 setup guide calls for an SSD on Jetson and at least 30 GB
free for its development environment. Allow additional space for this project's
builds, downloaded assets, Docker cache, recordings and the arm application;
30 GB is not a measured capacity requirement for our complete installation.
[Isaac ROS 3.2 environment setup](https://nvidia-isaac-ros.github.io/v/release-3.2/getting_started/dev_env_setup.html)

### What runs where

| Location | Responsibility |
|---|---|
| Jetson host | JetPack/kernel/GPU support, Docker, ROS Humble, `vio-launch`, bridge and its serial connection; MAVROS only for Path B |
| Isaac ROS container | RealSense driver, cuVSLAM, ROS Humble, source-overlay build and optional RViz |
| Mounted `isaac_ros-dev` workspace | Persistent NVIDIA source, project overrides, assets and generated overlay; visible inside the container as `/workspaces/isaac_ros-dev` |
| PX4 flight controller | Flight firmware, sensor fusion, height source, control and failsafes; not installed by Jetson bootstrap |
| Separate optional tools | RTK corrections/recording, arm application and remote desktop; not restored by these scripts |

Host and container each have their own `/opt/ros/humble`. Installing ROS only
inside Docker does not satisfy the host bridge. The container shares the host's
GPU/kernel facilities; it cannot repair a mismatched host BSP.

The last verified rear-camera SDK serial was `333422300181`. That belongs to
the physical camera, not the Jetson. Check the connected hardware; do not copy
this serial to a replacement camera or allow automatic first-camera selection.

Current Path A:

```text
D456 stereo -> cuVSLAM VO -> odom/drone_link -> continuity/heading bridge
            -> direct MAVLink HIL_GPS -> PX4
PX4 ATTITUDE -> bridge startup heading and GPS-silent recovery
```

Stereo-only **VO** is intentional. Do not enable the older RealSense IMU
experiment during recovery. No Micro XRCE-DDS or MAVROS is used in Path A.
MAVROS is a separate dependency for Path B; the current bootstrap nevertheless
creates/builds its workspace for both-path deployment.

Commit `fcc1222` also adds **optional VO heading over MAVLink ODOMETRY**, default
off. Leave it off when restoring GPS-only operation. Enabling it requires a
separate PX4 configuration and ground validation; see
[Path A external yaw](PATH_A_EXTERNAL_YAW.md). It does not enable camera IMU fusion.

## 3. What Git does and does not restore

| Item | Recovery source |
|---|---|
| Bridge, launcher, NVIDIA patches and dependency pins | This private repository |
| Home, camera serial, extrinsics, link, gate/recovery settings | **`~/.config/vio-launch/config.json`**, separately backed up |
| Older configuration, if present | `~/.config/vio-test/config.json`; retain for investigation, not blindly as current settings |
| Docker images/build caches | Existing SSD or fresh download/build; not in Git |
| Nvidia/ROS base platform and QSPI firmware | Provision separately, using matching releases |
| PX4 firmware, custom heading changes and aircraft parameters | Preserve exact `.px4` artifact, source revision and QGC `.params` export separately |
| Navigation logs and launcher state | `~/.local/state/vio-launch/`; archive logs, but do not restore stale process IDs as running services |
| RTK receiver, correction client and recorder tools | **`~/gnss-validation/`**, separate from this repo's bootstrap |
| Flight recordings | `~/gnss-validation/bags/`, QGC telemetry files and FC ULogs; preserve raw files |
| Arm application and its dependencies | Separate team repository/install; not restored by VIO bootstrap |
| Wi-Fi/SSH/AnyDesk configuration and credentials | System/user backup; do not commit secrets to Git |

Camera translations are metres, CLI angles are degrees. Extrinsics describe
`drone_link -> camera_link` using ROS FLU conventions. Restore the actual saved
configuration rather than reconstructing angles from an old chat/report. Do
not use fitted RTK trajectory rotations as camera calibration. Fresh defaults
have **empty camera serial and zero extrinsics**, and an example home location;
they are not the aircraft's working calibration.

RTK is a comparison instrument only, not required for navigation. Its deployment
also needs the u-blox ROS workspace/messages, USB access, the NTRIP/RTCM client,
base-station network reachability and credentials, and a recording process.
The latest receiver/caster setup is not provisioned by `bootstrap_*.sh`.
Securely preserve it before wiping anything. No passwords belong in this file.

## 4. Existing NVMe moved to a replacement

1. Image the original disk and retain an untouched copy.
2. Identify exact module/carrier, installed L4T and required boot firmware. Do
   not substitute a current JetPack release for the captured platform.
3. Boot with compatible firmware and inspect storage/mounts before rebuilding.
4. Verify user configuration, camera identity, serial-device permissions and
   network addressing (new interfaces/MAC addresses may change connectivity).
5. Inspect the existing Docker image/workspaces. Do not run bootstrap over a
   working installation merely because the module changed.
6. Perform the non-flight checks in section 6 before a controlled test.

## 5. Fresh disk / clean rebuild

### 5.1 Provision the host first

Do these installation steps on an idle replacement, not while VIO is running:

1. Install the matching **JetPack/BSP and boot firmware** for the exact
   module/carrier, with Ubuntu 22.04 on the intended disk. Some factory Orin
   Nano developer kits need a boot-firmware update before JetPack 6 can boot.
   Use NVIDIA/carrier installation instructions, not a generic desktop GPU
   driver installer. [JetPack installation](https://developer.nvidia.com/embedded/jetpack-sdk-621)
2. Install **host ROS 2 Humble**, its apt repository and development tools.
   `ros-humble-ros-base` plus `ros-dev-tools` is the foundation; desktop ROS is
   also suitable. Use the Ubuntu 22.04 ARM64 instructions and system Python.
   Follow their system-package prerequisites before installing ROS. Do not
   substitute ROS Jazzy or source a Conda/experimental ROS environment.
   [Official Humble installation](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)
3. Install/verify **Docker and NVIDIA Container Toolkit/runtime** for the
   JetPack platform. Confirm the operator can use Docker without `sudo`.
   Runtime configuration and Docker restarts are setup actions that can disrupt
   all containers: never do them during navigation.
   [NVIDIA runtime setup](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
4. Give the trusted operator appropriate **Docker, camera and serial access**.
   Check Docker group membership and the actual serial device's owning group
   (commonly `dialout`); log in again after group changes. Docker access is
   privileged host access. Do not run the launcher as root to bypass permissions.
5. Have a working **desktop session/display** for visible `gnome-terminal`
   windows and RViz. SSH/AnyDesk are access methods, not navigation dependencies;
   SSH alone does not supply a desktop display. The current visible-terminal
   workflow should not be assumed to work on a headless server install.
6. Provide internet/DNS and a correct system clock for GitHub, NGC/`nvcr.io`,
   NVIDIA/Ubuntu/ROS apt repositories and Python packages. Configure access to
   the private VIO repository. Keep SSH keys, registry and RTK credentials out
   of Git. Offline recovery needs preserved artifacts, not just source code.

Read-only prechecks on the replacement:

```bash
uname -m
cat /etc/nv_tegra_release
cat /etc/os-release
dpkg-query -W nvidia-l4t-core nvidia-jetpack
findmnt /
df -h / /var/lib/docker
free -h
id
test -r /opt/ros/humble/setup.bash
docker info
source /opt/ros/humble/setup.bash
python3 -c 'import rclpy; from nav_msgs.msg import Odometry; from sensor_msgs.msg import Imu; from rcl_interfaces.msg import Parameter'
```

Expect `aarch64`, Ubuntu 22.04, a reviewed 36.4.x BSP, accessible Docker and
successful ROS imports. An absent `nvidia-jetpack` meta-package alone does not
prove its component packages are missing; inspect the installed components.
`docker info` only checks the daemon, not CUDA access inside Isaac ROS.

### 5.2 Know what the bootstrap installs

| Stage | Actions |
|---|---|
| Must exist before bootstrap | Compatible OS/BSP/GPU stack, host Humble and ROS apt repository, Docker/NVIDIA runtime, user/device/display access |
| `bootstrap_isaac_ros.sh --install-deps` | Installs Git/Git LFS/curl/tar; checks architecture, L4T and Docker; retrieves pinned Common/VSLAM sources and NGC asset; calls the project installer |
| `bootstrap_jetson.sh --install-deps` | Installs pip, colcon, GeographicLib tools, tmux, gnome-terminal, MAVROS apt dependencies and `pymavlink`; retrieves EGM96-5 geoid data if missing |
| Project installation, with or without `--install-deps` | Clones/patches/builds MAVROS; installs Isaac launch helpers and Docker/VSLAM patches; builds the host bridge; creates launcher symlinks |
| First camera launch | Builds/pulls the Isaac image as needed, builds the cuVSLAM workspace overlay inside it, then runs the selected camera launch |
| Not provided | OS flashing, PX4 firmware/parameter changes, saved vehicle config, RTK/arm stack, network/remote-access setup or recovery of recordings |

The MAVROS apt packages supply dependencies; the separately patched source
workspace is still built. The full bootstrap builds MAVROS even for a user who
plans to use only Path A. It does not install or start Micro XRCE-DDS.

Also verify host serial support with `python3 -c 'import serial'`. If missing,
install Ubuntu's `python3-serial` package. Importing `pymavlink` alone does not
verify that its serial transport is usable; the installer does not explicitly
install/test PySerial separately.

### 5.3 Create the project and dependency workspaces

Use an approved repository commit (captured runtime baseline: `0606a1f`), and
review any newer changes before deployment. Fetching a branch does not itself
prove it matches the last flown installation.

```bash
mkdir -p ~/workspaces
cd ~/workspaces
git clone -b vio-as-gps git@github.com:AuWeerachai/VIO_PX4.git
cd VIO_PX4
git rev-parse HEAD
./bootstrap/bootstrap_isaac_ros.sh --install-deps
```

Read the selected commit's release/setup notes before running the last command;
the branch may have advanced beyond the recorded runtime baseline.

This full bootstrap checks the host/L4T prerequisites, clones pinned NVIDIA
sources, downloads assets, then calls `bootstrap_jetson.sh`. The latter patches
and builds MAVROS, installs our Isaac overrides, patches the Dockerfile/cuVSLAM
source, builds the bridge, and creates `~/vio-launch` and
`~/.local/bin/vio-launch`. The first camera launch may download/build the image;
later launches reuse caches and incrementally build the cuVSLAM overlay.

If compatible Isaac sources/assets are already present, use only
`./bootstrap/bootstrap_jetson.sh --install-deps`. Existing mismatched NVIDIA or
MAVROS checkouts are rejected rather than overwritten. Backups precede selected
override replacements. **Neither script is a dry run**; even without
`--install-deps`, cloning, patching, copying and builds can occur.

Restore/review `~/.config/vio-launch/config.json` from the secured backup before
launch. Check absolute paths if the username differs from `teamc`. Preserve
the sibling layout:

```text
~/workspaces/VIO_PX4
~/workspaces/isaac_ros-dev
~/workspaces/mavros
~/gnss-validation                 # separately restored RTK tooling
~/vio-launch                     # launcher link, not a second source copy
```

Internet/registry access is needed for fresh dependency retrieval. Not every
apt/pip package or container base is locked by digest, so pinned Git commits
alone do not establish byte-for-byte reproducibility. A retained disk image
is the strongest copy of the actual working installation.

### 5.4 Restore settings, then validate the container

Before starting Path A, review the saved camera SDK serial, XYZ/RPY extrinsics,
home latitude/longitude and altitude reference, FC device/baud/system ID, heading
source, physical gate limits and recovery settings. A successful install does
not recover these values. Close older launcher menus after an update, then use
`~/vio-launch`; both root and PATH launchers point to this repository.

The launcher opens `run_dev.sh` on the Jetson desktop and builds the VSLAM source
overlay before starting the camera. For setup inspection only, with no existing
camera session, the same development shell can be opened manually on the host:

```bash
export ISAAC_ROS_WS="$HOME/workspaces/isaac_ros-dev"
cd "$ISAAC_ROS_WS/src/isaac_ros_common"
./scripts/run_dev.sh
```

Inside that container, verify the build/overlay from a clean ROS environment:

```bash
cd /workspaces/isaac_ros-dev
source /opt/ros/humble/setup.bash
colcon build --packages-select isaac_ros_visual_slam
source install/setup.bash
ros2 pkg prefix isaac_ros_visual_slam
ros2 pkg prefix realsense2_camera
ros2 pkg prefix realsense2_camera_msgs
```

VSLAM should resolve into `/workspaces/isaac_ros-dev/install/`. In this VO
baseline, the RealSense driver **and messages** should resolve to the matching
base-image packages under `/opt/ros/humble`, not an experimental workspace
overlay. Also verify CUDA using the Isaac environment/GPU diagnostic tools;
being able to open a shell inside Docker does not establish working GPU access.

Confirm host/container ROS discovery using the actual body-odometry topic and
matching `ROS_DOMAIN_ID`/compatible middleware configuration. Preserve any
deployment-specific DDS settings; do not blindly copy stale `.bashrc` sources
from deleted workspaces. On the host, source Humble and the bridge overlay,
not the container's ARM camera-library installation.

### 5.5 Which changes require which rebuild?

| Change | Required action while stopped/on the ground |
|---|---|
| Bridge Python source | Build `vio_px4_bridge` on the host and restart its process; source the updated host overlay |
| cuVSLAM source or its installed launch files | Incremental colcon build inside Isaac, then source that overlay; the launcher does this on a new camera session |
| Project-owned Isaac override scripts | Reinstall reviewed overrides from this repo; editing a repo copy alone does not change its installed copy in Isaac |
| Dockerfile or base-image dependencies | Rebuild the relevant Docker image layers and recreate the container; source overlay rebuild alone is insufficient |
| Saved extrinsics or gate/heading settings | Restart the affected camera/bridge path as instructed by the CLI; existing processes do not automatically adopt every change |
| PX4 firmware/parameters | Separate FC update and validation; Jetson bootstrap does not flash or configure PX4 |

## 6. Acceptance checks before returning to use

- [ ] Boot Ubuntu successfully; confirm intended NVMe root and adequate space.
- [ ] Confirm actual repository revisions and intentional local modifications.
- [ ] Verify Docker GPU runtime and ROS communication across host/container.
- [ ] Verify rear D456 serial, USB 3, both stereo streams and exclusive camera
  ownership; do not let the arm application select the VIO camera.
- [ ] Confirm **VO**, `enable_imu_fusion=False`, `/visual_slam/tracking/odometry`
  with `child_frame_id=drone_link`, fresh samples, and correct extrinsics/TF.
- [ ] Verify source changes resolve to the built workspace package, not an
  unintended `/opt/ros` installation. Source the overlay after a successful build.
- [ ] Check RealSense driver/message compatibility as described in section 7;
  do not reuse an experimental `install/` tree merely to save build time.
- [ ] Restore vehicle-specific home and gate settings; verify actual PX4 heading.
- [ ] Check FC serial ownership, heartbeat and parameter readbacks. Path A uses
  `MAV_USEHILGPS=1`, `EKF2_GPS_CTRL=1` (horizontal only), and height reference
  barometer (`0`) or rangefinder (`2`). These are not a full aircraft parameter set.
- [ ] Keep optional external yaw off and `EKF2_EV_CTRL=0` for GPS-only recovery.
  If enabling it, follow the separate [external-yaw checklist](PATH_A_EXTERNAL_YAW.md)
  instead; do not combine settings from the two modes.
- [ ] With props removed/on the ground, validate RC channel 6 triggering,
  stationary spoof, camera checks, heading alignment and global-origin readback.
  Configured origin altitude is AMSL, not rangefinder AGL. Preserve the actual
  home; do not enable synthetic GPS height fusion to fix a reference problem.
- [ ] Confirm PX4 global/local validity separately. After an FC reboot, do not
  assume a bridge from the previous FC session has re-established its origin.
  The abandoned re-spoof implementation is **not** part of this baseline.
- [ ] Verify real-world direction/displacement against RViz and PX4, and test
  loss/quarantine/recovery without flight. No isolated-sample recovery shortcut.
- [ ] Restore RTK separately if comparison is needed: verify correction usage,
  actual Fixed/Float status, timestamps, heading capture and fresh initial offset.
- [ ] Resource-test VIO with the arm application before simultaneous use. Do
  not infer stability solely from a successful launch or visible camera image.

See [PX4 interface checklist](PX4_INTERFACE_CHECKLIST.md) for the bridge checks.
Passing software tests is not aircraft flight qualification.

## 7. Known deployment pitfalls

| Symptom | Check before changing estimator settings |
|---|---|
| Camera identity service times out or camera container segfaults | Explicit D456 serial, USB/ownership, driver/message library versions and startup log; RViz opening is not camera-health proof |
| Only one stereo image, or no features | USB 3 link and both infrared streams, then actual odometry; a visible left image is insufficient |
| `cudaErrorMemoryAllocation` / `GXF_OUT_OF_MEMORY` | Available shared memory and competing GPU processes, including arm software; swapping/log cleanup does not by itself fix GPU allocation failures |
| FC connection fails but camera works | Actual serial device, permissions, baud, port owner and PX4 heartbeat; do not restart the camera to repair FC connectivity |
| Edits do not take effect | Package prefix, installed override copy, overlay build/source order and still-running old processes |
| Global altitude is invalid with valid local height | Global-origin readback versus height-source fusion; do not turn on synthetic GPS altitude fusion as a workaround |

An earlier camera failure involved a **4.51.1 driver with 4.57.6 message
libraries** from a different overlay. Their `DeviceInfo` service layouts differed
(the newer response included `physical_port`); this is a binary/interface
compatibility problem, not a VO gate setting. Package names alone are not enough
to prove compatibility. Check the selected package prefixes/versions and, if
needed, the libraries actually loaded by the camera process.

Current startup rejects an explicitly sourced `.vio-imu` overlay, but that guard
does **not** catch every way to mix libraries through a normal workspace
`install/` or `LD_LIBRARY_PATH`. A fresh build must not inherit those experimental
artifacts. Preserve them in a backup rather than copying them into the new
runtime. This documentation does not claim that runtime guard has been expanded.

Use `lsusb -t` to check the D456 link (USB 3 should report at least `5000M`), and
inspect the SDK-reported serial. Test enumeration with the camera pipeline
stopped; do not start a second driver/viewer against an active navigation camera.
Keep the arm's separate camera selection explicit as well. The two applications
must also be tested together for RAM/GPU/USB load on the actual replacement.

## 8. Save an installation record and know what is verified

After acceptance, securely retain these alongside the backup, not as secrets or
large binaries in Git:

- Module/carrier identity, `/etc/nv_tegra_release`, relevant installed NVIDIA/ROS
  package versions and storage layout.
- Full VIO_PX4/Common/VSLAM/MAVROS commit IDs and any local patches.
- Docker image IDs/digests and, if offline recovery is required, image archives.
- Camera SDK serial, camera firmware/driver versions and verified stream setup.
- Saved launcher configuration and exact PX4 firmware artifact/hash plus params.
- RTK deployment/configuration and raw recordings, if comparison is required.

Review status for this documentation update:

- Local documentation cross-checked against the bootstrap, launcher and bridge.
- A clean replacement-Jetson rebuild has **not** been executed for this review.
- No live Jetson inspection or deployment was performed for this documentation
  update. Recover/compare the actual on-device config, uncommitted work, RTK tools
  and logs with the working NVMe/backups; do not assume Git contains them.
- Some dependency artifacts require online retrieval. Preserve the old disk
  image and exact firmware artifacts before attempting any reinstall.

References: [NVIDIA L4T 36.4 UEFI documentation](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/Bootloader/UEFI.html),
[NVIDIA flashing documentation](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/FlashingSupport.html),
[pinned Isaac bootstrap](ISAAC_ROS_BOOTSTRAP.md).
