# Jetson deployment files

This directory contains every project-owned override that otherwise lives
inside the external Isaac ROS workspace.

Run from a fresh clone on a Jetson that already has the Isaac ROS cuVSLAM
workspace:

```bash
cd ~/workspaces/VIO_PX4
./scripts/bootstrap_jetson.sh --install-deps
~/vio-launch
```

For a Jetson without the NVIDIA workspace, run
`./scripts/bootstrap_isaac_ros.sh --install-deps` instead. It selects only the
supported ROS 2 Humble Isaac ROS release for the detected JetPack/L4T version,
clones NVIDIA's `isaac_ros_common`, downloads the matching visual-SLAM assets,
and then invokes `bootstrap_jetson.sh`. It deliberately rejects unknown JetPack
versions rather than installing an incompatible current release.

The bootstrap script copies the reviewed overrides to:

```text
${ISAAC_ROS_WS}/src/isaac_ros_common/scripts/start_vio.sh
${ISAAC_ROS_WS}/src/isaac_ros_common/scripts/start_rviz.sh
${ISAAC_ROS_WS}/src/isaac_ros_common/scripts/.isaac_ros_common-config
```

It creates timestamped backups before replacing existing files. The Git copies
under `jetson/isaac_ros_overrides/` remain authoritative.

The default launch is the proven visual-only cuVSLAM configuration. The staged
RealSense IMU design is documented in `docs/STAGED_IMU_IMPLEMENTATION.md` and
must not become the default until it passes the documented bench tests.

The exact NVIDIA/JetPack pins, original manual layers, and clean-room rebuild
sequence are documented in `docs/ISAAC_ROS_BOOTSTRAP.md`.
