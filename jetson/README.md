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
