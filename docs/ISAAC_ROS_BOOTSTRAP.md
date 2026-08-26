# Proven Isaac ROS / cuVSLAM bootstrap

This deployment is intentionally frozen to the configuration observed on the
working `teamc-desktop` Jetson Orin Nano on 2026-08-26. It must not track the
current NVIDIA release: current Isaac ROS documentation has moved beyond this
ROS 2 Humble system.

## Captured platform

| Component | Proven value |
|---|---|
| Jetson Linux / L4T | `36.4.7` (JetPack 6.2.1 family) |
| Ubuntu / ROS | Ubuntu 22.04 / ROS 2 Humble |
| Isaac ROS | `release-3.2` |
| `isaac_ros_common` | `fcf4d9e17f8f0a7f47f1d22d6a18421ce3768c01` |
| `isaac_ros_visual_slam` | `e31f4cc1d41a329a01946e5fe63669f8b15da677` |
| Visual-SLAM NGC asset | `3.1.0` (the asset used by the 3.2 documentation) |
| librealsense source | `v2.55.1` |
| NVIDIA RealSense ROS branch | `release/4.51.1-isaac` |
| Docker image key | `aarch64.ros2_humble.realsense` |

The authoritative machine-readable pins are in
`jetson/isaac_ros_release.env`.

## Why the original setup was difficult to reproduce

The working workspace was assembled across several layers:

1. JetPack supplies the NVIDIA kernel, CUDA host mounts, Docker, and the
   NVIDIA container runtime.
2. `isaac_ros_common/run_dev.sh` selects and builds the Humble + RealSense
   development image.
3. A manual edit to NVIDIA's `Dockerfile.realsense` installs
   `isaac_ros_visual_slam`, `isaac_ros_examples`, and `isaac_ros_realsense`.
4. `isaac_ros_visual_slam` is a nested source checkout inside
   `isaac_ros_common`, rather than a normal sibling repository.
5. Two manual source changes select 30 FPS, enable IMU by default, and increase
   the image-jitter threshold to 60 ms.
6. Custom `start_vio.sh` and `start_rviz.sh` files live inside NVIDIA's checkout.
7. The NGC interface-spec asset is outside Git.

Items 3–7 were therefore easy to lose when rebuilding or cloning only one
repository. They are now represented by versioned manifests, patches, and
launch overrides in VIO_PX4.

## Fresh installation

Start from a Jetson already flashed with the matching JetPack 6.2.x family:

```bash
cd ~/workspaces
git clone -b vio-as-gps git@github.com:AuWeerachai/VIO_PX4.git
cd VIO_PX4
./scripts/bootstrap_isaac_ros.sh --install-deps
./vio-launch
```

The bootstrap performs these operations in order:

1. Refuses non-aarch64 systems and an L4T family other than 36.4.x.
2. Verifies Docker and the NVIDIA container environment are usable.
3. Clones both NVIDIA repositories at the captured commits.
4. Downloads the pinned NGC visual-SLAM asset.
5. Applies the captured Docker and cuVSLAM source patches idempotently.
6. Installs the project-owned body-frame/extrinsic launch scripts.
7. Builds the VIO/PX4 ROS package and exposes `~/vio-launch`.

The first `./vio-launch` invocation asks `run_dev.sh` to build/pull the large
Isaac ROS image. Docker caches unchanged layers, so subsequent launches should
reuse the image. Changing a relevant Dockerfile invalidates that layer and
causes a rebuild.

## Source edits versus launch configuration

The captured source patch is retained because the requirement is to reproduce
the proven machine exactly. For normal operation, VIO_PX4 also passes the
60 ms jitter threshold explicitly and starts visual-only mode explicitly. This
makes the operational setting visible at the launch boundary instead of relying
solely on modified NVIDIA defaults. IMU fusion remains staged until its separate
bench-validation checklist passes.

Do not copy generated `build/`, `install/`, Docker images, the 11 GB asset
directory, logs, core dumps, swap files, or RViz-generated frame diagrams into
Git. The bootstrap reconstructs the required generated state.
