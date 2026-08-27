# VIO_PX4

Jetson Orin Nano cuVSLAM bridge for PX4, supporting VIO-as-GPS and MAVROS
external-vision paths.

- Run the installed system with `./vio-launch`.
- ROS source is under `src/`.
- Operator launcher code is under `scripts/vio-launch/`.
- PX4 and architecture documentation is under `docs/`.
- Fresh-Jetson provisioning is self-contained under [`bootstrap/`](bootstrap/README.md).

After provisioning a flight Jetson, `bootstrap/` may be removed from that
device without affecting normal Path A or Path B operation. Keep it in Git and
on development machines for reconstruction and repair.
