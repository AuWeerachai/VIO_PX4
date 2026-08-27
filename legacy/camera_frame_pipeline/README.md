# Legacy camera-frame pipeline

These files preserve the earlier experimental flow that launched cuVSLAM in
`camera_link` and converted camera odometry afterward. They are retained only
for history and comparison.

Do not install or run this path with the production launcher. The current
architecture publishes the physical `drone_link -> camera_link` extrinsic
before cuVSLAM starts and asks cuVSLAM to publish vehicle-body odometry directly
with `base_frame:=drone_link`.

Production files live under:

```text
bootstrap/jetson-config/isaac_ros_overrides/
scripts/vio-launch/
src/vio_px4_bridge/
```
