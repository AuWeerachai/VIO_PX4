"""Heading-assisted translation, with an anchored horizontal noise band.

No sensor fusion or flight-controller position feedback lives here. Inputs must
already pass the raw pose gate. Rotations are Hamilton quaternions (w,x,y,z).
The epoch tilt is fixed; only the yaw correction changes during normal tracking.
"""
import math

from vio_px4_bridge import frame_alignment as frames
from vio_px4_bridge.geo_utils import wrap_pi


class StabilizedDisplacement:
    def __init__(self, noise_radius_m=0.01, heading_tau_s=0.5):
        if not math.isfinite(noise_radius_m) or not 0 <= noise_radius_m <= 0.1:
            raise ValueError("position noise radius must be 0..0.1 m")
        if not math.isfinite(heading_tau_s) or not 0 <= heading_tau_s <= 5:
            raise ValueError("heading smoothing time must be 0..5 s")
        self.noise_radius_m = noise_radius_m
        self.heading_tau_s = heading_tau_s
        self.reset()

    def reset(self):
        self.position = None
        self.filtered_input = None
        self.base_rotation = frames.IDENTITY
        self.timestamp = None
        self.correction = 0.0
        self.residual = 0.0
        self.held = False
        self.pending_distance_m = 0.0

    def seed(self, raw_position, rotation, output_position, timestamp):
        if not all(math.isfinite(v) for v in (*raw_position, *output_position, timestamp)):
            raise ValueError("nonfinite trajectory seed")
        self.reset()
        self.base_rotation = frames.normalize(rotation)
        self.filtered_input = frames.rotate(self.base_rotation, raw_position)
        self.position = tuple(output_position)
        self.timestamp = timestamp

    def update(self, raw_position, raw_orientation, heading_reference, timestamp):
        if self.position is None:
            raise ValueError("trajectory must be seeded before updating")
        if not all(math.isfinite(v) for v in (*raw_position, heading_reference, timestamp)):
            raise ValueError("nonfinite trajectory input")
        dt = timestamp - self.timestamp
        if dt <= 0:
            raise ValueError("trajectory timestamps must increase")
        q = frames.multiply(self.base_rotation, raw_orientation)
        # Filter the discrepancy, not the vehicle heading: a real turn shared
        # by VO and PX4 therefore has no smoothing lag.
        requested = wrap_pi(heading_reference - frames.yaw(q))
        alpha = 1.0 if self.heading_tau_s == 0 else -math.expm1(-dt/self.heading_tau_s)
        correction = wrap_pi(self.correction + alpha*wrap_pi(requested-self.correction))
        middle = wrap_pi(self.correction + .5*wrap_pi(correction-self.correction))
        p = frames.rotate(self.base_rotation, raw_position)
        dx, dy = p[0]-self.filtered_input[0], p[1]-self.filtered_input[1]
        distance = math.hypot(dx, dy)
        # A vector "play" filter: retain a fixed anchor while inside the band.
        # Do NOT drop each small frame-to-frame increment: arbitrarily slow
        # travel eventually crosses the band and is retained with <=radius lag.
        excess = max(0.0, distance-self.noise_radius_m)
        fraction = excess/distance if distance else 0.0
        sx, sy = dx*fraction, dy*fraction
        step = frames.rotate(frames.from_rpy(0, 0, middle), (sx, sy, 0.0))
        self.position = (self.position[0]+step[0], self.position[1]+step[1],
                         self.position[2]+p[2]-self.filtered_input[2])
        self.filtered_input = (self.filtered_input[0]+sx, self.filtered_input[1]+sy, p[2])
        self.pending_distance_m = min(distance, self.noise_radius_m)
        self.held = excess == 0.0
        self.correction, self.timestamp = correction, timestamp
        self.residual = wrap_pi(requested-correction)
        rotation = frames.multiply(frames.from_rpy(0, 0, correction), self.base_rotation)
        return self.position, rotation
