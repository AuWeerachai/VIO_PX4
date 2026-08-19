"""Local-tangent-plane helpers (WGS84, no extra geo deps)."""

from __future__ import annotations

import math

# WGS84 mean Earth radius used for small local NED <-> LLA mappings.
_EARTH_RADIUS_M = 6378137.0


def ned_offset_to_lla(
    origin_lat_deg: float,
    origin_lon_deg: float,
    origin_alt_m: float,
    north_m: float,
    east_m: float,
    down_m: float,
) -> tuple[float, float, float]:
    """Convert local NED offset from origin into WGS84 lat/lon/alt (MSL)."""
    lat0 = math.radians(origin_lat_deg)
    d_lat = north_m / _EARTH_RADIUS_M
    d_lon = east_m / (_EARTH_RADIUS_M * max(1e-6, math.cos(lat0)))
    lat_deg = origin_lat_deg + math.degrees(d_lat)
    lon_deg = origin_lon_deg + math.degrees(d_lon)
    alt_m = origin_alt_m - down_m
    return lat_deg, lon_deg, alt_m


def enu_offset_to_ned(east_m: float, north_m: float, up_m: float) -> tuple[float, float, float]:
    """ROS ENU vector -> PX4/NED vector."""
    return north_m, east_m, -up_m


def rotate_vector_by_quaternion_xyzw(
    vx: float,
    vy: float,
    vz: float,
    qx: float,
    qy: float,
    qz: float,
    qw: float,
) -> tuple[float, float, float]:
    """Rotate a vector by a normalized quaternion (body frame -> world frame)."""
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-9:
        raise ValueError("odometry quaternion has zero norm")
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    # Efficient q * v * conjugate(q).
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


def flu_velocity_to_ned(
    vx_flu: float,
    vy_flu: float,
    vz_flu: float,
    yaw_ned_rad: float,
) -> tuple[float, float, float]:
    """
    Body FLU velocity -> NED using yaw in NED (0 = north, positive to east).

    For heading-only rotation (level flight approximation). Prefer world-frame
    twist when the VIO topic already publishes ENU world velocity.
    """
    # FLU -> FRD
    vx_frd = vx_flu
    vy_frd = -vy_flu
    vz_frd = -vz_flu
    c = math.cos(yaw_ned_rad)
    s = math.sin(yaw_ned_rad)
    vn = c * vx_frd - s * vy_frd
    ve = s * vx_frd + c * vy_frd
    vd = vz_frd
    return vn, ve, vd


def course_over_ground_cdeg(vn_m_s: float, ve_m_s: float) -> int:
    """NED velocity -> HIL_GPS cog in centidegrees (0..35999), or 65535 if invalid."""
    speed = math.hypot(vn_m_s, ve_m_s)
    if speed < 0.05:
        return 65535
    # atan2(east, north) -> radians from north, clockwise-ish for NED course
    cog_rad = math.atan2(ve_m_s, vn_m_s)
    if cog_rad < 0.0:
        cog_rad += 2.0 * math.pi
    return int(round(math.degrees(cog_rad) * 100.0)) % 36000


def wrap_pi(angle_rad: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quaternion_xyzw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Return the Z-axis yaw of a body->world quaternion."""
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-9:
        raise ValueError("odometry quaternion has zero norm")
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    return math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


def rotate_ned_heading(
    north: float, east: float, down: float, heading_offset_rad: float
) -> tuple[float, float, float]:
    """Rotate a NED vector about Down by a north-heading offset."""
    c = math.cos(heading_offset_rad)
    s = math.sin(heading_offset_rad)
    return c * north - s * east, s * north + c * east, down


def tilt_compensated_compass_yaw(
    roll_rad: float,
    pitch_rad: float,
    mag_x_frd: float,
    mag_y_frd: float,
    mag_z_frd: float,
    declination_rad: float,
) -> float:
    """True-north NED body yaw from a calibrated body-FRD magnetometer.

    Roll and pitch may come from the FC attitude estimator. FC yaw is
    deliberately not an input, avoiding a feedback loop through injected GPS.
    This is the same leveling convention used by the Basalt runtime.
    """
    cr, sr = math.cos(roll_rad), math.sin(roll_rad)
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    # Rx(roll), then Ry(pitch), matching the proven Basalt implementation.
    x1 = mag_x_frd
    y1 = cr * mag_y_frd - sr * mag_z_frd
    z1 = sr * mag_y_frd + cr * mag_z_frd
    x_level = cp * x1 + sp * z1
    y_level = y1
    return wrap_pi(math.atan2(-y_level, x_level) + declination_rad)
