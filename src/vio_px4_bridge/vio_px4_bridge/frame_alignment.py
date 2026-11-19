"""Rigid frame rotations for recovery (Hamilton quaternions, w/x/y/z).

A quaternion maps body/vector coordinates into its parent frame. All helpers
normalize inputs; q and -q are equivalent. No camera mounting correction lives
here: cuVSLAM must already publish the Cube's drone_link body pose.
"""

import math


IDENTITY = (1.0, 0.0, 0.0, 0.0)


def normalize(q):
    if len(q) != 4 or not all(math.isfinite(v) for v in q):
        raise ValueError("nonfinite or malformed orientation")
    norm = math.hypot(*q)
    if not math.isfinite(norm) or norm < 1e-9:
        raise ValueError("orientation has zero norm")
    return tuple(v / norm for v in q)


def inverse(q):
    w, x, y, z = normalize(q)
    return w, -x, -y, -z


def multiply(a, b):
    w, x, y, z = normalize(a)
    v, i, j, k = normalize(b)
    return normalize((w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
                      w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v))


def rotate(q, vector):
    w, x, y, z = normalize(q)
    vx, vy, vz = vector
    tx, ty, tz = 2*(y*vz-z*vy), 2*(z*vx-x*vz), 2*(x*vy-y*vx)
    return (vx+w*tx+y*tz-z*ty, vy+w*ty+z*tx-x*tz, vz+w*tz+x*ty-y*tx)


def from_rpy(roll, pitch, yaw):
    """Rz(yaw) Ry(pitch) Rx(roll), in radians."""
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    return normalize((cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                      cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy))


def yaw(q):
    w, x, y, z = normalize(q)
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def angle_between(a, b):
    """Smallest 3-D rotation angle; insensitive to quaternion sign/wrap."""
    d = multiply(inverse(a), b)
    return 2*math.atan2(math.sqrt(sum(v*v for v in d[1:])), abs(d[0]))


def ros_to_ned_frd(q_enu_flu, child_to_body_yaw=0.0):
    """ENU-world/FLU-body -> NED-like-world/FRD-body, not earth alignment."""
    enu_to_ned = (0.0, math.sqrt(0.5), math.sqrt(0.5), 0.0)
    frd_to_flu = (0.0, 1.0, 0.0, 0.0)
    body = multiply(q_enu_flu, from_rpy(0.0, 0.0, child_to_body_yaw))
    return multiply(multiply(enu_to_ned, body), frd_to_flu)


def alignment(raw_body_orientation, desired_body_orientation):
    """Raw-world -> corrected-world; includes roll/pitch, not only yaw."""
    return multiply(desired_body_orientation, inverse(raw_body_orientation))
