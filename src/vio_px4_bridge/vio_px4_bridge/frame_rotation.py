"""Small, dependency-free SO(3) helpers. Quaternions are w,x,y,z throughout.

R maps child/body column vectors into parent/world coordinates. A change of
world basis multiplies on the left; a change of body basis on the right.
"""
import math

IDENTITY = (1.0, 0.0, 0.0, 0.0)


def normalize(q):
    if len(q) != 4 or not all(math.isfinite(x) for x in q):
        raise ValueError("nonfinite/malformed orientation")
    norm = math.hypot(*q)
    if not math.isfinite(norm) or norm < 1e-9:
        raise ValueError("zero orientation")
    return tuple(x/norm for x in q)


def inverse(q):
    w, x, y, z = normalize(q)
    return w, -x, -y, -z


def multiply(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return normalize((w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
                      w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v))


def from_rpy(roll, pitch, yaw):
    if not all(math.isfinite(x) for x in (roll, pitch, yaw)):
        raise ValueError("nonfinite attitude")
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    return normalize((cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                      cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy))


def matrix(q):
    w, x, y, z = normalize(q)
    return ((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)),
            (2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)),
            (2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)))


def rotate_matrix(r, v):
    return tuple(sum(row[k]*v[k] for k in range(3)) for row in r)


def rotate(q, v):
    return rotate_matrix(matrix(q), v)


def yaw(q):
    r = matrix(q)
    return math.atan2(r[1][0], r[0][0])


def distance(a, b):
    dot = abs(sum(x*y for x, y in zip(normalize(a), normalize(b))))
    return 2*math.acos(min(1.0, max(0.0, dot)))


def tilt(q):
    return math.acos(min(1.0, max(-1.0, matrix(q)[2][2])))


def tilt_disagreement(a, b):
    # World-down expressed in the body: independent of the world yaw gauge.
    za, zb = matrix(a)[2], matrix(b)[2]
    return math.acos(min(1.0, max(-1.0, sum(x*y for x, y in zip(za, zb)))))


def interpolate(a, b, fraction):
    a, b = normalize(a), normalize(b)
    dot = sum(x*y for x, y in zip(a, b))
    if dot < 0:
        b, dot = tuple(-x for x in b), -dot
    dot = min(1.0, max(0.0, dot))
    if dot > .9995:
        return normalize(tuple(x+(y-x)*fraction for x, y in zip(a, b)))
    angle = math.acos(dot)
    sa, sb = math.sin((1-fraction)*angle), math.sin(fraction*angle)
    return normalize(tuple((sa*x+sb*y)/math.sin(angle) for x, y in zip(a, b)))


def ros_to_ned_frd(q_xyzw):
    x, y, z, w = q_xyzw
    # ENU -> NED on the left, FRD -> FLU on the right. Both are proper rotations.
    enu_ned = (0.0, math.sqrt(.5), math.sqrt(.5), 0.0)
    frd_flu = (0.0, 1.0, 0.0, 0.0)
    return multiply(multiply(enu_ned, normalize((w, x, y, z))), frd_flu)
