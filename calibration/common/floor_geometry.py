"""Read-only, floor-based camera extrinsic estimation. No ROS dependency.

Matrices use column vectors: R_BC maps camera_link coordinates into drone_link.
The operator must independently establish that drone_link and the floor are level.
Yaw and translation are NOT observable from a floor normal and are not estimated.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np


class CalibrationError(ValueError):
    pass


def angle_change_lines(current, proposed):
    """Display the subtraction/addition using exactly the printed CLI values."""
    from decimal import Decimal
    lines = []
    for label, old, new in zip(('Roll', 'Pitch'), current[:2], proposed[:2]):
        a, b = Decimal(f'{old:.3f}'), Decimal(f'{new:.3f}')
        delta = b-a
        sign = '+' if delta >= 0 else '−'
        lines.append(f'{label}: {a:+.3f}° {sign} {abs(delta):.3f}° = {b:+.3f}°')
    return lines


def unit(v):
    v = np.asarray(v, dtype=float)
    length = np.linalg.norm(v)
    if v.shape != (3,) or not np.isfinite(v).all() or length < 1e-12:
        raise CalibrationError("invalid normal/vector")
    return v / length


def angle_deg(a, b):
    return math.degrees(math.acos(float(np.clip(unit(a) @ unit(b), -1, 1))))


def level_basis(up_body):
    """Body FLU -> a horizontal frame, using independent up, never the floor fit.

    Horizontal X follows projected body-forward. This is a model of a level-start
    reference, not a change to the recorded RViz map or to any live transform.
    """
    up = unit(up_body)
    x = unit(np.array([1., 0., 0.])-up*up[0])
    return np.stack((x, np.cross(up, x), up))


def rpy_matrix(roll_deg, pitch_deg, yaw_deg):
    r, p, y = np.radians([roll_deg, pitch_deg, yaw_deg])
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp, cp*sr, cp*cr]])


def validate_rotation(r):
    r = np.asarray(r, dtype=float)
    if (r.shape != (3, 3) or not np.isfinite(r).all()
            or not np.allclose(r.T @ r, np.eye(3), atol=1e-6)
            or not math.isclose(np.linalg.det(r), 1, abs_tol=1e-6)):
        raise CalibrationError("invalid rotation matrix")
    return r


def matrix_rpy(r):
    r = validate_rotation(r)
    p = math.atan2(-r[2, 0], math.hypot(r[0, 0], r[1, 0]))
    if abs(math.cos(p)) < 0.05:
        raise CalibrationError("near-nadir mounting: roll/yaw cannot be separated reliably")
    return np.degrees([math.atan2(r[2, 1], r[2, 2]), p, math.atan2(r[1, 0], r[0, 0])])


def quaternion_matrix(x, y, z, w):
    q = np.asarray([x, y, z, w], dtype=float)
    if not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise CalibrationError("invalid TF quaternion")
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def rotation_angle_deg(r):
    r = validate_rotation(r)
    return math.degrees(math.acos(float(np.clip((np.trace(r)-1)/2, -1, 1))))


def suggest_extrinsic(normal_body, current_rotation, fixed_yaw_deg=180.0,
                      desired_up_body=(0, 0, 1)):
    """Solve R_new R_old.T measured_up = independent physical up in body FLU.

    Flight mode supplies up from PX4 attitude; stationary mode requires a level
    body. Yaw is fixed, never inferred from the plane. Pick the solution nearest
    the current mounting, rejecting a nearly singular roll/pitch solution.
    """
    current_rotation = validate_rotation(current_rotation)
    camera_up = current_rotation.T @ unit(normal_body)
    target = rpy_matrix(0, 0, -fixed_yaw_deg) @ unit(desired_up_body)
    yz = math.hypot(camera_up[1], camera_up[2])
    if yz < 0.05 or abs(target[1]) >= yz-1e-8:
        raise CalibrationError("near-nadir mounting: floor normal does not resolve roll")
    alpha = math.atan2(camera_up[1], camera_up[2])
    beta = math.asin(float(target[1]/yz))
    candidates = []
    for r in (alpha-beta, alpha-(math.pi-beta)):
        rotated = rpy_matrix(math.degrees(r), 0, 0) @ camera_up
        p = math.atan2(target[0], target[2])-math.atan2(rotated[0], rotated[2])
        roll, pitch = [(math.degrees(v)+180) % 360-180 for v in (r, p)]
        candidate = rpy_matrix(roll, pitch, fixed_yaw_deg)
        candidates.append((rotation_angle_deg(candidate @ current_rotation.T), roll, pitch, candidate))
    _, roll, pitch, proposed = min(candidates, key=lambda item: item[0])
    matrix_rpy(proposed)  # Reject Euler singularity even if the vector fit exists.
    correction = proposed @ current_rotation.T
    if angle_deg(correction @ unit(normal_body), desired_up_body) > 1e-4:
        raise CalibrationError("internal rotation consistency check failed")
    return np.array([roll, pitch, fixed_yaw_deg]), correction


def fit_absolute_mount(camera_normals, body_up, fixed_yaw_deg=180.0):
    """Fit absolute camera->body R/P to independent up vectors.

    No current mount is an argument, prior, or regularizer. Callers remove the
    *recorded* mount once when recovering camera-frame normals. All samples have
    equal weight; a densely repeated cloud cannot dominate by point count.
    """
    from scipy.optimize import least_squares
    normals = np.asarray([unit(n) for n in camera_normals])
    targets = np.asarray([unit(n) for n in body_up])
    if len(normals) < 3 or normals.shape != targets.shape:
        raise CalibrationError('absolute fit requires at least three paired normals')
    # The rear/down mounting is away from the +/-90-degree Euler singularity.
    # Seed from independent per-sample absolute solutions, not the old settings.
    seed = []
    for n, up in zip(normals, targets):
        # A nominal rotation selects only the Euler branch; it contributes no
        # residual and no regularization term to the optimization below.
        a, _ = suggest_extrinsic(n, np.eye(3), fixed_yaw_deg, up)
        if abs(a[0]) >= 85 or abs(a[1]) >= 85:
            # The alternate Euler solution represents the same normal mapping.
            target = rpy_matrix(0, 0, -fixed_yaw_deg) @ up
            alpha = math.atan2(n[1], n[2])
            beta = math.asin(float(target[1]/math.hypot(n[1], n[2])))
            choices = []
            for roll in (alpha-beta, alpha-(math.pi-beta)):
                rotated = rpy_matrix(math.degrees(roll), 0, 0) @ n
                pitch = math.atan2(target[0], target[2])-math.atan2(rotated[0], rotated[2])
                rr, pp = [(math.degrees(v)+180) % 360-180 for v in (roll, pitch)]
                if abs(rr) < 85 and abs(pp) < 85:
                    choices.append([rr, pp, fixed_yaw_deg])
            if not choices:
                raise CalibrationError('floor normal does not resolve this mounting branch')
            a = choices[0]
        seed.append(a[:2])
    def residual(x):
        return (normals @ rpy_matrix(x[0], x[1], fixed_yaw_deg).T-targets).ravel()
    result = least_squares(residual, np.median(seed, axis=0), bounds=(-84.999, 84.999),
                           loss='soft_l1', f_scale=math.radians(.25),
                           xtol=1e-12, ftol=1e-12, gtol=1e-12)
    if not result.success or not np.isfinite(result.x).all():
        raise CalibrationError('absolute roll/pitch fit did not converge')
    angles = np.array([*result.x, fixed_yaw_deg])
    errors = np.array([angle_deg(rpy_matrix(*angles) @ n, up)
                       for n, up in zip(normals, targets)])
    return angles, errors


def read_xyz(msg):
    """PointCloud2 XYZ decoder, including endian, point padding and row padding."""
    if msg.width == 0 or msg.height == 0:
        return np.empty((0, 3))
    types = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}
    fields = {f.name: f for f in msg.fields}
    formats, offsets = [], []
    for name in ("x", "y", "z"):
        f = fields.get(name)
        if f is None or f.count != 1 or f.datatype not in types:
            raise CalibrationError("point cloud must contain scalar x/y/z fields")
        dtype = np.dtype((">" if msg.is_bigendian else "<") + types[f.datatype])
        if f.offset < 0 or f.offset + dtype.itemsize > msg.point_step:
            raise CalibrationError("invalid point cloud field offset")
        formats.append(dtype)
        offsets.append(f.offset)
    if msg.row_step < msg.width * msg.point_step or len(msg.data) < msg.row_step * msg.height:
        raise CalibrationError("truncated point cloud buffer")
    dtype = np.dtype(dict(names=["x", "y", "z"], formats=formats,
                         offsets=offsets, itemsize=msg.point_step))
    array = np.ndarray((msg.height, msg.width), dtype=dtype, buffer=bytes(msg.data),
                       strides=(msg.row_step, msg.point_step))
    points = np.column_stack([array[name].ravel().astype(float) for name in ("x", "y", "z")])
    return points[np.isfinite(points).all(axis=1)]


@dataclass
class Limits:
    height_m: float
    height_tolerance_m: float = 0.12
    distance_threshold_m: float = 0.025
    max_plane_rms_m: float = 0.015
    max_floor_tilt_deg: float = 25.0
    min_points: int = 50
    min_span_m: float = 0.35
    min_floor_fraction: float = 0.20
    max_range_m: float = 6.0
    min_frames: int = 10
    min_duration_s: float = 5.0
    min_good_fraction: float = 0.75
    max_normal_spread_deg: float = 1.0
    max_patch_disagreement_deg: float = 2.0
    max_body_motion_m: float = 0.05
    max_body_rotation_deg: float = 2.0
    max_correction_deg: float = 10.0
    ransac_trials: int = 250
    voxel_m: float = 0.025

    def validate(self):
        for key, value in asdict(self).items():
            if not math.isfinite(value) or value <= 0:
                raise CalibrationError(f"{key} must be positive and finite")
        if self.height_tolerance_m >= self.height_m:
            raise CalibrationError("height tolerance must be smaller than body height")
        if self.min_points < 12 or self.min_frames < 3:
            raise CalibrationError("at least 12 points and 3 frames are required")
        if self.min_good_fraction > 1 or self.min_floor_fraction > 1 or self.max_floor_tilt_deg >= 80:
            raise CalibrationError("invalid fraction or floor tilt limit")


def plane_svd(points):
    center = np.mean(points, axis=0)
    _, _, vt = np.linalg.svd(points-center, full_matrices=False)
    normal = unit(vt[-1])
    if normal[2] < 0:
        normal = -normal
    return normal, -float(normal @ center), vt[:2]


def fit_floor(points, limits, seed=0):
    """Find supported, near-horizontal planes at the independently measured height."""
    limits.validate()
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise CalibrationError("points must have shape N x 3")
    roi = points[np.isfinite(points).all(axis=1)]
    roi = roi[(roi[:, 2] < -0.04) & (np.linalg.norm(roi, axis=1) < limits.max_range_m)]
    if len(roi) < limits.min_points:
        raise CalibrationError("too few points below the body")
    # Repeated landmarks must not inflate the count or dominate a fit.
    _, index = np.unique(np.floor(roi / limits.voxel_m).astype(np.int64), axis=0, return_index=True)
    roi = roi[np.sort(index)]
    rng = np.random.default_rng(seed)
    if len(roi) > 4000:
        roi = roi[rng.choice(len(roi), 4000, replace=False)]
    candidates, remaining = [], roi
    cosine = math.cos(math.radians(limits.max_floor_tilt_deg))
    for _ in range(4):
        if len(remaining) < limits.min_points:
            break
        best_mask = None
        best_count = 0
        for _ in range(limits.ransac_trials):
            tri = remaining[rng.choice(len(remaining), 3, replace=False)]
            raw = np.cross(tri[1]-tri[0], tri[2]-tri[0])
            length = np.linalg.norm(raw)
            if length < 1e-8:
                continue
            n = raw/length
            if n[2] < 0:
                n = -n
            if n[2] < cosine:
                continue  # Not a floor: never pick the largest wall.
            d = -float(n @ tri[0])
            if abs(d/n[2] - limits.height_m) > limits.height_tolerance_m:
                continue
            mask = np.abs(remaining @ n+d) <= limits.distance_threshold_m
            count = int(mask.sum())
            if count > best_count:
                best_mask, best_count = mask, count
        if best_mask is None or best_count < limits.min_points:
            break
        for _ in range(3):
            n, d, axes = plane_svd(remaining[best_mask])
            best_mask = np.abs(remaining @ n+d) <= limits.distance_threshold_m
            if best_mask.sum() < limits.min_points:
                break
        inside = remaining[best_mask]
        remaining = remaining[~best_mask]
        if len(inside) < limits.min_points or n[2] < cosine:
            continue
        height = d/n[2]
        if abs(height-limits.height_m) > limits.height_tolerance_m:
            continue
        rms = float(np.sqrt(np.mean((inside @ n+d)**2)))
        coordinates = (inside - inside.mean(axis=0)) @ axes.T
        spans = np.percentile(coordinates, 90, axis=0) - np.percentile(coordinates, 10, axis=0)
        if rms > limits.max_plane_rms_m or min(spans) < limits.min_span_m:
            continue
        halves = coordinates[:, 0] <= np.median(coordinates[:, 0])
        patch_angle = angle_deg(plane_svd(inside[halves])[0], plane_svd(inside[~halves])[0])
        if patch_angle > limits.max_patch_disagreement_deg:
            continue  # Locally warped/sloping reconstruction: not one rigid tilt.
        candidates.append(dict(normal=n, d=d, points=inside, count=len(inside), rms=rms,
                               height_m=height, spans_m=spans, patch_angle_deg=patch_angle))
    if not candidates:
        raise CalibrationError("no sufficiently flat, broad floor at the measured body height")
    candidates.sort(key=lambda x: x["count"], reverse=True)
    best = candidates[0]
    if best["count"]/len(roi) < limits.min_floor_fraction:
        raise CalibrationError("floor has too little support relative to other points")
    for other in candidates[1:]:
        if other["count"] >= 0.6*best["count"]:
            raise CalibrationError("ambiguous: multiple plausible floor planes; change the view")
    best["roi"] = roi
    return best


def analyze_frames(frames, limits, fixed_yaw_deg=180.0):
    """Assess a stationary capture; return suggestions only when all checks pass."""
    limits.validate()
    errors, fits, rows = [], [], []
    if not frames:
        return {"accepted": False, "reasons": ["no captured frames"]}, []
    first = frames[0]
    r_bc = validate_rotation(first["r_bc"])
    for field in ("r_ob", "r_gb"):
        validate_rotation(first[field])
    previous_stamp = -float("inf")
    for index, frame in enumerate(frames):
        for field in ("r_bc", "r_ob", "r_gb"):
            validate_rotation(frame[field])
        stamp = float(frame["stamp"])
        if not math.isfinite(stamp) or stamp <= previous_stamp:
            errors.append("cloud timestamps duplicated or regressed")
        previous_stamp = stamp
        if (np.linalg.norm(frame["t_bc"]-first["t_bc"]) > 1e-5
                or rotation_angle_deg(frame["r_bc"] @ r_bc.T) > 0.01):
            errors.append("active camera extrinsic changed during capture")
        if np.linalg.norm(frame["t_ob"]-first["t_ob"]) > limits.max_body_motion_m:
            errors.append("body moved, VO drifted, or odom reset during capture")
        if rotation_angle_deg(frame["r_ob"] @ first["r_ob"].T) > limits.max_body_rotation_deg:
            errors.append("body rotated or VO attitude reset during capture")
        row = {"stamp": stamp, "frame_index": index}
        try:
            fit = fit_floor(frame["points_body"], limits, seed=index)
            rpy, correction = suggest_extrinsic(fit["normal"], r_bc, fixed_yaw_deg)
            row.update(accepted=True, roll_deg=float(rpy[0]), pitch_deg=float(rpy[1]),
                       plane_rms_m=fit["rms"], floor_height_m=fit["height_m"],
                       floor_tilt_deg=angle_deg(fit["normal"], [0, 0, 1]), count=fit["count"])
            fit["frame_index"] = index
            fits.append(fit)
        except CalibrationError as exc:
            row.update(accepted=False, reason=str(exc))
        rows.append(row)
    if len(fits) < limits.min_frames:
        errors.append(f"only {len(fits)} good frames; need {limits.min_frames}")
    if len(fits)/len(frames) < limits.min_good_fraction:
        errors.append("too many frames failed floor detection")
    good_stamps = [x["stamp"] for x in rows if x["accepted"]]
    if len(good_stamps) < 2 or good_stamps[-1]-good_stamps[0] < limits.min_duration_s:
        errors.append("insufficient duration of valid observations")
    report = dict(accepted=False, reasons=[], captured_frames=len(frames), good_frames=len(fits),
                  thresholds=asdict(limits), per_frame=rows,
                  yaw_deg_fixed=fixed_yaw_deg, yaw_is_estimated=False,
                  translation_is_estimated=False,
                  assumption="independently level drone_link above a genuinely horizontal floor",
                  caveat="A floor fit cannot distinguish mounting error from systematic stereo reconstruction error.")
    if fits:
        normals = np.stack([x["normal"] for x in fits])
        normal = unit(normals.mean(axis=0))
        deviations = np.array([angle_deg(n, normal) for n in normals])
        spread = float(np.max(deviations))
        if spread > limits.max_normal_spread_deg:
            errors.append(f"floor normal changes over time ({spread:.2f} deg)")
        try:
            rpy, correction = suggest_extrinsic(normal, r_bc, fixed_yaw_deg)
            old_rpy = matrix_rpy(r_bc)
            change = rotation_angle_deg(correction)
            if change > limits.max_correction_deg:
                errors.append(f"required mounting change {change:.2f} deg exceeds {limits.max_correction_deg:g} deg")
            report.update(current_rpy_deg=old_rpy.tolist(), candidate_rpy_deg=rpy.tolist(), unchanged_xyz_m=first["t_bc"].tolist(),
                          measured_floor_normal_body=normal.tolist(), normal_max_deviation_deg=spread,
                          measured_tilt_body_deg=angle_deg(normal, [0, 0, 1]),
                          forward_up_slope_deg=math.degrees(math.atan2(-normal[0], normal[2])),
                          left_up_slope_deg=math.degrees(math.atan2(-normal[1], normal[2])),
                          repeatability_std_roll_pitch_deg=np.std(
                              [[x["roll_deg"], x["pitch_deg"]] for x in rows if x["accepted"]], axis=0).tolist(),
                          mean_floor_rms_m=float(np.mean([x["rms"] for x in fits])),
                          correction_angle_deg=change)
            if not errors:
                report.update(accepted=True, suggested_rpy_deg=rpy.tolist(),
                              correction_rotation=correction.tolist())
        except CalibrationError as exc:
            errors.append(str(exc))
    report["reasons"] = list(dict.fromkeys(errors))
    return report, fits
