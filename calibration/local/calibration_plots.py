"""Recorded RViz geometry and a separate physical mounting/level-start prediction."""
import numpy as np

from floor_geometry import CalibrationError, level_basis, angle_change_lines


def directional_tilts(normal):
    """Signed plane-section angles, not the combined 3-D tilt magnitude.

    Y-Z: positive about +X (roll) rises toward +Y.
    X-Z: positive about +Y (pitch) falls toward +X, following ROS FLU.
    These are projected section tilts, not a full Euler decomposition.
    """
    n = np.asarray(normal, dtype=float)
    if n.shape != (3,) or not np.isfinite(n).all() or abs(n[2]) < 1e-12:
        raise CalibrationError('plane cannot be expressed as horizontal section slopes')
    if n[2] < 0:
        n = -n
    return np.degrees([np.arctan2(-n[1], n[2]), np.arctan2(n[0], n[2])])


def plot_scene(report, fit, frame, reference='level_start'):
    """The same rigid camera-origin correction is applied to points AND plane.

    For 'recorded', preserve the actual RViz frame. For 'level_start', use PX4
    physical up (or independently confirmed stationary level) to establish the
    predicted horizontal reference. It never comes from the fitted floor normal.
    This predicts geometry with a new mount and level startup, not old-map repair.
    """
    normal, offset = fit['normal'], fit['d']
    ground = fit['points']
    cloud = frame['points_body']
    cloud = cloud[np.isfinite(cloud).all(axis=1)]
    cloud = cloud[np.linalg.norm(cloud, axis=1) < report['thresholds']['max_range_m']]
    if len(cloud) > 2000:
        cloud = cloud[np.linspace(0, len(cloud)-1, 2000, dtype=int)]
    x = np.linspace(*np.percentile(ground[:, 0], [5, 95]), 9)
    y = np.linspace(*np.percentile(ground[:, 1], [5, 95]), 9)
    xx, yy = np.meshgrid(x, y)
    zz = -(normal[0]*xx+normal[1]*yy+offset)/normal[2]
    surface = np.stack((xx, yy, zz), axis=-1)
    if reference == 'recorded':
        r, t = frame['r_gb'], frame['t_gb']
    elif reference == 'level_start':
        r = level_basis(report['pictured_plane_fit']['target_up_body'])
        t = np.zeros(3)
    else:
        raise CalibrationError('unknown plot reference')
    odom_rotation = r @ frame['r_ob'].T
    centre = np.mean(ground @ r.T+t, axis=0)
    correction = np.array(report['pictured_plane_fit']['correction_rotation'])
    camera = frame['t_bc']
    cases = []
    for after in (False, True):
        def transform(p):
            if after:
                p = (p-camera) @ correction.T+camera
            return p @ r.T+t-centre
        n = r @ (correction @ normal if after else normal)
        d = offset
        if after:
            d -= float((correction @ normal) @ (camera-correction @ camera))
        d = d-float(n @ t)+float(n @ centre)
        cases.append(dict(cloud=transform(cloud), ground=transform(ground),
                          surface=transform(surface), normal=n, offset=d,
                          odom_rotation=odom_rotation))
    g = cases[0]['ground']
    gx, gy = np.meshgrid(np.linspace(*np.percentile(g[:, 0], [2, 98]), 7),
                        np.linspace(*np.percentile(g[:, 1], [2, 98]), 7))
    # Recorded RViz z=0 or the explicitly labelled predicted level-start z=0.
    grid = np.stack((gx, gy, np.full_like(gx, -centre[2])), axis=-1)
    limits = np.vstack([grid.reshape(-1, 3)]+[np.percentile(c['ground'], [1, 99], axis=0) for c in cases])
    middle = (limits.min(axis=0)+limits.max(axis=0))/2
    half = .58*max(float(np.ptp(limits, axis=0).max()), .5)
    return cases, grid, middle, half, centre


def draw_view(ax, case, grid, middle, half, view):
    cloud, ground, surface = case['cloud'], case['ground'], case['surface']
    if view == 0:
        ax.set_proj_type('ortho')
        ax.scatter(*cloud.T, s=2, c='#9da8b3', alpha=.35, depthshade=False)
        ax.scatter(*ground.T, s=6, c='#277454', alpha=.8, depthshade=False)
        ax.plot_surface(*[surface[:, :, j] for j in range(3)], color='#e7a04e',
                        alpha=.45, edgecolor='#ab6d25', linewidth=.35, shade=False)
        ax.plot_wireframe(*[grid[:, :, j] for j in range(3)], color='#477dbb', linewidth=.7, alpha=.8)
        ax.set(xlim=(middle[0]-half, middle[0]+half), ylim=(middle[1]-half, middle[1]+half),
               zlim=(middle[2]-half, middle[2]+half), xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)')
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=24, azim=-55)
        ax.tick_params(labelsize=8, pad=1)
        # Direction-only key, not a claim about the odom origin's location.
        anchor = middle + half*np.array([-.65, -.65, .55])
        length = half*.32
        for axis, color in enumerate(('#be3838', '#25864c', '#356bc4')):
            direction = case['odom_rotation'][:, axis]*length
            ax.quiver(*anchor, *direction, color=color, linewidth=1.4,
                      arrow_length_ratio=.18)
            tip = anchor+direction*1.2
            ax.text(*tip, 'XYZ'[axis], color=color, fontsize=7)
        ax.text2D(.05, .96, 'Recorded odom directions', transform=ax.transAxes,
                  color='#555555', fontsize=8)
    else:
        h = 1 if view == 1 else 0
        hidden = 1-h
        ax.scatter(cloud[:, h], cloud[:, 2], s=3, c='#9da8b3', alpha=.4)
        ax.scatter(ground[:, h], ground[:, 2], s=6, c='#277454', alpha=.8)
        horizontal = np.linspace(*np.percentile(ground[:, h], [2, 98]), 100)
        n = case['normal']
        z = -(n[h]*horizontal+n[hidden]*np.median(ground[:, hidden])+case['offset'])/n[2]
        ax.plot(horizontal, z, color='#ce7f19', linewidth=2.2)
        ax.plot(horizontal, np.full_like(horizontal, grid[0, 0, 2]), color='#477dbb', linewidth=1.5)
        ax.set(xlim=(middle[h]-half, middle[h]+half), ylim=(middle[2]-half, middle[2]+half),
               xlabel=('Y (m)' if view == 1 else 'X (m)'), ylabel='Z (m)')
        if view == 2:
            ax.invert_xaxis()  # True view from +Y, +Z upward.
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=.15)
        ax.tick_params(labelsize=8)


def save_plane_plots(folder, report, fits, frames, grid_frame):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    if not fits or 'pictured_plane_fit' not in report:
        return []  # No invented geometry or angles when a plane was not fitted.
    fit = min(fits, key=lambda item: item['rms'])
    frame = frames[fit['frame_index']]
    pictured = report['pictured_plane_fit']
    r, p, y = pictured['rpy_deg']
    before = pictured['before_tilt_to_physical_level_deg']
    after = pictured['after_tilt_to_physical_level_deg']
    # Three-decimal CLI rounding leaves a tiny real residual. Report it, do not
    # rotate/flatten the points again to force a cosmetically perfect zero.
    if not np.isfinite(after):
        raise CalibrationError('invalid predicted floor angle')
    plots = []
    for rows, name, reference in [([0], 'current_floor_view.png', 'recorded'),
                                  ([1], 'predicted_floor_view.png', 'level_start'),
                                  ([0, 1], 'calibration_before_after.png', 'level_start')]:
        cases, grid, middle, half, centre = plot_scene(report, fit, frame, reference)
        multi = len(rows) == 2
        fig = plt.figure(figsize=(14.4, 9.1 if multi else 5.3))
        heading = 'Recorded RViz floor' if reference == 'recorded' else 'VO extrinsics — level-start prediction'
        if report.get('synthetic_test'):
            heading = 'SYNTHETIC TEST — '+heading
        fig.suptitle(heading, fontsize=17, y=.978)
        if reference == 'recorded':
            old = report['current_rpy_deg']
            fig.text(.5, .90, f'Active: Roll {old[0]:+.3f}°   Pitch {old[1]:+.3f}°   Yaw {old[2]:+.3f}°',
                     ha='center', fontsize=13, color='#205d7a')
        else:
            fig.text(.5, .929 if multi else .90,
                     'Current CLI angle ± adjustment = new CLI angle · Yaw and XYZ unchanged',
                     ha='center', fontsize=11, color='#205d7a')
        changes = angle_change_lines(report['current_rpy_deg'], pictured['rpy_deg'])
        for row, case_index in enumerate(rows):
            for view, view_name in enumerate(('Isometric', 'Along +X (Y–Z)', 'Along +Y (X–Z)')):
                ax = fig.add_subplot(len(rows), 3, row*3+view+1, projection='3d' if view == 0 else None)
                draw_view(ax, cases[case_index], grid, middle, half, view)
                stage = ('Recorded floor' if reference == 'recorded' else
                         ('Current floor' if case_index == 0 else 'Predicted after'))
                if view == 0:
                    ax.set_title(stage+' · Isometric', fontsize=11, pad=8)
                else:
                    ax.set_title(stage+' · '+view_name+'\nCLI '+changes[view-1],
                                 fontsize=11, pad=8)
        fig.legend(handles=[Line2D([0], [0], marker='.', linestyle='none', color='#277454', label='Floor points'),
                            Patch(facecolor='#e7a04e', edgecolor='#ab6d25', alpha=.65, label='Fitted plane'),
                            Line2D([0], [0], color='#477dbb',
                                   label='RViz grid' if reference == 'recorded' else 'Predicted level-start grid')],
                   loc='lower center', bbox_to_anchor=(.5, .025), ncol=3, frameon=False, fontsize=10)
        footer = ('Original cloud and grid; unchanged.' if reference == 'recorded' else
                  'Same camera-relative cloud; independent PX4 tilt; level-start model, not a new VO run.')
        footer += ' Nothing applied.'
        fig.text(.5, .012, footer,
                 ha='center', fontsize=8, color='#555555')
        fig.subplots_adjust(left=.045, right=.99, bottom=.115 if multi else .17,
                            top=.83 if multi else .77,
                            hspace=.42, wspace=.22)
        fig.savefig(folder/name, dpi=170)
        plt.close(fig)
        plots.append(name)
    return plots
