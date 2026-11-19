"""No flight-accuracy claims: deterministic geometry, noise, and gate regression tests."""
import math
import random

import pytest

from vio_px4_bridge import frame_alignment as f
from vio_px4_bridge.local_continuity import LocalPoseContinuity
from vio_px4_bridge.stabilized_displacement import StabilizedDisplacement


@pytest.mark.parametrize('radius,tau', [(-1,.5),(.101,.5),(float('nan'),.5),
                                     (.01,-1),(.01,float('inf'))])
def test_invalid_limits(radius, tau):
    with pytest.raises(ValueError):
        StabilizedDisplacement(radius, tau)


@pytest.mark.parametrize('initial', [(0,0,0),(50,-30,2),(1e6,-1e6,0)])
def test_heading_alone_never_moves_old_position(initial):
    s = StabilizedDisplacement()
    s.seed(initial, f.IDENTITY, initial, 0.)
    for i in range(1,1001):
        out, _ = s.update(initial, f.from_rpy(0,0,i*.003), -i*.005, i*.02)
        assert out == initial


@pytest.mark.parametrize('mount', [(0,0,0),(.3,.5,math.pi),(1.2,1.56,-2.8)])
def test_stationary_noise_and_independent_heading_noise_have_zero_horizontal_walk(mount):
    rng = random.Random(718)
    basis = f.from_rpy(*mount)
    s = StabilizedDisplacement(.01,.5)
    origin, target = (50,-30,2), (3,4,0)
    s.seed(origin, basis, target, 0.)
    for i in range(1,5001):
        # Noise is bounded by 5 mm in the leveled horizontal plane. It must
        # not be integrated into drift as heading differences vary.
        perturbation = f.rotate(f.inverse(basis), (rng.uniform(-.005,.005),rng.uniform(-.005,.005),0))
        p = tuple(origin[k]+perturbation[k] for k in range(3))
        q = f.multiply(f.inverse(basis), f.from_rpy(0,0,rng.uniform(-.1,.1)))
        out, _ = s.update(p,q, rng.uniform(-.1,.1), i/90.)
        assert out[:2] == target[:2]
        assert s.held


@pytest.mark.parametrize('hz', [10,30,90,200])
def test_slow_motion_not_lost_to_per_sample_deadband(hz):
    s = StabilizedDisplacement(.01,.5)
    s.seed((0,0,0), f.IDENTITY, (0,0,0), 0)
    for i in range(1,20*hz+1):
        t=i/hz
        out,_=s.update((.001*t,0,0), f.IDENTITY,0,t)
    assert out == pytest.approx((.01,0,0),abs=1e-10)
    assert s.pending_distance_m == pytest.approx(.01)


def test_no_step_at_noise_band_boundary_and_no_velocity_assumption():
    s=StabilizedDisplacement(.01,0)
    s.seed((0,0,0),f.IDENTITY,(0,0,0),0)
    a,_=s.update((0,.009999,0), f.IDENTITY,0,1)
    b,_=s.update((0,.010001,0), f.IDENTITY,0,2)
    assert a==(0,0,0)
    assert b==pytest.approx((0,.000001,0))  # continuous, not a one-cm jump


def test_real_turns_and_yaw_wrap_are_not_smoothed():
    s=StabilizedDisplacement(0,.5)
    s.seed((2,0,0),f.IDENTITY,(2,0,0),0)
    for i in range(1,2001):
        angle=math.tau*i/2000
        p=(2*math.cos(angle),2*math.sin(angle),0)
        yaw=angle+math.pi/2
        out,_=s.update(p,f.from_rpy(0,0,yaw),yaw,i*.02)
        assert out==pytest.approx(p,abs=1e-10)
        assert abs(s.correction)<1e-12


def test_slow_yaw_drift_corrected_without_retroactive_path_rotation():
    s=StabilizedDisplacement(0,.5)
    s.seed((0,0,0),f.IDENTITY,(0,0,0),0)
    raw=[0.,0.,0.]
    for i in range(1,2001):
        yaw=.01*i*.01
        raw[0]+=.01*math.cos(yaw-.00005)
        raw[1]+=.01*math.sin(yaw-.00005)
        out,_=s.update(raw,f.from_rpy(0,0,yaw),0,i*.01)
    assert abs(raw[1])>1.9
    assert abs(out[1])<.11  # finite smoothing lag, not claimed zero error
    assert abs(out[0]-20)<.002
    old=out
    for i in range(1,501):
        out,_=s.update(raw,f.from_rpy(0,0,.2+i*.001),0,20+i*.01)
        assert out==old


def gate(radius=.01):
    return LocalPoseContinuity(max_speed_mps=15,max_acceleration_mps2=40,
                               confirmation_samples=3,recovery_samples=4,
                               stabilizer=StabilizedDisplacement(radius,.5))


def feed(g,p,t,q=f.IDENTITY,heading=0,**kwargs):
    return g.update(p,f.yaw(q),(0,0,0),0,t,'odom|drone_link',orientation=q,
                    heading_reference=heading,require_heading=True,**kwargs)


def test_unpaired_samples_are_checked_but_displacement_is_not_discarded():
    g=gate(0)
    feed(g,(0,0,0),1)
    assert feed(g,(.1,0,0),1.1,heading=None).waiting_for_heading
    assert feed(g,(.2,0,0),1.2,heading=None).waiting_for_heading
    out=feed(g,(.3,0,0),1.3)
    assert out.position==pytest.approx((.3,0,0))


def test_spike_is_quarantined_before_noise_suppression_even_between_heading_pairs():
    g=gate(.1)
    feed(g,(0,0,0),1)
    spike=feed(g,(10,0,0),1.01,heading=None)
    assert spike.recovering and spike.reason=='position_jump'
    back=feed(g,(0,0,0),1.02)
    assert back.recovering  # absolutely no return-to-old-path shortcut


def test_recovery_reseeds_noise_and_heading_without_integrating_coordinate_jump():
    g=gate()
    feed(g,(0,0,0),1)
    before=feed(g,(1,0,0),1.1).position
    assert feed(g,(50,20,5),1.2,q=f.from_rpy(.3,.4,1)).recovering
    q=f.from_rpy(.3,.4,1)
    for i in range(1,12):
        out=feed(g,(50,20,5),1.2+i*.1,q=q,
                 recovery_displacement=(2,3,0),recovery_orientation=f.IDENTITY)
    assert not out.recovering
    assert out.position==pytest.approx((before[0]+2,before[1]+3,0))
    # A real 10 cm body-forward move after arbitrary 3-D epoch change.
    delta=f.rotate(q,(.1,0,0))
    out=feed(g,tuple((50,20,5)[k]+delta[k] for k in range(3)),2.5,q=q)
    assert out.position==pytest.approx((before[0]+2+.09,before[1]+3,0),abs=1e-9)


def test_recovery_cannot_resume_without_matched_heading():
    g=gate()
    feed(g,(0,0,0),1)
    feed(g,(50,0,0),1.1)
    for i in range(1,20):
        out=feed(g,(50,0,0),1.1+i*.1,heading=None,
                 recovery_displacement=(0,0,0),recovery_orientation=f.IDENTITY)
        assert out.recovering
    out=feed(g,(50,0,0),3.2,recovery_displacement=(0,0,0),recovery_orientation=f.IDENTITY)
    assert not out.recovering and out.position==(0,0,0)
