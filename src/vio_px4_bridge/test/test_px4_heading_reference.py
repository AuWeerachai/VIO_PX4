import math

import pytest

from vio_px4_bridge import frame_alignment as f
from vio_px4_bridge.px4_heading_reference import ClockSync, Px4HeadingReference, interpolate


def synced(reference, now=10.):
    c=reference.clock
    for i in range(5):
        sent=now-.1+i*.01
        token=c.request(sent)
        assert c.observe(int((sent-5+.001)*1e9),token,sent+.002)
    assert c.valid(now)
    assert c.offset==pytest.approx(5)


def test_clock_rejects_unrequested_old_or_slow_replies():
    c=ClockSync()
    assert not c.observe(100,99,10)
    token=c.request(10)
    assert not c.observe(int(5e9),token,10.2)
    token=c.request(11)
    assert c.observe(int(6.001e9),token,11.002)
    assert not c.observe(int(6.001e9),token,11.003)
    assert not c.valid(11.01)  # one reply cannot establish synchronization


def test_reboot_invalidates_offset_until_relearned():
    r=Px4HeadingReference()
    synced(r)
    token=r.clock.request(10)
    assert r.clock.observe(int(.001e9),token,10.002)
    assert r.clock.generation==1
    assert not r.clock.valid(10.002)


def test_pair_uses_capture_time_interpolates_wrap_and_does_not_extrapolate():
    r=Px4HeadingReference()
    synced(r)
    r.observe(9.98,4.97,f.from_rpy(0,0,math.radians(179)),0)
    r.observe(10.03,5.02,f.from_rpy(0,0,math.radians(-179)),0)
    result=r.match(1009.995,1010.04,10.04)
    assert abs(result)==pytest.approx(math.pi)
    assert r.match(1010.035,1010.04,10.04) is None
    assert r.reason=='waiting_for_time_matched_heading'


def test_invalid_packets_and_duplicate_timestamps_never_refresh_freshness():
    r=Px4HeadingReference()
    synced(r)
    r.observe(10,5,f.IDENTITY,0)
    r.observe(10.2,5,f.IDENTITY,0)
    assert r.history[-1][0]==10
    assert not r.fresh(10.3)
    assert r.observe(10.31,5.3,(0,0,0,0),0)=='invalid_px4_heading'
    assert not r.history


@pytest.mark.parametrize('stamp,reset,q,reason', [
    (5.05,1,f.IDENTITY,'px4_estimator_reset'),
    (5.0,1,f.IDENTITY,'px4_estimator_reset'),
    (4.,0,f.IDENTITY,'px4_heading_clock_regression'),
    (5.5,0,f.IDENTITY,'px4_heading_message_gap'),
    (5.01,0,f.from_rpy(0,0,1),'px4_attitude_rate_limit'),
])
def test_discontinuity_is_explicit_and_old_pairing_cleared(stamp,reset,q,reason):
    r=Px4HeadingReference()
    synced(r)
    r.observe(10,5,f.IDENTITY,0)
    assert r.observe(10.05,stamp,q,reset)==reason
    assert len(r.history)==1
    assert r.last_match is None


def test_queued_old_telemetry_is_not_fresh_because_it_just_arrived():
    r=Px4HeadingReference()
    synced(r)
    r.observe(10,4,f.IDENTITY,0)
    assert not r.fresh(10)


def test_camera_clock_change_cannot_silently_repair_pairing():
    r=Px4HeadingReference()
    synced(r)
    r.observe(9.98,4.97,f.IDENTITY,0)
    r.observe(10.03,5.02,f.IDENTITY,0)
    assert r.match(1009.995,1010.04,10.04)==0
    assert r.match(1010.995,1011.04,10.04) is None
    assert r.reason=='camera_clock_changed' and not r.history


def test_motion_boundary_is_interpolated_only_for_recovery():
    r=Px4HeadingReference()
    synced(r)
    a=(9.98,4.97,10,20,1,2,.1,.2,0)
    b=(10.03,5.02,10.05,20.10,1,2,.2,.1,0)
    r.observe(a[0],a[1],f.IDENTITY,0,a)
    r.observe(b[0],b[1],f.IDENTITY,0,b)
    out=r.motion_at(9.995,10.04)
    assert out[2:4]==pytest.approx((10.025,20.05))
    assert out[6:8]==(.2,.2)
    assert r.motion_at(10.04,10.04) is None


def test_quaternion_sign_change_is_not_a_turn():
    q=f.from_rpy(.1,.2,3.13)
    assert f.angle_between(interpolate(q,tuple(-v for v in q),.5),q)<1e-12


def test_clock_uncertainty_does_not_manufacture_stale_heading():
    r = Px4HeadingReference()
    synced(r)
    r.observe(10, 5.001, f.IDENTITY, 0)
    assert r.fresh(10)
    r.observe(10, 5.04, f.IDENTITY, 0)
    assert not r.fresh(10)  # beyond the bounded clock uncertainty
