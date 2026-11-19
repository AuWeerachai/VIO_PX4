"""Production dispatch/callback/output tests, with no serial or ROS runtime."""
import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vio_px4_bridge import frame_alignment as f
from vio_px4_bridge.geo_utils import ned_offset_to_lla
from vio_px4_bridge.local_continuity import LocalPoseContinuity
from vio_px4_bridge.px4_heading_reference import Px4HeadingReference
from vio_px4_bridge.stabilized_displacement import StabilizedDisplacement
from test_px4_recovery import bridge
from test_full3d_recovery import message
from test_px4_heading_reference import synced


@pytest.fixture
def stabilized(bridge):
    b=bridge
    class PhaseLabel(str):
        @property
        def value(self):
            return str(self)
    b.phase=PhaseLabel('live')
    b.stabilized_heading_enabled=True
    b.inertial_active=b.continuity_recovering=False
    b.have_vio=False
    b.last_vio_time=None
    b.last_heading_output_time=None
    b.heading_reference=Px4HeadingReference()
    b.continuity=LocalPoseContinuity(confirmation_samples=3,recovery_samples=4,
                                   stabilizer=StabilizedDisplacement())
    b.get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(
        nanoseconds=round((1000+b.clock.now)*1e9)))
    b.rc_trigger_enabled=False
    b._mav=SimpleNamespace(mav=Mock(),recv_match=Mock(return_value=None))
    b.last_timesync_request=float('-inf')
    return b


def telemetry(b,t,yaw=0.,reset=0,north=0.,east=0.,vn=0.,ve=0.,source=1,
              capture=None, sync=True, variance=.01):
    b.clock.now=t
    if sync:
        synced(b.heading_reference,t)
    boot=(t if capture is None else capture)-5
    odom=SimpleNamespace(
        get_type=lambda:'ODOMETRY',get_srcSystem=lambda:1,get_srcComponent=lambda:source,
        time_usec=round(boot*1e6),frame_id=1,child_frame_id=1,estimator_type=8,
        quality=0,x=north,y=east,vx=vn,vy=ve,vz=0.,q=f.from_rpy(0,0,yaw),
        pose_covariance=[variance]*21,reset_counter=reset)
    att=SimpleNamespace(
        get_type=lambda:'ATTITUDE',get_srcSystem=lambda:1,get_srcComponent=lambda:source,
        time_boot_ms=round(boot*1000),roll=0.,pitch=0.,yaw=yaw)
    b._mav.recv_match=Mock(side_effect=[att,odom,None])
    b.poll_real()


def camera(b,t,p=(0,0,0),q=f.IDENTITY):
    b.clock.now=t
    b._odom_callback(message(1000+t-.02,p,q,body_twist=False))


def prime(b):
    telemetry(b,9.95,yaw=b.heading_offset_rad)
    telemetry(b,10.,yaw=b.heading_offset_rad)
    camera(b,10.)
    b._tick()
    assert not b.continuity_recovering
    assert b.have_vio
    b._publish_fix.reset_mock()


def test_raw_noise_and_heading_changes_do_not_change_transmitted_gps(stabilized):
    b=stabilized
    prime(b)
    for i in range(1,401):
        t=10+i*.05
        telemetry(b,t,yaw=.04*math.sin(i*.03))
        camera(b,t,(.003*math.sin(i*.7),.003*math.cos(i*.4),0),
               f.from_rpy(0,0,.002*math.sin(i*.5)))
        b._tick()
        assert not b.continuity_recovering
        fix=b._publish_fix.call_args.args[0]
        assert (fix['lat'],fix['lon'])==(b.home_lat,b.home_lon)
        assert (fix['vn'],fix['ve'],fix['vd'])==(0.,0.,0.)
        assert b.pose_jump_count==0


def test_heading_source_is_px4_not_px4_position(stabilized):
    b=stabilized
    b.heading_offset_rad=math.pi/2
    prime(b)
    for i in range(1,41):
        t=10+i*.05
        # Deliberately change FCU position, but no recovery occurs. This must
        # NOT drag the bridge. Only matched heading and VO movement are used.
        telemetry(b,t,yaw=math.pi/2,north=100+i,east=-100-i)
        camera(b,t,(0,0,0))
    initial=dict(b.latest_fix)
    # After discrepancy settles, one metre local forward must point east.
    for i in range(1,101):
        t=12+i*.05
        telemetry(b,t,yaw=math.pi/2,north=200+i,east=-200-i)
        camera(b,t,(i*.01,0,0))
    expected=ned_offset_to_lla(b.home_lat,b.home_lon,b.home_alt,0,.99,0)
    assert b.latest_fix['lat']==pytest.approx(expected[0],abs=2e-8)
    assert b.latest_fix['lon']==pytest.approx(expected[1],abs=2e-8)
    assert initial['lat']==b.home_lat and initial['lon']==b.home_lon
    assert not b.continuity_recovering


def test_stale_reference_stops_output_then_recovers_without_restart(stabilized):
    b=stabilized
    prime(b)
    b.clock.now=10.4
    b._tick()
    assert b.continuity_recovering
    b._publish_fix.assert_not_called()
    for i in range(30):
        t=10.45+i*.05
        telemetry(b,t)
        camera(b,t)
        b._tick()
    assert not b.continuity_recovering
    assert b._publish_fix.called


def test_bad_camera_sample_between_matched_samples_cannot_leak(stabilized):
    b=stabilized
    prime(b)
    camera(b,10.01,(100,0,0))
    b._tick()
    assert b.continuity_recovering
    b._publish_fix.assert_not_called()
    assert b.pose_jump_count==1
    for i in range(1,25):
        t=10.01+i*.05
        telemetry(b,t)
        camera(b,t,(100,0,0))
        b._tick()
    assert not b.continuity_recovering
    assert b.latest_fix['lat']==b.home_lat
    assert b.latest_fix['lon']==b.home_lon


def test_px4_reset_stops_output_even_if_vo_is_stationary(stabilized):
    b=stabilized
    prime(b)
    telemetry(b,10.05,reset=1,yaw=.5)
    b._tick()
    assert b.continuity_recovering
    b._publish_fix.assert_not_called()
    for i in range(1,30):
        t=10.05+i*.05
        telemetry(b,t,reset=1,yaw=.5)
        camera(b,t)
        b._tick()
    assert not b.continuity_recovering
    assert b.latest_fix['lat']==b.home_lat
    assert b.latest_fix['lon']==b.home_lon


def test_other_component_cannot_change_heading_or_trigger_reset(stabilized):
    b=stabilized
    prime(b)
    count=len(b.heading_reference.history)
    telemetry(b,10.05,yaw=1.,reset=3,source=191)
    assert len(b.heading_reference.history)==count
    assert not b.continuity_recovering


def test_unknown_timesync_token_does_not_replace_clock(stabilized):
    b=stabilized
    prime(b)
    msg=SimpleNamespace(get_type=lambda:'TIMESYNC',get_srcSystem=lambda:1,
                        get_srcComponent=lambda:1,tc1=int(3e9),ts1=123)
    b._mav.recv_match=Mock(side_effect=[msg,None])
    b.poll_real()
    assert b.heading_reference.clock.offset==pytest.approx(5)
    assert not b.continuity_recovering


def test_normal_heading_pairing_wait_does_not_refresh_output_or_quarantine(stabilized):
    b=stabilized
    prime(b)
    accepted=b.last_vio_time
    camera(b,10.04,(.02,0,0))  # capture is beyond last FCU sample; no extrapolation
    assert b.last_vio_time==accepted
    assert not b.continuity_recovering
    telemetry(b,10.05)
    camera(b,10.05,(.03,0,0))
    assert b.last_vio_time>accepted
    assert b.continuity.last_output_position[0]==pytest.approx(.02)


@pytest.mark.parametrize('camera_latency', [.005, .03, .1, .2])
def test_90hz_camera_20hz_fc_unequal_latency_no_false_quarantine(stabilized, camera_latency):
    b = stabilized
    prime(b)
    events = []
    for i in range(1, 901):
        capture = 10+i/90
        latency = camera_latency*(1+.05*math.sin(i*.33))
        events.append((capture+latency, 'camera', capture))
    for i in range(1, 206):
        capture = 10+i/20
        events.append((capture+.008+.002*math.sin(i*.37), 'px4', capture))
    for i in range(1, 103):
        events.append((10+i/10, 'gps', None))
    last_sync = 10.
    output_stamps = []
    for now, kind, capture in sorted(events):
        b.clock.now = now
        if now-last_sync >= .5:
            synced(b.heading_reference, now)
            last_sync = now
        if kind == 'px4':
            telemetry(b, now, capture=capture, sync=False,
                      yaw=.01*math.sin(capture))
            b._drain_pending_odometry()
        elif kind == 'camera':
            p = (.002*math.sin(capture*2), .002*math.cos(capture*3), 0.)
            b._odom_callback(message(1000+capture, p, f.IDENTITY))
        else:
            b._tick()
            if b._publish_fix.called:
                output_stamps.append(b.last_vio_time)
                fix = b._publish_fix.call_args.args[0]
                assert (fix['lat'], fix['lon']) == (b.home_lat, b.home_lon)
        assert not b.continuity_recovering
        assert len(b.pending_odometry) < 30
    assert b.pose_jump_count == 0
    assert len(set(output_stamps)) > 90
    assert not b.pending_odometry


def test_pending_jump_is_not_replaced_by_good_sample(stabilized):
    b = stabilized
    prime(b)
    camera(b, 10.04, (100, 0, 0))
    camera(b, 10.045, (0, 0, 0))
    assert len(b.pending_odometry) == 2
    telemetry(b, 10.05)
    b._tick()
    assert b.continuity_recovering
    assert b.pose_jump_count == 1
    b._publish_fix.assert_not_called()


def test_negative_fc_covariance_and_invalid_body_frame_stop_output(stabilized):
    b = stabilized
    prime(b)
    telemetry(b, 10.05, variance=-.01)
    b._tick()
    assert b.continuity_recovering
    b._publish_fix.assert_not_called()


def test_wrong_body_frame_cannot_use_last_good_gps(stabilized):
    b = stabilized
    prime(b)
    msg = message(1009.99, (0, 0, 0), f.IDENTITY)
    msg.child_frame_id = 'camera_link'
    b._odom_callback(msg)
    b._tick()
    assert b.continuity_recovering
    b._publish_fix.assert_not_called()


@pytest.mark.parametrize('angles', [(2.3, .2, -.9), (0, math.pi/2, 0), (math.pi, 0, 0)])
def test_full3d_reset_then_rectangle_preserves_position(stabilized, angles):
    b = stabilized
    prime(b)
    jump = f.from_rpy(*angles)
    shift = (100, -200, 300)
    for i in range(1, 31):
        t = 10+i*.05
        telemetry(b, t)
        camera(b, t, shift, jump)
        b._tick()
    assert not b.continuity_recovering
    assert b.latest_fix['lat'] == b.home_lat
    assert b.latest_fix['lon'] == b.home_lon
    n = e = 0.
    for vn, ve in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
        for _ in range(20):
            t += .05
            n, e = n+vn*.05, e+ve*.05
            p = tuple(a+c for a, c in zip(shift, f.rotate(jump, (n, e, 0))))
            telemetry(b, t)
            camera(b, t, p, jump)
            assert not b.continuity_recovering
            out = b.continuity.last_output_position
            assert math.hypot(out[0]-n, out[1]-e) < .011
    assert math.hypot(*b.continuity.last_output_position[:2]) < .011
    assert b.pose_jump_count == 1


def test_moving_recovery_uses_capture_boundary_and_no_motion_double_count(stabilized):
    b = stabilized
    prime(b)
    # Position is stationary through t=10; afterwards the FCU travels north.
    # Camera captures are 20 ms older than receipt. The final anchor must use
    # that older capture, not add an extra 20 ms of travel at handoff.
    speed = .1
    resumed = None
    for i in range(1, 31):
        t = 10+i*.05
        telemetry(b, t, north=speed*(t-10), vn=speed)
        capture = t-.02
        p = (100+speed*(capture-10), 0, 0)
        b._odom_callback(message(1000+capture, p, f.IDENTITY,
                                world_velocity=(speed, 0, 0), body_twist=False))
        if i == 1:
            assert b.continuity_recovering
        if not b.continuity_recovering:
            resumed = t
            assert b.continuity.last_output_position[0] == pytest.approx(speed*(capture-10), abs=1e-8)
            break
    assert resumed is not None
    for i in range(1, 21):
        t = resumed+i*.05
        capture = t-.02
        telemetry(b, t, north=speed*(t-10), vn=speed)
        b._odom_callback(message(1000+capture, (100+speed*(capture-10), 0, 0), f.IDENTITY,
                                world_velocity=(speed, 0, 0), body_twist=False))
    assert not b.continuity_recovering
    assert b.continuity.last_output_position[0] == pytest.approx(speed*(capture-10)-.01, abs=1e-8)
