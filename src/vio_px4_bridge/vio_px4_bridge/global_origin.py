"""Bounded, disarmed-only global-reference setup (not height measurement fusion)."""


class GlobalOriginSetup:
    def __init__(self, latitude, longitude, altitude):
        # MAVLink wire units: degrees * 1e7 and millimetres AMSL.
        self.target = (round(latitude * 1e7), round(longitude * 1e7), round(altitude * 1000))
        self.started = None
        self.last_request = None
        self.last_set = None
        self.attempts = 0
        self.first_match = None
        self.last_match = None
        self.confirmed = False
        self.failure = None

    def observe(self, now, latitude, longitude, altitude):
        if self.started is None or self.last_request is None or self.failure:
            return
        matches = (abs(latitude - self.target[0]) <= 2
                   and abs(longitude - self.target[1]) <= 2
                   and abs(altitude - self.target[2]) <= 100)
        if not matches:
            self.first_match = self.last_match = None
            self.confirmed = False
            return
        if self.first_match is None:
            self.first_match = now
        self.last_match = now
        # An immediate response can precede PX4's command processing. Require
        # another matching readback later, not merely a successful send/ACK.
        self.confirmed = now - self.first_match >= 1.0

    def step(self, now, disarmed):
        """Return (set_origin, request_readback); caller must be in startup spoof."""
        if self.started is None:
            self.started = now
        if self.failure:
            return False, False
        if disarmed is False:
            self.failure = "origin setup requires a fresh DISARMED PX4 heartbeat; restart while disarmed"
            return False, False
        if disarmed is None:
            if now - self.started >= 10.0:
                self.failure = "PX4 heartbeat missing/stale during origin setup"
            return False, False
        if self.reason(now) is None:
            # Continue querying through spoof so a later startup reset is seen.
            pass
        elif now - self.started >= 10.0:
            self.failure = "PX4 global origin not verified within 10 seconds; GPS handoff blocked"
            return False, False
        request = self.last_request is None or now - self.last_request >= 1.0
        if request:
            self.last_request = now
        # Query first: do not reset an already correct origin. Allow asynchronous
        # PX4 processing and limit write attempts even if replies never arrive.
        set_origin = (self.first_match is None and now - self.started >= 1.0
                      and self.attempts < 5
                      and (self.last_set is None or now - self.last_set >= 2.0))
        if set_origin:
            self.last_set = now
            self.attempts += 1
        return set_origin, request

    def reason(self, now):
        if self.failure:
            return self.failure
        if not self.confirmed or self.last_match is None or now - self.last_match > 2.5:
            return "waiting for PX4 global-origin readback"
        return None
