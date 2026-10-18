"""Time source for the control loop: real (perf_counter) or simulated (tests run 60 s trajectories instantly)."""

from __future__ import annotations

import time
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float: ...

    def sleep_until(self, deadline_s: float) -> None: ...

    def elapse(self, duration_s: float) -> None:
        """Account for time a (mock) device transaction takes. A no-op on the real clock, where it really takes time."""


class RealClock:
    """perf_counter time. `sleep_until` sleeps coarsely, then spins for the last ~1 ms (sleep overshoots 0.1-1 ms)."""

    spin_margin_s = 0.001

    def now(self) -> float:
        return time.perf_counter()

    def sleep_until(self, deadline_s: float) -> None:
        remaining = deadline_s - time.perf_counter()
        if remaining > self.spin_margin_s:
            time.sleep(remaining - self.spin_margin_s)
        while time.perf_counter() < deadline_s:
            pass

    def elapse(self, duration_s: float) -> None:
        return None


class SimClock:
    """Virtual time: `sleep_until` jumps forward, `elapse` advances by a transaction's duration."""

    def __init__(self, start_s: float = 1000.0) -> None:
        self._now = start_s

    def now(self) -> float:
        return self._now

    def sleep_until(self, deadline_s: float) -> None:
        self._now = max(self._now, deadline_s)

    def elapse(self, duration_s: float) -> None:
        self._now += duration_s


if __name__ == "__main__":
    real, sim = RealClock(), SimClock()
    start = real.now()
    real.sleep_until(start + 0.01)
    assert 0.0099 < real.now() - start < 0.012, "RealClock.sleep_until should land within ~2 ms"
    sim.sleep_until(1001.0)
    sim.elapse(0.5)
    assert sim.now() == 1001.5
    print("clock sanity checks passed")
