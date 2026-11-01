"""Behavioral model of one STS3215 joint in position mode, and its fitter.

Why this exists: the servo's internal P-loop hides J, b and friction torque, so physical sysid is not identifiable from
position logs (docs/POSTMORTEM.md). What *is* identifiable is the closed-loop behavior:

    tau * dq/dt + q = Deadband_d( q_cmd(t - Td) + g(q) )

    Td   transport delay (s)              bus write -> servo acts on it
    tau  first-order lag time constant (s)
    d    deadband half-width (rad)        modelled as a "play" operator (hysteresis), not a dead zone around zero
         optionally load dependent:       d(q) = d0 + d1 * |cos(q - q_horiz)|   (stiction grows with gravity load;
         shoulder_lift fitted d = 3.9 / 14 / 19 ticks at |cos| = 0.04 / 0.82 / 0.99)
    g(q) command offset (rad) from gravity/steady-state error:  g0                 (horizontal joints)
                                                               g0 * cos(q - q_horiz) (gravity-loaded joints)

Order of the blocks is delay -> (command + g) -> play -> lag. `BehaviorFilter` is the single stateful implementation;
`simulate` (pure model on a recorded time grid) and `STS3215SimWrapper` (inside MuJoCo) both use it, so what is fitted
is exactly what is simulated.

Units: seconds and radians throughout; `rms_error_ticks` converts to encoder ticks for reporting only.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from autosysid.data import RawLog, zero_order_hold
from autosysid.errors import ConfigError, DataError
from autosysid.hardware.calibration import RAD_PER_TICK

__all__ = [
    "RAD_PER_TICK",
    "BehaviorFilter",
    "BehaviorFit",
    "BehaviorParams",
    "fit_behavior",
    "rms_error_ticks",
    "simulate",
]

LOGGER = logging.getLogger("autosysid.behavior")

CONFIG = {
    # 5 ms: the lag is discretised exactly (valid for any dt), the 10 ms log ticks fall on the grid, and a 12 s log
    # is only 2400 steps, which keeps a multi-start fit to seconds. Finer grids change fitted delay by < 0.5 ms.
    "sim_dt_s": 0.005,
    # Bounds are wide enough to contain every value seen on hardware (Td 10-30 ms, tau 50-70 ms, d 3.5-6 ticks)
    # so a fit that sits on a bound is a signal, not a constraint.
    "bounds": {
        "delay_s": (0.0, 0.08),
        "tau_s": (0.005, 0.3),
        "deadband_rad": (0.0, 0.05),
        "gravity_rad": (-0.1, 0.1),
        "deadband_gravity_rad": (0.0, 0.05),
    },
    # Parameter scales for the optimizer: roughly the size of a typical value, so finite-difference steps are
    # comparable across parameters.
    "scales": {
        "delay_s": 0.01,
        "tau_s": 0.05,
        "deadband_rad": 0.005,
        "gravity_rad": 0.005,
        "deadband_gravity_rad": 0.005,
    },
    # The deadband makes the cost piecewise smooth; a 5% relative step crosses the quantisation of the data instead of
    # differentiating noise.
    "diff_step": 0.05,
    # Multi-start grid. Delay and tau trade off, so start from several (Td, tau) pairs.
    "starts": [(0.005, 0.04), (0.015, 0.06), (0.030, 0.08)],
    "max_nfev": 60,
}


@dataclass(frozen=True)
class BehaviorParams:
    """Per-joint behavioral parameters. `q_horiz_rad is None` means a constant gravity offset `gravity_rad`."""

    delay_s: float
    tau_s: float
    deadband_rad: float
    gravity_rad: float = 0.0
    q_horiz_rad: float | None = None
    deadband_gravity_rad: float = 0.0  # d1: extra deadband at full gravity load; needs q_horiz_rad

    def __post_init__(self) -> None:
        for name in ("delay_s", "tau_s", "deadband_rad", "gravity_rad", "deadband_gravity_rad"):
            if not math.isfinite(getattr(self, name)):
                raise ConfigError(f"{name} must be finite, got {getattr(self, name)}")
        if self.q_horiz_rad is not None and not math.isfinite(self.q_horiz_rad):
            raise ConfigError(f"q_horiz_rad must be finite or None, got {self.q_horiz_rad}")
        if self.delay_s < 0:
            raise ConfigError(f"delay_s must be >= 0, got {self.delay_s}")
        if self.tau_s <= 0:
            raise ConfigError(f"tau_s must be > 0, got {self.tau_s}")
        if self.deadband_rad < 0:
            raise ConfigError(f"deadband_rad must be >= 0, got {self.deadband_rad}")
        if self.deadband_gravity_rad < 0:
            raise ConfigError(f"deadband_gravity_rad must be >= 0, got {self.deadband_gravity_rad}")
        if self.deadband_gravity_rad != 0 and self.q_horiz_rad is None:
            raise ConfigError("deadband_gravity_rad needs q_horiz_rad (the load angle it depends on)")

    def gravity_offset(self, q: float) -> float:
        """Command offset g(q) at joint angle `q` (rad)."""
        if self.q_horiz_rad is None:
            return self.gravity_rad
        return self.gravity_rad * math.cos(q - self.q_horiz_rad)

    def deadband_at(self, q: float) -> float:
        """Deadband half-width (rad) at joint angle `q`: d0 + d1*|cos(q - q_horiz)|."""
        if self.deadband_gravity_rad == 0.0 or self.q_horiz_rad is None:
            return self.deadband_rad
        return self.deadband_rad + self.deadband_gravity_rad * abs(math.cos(q - self.q_horiz_rad))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> BehaviorParams:
        try:
            return cls(**json.loads(Path(path).read_text()))
        except (OSError, ValueError, TypeError) as error:
            raise ConfigError(f"cannot read behavior parameters {path}: {error}") from error


class BehaviorFilter:
    """Stateful discrete implementation of the model. Call `step(command)` once per `dt` seconds.

    `step` returns the lagged position `s`, i.e. the model's prediction of the joint angle. Inside MuJoCo, `s` is
    written to the (stiff) position actuator and the true `qpos` is passed back as `q` so gravity sees the real
    configuration; standalone, the filter feeds its own state back.
    """

    def __init__(self, params: BehaviorParams, dt: float, q0: float, initial_command: float | None = None) -> None:
        if not (math.isfinite(dt) and dt > 0):
            raise ConfigError(f"dt must be finite and > 0, got {dt}")
        self.params = params
        self.dt = dt
        self._alpha = 1.0 - math.exp(-dt / params.tau_s)  # exact for a piecewise-constant input
        delay_steps = params.delay_s / dt
        self._delay_whole = int(math.floor(delay_steps))
        self._delay_frac = delay_steps - self._delay_whole
        self._history = [0.0] * (self._delay_whole + 2)  # ring buffer, newest at `_head`
        self._head = 0
        self.reset(q0, initial_command)

    def reset(self, q0: float, initial_command: float | None = None) -> None:
        """Rest state: joint at `q0`, and the command history holding `initial_command`.

        Default history is the command that holds the joint at q0 (`q0 - g(q0)`).
        """
        if not math.isfinite(q0):
            raise DataError(f"q0 must be finite, got {q0}")
        held = q0 - self.params.gravity_offset(q0) if initial_command is None else float(initial_command)
        if not math.isfinite(held):
            raise DataError(f"initial_command must be finite, got {initial_command}")
        self._history = [held] * len(self._history)
        self._head = 0
        self._play = q0  # deadband ("play") output
        self.state = q0  # lagged position, the model output

    def step(self, command: float, q: float | None = None) -> float:
        """Advance one `dt`. `command` is the commanded position (rad); `q` the measured angle for the gravity term."""
        if not math.isfinite(command):
            raise DataError(f"command must be finite, got {command}")
        history = self._history
        size = len(history)
        self._head = (self._head + 1) % size
        history[self._head] = command
        newer = history[(self._head - self._delay_whole) % size]
        older = history[(self._head - self._delay_whole - 1) % size]
        delayed = (1.0 - self._delay_frac) * newer + self._delay_frac * older

        params = self.params
        configuration = self.state if q is None else q
        effective = delayed + params.gravity_offset(configuration)

        band = params.deadband_at(configuration)
        if effective - self._play > band:
            self._play = effective - band
        elif effective - self._play < -band:
            self._play = effective + band
        self.state += self._alpha * (self._play - self.state)
        return self.state


def simulate(
    params: BehaviorParams,
    times: np.ndarray,
    commands: np.ndarray,
    q0: float,
    dt: float | None = None,
    initial_command: float | None = None,
) -> np.ndarray:
    """Predicted joint angle at each log time for a recorded command log.

    Commands are zero-order-held between log ticks (the servo keeps the last goal). The model runs on its own fine
    grid `dt` and is linearly interpolated back to the log timestamps, so jittery stamps do not alias.
    `initial_command` defaults to `commands[0]` (the command had been held before the log began).
    """
    times = np.asarray(times, dtype=float)
    commands = np.asarray(commands, dtype=float)
    if times.ndim != 1 or commands.shape != times.shape or len(times) < 2:
        raise DataError(f"times {times.shape} and commands {commands.shape} must be matching 1-D arrays (>= 2 samples)")
    if not (np.all(np.isfinite(times)) and np.all(np.isfinite(commands))):
        raise DataError("times/commands contain NaN or inf")
    if np.any(np.diff(times) <= 0):
        raise DataError("times must be strictly increasing")
    step_s = CONFIG["sim_dt_s"] if dt is None else dt

    relative = times - times[0]
    n_steps = int(math.ceil(relative[-1] / step_s - 1e-9))
    grid = np.arange(n_steps) * step_s
    grid_commands = zero_order_hold(grid, relative, commands)
    behavior_filter = BehaviorFilter(
        params, dt=step_s, q0=q0, initial_command=float(commands[0]) if initial_command is None else initial_command
    )
    states = np.empty(n_steps + 1)
    states[0] = q0
    for index, command in enumerate(grid_commands.tolist()):
        states[index + 1] = behavior_filter.step(command)
    return np.interp(relative, np.arange(n_steps + 1) * step_s, states)


def rms_error_ticks(predicted_rad: np.ndarray, measured_rad: np.ndarray) -> float:
    """Root-mean-square position error in encoder ticks (the reporting unit used throughout the post-mortem)."""
    predicted = np.asarray(predicted_rad, dtype=float)
    measured = np.asarray(measured_rad, dtype=float)
    if predicted.shape != measured.shape or predicted.size == 0:
        raise DataError(f"shape mismatch or empty: {predicted.shape} vs {measured.shape}")
    return float(np.sqrt(np.mean((predicted - measured) ** 2)) / RAD_PER_TICK)


@dataclass(frozen=True)
class BehaviorFit:
    params: BehaviorParams
    rms_ticks: float  # training RMS over all logs
    n_samples: int
    at_bounds: tuple[str, ...]  # parameters that finished on a bound: treat the fit as unreliable


_FIT_NAMES = ("delay_s", "tau_s", "deadband_rad", "gravity_rad", "deadband_gravity_rad")


def fit_behavior(
    logs: list[RawLog],
    joint: str,
    fit_gravity: bool = True,
    q_horiz_rad: float | None = None,
    fit_deadband_gravity: bool = False,
) -> BehaviorFit:
    """Least-squares fit of Td, tau, d and (optionally) g0 for `joint` over all `logs`.

    Each log starts from its own measured q0 and first command. `q_horiz_rad` is NOT fitted: over the small
    excursions of a sweep it is not separable from g0, so pass the geometry value (None = constant offset).
    `fit_deadband_gravity` adds d1 (load-dependent deadband); it needs `q_horiz_rad`, `fit_gravity`, and logs from
    base poses with different load, otherwise d0 and d1 are not separable.
    """
    if fit_deadband_gravity and (q_horiz_rad is None or not fit_gravity):
        raise ConfigError("fit_deadband_gravity needs q_horiz_rad and fit_gravity=True")
    if not logs:
        raise DataError("no logs to fit")
    series = []
    for log in logs:
        column = log.column(joint)  # DataError if the joint is missing
        series.append((log.times, log.target_rad[:, column], log.measured_rad[:, column]))
    n_samples = sum(len(measured) for _, _, measured in series)

    names = _FIT_NAMES[: 5 if fit_deadband_gravity else 4 if fit_gravity else 3]
    scales = np.array([CONFIG["scales"][name] for name in names])
    lower = np.array([CONFIG["bounds"][name][0] for name in names]) / scales
    upper = np.array([CONFIG["bounds"][name][1] for name in names]) / scales

    def to_params(scaled: np.ndarray) -> BehaviorParams:
        values = dict(zip(names, (scaled * scales).tolist(), strict=True))
        return BehaviorParams(**values, q_horiz_rad=q_horiz_rad if fit_gravity else None)

    def residuals(scaled: np.ndarray) -> np.ndarray:
        params = to_params(scaled)
        return np.concatenate(
            [simulate(params, times, target, q0=measured[0]) - measured for times, target, measured in series]
        )

    best = None
    for delay_start, tau_start in CONFIG["starts"]:
        start = [delay_start, tau_start, 0.004, 0.001, 0.004][: len(names)]
        scaled_start = np.clip(np.array(start) / scales, lower + 1e-6, upper - 1e-6)
        result = least_squares(
            residuals,
            scaled_start,
            bounds=(lower, upper),
            diff_step=CONFIG["diff_step"],
            max_nfev=CONFIG["max_nfev"],
        )
        LOGGER.debug("start (Td=%.3f, tau=%.3f): rms %.3f ticks", delay_start, tau_start, _rms_ticks(result.fun))
        if best is None or result.cost < best.cost:
            best = result
    assert best is not None  # starts is non-empty

    margin = 1e-3
    at_bounds = tuple(
        name
        for name, value, low, high in zip(names, best.x, lower, upper, strict=True)
        if value <= low + margin or value >= high - margin
    )
    if at_bounds:
        LOGGER.warning("fit for %s finished on a bound: %s", joint, ", ".join(at_bounds))
    return BehaviorFit(
        params=to_params(best.x), rms_ticks=_rms_ticks(best.fun), n_samples=n_samples, at_bounds=at_bounds
    )


def _rms_ticks(residual_rad: np.ndarray) -> float:
    return float(np.sqrt(np.mean(residual_rad**2)) / RAD_PER_TICK)


if __name__ == "__main__":
    truth = BehaviorParams(delay_s=0.016, tau_s=0.062, deadband_rad=0.006, gravity_rad=0.004)
    sweep_times = np.arange(0, 12.0, 0.01)
    sweep_commands = 0.12 * np.sin(2 * np.pi * (0.3 + 0.5 * sweep_times / 12.0) * sweep_times)
    sweep_measured = simulate(truth, sweep_times, sweep_commands, q0=0.0)
    synthetic = RawLog(
        times=sweep_times,
        joints=("demo",),
        target_rad=sweep_commands[:, None],
        measured_rad=sweep_measured[:, None],
    )
    recovered = fit_behavior([synthetic], "demo")
    print(f"truth     : {truth}")
    print(f"recovered : {recovered.params}")
    print(f"train RMS : {recovered.rms_ticks:.3f} ticks, at bounds: {recovered.at_bounds}")
    assert recovered.rms_ticks < 0.2, "noise-free synthetic data should be fitted almost exactly"
    print("ok")
