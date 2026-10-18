"""Pure excitation-signal generators (radians, seconds). Shared by the mock arm and hardware/trajectories.

Every generator starts and ends at `center_rad` (tapered), so a trajectory can always be entered and left gently.
"""

from __future__ import annotations

import numpy as np

from autosysid.errors import ConfigError


def _taper(elapsed: np.ndarray, duration: float, taper_s: float) -> np.ndarray:
    if taper_s < 0 or 2 * taper_s > duration:
        raise ConfigError(f"taper_s={taper_s} must be in [0, duration/2={duration / 2}]")
    if taper_s == 0:
        return np.ones_like(elapsed)
    ramp_in = np.clip(elapsed / taper_s, 0.0, 1.0)
    ramp_out = np.clip((duration - elapsed) / taper_s, 0.0, 1.0)
    # raised-cosine ramps: zero slope at both ends so the servo never sees a velocity step
    return 0.5 * (1 - np.cos(np.pi * ramp_in)) * 0.5 * (1 - np.cos(np.pi * ramp_out))


def chirp(
    times: np.ndarray,
    f_start_hz: float,
    f_end_hz: float,
    amplitude_rad: float,
    taper_s: float = 1.0,
    center_rad: float = 0.0,
) -> np.ndarray:
    """Linear-frequency sweep, amplitude-tapered at both ends. Peak deviation from center is `amplitude_rad`."""
    times = np.asarray(times, dtype=float)
    if not (0 < f_start_hz < f_end_hz):
        raise ConfigError("need 0 < f_start_hz < f_end_hz")
    elapsed = times - times[0]
    duration = float(elapsed[-1])
    phase = 2 * np.pi * (f_start_hz * elapsed + 0.5 * (f_end_hz - f_start_hz) / duration * elapsed**2)
    return center_rad + amplitude_rad * _taper(elapsed, duration, taper_s) * np.sin(phase)


def multisine(
    times: np.ndarray,
    frequencies_hz: np.ndarray,
    amplitude_rad: float,
    rng: np.random.Generator,
    taper_s: float = 1.0,
    center_rad: float = 0.0,
) -> np.ndarray:
    """Sum of sinusoids with random phases, normalized so the untapered peak is `amplitude_rad`."""
    times = np.asarray(times, dtype=float)
    frequencies_hz = np.asarray(frequencies_hz, dtype=float)
    if frequencies_hz.ndim != 1 or frequencies_hz.size == 0 or np.any(frequencies_hz <= 0):
        raise ConfigError("frequencies_hz must be a non-empty 1-D array of positive values")
    elapsed = times - times[0]
    phases = rng.uniform(0, 2 * np.pi, frequencies_hz.size)
    signal = np.sin(2 * np.pi * np.outer(elapsed, frequencies_hz) + phases).sum(axis=1)
    signal /= np.max(np.abs(signal))
    return center_rad + amplitude_rad * _taper(elapsed, float(elapsed[-1]), taper_s) * signal


def triangle(times: np.ndarray, amplitude_rad: float, period_s: float, center_rad: float = 0.0) -> np.ndarray:
    """Slow triangle wave starting at the center: constant-velocity ramps for breakaway friction and bias."""
    if period_s <= 0:
        raise ConfigError("period_s must be > 0")
    elapsed = np.asarray(times, dtype=float) - times[0]
    return center_rad + amplitude_rad * (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * elapsed / period_s))


def step_train(
    times: np.ndarray,
    levels_rad: np.ndarray,
    hold_s: float,
    max_rate_rad_s: float,
    center_rad: float = 0.0,
    max_accel_rad_s2: float | None = None,
) -> np.ndarray:
    """Goal moves to each level in turn (held `hold_s`), slew-limited to `max_rate_rad_s`; starts at center.

    The slew limit is the same one the safety layer enforces on hardware, so what the model sees here is what the
    arm will be asked to do. Large levels briefly saturate the actuator, which is what makes kv and armature visible.
    With `max_accel_rad_s2` the profile is also acceleration-limited (trapezoidal velocity, braking before
    each level), so
    it is already admissible for the safety limiter, which would otherwise lag a corner and overshoot the level.
    """
    times = np.asarray(times, dtype=float)
    levels = np.asarray(levels_rad, dtype=float)
    if hold_s <= 0 or max_rate_rad_s <= 0 or levels.size == 0:
        raise ConfigError("need hold_s > 0, max_rate_rad_s > 0 and at least one level")
    if max_accel_rad_s2 is not None and max_accel_rad_s2 <= 0:
        raise ConfigError("max_accel_rad_s2 must be > 0")
    elapsed = times - times[0]
    raw = center_rad + levels[np.minimum((elapsed // hold_s).astype(int), levels.size - 1)]
    goal = np.empty_like(raw)
    goal[0] = center_rad
    velocity = 0.0
    for i in range(1, len(raw)):
        dt = times[i] - times[i - 1]
        error = raw[i] - goal[i - 1]
        if max_accel_rad_s2 is None:
            goal[i] = goal[i - 1] + np.clip(error, -max_rate_rad_s * dt, max_rate_rad_s * dt)
            continue
        braking_speed = np.sqrt(
            2.0 * 0.5 * max_accel_rad_s2 * abs(error)
        )  # half the cap for braking: ticks are discrete
        wanted = np.sign(error) * min(abs(error) / dt, max_rate_rad_s, braking_speed)
        velocity = np.clip(wanted, velocity - max_accel_rad_s2 * dt, velocity + max_accel_rad_s2 * dt)
        goal[i] = goal[i - 1] + velocity * dt
    return goal


# The three-part single-joint recipe (docs/data-contract.md section 4): part -> (stage name, nominal duration in s).
# r = slow ramps (friction, bias), c = unsaturated chirp (delay, tracking),
# s = slew-limited saturating steps (kv, armature).
PROTOCOL_PARTS: dict[str, tuple[str, float]] = {"r": ("ramp", 16.0), "c": ("chirp", 15.0), "s": ("steps", 14.0)}
_STEP_LEVELS_RAD = np.array([0.3, -0.3, 0.3, -0.3, 0.15, -0.15, 0.0])


def protocol_signal(
    part: str,
    times: np.ndarray,
    amplitude_scale: float = 1.0,
    max_rate_rad_s: float = 2.0,
    max_accel_rad_s2: float | None = None,
) -> np.ndarray:
    """Excitation (offset from the center pose, radians) for one part of the recipe.

    Amplitude and step slew rate scale together; `max_accel_rad_s2` makes the steps admissible for the safety
    limiter."""
    if part not in PROTOCOL_PARTS:
        raise ConfigError(f"unknown protocol part '{part}'; valid: {sorted(PROTOCOL_PARTS)}")
    if part == "r":
        return triangle(times, 0.08 * amplitude_scale, 8.0)
    if part == "c":
        return chirp(times, 0.1, 1.5, 0.2 * amplitude_scale, taper_s=1.5)
    return step_train(
        times,
        _STEP_LEVELS_RAD * amplitude_scale,
        2.0,
        max_rate_rad_s * amplitude_scale,
        max_accel_rad_s2=max_accel_rad_s2,
    )


if __name__ == "__main__":
    t = np.arange(0, 12, 0.01)
    y = chirp(t, 0.2, 4.0, 0.35)
    assert abs(y[0]) < 1e-12 and abs(y[-1]) < 1e-9 and np.max(np.abs(y)) <= 0.35 + 1e-12
    m = multisine(t, np.array([0.3, 0.7, 1.9]), 0.2, np.random.default_rng(0))
    assert np.max(np.abs(m)) <= 0.2 + 1e-12
    tri = triangle(t, 0.08, 8.0)
    assert abs(tri[0]) < 1e-12 and np.max(np.abs(tri)) <= 0.08 + 1e-12
    steps = step_train(t, np.array([0.3, -0.3, 0.0]), 3.0, 2.0)
    assert steps[0] == 0.0 and np.max(np.abs(np.diff(steps))) <= 2.0 * 0.01 + 1e-12, "slew limit violated"
    print(f"chirp peak {np.max(np.abs(y)):.3f} rad, multisine peak {np.max(np.abs(m)):.3f} rad: ok")
