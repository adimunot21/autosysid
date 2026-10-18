"""Single source of truth for every tunable value.

All defaults carry a justification. Values marked "bring-up" are deliberately conservative guesses to be tightened
(or confirmed) during hardware bring-up, not measured facts. Configs are frozen, validated on construction, and are
saved next to every run (`AutoSysIDConfig.save`) so a result can always be traced to the settings that produced it.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from autosysid.errors import ConfigError

# Order = servo IDs 1..6 on the SO-101 bus (docs/data-contract.md) and the actuator names in the SO-ARM100 MJCF.
JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
ARM_JOINT_NAMES: tuple[str, ...] = JOINT_NAMES[:5]  # v0.1 scope: gripper excluded (plastic fingers, least informative)
SERVO_IDS: dict[str, int] = {name: index + 1 for index, name in enumerate(JOINT_NAMES)}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


@dataclass(frozen=True)
class SafetyLimits:
    """Hard limits enforced on every command and every state read (hardware/safety.py)."""

    soft_limit_fraction: float = 0.8  # use the central 80% of the calibrated range; keeps clear of mechanical stops
    max_goal_rate_rad_s: float = 2.0  # bring-up: ~115 deg/s cap on commanded velocity
    # bring-up. NOT a shaping limit: armature is only visible when the goal's speed changes faster than the servo can
    # accelerate (Tmax/J ~ 150 rad/s^2), so this must stay high enough for a 2 rad/s change in one tick (Phase 4).
    # Safety comes from the rate cap, soft limits, torque limit and the tracking watchdog; this is a net against bugs.
    max_goal_accel_rad_s2: float = 400.0  # = 2 x rate cap / 10 ms tick: an instant 0 -> 2 rad/s change passes
    # fraction of the 1000-count Torque_Limit; bring-up. 0.35 left a ~1.2 N.m clamp that saturated even half-amplitude
    # chirps (Phase 4), which makes the fit blind to kp/kv; 0.7 keeps identification workable and the arm is light.
    torque_limit_fraction: float = 0.7
    max_temperature_c: int = 55  # bring-up: servos idle at ~23-28 C; abort well before thermal protection
    min_voltage_v: float = 10.5  # supply reads 12.0 V; a sag below this means the PSU is struggling
    max_tracking_error_rad: float = 0.5  # |goal - present| beyond this means stall/collision or a runaway
    max_loop_jitter_s: float = 0.02  # a gap this large (2 periods at 100 Hz) means the control loop stalled
    comm_timeout_s: float = 0.1  # consecutive failed reads for this long => safe stop
    safe_stop_rate_rad_s: float = 0.3  # bring-up: slow return to the start pose after a trip (~17 deg/s)

    def __post_init__(self) -> None:
        _require(0.0 < self.soft_limit_fraction <= 1.0, "soft_limit_fraction must be in (0, 1]")
        _require(self.max_goal_rate_rad_s > 0, "max_goal_rate_rad_s must be > 0")
        _require(self.max_goal_accel_rad_s2 > 0, "max_goal_accel_rad_s2 must be > 0")
        _require(0.0 < self.torque_limit_fraction <= 1.0, "torque_limit_fraction must be in (0, 1]")
        _require(self.max_temperature_c > 0, "max_temperature_c must be > 0")
        _require(self.min_voltage_v > 0, "min_voltage_v must be > 0")
        _require(self.max_tracking_error_rad > 0, "max_tracking_error_rad must be > 0")
        _require(self.max_loop_jitter_s > 0, "max_loop_jitter_s must be > 0")
        _require(self.comm_timeout_s > 0, "comm_timeout_s must be > 0")
        _require(0.0 < self.safe_stop_rate_rad_s <= self.max_goal_rate_rad_s, "safe_stop_rate_rad_s out of range")


@dataclass(frozen=True)
class RecordConfig:
    """How data is collected from the arm."""

    port: str = "/dev/so101-follower"  # stable udev alias for the follower arm's CH340 board
    baudrate: int = 1_000_000  # STS3215 factory/LeRobot default; Baud_Rate register reads 0
    control_rate_hz: float = 100.0  # sync-read is 1.3 ms p99 (measured), so 100 Hz uses ~13% of the period
    joints: tuple[str, ...] = ARM_JOINT_NAMES
    amplitude_scale: float = 1.0  # global multiplier on all excitation amplitudes; use < 1 for first bring-up runs
    safety: SafetyLimits = field(default_factory=SafetyLimits)
    seed: int = 0  # multisine phases; fixed so runs are repeatable

    def __post_init__(self) -> None:
        _require(self.baudrate > 0, "baudrate must be > 0")
        _require(10.0 <= self.control_rate_hz <= 500.0, "control_rate_hz must be in [10, 500] (sync-read ~1.3 ms)")
        _require(len(self.joints) > 0, "joints must not be empty")
        unknown = [name for name in self.joints if name not in JOINT_NAMES]
        _require(not unknown, f"unknown joints {unknown}; valid: {list(JOINT_NAMES)}")
        _require(0.0 < self.amplitude_scale <= 1.0, "amplitude_scale must be in (0, 1]")


@dataclass(frozen=True)
class FitConfig:
    """How the logged data is fitted with mujoco.sysid."""

    optimizer: Literal["mujoco"] = "mujoco"  # scipy backends failed to converge in the Phase 0 smoke test
    max_iters: int = 60  # converged runs in the smoke test needed 14-27 iterations
    bound_factor: float = 10.0  # multiplier bounds [1/10, 10] x nominal; wide enough to be uninformative priors
    max_delay_s: float = 0.1  # latency upper bound; also the span skipped at the start of every record
    init_delay_s: float = 0.01  # optimizer start for latency
    max_restarts: int = 2  # extra fits from perturbed starts; all are run and the lowest cost wins (local minima exist)
    noise_floor_margin: float = 5.0  # a fit is accepted if cost <= margin x expected quantization+noise floor
    prior_sigma_factor: float = 3.0  # log-normal prior on each dynamics multiplier, 1 sigma = x3 (0 disables)
    max_relative_ci: float = 0.5  # a dynamics multiplier whose 95% half-width exceeds this is "not identified"
    max_delay_ci_s: float = 0.005  # same for latency: 5 ms is half a control period at 100 Hz
    staged: bool = True  # fit delay/armature/damping first, then everything (see optimizer.fit_joint)
    encoder_ticks: int = 4096  # STS3215 12-bit position
    disable_contacts: bool = True  # free-air identification; contacts only cost time
    seed: int = 0  # restart perturbations

    def __post_init__(self) -> None:
        _require(self.max_iters > 0, "max_iters must be > 0")
        _require(self.bound_factor > 1.0, "bound_factor must be > 1")
        _require(0.0 < self.init_delay_s < self.max_delay_s, "need 0 < init_delay_s < max_delay_s")
        _require(self.max_restarts >= 0, "max_restarts must be >= 0")
        _require(self.noise_floor_margin >= 1.0, "noise_floor_margin must be >= 1")
        _require(self.prior_sigma_factor == 0 or self.prior_sigma_factor > 1, "prior_sigma_factor must be 0 or > 1")
        _require(self.max_relative_ci > 0 and self.max_delay_ci_s > 0, "CI thresholds must be > 0")
        _require(self.encoder_ticks > 0, "encoder_ticks must be > 0")


@dataclass(frozen=True)
class AutoSysIDConfig:
    record: RecordConfig = field(default_factory=RecordConfig)
    fit: FitConfig = field(default_factory=FitConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> AutoSysIDConfig:
        try:
            record_raw = dict(raw.get("record", {}))
            safety = SafetyLimits(**record_raw.pop("safety", {}))
            if "joints" in record_raw:
                record_raw["joints"] = tuple(record_raw["joints"])
            return cls(record=RecordConfig(safety=safety, **record_raw), fit=FitConfig(**raw.get("fit", {})))
        except TypeError as error:  # unknown / misspelled key
            raise ConfigError(f"invalid config: {error}") from error

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> AutoSysIDConfig:
        try:
            return cls.from_dict(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError) as error:
            raise ConfigError(f"cannot read config {path}: {error}") from error


if __name__ == "__main__":
    config = AutoSysIDConfig()
    restored = AutoSysIDConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert restored == config, "config must round-trip through JSON"
    print(json.dumps(config.to_dict(), indent=2))
    print("config sanity checks passed")
