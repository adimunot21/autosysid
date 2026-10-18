"""Servo ticks <-> MJCF joint radians, and per-joint position limits, from a LeRobot calibration file.

Contract (docs/data-contract.md section 1; lerobot `set_half_turn_homings`): LeRobot writes each servo's `Homing_Offset`
so that the calibration pose reads exactly half a turn (2047 ticks), and `Present_Position` / `Goal_Position` are
already in
that frame. The SO-ARM100 MJCF defines its zero pose as that same calibration pose, hence
`rad = sign * (ticks - 2047) * 2*pi/4096`. This is an ASSUMPTION until checked on hardware with a hand-posed arm
(Phase 6);
`zero_ticks` is a field precisely so a measured value can replace it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from autosysid.config import JOINT_NAMES, SERVO_IDS
from autosysid.errors import ConfigError

TICKS_PER_REVOLUTION = 4096  # STS3215, 12-bit
RAD_PER_TICK = 2 * math.pi / TICKS_PER_REVOLUTION
HALF_TURN_TICKS = 2047  # what LeRobot's homing makes the calibration pose read


@dataclass(frozen=True)
class JointCalibration:
    name: str
    servo_id: int
    drive_mode: int  # 1 reverses the direction (LeRobot convention)
    homing_offset: int  # informational: already applied by the servo, not by us
    range_min: int  # recorded limits in the homed frame (ticks)
    range_max: int
    zero_ticks: float = float(HALF_TURN_TICKS)

    def __post_init__(self) -> None:
        if self.drive_mode not in (0, 1):
            raise ConfigError(f"{self.name}: drive_mode must be 0 or 1, got {self.drive_mode}")
        if not 0 <= self.range_min < self.range_max <= TICKS_PER_REVOLUTION - 1:
            raise ConfigError(f"{self.name}: bad range [{self.range_min}, {self.range_max}]")

    @property
    def sign(self) -> float:
        return -1.0 if self.drive_mode else 1.0

    def ticks_to_rad(self, ticks: np.ndarray | float) -> np.ndarray | float:
        return self.sign * (np.asarray(ticks, dtype=float) - self.zero_ticks) * RAD_PER_TICK

    def rad_to_ticks(self, radians: np.ndarray | float) -> np.ndarray:
        ticks = self.zero_ticks + self.sign * np.asarray(radians, dtype=float) / RAD_PER_TICK
        return np.clip(np.rint(ticks), 0, TICKS_PER_REVOLUTION - 1).astype(int)

    def range_rad(self) -> tuple[float, float]:
        low, high = sorted(float(self.ticks_to_rad(t)) for t in (self.range_min, self.range_max))
        return low, high


class ArmCalibration:
    """Calibration of every joint on the bus, in bus order."""

    def __init__(self, joints: list[JointCalibration]) -> None:
        if not joints or len({j.name for j in joints}) != len(joints):
            raise ConfigError("calibration needs at least one joint and unique names")
        self.joints = {joint.name: joint for joint in joints}
        self.names: tuple[str, ...] = tuple(self.joints)

    @classmethod
    def from_lerobot_json(cls, path: str | Path, names: tuple[str, ...] = JOINT_NAMES) -> ArmCalibration:
        path = Path(path)
        try:
            raw = json.loads(path.read_text())
            joints = [
                JointCalibration(
                    name=name,
                    servo_id=int(raw[name]["id"]),
                    drive_mode=int(raw[name]["drive_mode"]),
                    homing_offset=int(raw[name]["homing_offset"]),
                    range_min=int(raw[name]["range_min"]),
                    range_max=int(raw[name]["range_max"]),
                )
                for name in names
            ]
        except (OSError, KeyError, ValueError, TypeError) as error:
            raise ConfigError(f"cannot read LeRobot calibration {path}: {error!r}") from error
        for joint in joints:
            if SERVO_IDS.get(joint.name) not in (None, joint.servo_id):
                raise ConfigError(f"{joint.name}: calibration id {joint.servo_id} != expected {SERVO_IDS[joint.name]}")
        return cls(joints)

    @classmethod
    def symmetric(cls, names: tuple[str, ...], half_range_ticks: int = 1300) -> ArmCalibration:
        """Synthetic calibration for tests/mock: every joint centred on the half turn with equal travel each side."""
        return cls(
            [
                JointCalibration(n, i + 1, 0, 0, HALF_TURN_TICKS - half_range_ticks, HALF_TURN_TICKS + half_range_ticks)
                for i, n in enumerate(names)
            ]
        )

    def ticks_to_rad(self, ticks: np.ndarray, names: tuple[str, ...] | None = None) -> np.ndarray:
        names = names or self.names
        return np.array([self.joints[n].ticks_to_rad(t) for n, t in zip(names, ticks, strict=True)])

    def rad_to_ticks(self, radians: np.ndarray, names: tuple[str, ...] | None = None) -> np.ndarray:
        names = names or self.names
        return np.array([self.joints[n].rad_to_ticks(r) for n, r in zip(names, radians, strict=True)], dtype=int)

    def soft_limits_rad(self, fraction: float) -> tuple[np.ndarray, np.ndarray]:
        """Lower/upper bounds: the central `fraction` of each joint's calibrated range."""
        lower, upper = [], []
        for joint in self.joints.values():
            low, high = joint.range_rad()
            centre, half = 0.5 * (low + high), 0.5 * (high - low) * fraction
            lower.append(centre - half)
            upper.append(centre + half)
        return np.array(lower), np.array(upper)

    def hard_limits_rad(self) -> tuple[np.ndarray, np.ndarray]:
        return self.soft_limits_rad(1.0)


if __name__ == "__main__":
    calibration = ArmCalibration.symmetric(("a", "b"))
    ticks = np.array([2047, 3000])
    assert np.array_equal(calibration.rad_to_ticks(calibration.ticks_to_rad(ticks)), ticks), "round trip failed"
    low, high = calibration.soft_limits_rad(0.8)
    assert np.all(low < 0) and np.all(high > 0) and np.allclose(high, -low)
    print(f"1 tick = {RAD_PER_TICK * 1e3:.3f} mrad; soft limits {low[0]:.3f}..{high[0]:.3f} rad: ok")
