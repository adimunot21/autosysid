"""Whole pipeline minus the arm: plan -> pre-flight -> safety session -> recorder -> MockBus (physics, delay,
quantization, wake-up jitter) -> RawLog -> fit_joint -> held-out validation.

The criterion is prediction of UNSEEN motion, not parameter equality: kp/kv/friction barely affect a servo that is
saturated or sub-tick accurate most of the time (Phase 4 finding), so their values may legitimately differ from the
plant's while the model still predicts well. Individual parameters are asserted only where the physics identifies them.
"""

import numpy as np
import pytest

from autosysid.config import FitConfig, RecordConfig, SafetyLimits
from autosysid.hardware.bus import MockBus
from autosysid.hardware.calibration import ArmCalibration
from autosysid.hardware.clock import SimClock
from autosysid.hardware.recorder import Recorder
from autosysid.hardware.trajectories import build_plan
from autosysid.mock import MINI_ARM_XML
from autosysid.optimizer import fit_joint, validate_joint
from conftest import JOINT, TRUE_DELAY_S

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex")


class JitteryClock(SimClock):
    """Wakes up 0-0.8 ms late, like time.sleep + spin under a busy desktop."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self._rng = np.random.default_rng(seed)

    def sleep_until(self, deadline_s: float) -> None:
        super().sleep_until(deadline_s + abs(self._rng.normal(0.0, 0.0004)))


@pytest.mark.slow
def test_recorded_logs_give_a_model_that_predicts_held_out_motion(truth):
    calibration = ArmCalibration.symmetric(JOINTS)
    clock = JitteryClock()
    bus = MockBus(calibration, clock, truth={JOINT: truth}, delays_s={JOINT: TRUE_DELAY_S}, seed=1)
    # 0.7 of full torque: at the 0.35 bring-up value this plant is saturated most of the time (see test_trajectories)
    config = RecordConfig(joints=(JOINT,), amplitude_scale=0.8, safety=SafetyLimits(torque_limit_fraction=0.7))
    plan = build_plan((JOINT,), JOINTS, config.control_rate_hz, config.amplitude_scale, config.safety, stages=(2, 3))
    logs = Recorder(bus, calibration, config, clock).run(plan, xml_path=MINI_ARM_XML)
    fit_logs = [log for log in logs if log.meta["stage"] == "stage2"]
    held_out = [log for log in logs if log.meta["role"] == "holdout"]
    assert len(fit_logs) == 3 and len(held_out) == 1

    result = fit_joint(MINI_ARM_XML, fit_logs, JOINT, FitConfig(max_restarts=2))
    validation = validate_joint(MINI_ARM_XML, result, held_out)

    print(
        f"\ne2e: cost {result.cost / result.noise_floor:.2f}x floor, delay {result.delay_s * 1e3:.2f} ms, "
        f"held-out rms {validation.rms_fitted_rad * 1e3:.2f} mrad "
        f"(nominal {validation.rms_nominal_rad * 1e3:.1f} mrad), "
        f"unidentified {result.unidentifiable}, dynamics/truth "
        + str({n: round(getattr(result.dynamics, n) / getattr(truth, n), 2) for n in truth.to_dict()})
    )
    assert result.accepted, f"cost {result.cost:.2e} vs noise floor {result.noise_floor:.2e}"
    assert abs(result.delay_s - TRUE_DELAY_S) < 0.0015, result.delay_s
    assert validation.rms_fitted_rad < 0.0025, f"{validation.rms_fitted_rad * 1e3:.2f} mrad on unseen motion"
    assert validation.improvement > 4, f"only {validation.improvement:.1f}x better than the nominal model"
    # inertia and viscous damping set the saturated response, so they must be right wherever they are reported
    for name in ("damping", "armature"):
        if name not in result.unidentifiable:
            assert getattr(result.dynamics, name) == pytest.approx(getattr(truth, name), rel=0.25), name
