"""Held-out evaluation of the competing models (hold-still, stock MJCF, stock + delay, 3/4-parameter behavior)."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

from autosysid.behavior import RAD_PER_TICK, BehaviorParams, simulate
from autosysid.data import RawLog
from autosysid.errors import DataError
from autosysid.evaluation import evaluate_joint, split_logs
from autosysid.mock import MINI_ARM_XML

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

JOINT = "shoulder_pan"
TRUTH = BehaviorParams(delay_s=0.016, tau_s=0.06, deadband_rad=4 * RAD_PER_TICK, gravity_rad=0.004)


def _log(stage: str, seed: int, role: str | None = None) -> RawLog:
    rng = np.random.default_rng(seed)
    times = np.arange(0, 10.0, 0.01)
    target = 0.12 * np.sin(2 * np.pi * (0.3 + 0.4 * times / 10) * times + rng.uniform(0, 1))
    target = np.round(target / RAD_PER_TICK) * RAD_PER_TICK
    measured = simulate(TRUTH, times, target, q0=0.0) + rng.normal(0, 0.0003, len(times))
    meta = {"excited": [JOINT]}
    if role:
        meta["role"] = role
    return RawLog(times, (JOINT,), target[:, None], measured[:, None], stage=stage, meta=meta)


class TestSplit:
    def test_prefers_recorded_holdout_role(self):
        logs = [_log("stage2_ramp", 1), _log("stage2_steps", 2), _log("stage3_multisine", 3, role="holdout")]
        fit, holdout = split_logs(logs, JOINT)
        assert [log.stage for log in holdout] == ["stage3_multisine"]
        assert len(fit) == 2

    def test_falls_back_to_steps_segment(self):
        logs = [_log("stage1_step", 1), _log("stage2_chirp", 2), _log("stage2_steps", 3)]
        fit, holdout = split_logs(logs, JOINT)
        assert [log.stage for log in holdout] == ["stage2_steps"]
        assert [log.stage for log in fit] == ["stage1_step", "stage2_chirp"]

    def test_no_holdout_available_raises(self):
        with pytest.raises(DataError):
            split_logs([_log("stage2_chirp", 1)], JOINT)

    def test_joint_not_excited_raises(self):
        with pytest.raises(DataError):
            split_logs([_log("stage2_steps", 1)], "elbow_flex")


@pytest.fixture(scope="module")
def result():
    model = mujoco.MjModel.from_xml_path(str(MINI_ARM_XML))
    logs = [_log("stage1_step", 1), _log("stage2_chirp", 2), _log("stage2_ramp", 4), _log("stage2_steps", 3)]
    return evaluate_joint(model, logs, JOINT)


class TestEvaluate:
    def test_all_models_reported_in_ticks(self, result):
        assert set(result.errors_ticks) == {
            "hold_still",
            "command_as_position",
            "stock_mjcf",
            "stock_plus_delay",
            "behavior_3param",
            "behavior_4param",
            "wrapped_mjcf",
        }
        assert all(np.isfinite(value) and value >= 0 for value in result.errors_ticks.values())

    def test_ordering_on_data_generated_by_the_model(self, result):
        errors = result.errors_ticks
        assert errors["behavior_4param"] <= errors["behavior_3param"] + 0.05
        assert errors["behavior_4param"] < 0.4 * errors["stock_mjcf"]
        assert errors["stock_plus_delay"] <= errors["stock_mjcf"] + 1e-9
        assert errors["behavior_4param"] < errors["hold_still"]
        assert errors["wrapped_mjcf"] < 0.4 * errors["stock_mjcf"]

    def test_holds_predictions_for_plots(self, result):
        assert result.predictions["behavior_4param"].shape == result.measured.shape
        assert result.times.shape == result.measured.shape


def _two_pose_logs(truth: BehaviorParams) -> list[RawLog]:
    logs = []
    for base, seed in ((0.0, 1), (-1.3, 2)):
        for stage, offset in (("stage1_step", 0), ("stage2_chirp", 5), ("stage2_steps", 9)):
            rng = np.random.default_rng(seed + offset)
            times = np.arange(0, 10.0, 0.01)
            target = base + 0.12 * np.sin(2 * np.pi * (0.3 + 0.4 * times / 10) * times + rng.uniform(0, 1))
            target = np.round(target / RAD_PER_TICK) * RAD_PER_TICK
            measured = simulate(truth, times, target, q0=base) + rng.normal(0, 0.0003, len(times))
            logs.append(RawLog(times, (JOINT,), target[:, None], measured[:, None], stage, {"excited": [JOINT]}))
    return logs


def test_load_dependent_deadband_is_reported_and_used_by_the_wrapped_model():
    truth = BehaviorParams(0.016, 0.06, 0.003, gravity_rad=-0.03, q_horiz_rad=-1.4, deadband_gravity_rad=0.016)
    model = mujoco.MjModel.from_xml_path(str(MINI_ARM_XML))
    result = evaluate_joint(model, _two_pose_logs(truth), JOINT, q_horiz_rad=-1.4, load_dependent_deadband=True)
    assert result.behavior_5param is not None and "behavior_5param" in result.errors_ticks
    assert result.errors_ticks["behavior_5param"] <= result.errors_ticks["behavior_4param"] + 0.1
    assert result.errors_ticks["wrapped_mjcf"] < 0.4 * result.errors_ticks["stock_mjcf"]
    assert result.behavior_5param.deadband_gravity_rad > 0


def test_default_evaluation_has_no_5param_entry(result):
    assert result.behavior_5param is None and "behavior_5param" not in result.errors_ticks


def test_shared_logs_are_added_for_every_excited_joint(tmp_path):
    from reproduce_postmortem import load_joint_logs

    own = tmp_path / "joint"
    shared = tmp_path / "multijoint"
    own.mkdir()
    shared.mkdir()
    _log("stage2_steps", 1).save(own / "a.npz")
    both = _log("stage3_multisine", 2, role="holdout")
    both.save(shared / "b.npz")
    other = _log("stage3_multisine", 3, role="holdout")
    other.meta["excited"] = ["elbow_flex"]
    other.save(shared / "c.npz")
    logs = load_joint_logs(own, shared_dir=shared, joint=JOINT)
    assert sorted(log.stage for log in logs) == ["stage2_steps", "stage3_multisine"]
