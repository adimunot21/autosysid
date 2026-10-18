import mujoco
import numpy as np
import pytest

from autosysid.data import RawLog, build_model_sequences, control_grid, zero_order_hold
from autosysid.errors import DataError
from autosysid.mock import MINI_ARM_XML
from autosysid.model import load_spec


def make_log(n=200, joints=("shoulder_pan",), jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    times = np.maximum.accumulate(np.arange(n) * 0.01 + rng.normal(0, jitter, n))
    target = np.tile(np.sin(times)[:, None], (1, len(joints)))
    return RawLog(times, joints, target, target.copy(), stage="t", meta={"a": 1})


def test_save_load_round_trip(tmp_path):
    log = make_log()
    log.save(tmp_path / "x" / "log.npz")
    restored = RawLog.load(tmp_path / "x" / "log.npz")
    assert np.array_equal(restored.times, log.times) and restored.joints == log.joints
    assert restored.stage == "t" and restored.meta == {"a": 1}


@pytest.mark.parametrize(
    "mutate",
    [
        lambda kw: kw.update(times=kw["times"][::-1]),  # decreasing
        lambda kw: kw["measured_rad"].__setitem__((3, 0), np.nan),
        lambda kw: kw.update(joints=("a", "a")),
        lambda kw: kw.update(target_rad=kw["target_rad"][:-1]),
        lambda kw: kw.update(
            times=kw["times"][:5], target_rad=kw["target_rad"][:5], measured_rad=kw["measured_rad"][:5]
        ),
    ],
)
def test_invalid_logs_are_rejected(mutate):
    base = make_log()
    kwargs = {
        "times": base.times.copy(),
        "joints": base.joints,
        "target_rad": base.target_rad.copy(),
        "measured_rad": base.measured_rad.copy(),
    }
    mutate(kwargs)
    with pytest.raises(DataError):
        RawLog(**kwargs)


def test_load_missing_file(tmp_path):
    with pytest.raises(DataError):
        RawLog.load(tmp_path / "missing.npz")


def test_zero_order_hold_holds_until_next_tick():
    ticks = np.array([0.0, 1.0, 2.0])
    values = np.array([10.0, 20.0, 30.0])
    out = zero_order_hold(np.array([-1.0, 0.0, 0.99, 1.0, 5.0]), ticks, values)
    assert list(out) == [10.0, 10.0, 10.0, 20.0, 30.0]


def test_control_grid_is_exact_even_when_the_log_end_is_not_a_multiple_of_dt():
    """Regression: TimeSeries.resample(target_dt) stretches the axis when the last stamp is not k*dt (Phase 0)."""
    model = load_spec(MINI_ARM_XML).compile()
    log = make_log(joints=("shoulder_pan", "shoulder_lift", "elbow_flex"), jitter=0.0004)
    assert (log.times[-1] - log.times[0]) % model.opt.timestep > 1e-6, "test needs an off-grid end time"
    grid, ctrl = control_grid(model, log)
    assert np.allclose(np.diff(grid), model.opt.timestep, rtol=0, atol=1e-12)
    assert grid[-1] <= log.times[-1] - log.times[0] + 1e-9
    assert ctrl.shape == (len(grid), model.nu)


def test_control_grid_needs_every_actuator_in_the_log():
    model = load_spec(MINI_ARM_XML).compile()
    with pytest.raises(DataError):
        control_grid(model, make_log(joints=("shoulder_pan",)))


def test_build_model_sequences_accepts_multiple_logs():
    spec = load_spec(MINI_ARM_XML, sensed_joints=("shoulder_pan",))
    joints = ("shoulder_pan", "shoulder_lift", "elbow_flex")
    sequences = build_model_sequences(spec, [make_log(joints=joints), make_log(joints=joints, seed=1)], "shoulder_pan")
    assert len(sequences.measured_rollout) == 2
    assert isinstance(sequences.gt_model, mujoco.MjModel)
