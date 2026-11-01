"""STS3215SimWrapper: the behavioral filter in front of a MuJoCo position actuator."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from autosysid.behavior import RAD_PER_TICK, BehaviorParams, rms_error_ticks, simulate
from autosysid.errors import ConfigError, DataError
from autosysid.mock import MINI_ARM_XML
from autosysid.sim_wrapper import STS3215SimWrapper, ideal_actuator_model

PAN = "shoulder_pan"  # vertical axis: no gravity torque, isolates the filter
LIFT = "shoulder_lift"  # horizontal axis: real gravity torque


@pytest.fixture
def stock_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(MINI_ARM_XML))


@pytest.fixture
def model(stock_model) -> mujoco.MjModel:
    return ideal_actuator_model(stock_model)


def _wrapper(model, params: dict[str, BehaviorParams]) -> STS3215SimWrapper:
    return STS3215SimWrapper(model, mujoco.MjData(model), params)


def _hold(wrapper: STS3215SimWrapper, name: str, value: float, seconds: float) -> np.ndarray:
    target = np.zeros(wrapper.model.nu)
    target[wrapper.actuator_index(name)] = value
    trace = []
    for _ in range(int(seconds / wrapper.model.opt.timestep)):
        wrapper.step(target)
        trace.append(wrapper.joint_position(name))
    return np.array(trace)


class TestStockEquivalence:
    def test_no_params_is_plain_mujoco(self, model):
        wrapper = _wrapper(model, {})
        reference = mujoco.MjData(model)
        target = np.array([0.1, -0.2, 0.3])
        for _ in range(200):
            wrapper.step(target)
            reference.ctrl[:] = target
            mujoco.mj_step(model, reference)
        np.testing.assert_allclose(wrapper.data.qpos, reference.qpos, atol=1e-12)


class TestBehavior:
    def test_transport_delay(self, model):
        params = BehaviorParams(delay_s=0.020, tau_s=0.002, deadband_rad=0.0)
        wrapper = _wrapper(model, {PAN: params})
        trace = _hold(wrapper, PAN, 0.5, 0.2)
        first_move = int(np.argmax(trace > 0.01)) * model.opt.timestep
        assert first_move == pytest.approx(0.020, abs=0.006)

    def test_first_order_lag(self, model):
        tau = 0.06
        wrapper = _wrapper(model, {PAN: BehaviorParams(delay_s=0.0, tau_s=tau, deadband_rad=0.0)})
        trace = _hold(wrapper, PAN, 0.3, 0.5)
        reached = int(np.argmax(trace >= 0.3 * (1 - np.exp(-1)))) * model.opt.timestep
        assert reached == pytest.approx(tau, abs=0.01)

    def test_deadband_holds_still_then_settles_short(self, model):
        d = 4 * RAD_PER_TICK
        wrapper = _wrapper(model, {PAN: BehaviorParams(delay_s=0.0, tau_s=0.02, deadband_rad=d)})
        assert np.max(np.abs(_hold(wrapper, PAN, 0.8 * d, 0.5))) < 1e-4
        settled = _hold(wrapper, PAN, 0.3, 1.0)[-1]
        assert settled == pytest.approx(0.3 - d, abs=3e-4)

    def test_gravity_offset_moves_resting_pose(self, model):
        offset = 0.02
        wrapper = _wrapper(model, {PAN: BehaviorParams(0.0, 0.02, 0.0, gravity_rad=offset)})
        assert _hold(wrapper, PAN, 0.0, 0.5)[-1] == pytest.approx(offset, abs=3e-4)

    def test_cos_gravity_uses_true_configuration(self, model):
        params = BehaviorParams(0.0, 0.02, 0.0, gravity_rad=0.02, q_horiz_rad=0.0)
        wrapper = _wrapper(model, {PAN: params})
        final = _hold(wrapper, PAN, 0.8, 1.0)[-1]
        assert final == pytest.approx(0.8 + 0.02 * np.cos(final), abs=5e-4)

    def test_other_joints_untouched(self, model):
        wrapper = _wrapper(model, {PAN: BehaviorParams(0.02, 0.06, 0.005)})
        reference = mujoco.MjData(model)
        target = np.array([0.4, 0.2, -0.1])
        for _ in range(300):
            wrapper.step(target)
            reference.ctrl[:] = target
            mujoco.mj_step(model, reference)
        for name in (LIFT, "elbow_flex"):
            index = wrapper.actuator_index(name)
            assert wrapper.data.qpos[index] == pytest.approx(reference.qpos[index], abs=1e-8)


class TestIdealActuators:
    def test_original_model_untouched(self, stock_model):
        before = stock_model.actuator_gainprm.copy(), stock_model.actuator_forcerange.copy()
        ideal = ideal_actuator_model(stock_model)
        np.testing.assert_array_equal(stock_model.actuator_gainprm, before[0])
        np.testing.assert_array_equal(stock_model.actuator_forcerange, before[1])
        assert ideal.actuator_gainprm[0, 0] == pytest.approx(10 * before[0][0, 0])
        assert ideal.actuator_forcerange[0, 1] == pytest.approx(100 * before[1][0, 1])

    def test_large_step_not_force_saturated(self, stock_model):
        """The stock actuator slews a 0.5 rad step (forcerange); under the wrapper it must follow the pure model."""
        params = BehaviorParams(0.015, 0.06, 4 * RAD_PER_TICK)
        times = np.arange(60) * 0.01
        targets = np.zeros((60, stock_model.nu))
        targets[10:, 0] = 0.5
        reference = simulate(params, times, targets[:, 0], 0.0, dt=stock_model.opt.timestep)
        ideal = _wrapper(ideal_actuator_model(stock_model), {PAN: params}).rollout(times, targets)[:, 0]
        stock = _wrapper(stock_model, {PAN: params}).rollout(times, targets)[:, 0]
        assert np.max(np.abs(ideal - reference)) / RAD_PER_TICK < 10
        assert np.max(np.abs(stock - reference)) / RAD_PER_TICK > 3 * np.max(np.abs(ideal - reference)) / RAD_PER_TICK

    def test_rejects_non_position_actuator(self, stock_model):
        stock_model.actuator_gaintype[0] = mujoco.mjtGain.mjGAIN_AFFINE
        with pytest.raises(ConfigError):
            ideal_actuator_model(stock_model)

    @pytest.mark.parametrize("kwargs", [{"stiffness_scale": 0.0}, {"force_scale": float("nan")}])
    def test_rejects_bad_scales(self, stock_model, kwargs):
        with pytest.raises(ConfigError):
            ideal_actuator_model(stock_model, **kwargs)


class TestRollout:
    def test_matches_reference_model_on_log_grid(self, model):
        """Wrapper+stiff actuator agrees with the pure behavioral model to within a tick or so."""
        params = BehaviorParams(delay_s=0.014, tau_s=0.06, deadband_rad=4 * RAD_PER_TICK)
        times = np.arange(0, 6.0, 0.01)
        command = 0.15 * np.sin(2 * np.pi * (0.2 + 0.3 * times / 6.0) * times)
        targets = np.zeros((len(times), model.nu))
        targets[:, 0] = command
        wrapper = _wrapper(model, {PAN: params})
        simulated = wrapper.rollout(times, targets)[:, 0]
        reference = simulate(params, times, command, q0=0.0, dt=model.opt.timestep)
        assert simulated.shape == times.shape
        assert rms_error_ticks(simulated, reference) < 1.5

    def test_rollout_is_repeatable(self, model):
        times = np.arange(0, 1.0, 0.01)
        targets = np.zeros((len(times), model.nu))
        targets[:, 0] = 0.1
        wrapper = _wrapper(model, {PAN: BehaviorParams(0.01, 0.05, 0.003)})
        first = wrapper.rollout(times, targets)
        np.testing.assert_array_equal(first, wrapper.rollout(times, targets))

    def test_rollout_starts_at_given_pose(self, model):
        times = np.arange(0, 0.5, 0.01)
        targets = np.full((len(times), model.nu), 0.2)
        traj = _wrapper(model, {}).rollout(times, targets, initial_qpos=np.full(model.nq, 0.2))
        assert traj[0, 0] == pytest.approx(0.2, abs=1e-6)


class TestValidation:
    def test_unknown_joint_in_params(self, model):
        with pytest.raises(ConfigError):
            _wrapper(model, {"wrist_roll": BehaviorParams(0.01, 0.05, 0.0)})

    def test_bad_target_shape(self, model):
        with pytest.raises(DataError):
            _wrapper(model, {}).step(np.zeros(model.nu + 1))

    def test_non_finite_target(self, model):
        target = np.zeros(model.nu)
        target[1] = np.nan
        with pytest.raises(DataError):
            _wrapper(model, {}).step(target)

    def test_unknown_actuator_name(self, model):
        with pytest.raises(ConfigError):
            _wrapper(model, {}).actuator_index("wrist_roll")

    def test_rollout_shape_mismatch(self, model):
        with pytest.raises(DataError):
            _wrapper(model, {}).rollout(np.arange(10) * 0.01, np.zeros((9, model.nu)))

    def test_from_json_round_trip(self, model, tmp_path):
        params = {PAN: BehaviorParams(0.012, 0.06, 0.006, gravity_rad=0.004)}
        STS3215SimWrapper.save_params(tmp_path / "p.json", params)
        loaded = STS3215SimWrapper.from_json(model, mujoco.MjData(model), tmp_path / "p.json")
        assert loaded.params == params

    def test_from_json_missing(self, model, tmp_path):
        with pytest.raises(ConfigError):
            STS3215SimWrapper.from_json(model, mujoco.MjData(model), tmp_path / "missing.json")
