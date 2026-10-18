import numpy as np
import pytest

from autosysid.config import FitConfig
from autosysid.errors import DataError
from autosysid.mock import MINI_ARM_XML
from autosysid.optimizer import PARAMETER_NAMES, _JointProblem, expected_noise_floor, fit_all, fit_joint
from conftest import JOINT, TRUE_DELAY_S, TRUTH_SCALE


@pytest.fixture(scope="module")
def problem_and_logs(make_logs):
    _, logs = make_logs(seed=0, parts="c", scale=0.6)
    config = FitConfig()
    return _JointProblem(MINI_ARM_XML, logs, JOINT, config), logs, config


def truth_start():
    return {**TRUTH_SCALE, "delay": TRUE_DELAY_S * 1e3}


def test_cost_at_truth_is_near_the_noise_floor(problem_and_logs):
    problem, logs, config = problem_and_logs
    cost = problem.cost(problem.make_params(truth_start(), set(PARAMETER_NAMES)))
    floor = expected_noise_floor(logs, JOINT, config)
    assert floor < cost < config.noise_floor_margin * floor  # 0.5 mrad read noise sits on top of quantization


def test_cost_is_far_above_floor_at_nominal(problem_and_logs):
    problem, logs, config = problem_and_logs
    nominal = {"kp": 1, "kv": 1, "damping": 1, "frictionloss": 1, "armature": 1, "delay": 10.0}
    cost = problem.cost(problem.make_params(nominal, set(PARAMETER_NAMES)))
    assert cost > 100 * expected_noise_floor(logs, JOINT, config)


def test_frozen_parameters_are_still_applied(problem_and_logs):
    """Regression: sysid's default build skips frozen parameters, silently reverting a pinned value to nominal."""
    problem, _, _ = problem_and_logs
    all_free = problem.cost(problem.make_params(truth_start(), set(PARAMETER_NAMES)))
    only_delay_free = problem.cost(problem.make_params(truth_start(), {"delay"}))
    assert only_delay_free == pytest.approx(all_free, rel=1e-9)


def test_residual_length_does_not_depend_on_delay(problem_and_logs):
    """Regression: a window that moved with the delay changed the sample count and broke finite differences."""
    problem, _, _ = problem_and_logs
    lengths = set()
    for delay_ms in (0.0, 10.0, 50.0, 99.0):
        params = problem.make_params({**truth_start(), "delay": delay_ms}, set(PARAMETER_NAMES))
        lengths.add(sum(r.size for r in problem.data_residual_fn(params.as_vector(), params)[0]))
    assert len(lengths) == 1


def test_prior_adds_terms_only_for_free_dynamics_parameters(problem_and_logs):
    problem, _, _ = problem_and_logs
    for free, n_prior in ((set(PARAMETER_NAMES), 5), ({"delay"}, 0), ({"damping", "delay"}, 1)):
        params = problem.make_params(truth_start(), free)
        data = problem.data_residual_fn(params.as_vector(), params)[0]
        full = problem.residual_fn(params.as_vector(), params)[0]
        assert len(full) - len(data) == (1 if n_prior else 0)
        if n_prior:
            assert full[-1].shape == (n_prior,)


def test_prior_is_negligible_at_nominal_and_pulls_away_from_it(problem_and_logs):
    problem, _, _ = problem_and_logs
    params = problem.make_params(
        {"kp": 1, "kv": 1, "damping": 1, "frictionloss": 1, "armature": 1, "delay": 10.0}, set(PARAMETER_NAMES)
    )
    assert np.allclose(problem.residual_fn(params.as_vector(), params)[0][-1], 0.0)  # ln(1) = 0
    params = problem.make_params(truth_start(), set(PARAMETER_NAMES))
    assert np.all(np.abs(problem.residual_fn(params.as_vector(), params)[0][-1]) > 0)


def test_fit_rejects_empty_joint_list():
    with pytest.raises(DataError):
        fit_all(MINI_ARM_XML, [], [], FitConfig())


@pytest.mark.slow
def test_fit_recovers_known_parameters(make_logs, truth):
    _, logs = make_logs(seed=0, parts="rcs")
    result = fit_joint(MINI_ARM_XML, logs, JOINT, FitConfig(max_restarts=2))
    assert result.accepted
    assert abs(result.delay_s - TRUE_DELAY_S) < 0.0015
    for name in ("kp", "kv", "damping", "armature"):
        assert getattr(result.dynamics, name) == pytest.approx(getattr(truth, name), rel=0.15), name
    # friction is only weakly identified: it may be off, but then it must say so
    friction_error = abs(result.dynamics.frictionloss / truth.frictionloss - 1)
    ci = result.relative_ci["frictionloss"]
    assert friction_error < 0.3 or ci > friction_error
    assert min(result.attempt_costs) == pytest.approx(result.cost)


def test_validation_scores_a_correct_model_far_better_than_the_nominal_one(make_logs, truth):
    from autosysid.model import load_spec, read_dynamics
    from autosysid.optimizer import JointFit, validate_joint

    nominal = read_dynamics(load_spec(MINI_ARM_XML), JOINT)
    fit = JointFit(JOINT, nominal, truth, TRUE_DELAY_S, {}, (), 1.0, 1.0, True, (1.0,), 0.0)
    _, held_out = make_logs(seed=7, parts="c", scale=0.6)
    result = validate_joint(MINI_ARM_XML, fit, held_out)
    assert result.rms_fitted_rad < 0.0015, "a correct model should be within ~1 tick"
    assert result.improvement > 5 and result.n_samples > 500
