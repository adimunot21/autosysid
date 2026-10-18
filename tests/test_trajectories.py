import numpy as np
import pytest

from autosysid.config import SafetyLimits
from autosysid.errors import ConfigError
from autosysid.excitation import step_train
from autosysid.hardware.calibration import ArmCalibration
from autosysid.hardware.trajectories import build_plan, limited_goals, preflight
from autosysid.mock import MINI_ARM_XML

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex")
LIMITS = SafetyLimits()
FULL_TORQUE = SafetyLimits(torque_limit_fraction=1.0)  # isolates the plan from the effect of the reduced torque limit


@pytest.fixture(scope="module")
def calibration():
    return ArmCalibration.symmetric(JOINTS)


def plan(scale=0.5, excited=("shoulder_pan", "shoulder_lift"), stages=(1, 2, 3), seed=0):
    return build_plan(excited, JOINTS, 100.0, scale, LIMITS, seed=seed, stages=stages)


def test_plan_structure_and_ordering():
    segments = plan()
    names = [s.name for s in segments]
    assert names[:2] == ["stage1_step_shoulder_pan", "stage1_ramp_shoulder_pan"]
    assert names[-2:] == ["stage3_multisine_fit", "stage3_multisine_holdout"]
    assert [s.role for s in segments if s.stage == "stage3"] == ["fit", "holdout"]
    assert sum(s.stage == "stage2" for s in segments) == 6  # 3 parts x 2 joints
    for segment in segments:
        assert segment.offsets_rad.shape == (len(segment.times_s), 3)
        assert segment.times_s[0] == 0.0 and np.allclose(np.diff(segment.times_s), 0.01)


def test_single_joint_segments_hold_the_other_joints_still():
    for segment in plan():
        if segment.stage != "stage3":
            held = [i for i, name in enumerate(JOINTS) if name not in segment.excited]
            assert np.all(segment.offsets_rad[:, held] == 0.0)


def test_every_segment_starts_and_ends_at_the_centre_pose():
    for segment in plan():
        assert np.allclose(segment.offsets_rad[0], 0.0, atol=1e-9)
        assert np.allclose(segment.offsets_rad[-1], 0.0, atol=2e-3), segment.name


def test_holdout_differs_from_the_fit_segment_and_is_reproducible():
    a, b = plan(stages=(3,), seed=1), plan(stages=(3,), seed=1)
    assert np.array_equal(a[0].offsets_rad, b[0].offsets_rad)
    assert not np.allclose(a[0].offsets_rad, a[1].offsets_rad), "the held-out segment must not repeat the fitted one"


def test_amplitude_scale_scales_the_excitation():
    full, half = plan(scale=1.0, stages=(2,)), plan(scale=0.5, stages=(2,))
    chirp_full = next(s for s in full if "chirp" in s.name).offsets_rad
    chirp_half = next(s for s in half if "chirp" in s.name).offsets_rad
    assert np.max(np.abs(chirp_half)) == pytest.approx(0.5 * np.max(np.abs(chirp_full)), rel=1e-6)


def test_invalid_plans_are_rejected():
    with pytest.raises(ConfigError):
        build_plan(("nope",), JOINTS, 100.0, 0.5, LIMITS)
    with pytest.raises(ConfigError):
        build_plan(("shoulder_pan",), JOINTS, 100.0, 1.5, LIMITS)


def test_default_plan_is_admissible_for_the_limiter(calibration):
    """The limiter must be a net, not part of the plan: nothing it sends should differ from what was planned."""
    for segment in plan(scale=1.0):
        goals, n_limited, clipped = limited_goals(segment, np.zeros(3), calibration, LIMITS)
        assert not clipped and n_limited == 0, segment.name
        assert np.max(np.abs(goals - segment.offsets_rad)) < 0.004, segment.name  # tick quantization + 1-tick lag


def test_accel_limited_step_train_respects_both_limits():
    t = np.arange(0, 14, 0.01)
    goal = step_train(t, np.array([0.3, -0.3, 0.0]), 4.0, 1.0, max_accel_rad_s2=10.0)
    velocity = np.diff(goal) / 0.01
    assert np.max(np.abs(velocity)) <= 1.0 + 1e-9
    assert np.max(np.abs(np.diff(velocity))) / 0.01 <= 10.0 + 1e-6
    assert goal.max() <= 0.3 + 1e-3 and goal.min() >= -0.3 - 1e-3, "no overshoot past the level"


def test_preflight_passes_a_gentle_plan(calibration):
    report = preflight(plan(excited=("shoulder_pan",)), MINI_ARM_XML, JOINTS, np.zeros(3), calibration, FULL_TORQUE)
    assert report.ok, report.summary()
    assert all(r.max_tracking_error_rad < 0.01 for r in report.segments)
    assert "ok" in report.summary()


def test_preflight_reports_saturation_of_the_step_segments(calibration):
    report = preflight(
        plan(scale=0.5, excited=("shoulder_pan",), stages=(2,)),
        MINI_ARM_XML,
        JOINTS,
        np.zeros(3),
        calibration,
        FULL_TORQUE,
    )
    steps = next(r for r in report.segments if "steps" in r.name)
    chirp = next(r for r in report.segments if "chirp" in r.name)
    assert steps.saturated_fraction["shoulder_pan"] > 0.01, (
        "steps are meant to briefly saturate (that is what identifies kv)"
    )
    assert chirp.saturated_fraction["shoulder_pan"] < 0.01 and not chirp.warnings


def test_preflight_warns_when_a_full_amplitude_chirp_saturates_the_nominal_model(calibration):
    """Finding: with nominal kp=998 and forcerange 3.35 the 0.2 rad chirp needs >3.4 mrad of error at its peaks, i.e. it
    saturates ~15% of the time; the plan is still allowed, but the user is told to reduce amplitude_scale."""
    report = preflight(
        plan(scale=1.0, excited=("shoulder_pan",), stages=(2,)),
        MINI_ARM_XML,
        JOINTS,
        np.zeros(3),
        calibration,
        FULL_TORQUE,
    )
    chirp = next(r for r in report.segments if "chirp" in r.name)
    assert report.ok and chirp.saturated_fraction["shoulder_pan"] > 0.05
    assert any("saturates" in w for w in chirp.warnings)


def test_preflight_blocks_a_start_pose_that_pushes_a_plan_past_the_soft_limit(calibration):
    near_limit = np.array([1.55, 0.0, 0.0])  # soft limit is 0.8 * 1.994 = 1.595 rad
    report = preflight(
        plan(scale=1.0, excited=("shoulder_pan",)), MINI_ARM_XML, JOINTS, near_limit, calibration, LIMITS
    )
    assert not report.ok
    assert any("soft position limit" in v for r in report.segments for v in r.violations)


def test_preflight_blocks_predicted_watchdog_trips(calibration):
    strict = SafetyLimits(max_tracking_error_rad=0.004)  # a budget of 2 mrad that the step segments cannot meet
    report = preflight(
        plan(scale=1.0, excited=("shoulder_pan",), stages=(2,)), MINI_ARM_XML, JOINTS, np.zeros(3), calibration, strict
    )
    assert not report.ok and any("watchdog" in v for r in report.segments for v in r.violations)


def test_preflight_uses_the_reduced_torque_limit_the_arm_will_run_with(calibration):
    """Finding (Phase 4): the 35% bring-up Torque_Limit leaves a 1.2 N.m clamp, so even a half-amplitude chirp saturates
    most of the time. Pre-flight must predict that, not the datasheet's 3.35 N.m."""
    segments = plan(scale=0.5, excited=("shoulder_pan",), stages=(2,))
    default = preflight(
        segments, MINI_ARM_XML, JOINTS, np.zeros(3), calibration, SafetyLimits(torque_limit_fraction=0.35)
    )
    full = preflight(segments, MINI_ARM_XML, JOINTS, np.zeros(3), calibration, FULL_TORQUE)
    chirp_default = next(r for r in default.segments if "chirp" in r.name)
    chirp_full = next(r for r in full.segments if "chirp" in r.name)
    assert chirp_default.saturated_fraction["shoulder_pan"] > 5 * max(
        chirp_full.saturated_fraction["shoulder_pan"], 0.01
    )
    assert any("saturates" in w for w in chirp_default.warnings)
