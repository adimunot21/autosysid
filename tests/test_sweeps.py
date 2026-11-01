"""Per-joint sweep amplitudes for the gravity-loaded joints and the gripper, and the MJCF gravity phase."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from autosysid.config import JOINT_NAMES, RecordConfig, SafetyLimits
from autosysid.errors import ConfigError
from autosysid.gravity import gravity_phase_from_mjcf
from autosysid.hardware.calibration import ArmCalibration
from autosysid.hardware.sweeps import BEHAVIOR_SWEEP_FACTORS, gravity_poses_rad
from autosysid.hardware.trajectories import build_plan, limited_goals
from autosysid.mock import MINI_ARM_XML

LIMITS = SafetyLimits()


def _plan(factors, excited=("shoulder_lift",), stages=(1, 2, 3)):
    return build_plan(excited, JOINT_NAMES, 100.0, 0.5, LIMITS, stages=stages, joint_amplitude_factors=factors)


def _peak(plan, joint):
    column = JOINT_NAMES.index(joint)
    return max(float(np.max(np.abs(segment.offsets_rad[:, column]))) for segment in plan)


class TestAmplitudeFactors:
    def test_factor_scales_only_that_joint(self):
        full = _plan(None, ("shoulder_lift", "elbow_flex"))
        scaled = _plan({"shoulder_lift": 0.5}, ("shoulder_lift", "elbow_flex"))
        assert _peak(scaled, "shoulder_lift") == pytest.approx(0.5 * _peak(full, "shoulder_lift"), rel=1e-6)
        assert _peak(scaled, "elbow_flex") == pytest.approx(_peak(full, "elbow_flex"), rel=1e-6)

    @pytest.mark.parametrize("factor", [0.0, -0.1, 1.5, float("nan")])
    def test_rejects_bad_factor(self, factor):
        with pytest.raises(ConfigError):
            _plan({"shoulder_lift": factor})

    def test_rejects_unknown_joint(self):
        with pytest.raises(ConfigError):
            _plan({"wrist_spin": 0.5})

    def test_record_config_validates_factors(self):
        with pytest.raises(ConfigError):
            RecordConfig(joint_amplitude_factors={"shoulder_lift": 2.0})
        assert RecordConfig(joint_amplitude_factors={"gripper": 0.5}).joint_amplitude_factors == {"gripper": 0.5}


class TestBehaviorSweeps:
    def test_factors_cover_the_untested_joints(self):
        assert {"shoulder_lift", "elbow_flex", "gripper"} <= set(BEHAVIOR_SWEEP_FACTORS)
        assert all(0 < value <= 1 for value in BEHAVIOR_SWEEP_FACTORS.values())

    @pytest.mark.parametrize("joint", ["shoulder_lift", "elbow_flex", "gripper"])
    def test_plans_pass_through_the_safety_limiter_unclipped(self, joint):
        calibration = ArmCalibration.symmetric(JOINT_NAMES)
        center = np.zeros(len(JOINT_NAMES))
        for segment in _plan(BEHAVIOR_SWEEP_FACTORS, (joint,)):
            goals, n_limited, clipped = limited_goals(segment, center, calibration, LIMITS)
            assert not clipped, segment.name
            rate = np.max(np.abs(np.diff(goals, axis=0))) * 100.0
            assert rate <= LIMITS.max_goal_rate_rad_s * 1.05, (segment.name, rate)

    def test_gravity_poses_are_distinct_and_inside_limits(self):
        poses = gravity_poses_rad("shoulder_lift", lower=-1.2, upper=1.2, count=3)
        assert len(poses) == 3 and len(set(poses)) == 3
        assert min(poses) >= -1.2 and max(poses) <= 1.2
        assert poses == sorted(poses)

    def test_gravity_poses_reject_bad_input(self):
        with pytest.raises(ConfigError):
            gravity_poses_rad("shoulder_lift", lower=1.0, upper=-1.0, count=3)
        with pytest.raises(ConfigError):
            gravity_poses_rad("shoulder_lift", lower=-1.0, upper=1.0, count=1)


class TestGravityPhase:
    def test_recovers_analytic_phase_on_mini_arm(self):
        """mini_arm's upper_arm COM sits above the y-axis, so holding torque is +-m g r sin(q): peak at q = +-pi/2."""
        model = mujoco.MjModel.from_xml_path(str(MINI_ARM_XML))
        phase = gravity_phase_from_mjcf(model, "shoulder_lift")
        assert phase.torque_amplitude_nm > 0
        for candidate in (phase.q_horiz_rad, phase.q_horiz_rad + 2 * np.pi, phase.q_horiz_rad - 2 * np.pi):
            if abs(abs(candidate) - np.pi / 2) < 0.05:
                break
        else:
            pytest.fail(f"q_horiz {phase.q_horiz_rad} is not +-pi/2")
        assert phase.r_squared > 0.999

    def test_pan_axis_has_no_gravity_torque(self):
        model = mujoco.MjModel.from_xml_path(str(MINI_ARM_XML))
        assert gravity_phase_from_mjcf(model, "shoulder_pan").torque_amplitude_nm < 1e-6

    def test_unknown_joint(self):
        model = mujoco.MjModel.from_xml_path(str(MINI_ARM_XML))
        with pytest.raises(ConfigError):
            gravity_phase_from_mjcf(model, "wrist_roll")
