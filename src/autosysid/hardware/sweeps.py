"""Sweep settings for the joints the first hardware sessions did not cover: shoulder_lift, elbow_flex, gripper.

Nothing here moves the arm. It only shapes the plan that `autosysid record --behavior-sweep` hands to the usual
pipeline, so every goal still passes pre-flight, `GoalLimiter` (soft limits, rate and acceleration caps) and the
watchdogs unchanged.
"""

from __future__ import annotations

import numpy as np

from autosysid.errors import ConfigError

# Multipliers on top of --amplitude-scale (CLI default 0.5). First-use values, chosen so the stage-2 chirp (0.2 rad
# x 0.5 x factor) peaks at ~0.06 rad = 3.4 deg for the heavy joints, i.e. no bigger than the 0.05 rad stage-1 moves
# that were proven safe on wrist/pan joints; the gripper (plastic fingers, no gravity load, but a hard stop at each end
# of its short 1489-tick range) gets 0.5. Raise through --amplitude-scale only after a clean pre-flight and a first run.
BEHAVIOR_SWEEP_FACTORS: dict[str, float] = {"shoulder_lift": 0.6, "elbow_flex": 0.6, "gripper": 0.5}


def gravity_poses_rad(joint: str, lower: float, upper: float, count: int = 3) -> list[float]:
    """Base poses at which to repeat a gravity-loaded joint's sweep, spread over the central 60% of [lower, upper].

    g0*cos(q - q_horiz) is only identifiable if the same joint is swept at several configurations: within one
    ~0.1 rad sweep cos is nearly constant. Each pose is a separate `autosysid record` run started at that pose
    (the recorder centres the plan on the pose it finds); the operator places the arm, with it supported.
    """
    if not (np.isfinite(lower) and np.isfinite(upper) and lower < upper):
        raise ConfigError(f"{joint}: need finite lower < upper, got [{lower}, {upper}]")
    if count < 2:
        raise ConfigError(f"{joint}: need at least 2 poses to separate gravity from offset, got {count}")
    return [float(pose) for pose in lower + (upper - lower) * np.linspace(0.2, 0.8, count)]
