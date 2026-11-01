#!/usr/bin/env python
"""Print the base poses (rad and ticks) at which to repeat each gravity-loaded joint's sweep, and q_horiz from the MJCF.

Read-only: parses the LeRobot calibration file and the MJCF; opens no serial port and moves nothing.

    python scripts/print_gravity_poses.py --xml runs/model/so101/scene.xml
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import mujoco

from autosysid.config import JOINT_NAMES, SafetyLimits
from autosysid.gravity import gravity_phase_from_mjcf
from autosysid.hardware.calibration import ArmCalibration
from autosysid.hardware.sweeps import gravity_poses_rad
from autosysid.pipeline import DEFAULT_CALIBRATION

GRAVITY_LOADED = ("shoulder_lift", "elbow_flex")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args()

    calibration = ArmCalibration.from_lerobot_json(args.calibration)
    lower, upper = calibration.soft_limits_rad(SafetyLimits().soft_limit_fraction)
    model = mujoco.MjModel.from_xml_path(str(args.xml))
    for joint in GRAVITY_LOADED:
        index = JOINT_NAMES.index(joint)
        phase = gravity_phase_from_mjcf(model, joint)
        print(
            f"{joint}: q_horiz {phase.q_horiz_rad:+.3f} rad ({math.degrees(phase.q_horiz_rad):+.0f} deg), "
            f"peak holding torque {phase.torque_amplitude_nm:.2f} N.m (other joints at the MJCF reference pose)"
        )
        for pose in gravity_poses_rad(joint, float(lower[index]), float(upper[index]), args.count):
            ticks = calibration.rad_to_ticks(_pose_vector(index, pose), None)[index]
            print(f"    base pose {pose:+.3f} rad = {math.degrees(pose):+6.1f} deg = {ticks} ticks")


def _pose_vector(index: int, value: float):
    import numpy as np

    pose = np.zeros(len(JOINT_NAMES))
    pose[index] = value
    return pose


if __name__ == "__main__":
    main()
