#!/usr/bin/env python
"""Real arm vs stock MJCF vs 4-parameter wrapped MJCF: does the wrapper cut tracking error by >= 4x?

    python scripts/validate_wrapper.py                        # held-out segment of every joint in hardware_logs/
    python scripts/validate_wrapper.py --trajectory run.npz   # a recorded multi-joint trajectory (any RawLog)

Per joint, parameters are fitted on that joint's FIT logs only, then the stock MJCF and the wrapped MJCF
(`STS3215SimWrapper` over `ideal_actuator_model`) are both driven by the logged targets and scored against the
measured angles. With --trajectory, parameters for each excited joint come from its fit logs in hardware_logs/ and all
joints are wrapped at once, so cross-joint coupling in the MJCF is exercised.

Exit code 1 if any joint's ratio stock/wrapped is below --min-ratio. The 4x target is a check, not an assumption.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import mujoco
import numpy as np

from autosysid.behavior import BehaviorParams, fit_behavior, rms_error_ticks
from autosysid.data import RawLog, zero_order_hold
from autosysid.errors import AutoSysIDError
from autosysid.evaluation import split_logs
from autosysid.sim_wrapper import STS3215SimWrapper, ideal_actuator_model

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reproduce_postmortem import find_xml, load_joint_logs  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGGER = logging.getLogger("validate_wrapper")


def rollout_all(model: mujoco.MjModel, params: dict[str, BehaviorParams], log: RawLog) -> np.ndarray:
    """Every actuator's simulated angle at the log times; start pose = first measured angles (zero if not logged)."""
    wrapper = STS3215SimWrapper(model, mujoco.MjData(model), params)
    relative = log.times - log.times[0]
    targets = np.zeros((len(log.times), model.nu))
    initial_qpos = np.zeros(model.nq)
    for index, name in enumerate(wrapper.actuator_names):
        if name in log.joints:
            column = log.column(name)
            targets[:, index] = zero_order_hold(relative, relative, log.target_rad[:, column])
            initial_qpos[wrapper.qpos_address[index]] = log.measured_rad[0, column]
    return wrapper.rollout(log.times, targets, initial_qpos=initial_qpos)


def score(model: mujoco.MjModel, params: dict[str, BehaviorParams], log: RawLog, joints: list[str]) -> dict[str, float]:
    ideal = ideal_actuator_model(model)
    names = [model.actuator(i).name for i in range(model.nu)]
    stock = rollout_all(model, {}, log)
    wrapped = rollout_all(ideal, params, log)
    out = {}
    for joint in joints:
        measured = log.measured_rad[:, log.column(joint)]
        out[joint] = (
            rms_error_ticks(stock[:, names.index(joint)], measured),
            rms_error_ticks(wrapped[:, names.index(joint)], measured),
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs", type=Path, default=REPO_ROOT / "hardware_logs")
    parser.add_argument("--xml", type=Path, default=None)
    parser.add_argument("--trajectory", type=Path, default=None, help="multi-joint RawLog .npz to validate on")
    parser.add_argument("--min-ratio", type=float, default=4.0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    model = mujoco.MjModel.from_xml_path(str(find_xml(args.xml)))
    joint_dirs = sorted(path.name for path in args.logs.iterdir() if path.is_dir())

    fits: dict[str, BehaviorParams] = {}
    held: dict[str, list[RawLog]] = {}
    for joint in joint_dirs:
        try:
            fit_logs, held[joint] = split_logs(load_joint_logs(args.logs / joint), joint)
        except AutoSysIDError as error:
            LOGGER.warning("%s skipped: %s", joint, error)
            continue
        fits[joint] = fit_behavior(fit_logs, joint).params

    results: dict[str, tuple[float, float]] = {}
    if args.trajectory is not None:
        log = RawLog.load(args.trajectory)
        joints = [joint for joint in log.meta.get("excited", []) if joint in fits]
        if not joints:
            raise SystemExit(f"none of the excited joints {log.meta.get('excited')} have fitted parameters")
        LOGGER.info("trajectory %s: joints %s", args.trajectory.name, joints)
        results = score(model, {joint: fits[joint] for joint in joints}, log, joints)
    else:
        for joint, logs in held.items():
            for log in logs:
                results[f"{joint} ({log.stage})"] = score(model, {joint: fits[joint]}, log, [joint])[joint]

    print(f"{'case':44s} {'stock':>8s} {'wrapped':>8s} {'ratio':>7s}")
    failed = False
    for case, (stock_error, wrapped_error) in results.items():
        ratio = stock_error / wrapped_error
        failed |= ratio < args.min_ratio
        verdict = "ok" if ratio >= args.min_ratio else "BELOW TARGET"
        print(f"{case:44s} {stock_error:8.2f} {wrapped_error:8.2f} {ratio:6.1f}x {verdict}")
    if args.trajectory is None:
        print("\nNote: single-joint recordings only; multi-joint needs a hardware run (--trajectory).")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
