#!/usr/bin/env python
"""Regenerate the post-mortem error table and residual plots from the committed hardware logs.

    python scripts/reproduce_postmortem.py                      # all joints in hardware_logs/
    python scripts/reproduce_postmortem.py --joints wrist_roll  # one joint
    python scripts/reproduce_postmortem.py --rerun-sysid        # also re-run mujoco.sysid (~10 min per joint)

Inputs
  hardware_logs/<joint>/*.npz   raw logs from `autosysid record` (see hardware_logs/README.md)
  SO-101 MJCF                   --xml, $AUTOSYSID_SO101_XML or runs/model/so101/scene.xml. Not in the repo (16 MB of
                                meshes): clone https://github.com/TheRobotStudio/SO-ARM100 @ a758567 and use
                                Simulation/SO101/scene.xml.
Outputs (--out, default results/)
  comparison.md / comparison.json   held-out RMS error (ticks) per joint and model, plus fitted parameters
  behavior_params.json              4-parameter fit on ALL logs per joint; input to STS3215SimWrapper
  residuals_<joint>.png             held-out prediction and residuals for stock / stock+delay / 4-parameter model

Held-out data is the recorded stage-3 multisine when a joint has one, otherwise its stage-2 "steps" segment
(autosysid.evaluation.split_logs). The reference `mujoco.sysid` number is read from results/sysid_reference.json
unless --rerun-sysid is given, and is flagged in the table because its split was not saved.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402

from autosysid.behavior import RAD_PER_TICK, fit_behavior  # noqa: E402
from autosysid.data import RawLog  # noqa: E402
from autosysid.errors import AutoSysIDError  # noqa: E402
from autosysid.evaluation import JointEvaluation, evaluate_joint  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGGER = logging.getLogger("reproduce_postmortem")

CONFIG = {
    # Table column order: the story from "do nothing" to the model, left to right.
    "columns": [
        ("hold_still", "Hold still"),
        ("stock_mjcf", "Stock MJCF"),
        ("stock_plus_delay", "Stock + best delay"),
        ("mujoco_sysid", "mujoco.sysid*"),
        ("behavior_3param", "3-param"),
        ("behavior_4param", "4-param"),
        ("wrapped_mjcf", "4-param in MJCF"),
    ],
    # Colour-blind-safe (Okabe-Ito) hues for the three plotted models.
    "plot_colors": {"stock_mjcf": "#D55E00", "stock_plus_delay": "#E69F00", "behavior_4param": "#0072B2"},
    "plot_window_s": 12.0,  # seconds of the first held-out segment shown in the top panel
}


def find_xml(cli_value: Path | None) -> Path:
    candidates = [cli_value, Path(os.environ["AUTOSYSID_SO101_XML"]) if "AUTOSYSID_SO101_XML" in os.environ else None]
    candidates.append(REPO_ROOT / "runs" / "model" / "so101" / "scene.xml")
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    raise SystemExit(
        "SO-101 MJCF not found. Clone https://github.com/TheRobotStudio/SO-ARM100 @ a758567 and pass "
        "--xml <clone>/Simulation/SO101/scene.xml (or set AUTOSYSID_SO101_XML)."
    )


def load_joint_logs(directory: Path) -> list[RawLog]:
    files = sorted(directory.rglob("*.npz"))  # subfolders = extra base poses of the same joint
    if not files:
        raise SystemExit(f"no .npz logs in {directory}")
    return [RawLog.load(path) for path in files]


def plot_residuals(evaluation: JointEvaluation, path: Path) -> None:
    colors = CONFIG["plot_colors"]
    window = evaluation.times <= evaluation.times[0] + CONFIG["plot_window_s"]
    time_axis = evaluation.times[window]
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    top.plot(time_axis, evaluation.measured[window] / RAD_PER_TICK, color="black", lw=1.4, label="measured")
    for name, color in colors.items():
        top.plot(time_axis, evaluation.predictions[name][window] / RAD_PER_TICK, color=color, lw=1.0, label=name)
        residual = (evaluation.predictions[name] - evaluation.measured)[window] / RAD_PER_TICK
        error = evaluation.errors_ticks[name]
        bottom.plot(time_axis, residual, color=color, lw=0.9, label=f"{name}  (RMS {error:.2f} ticks, all held-out)")
    top.set_ylabel("position (ticks)")
    bottom.set_ylabel("prediction - measured (ticks)")
    bottom.set_xlabel("held-out time (s)")
    bottom.axhline(0, color="grey", lw=0.5)
    top.legend(loc="upper right", ncols=4, fontsize=8)
    bottom.legend(loc="upper right", fontsize=8)
    top.set_title(f"{evaluation.joint}: held-out {', '.join(evaluation.holdout_stages)}")
    figure.savefig(path, dpi=130)
    plt.close(figure)


def sysid_numbers(joint: str, rerun: bool, xml: Path, logs: list[RawLog]) -> float | None:
    """Held-out RMS (ticks) of the best `mujoco.sysid` fit, or None when unavailable for this joint."""
    if rerun:
        from autosysid.evaluation import split_logs
        from autosysid.optimizer import fit_joint, validate_joint

        fit_logs, held_logs = split_logs(logs, joint)
        validation = validate_joint(xml, fit_joint(xml, fit_logs, joint), held_logs)
        return validation.rms_fitted_rad / RAD_PER_TICK
    reference_path = REPO_ROOT / "results" / "sysid_reference.json"
    if reference_path.is_file():
        reference = json.loads(reference_path.read_text())
        if reference["joint"] == joint:
            return float(reference["heldout_sysid_ticks"])
    return None


def render_markdown(rows: dict[str, dict[str, float | None]], notes: list[str]) -> str:
    header = ["Joint", *[label for _, label in CONFIG["columns"]], "Stock / 4-param", "(Stock+delay) / 4-param"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for joint, errors in rows.items():
        cells = [f"{errors[key]:.2f}" if errors.get(key) is not None else "n/a" for key, _ in CONFIG["columns"]]
        ratio = errors["stock_mjcf"] / errors["behavior_4param"]
        ratio_delay = errors["stock_plus_delay"] / errors["behavior_4param"]
        lines.append(f"| {joint} | " + " | ".join(cells) + f" | {ratio:.1f}x | {ratio_delay:.1f}x |")
    return (
        "Held-out RMS position error, encoder ticks (1 tick = 1.534 mrad)\n\n"
        + "\n".join(lines)
        + "\n\n"
        + "\n".join(notes)
        + "\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs", type=Path, default=REPO_ROOT / "hardware_logs")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--xml", type=Path, default=None)
    parser.add_argument("--joints", nargs="*", default=None, help="default: every subdirectory of --logs")
    parser.add_argument("--rerun-sysid", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    xml = find_xml(args.xml)
    model = mujoco.MjModel.from_xml_path(str(xml))
    joints = args.joints or sorted(path.name for path in args.logs.iterdir() if path.is_dir())
    args.out.mkdir(parents=True, exist_ok=True)

    rows: dict[str, dict[str, float | None]] = {}
    details: dict[str, dict] = {}
    final_params: dict[str, dict] = {}
    for joint in joints:
        logs = load_joint_logs(args.logs / joint)
        try:
            evaluation = evaluate_joint(model, logs, joint)
        except AutoSysIDError as error:
            LOGGER.error("%s: skipped: %s", joint, error)
            continue
        errors: dict[str, float | None] = dict(evaluation.errors_ticks)
        errors["mujoco_sysid"] = sysid_numbers(joint, args.rerun_sysid, xml, logs)
        rows[joint] = errors
        details[joint] = {
            "heldout_stages": list(evaluation.holdout_stages),
            "errors_ticks": errors,
            "behavior_3param": evaluation.behavior_3param.__dict__,
            "behavior_4param": evaluation.behavior_4param.__dict__,
            "stock_delay_s": evaluation.stock_delay_s,
        }
        every_log = [log for log in logs if joint in log.meta.get("excited", [])]
        final_params[joint] = fit_behavior(every_log, joint).params.__dict__
        plot_residuals(evaluation, args.out / f"residuals_{joint}.png")
        LOGGER.info("%s: " + ", ".join(f"{k}={v:.2f}" for k, v in errors.items() if v is not None), joint)

    if not rows:
        LOGGER.error("nothing evaluated")
        return 1
    notes = [
        "Held-out = stage-3 multisine where recorded, else the stage-2 'steps' segment (see evaluation.split_logs).",
        "'4-param in MJCF' = the 4-parameter model in front of an ideal-actuator copy of the stock MJCF (sim_wrapper).",
        "*mujoco.sysid is the wrist_roll number from results/sysid_reference.json; its split was not saved, so it is "
        "not directly comparable (re-run with --rerun-sysid for a like-for-like value).",
    ]
    (args.out / "comparison.md").write_text(render_markdown(rows, notes))
    (args.out / "comparison.json").write_text(json.dumps(details, indent=2) + "\n")
    (args.out / "behavior_params.json").write_text(json.dumps(final_params, indent=2) + "\n")
    print((args.out / "comparison.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
