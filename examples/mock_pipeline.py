"""End-to-end identification on a synthetic arm: no hardware needed.

Builds a "truth" arm whose servo dynamics differ from the MJCF, records the three-part excitation protocol with
12-bit quantization, read noise, timing jitter and 26 ms command latency, then fits each joint and shows how close
the recovered values are to the known truth.

    python examples/mock_pipeline.py                      # shoulder_pan, ~1.5 min
    python examples/mock_pipeline.py --joints shoulder_pan shoulder_lift --restarts 1
"""

from __future__ import annotations

import argparse

import numpy as np
from rich.console import Console
from rich.table import Table

from autosysid.config import FitConfig
from autosysid.logging_utils import setup_logging
from autosysid.mock import MINI_ARM_XML, MockArm, MockSettings, record_protocol
from autosysid.model import DYNAMICS_FIELDS, JointDynamics
from autosysid.optimizer import fit_joint

CONFIG = {
    "truth_scale": {"kp": 0.7, "kv": 1.5, "damping": 1.4, "frictionloss": 1.5, "armature": 0.8},  # truth / MJCF nominal
    "truth_delay_s": 0.026,  # 13 sim steps; typical order of magnitude for a USB serial servo loop
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--joints", nargs="+", default=["shoulder_pan"])
    parser.add_argument("--restarts", type=int, default=2, help="extra fits from perturbed starts (best cost wins)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    setup_logging()
    console = Console()

    nominal_arm = MockArm(MINI_ARM_XML)
    truth = {}
    for joint in args.joints:
        nominal = nominal_arm.nominal[joint]
        truth[joint] = JointDynamics(**{f: getattr(nominal, f) * k for f, k in CONFIG["truth_scale"].items()})
    arm = MockArm(
        MINI_ARM_XML,
        truth=truth,
        delays_s={joint: CONFIG["truth_delay_s"] for joint in args.joints},
        settings=MockSettings(seed=args.seed),
    )
    for joint in args.joints:
        logs = record_protocol(arm, joint, np.random.default_rng(args.seed))
        console.print(f"[bold]{joint}[/bold]: {len(logs)} sequences, {sum(len(log.times) for log in logs)} samples")
        fit = fit_joint(MINI_ARM_XML, logs, joint, FitConfig(max_restarts=args.restarts, seed=args.seed))

        table = Table("parameter", "nominal", "truth", "fitted", "error", "95% CI (+-)", "truth in CI", title=joint)
        for name in DYNAMICS_FIELDS:
            nominal_value = getattr(fit.nominal, name)
            truth_value = getattr(truth[joint], name)
            fitted_value = getattr(fit.dynamics, name)
            half_width = fit.relative_ci.get(name, float("inf")) * fitted_value  # CI is in multiplier units
            inside = abs(fitted_value - truth_value) <= half_width
            table.add_row(
                name,
                f"{nominal_value:.4g}",
                f"{truth_value:.4g}",
                f"{fitted_value:.4g}",
                f"{100 * (fitted_value / truth_value - 1):+.1f}%",
                f"{half_width:.3g}",
                "[green]yes[/green]" if inside else "[red]no[/red]",
            )
        delay_ci = fit.relative_ci.get("delay", float("inf"))
        table.add_row(
            "delay (ms)",
            "-",
            f"{1e3 * CONFIG['truth_delay_s']:.1f}",
            f"{1e3 * fit.delay_s:.2f}",
            f"{1e3 * (fit.delay_s - CONFIG['truth_delay_s']):+.2f} ms",
            f"{1e3 * delay_ci:.2f}",
            "",
        )
        console.print(table)
        costs = ", ".join(f"{cost / fit.noise_floor:.2f}" for cost in fit.attempt_costs)
        console.print(
            f"cost / noise floor per start: [{costs}] -> best {fit.cost / fit.noise_floor:.2f}x "
            f"({'accepted' if fit.accepted else 'NOT accepted'}), "
            f"not identified: {list(fit.unidentifiable) or 'none'}, {fit.seconds:.0f} s"
        )


if __name__ == "__main__":
    main()
