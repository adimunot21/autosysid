from typer.testing import CliRunner

from autosysid import __version__
from autosysid.cli import app

runner = CliRunner()


def test_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_info_reports_sysid_available():
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "mujoco.sysid" in result.output
    assert "NOT INSTALLED" not in result.output


# ---- pipeline commands (all on the simulated arm; nothing opens a real port) -------------------------------------

import json  # noqa: E402
import shutil  # noqa: E402

import pytest  # noqa: E402

from autosysid.errors import DataError, HardwareError  # noqa: E402
from autosysid.mock import MINI_ARM_XML  # noqa: E402
from autosysid.model import JointDynamics  # noqa: E402
from autosysid.optimizer import JointFit  # noqa: E402
from autosysid.pipeline import parse_joints, parse_stages  # noqa: E402


def test_doctor_on_the_mock_arm_is_read_only_and_reports_every_joint():
    result = runner.invoke(app, ["doctor", "--mock"])
    assert result.exit_code == 0, result.output
    for joint in ("shoulder_pan", "shoulder_lift", "elbow_flex"):
        assert joint in result.output
    assert "sync read" in result.output


def test_release_on_the_mock_arm_turns_torque_off():
    result = runner.invoke(app, ["release", "--mock"])
    assert result.exit_code == 0, result.output
    assert "limp" in result.output


def test_release_without_hardware_fails_cleanly():
    result = runner.invoke(app, ["release", "--port", "/dev/does-not-exist", "--yes"])
    assert result.exit_code != 0 and "Traceback" not in result.output


def test_doctor_without_hardware_fails_with_a_clear_error_and_opens_nothing():
    result = runner.invoke(app, ["doctor", "--port", "/dev/does-not-exist"])
    assert result.exit_code != 0 and isinstance(result.exception, HardwareError)
    assert "cannot open" in str(result.exception)


def test_record_mock_saves_one_log_per_segment_and_the_config(tmp_path):
    shutil.copy(MINI_ARM_XML, tmp_path / "arm.xml")
    run = tmp_path / "run"
    result = runner.invoke(
        app, ["record", "--mock", "--xml", str(tmp_path / "arm.xml"), "--run-dir", str(run), "--stages", "1"]
    )
    assert result.exit_code == 0, result.output
    assert len(list((run / "raw").glob("*.npz"))) == 2  # stage-1 step + ramp for the one excited joint
    assert json.loads((run / "record.json").read_text())["stages"] == [1]
    assert "pre-flight" in result.output.lower()


def test_record_refuses_when_preflight_fails_and_moves_nothing(tmp_path):
    shutil.copy(MINI_ARM_XML, tmp_path / "arm.xml")
    result = runner.invoke(
        app,
        [
            "record",
            "--mock",
            "--xml",
            str(tmp_path / "arm.xml"),
            "--run-dir",
            str(tmp_path / "r"),
            "--stages",
            "2",
            "--amplitude-scale",
            "1.0",
            "--torque-limit",
            "0.05",
        ],
    )
    assert result.exit_code != 0 and "pre-flight failed" in str(result.exception)
    assert not (tmp_path / "r" / "raw").exists()


def test_record_prompt_can_be_declined(tmp_path):
    """The real-arm path asks first; a "no" must stop everything before the arm moves."""
    from autosysid import cli

    confirm = cli._confirm_factory(mock=False, yes=False, config=cli.RecordConfig(), stages=(1,))

    class Report:
        def summary(self):
            return "ok"

    result = runner.invoke(app, ["--help"])  # keep the runner warm; the prompt itself is exercised below
    assert result.exit_code == 0
    import typer

    original = typer.confirm
    typer.confirm = lambda *a, **k: False
    try:
        assert confirm(Report()) is False
    finally:
        typer.confirm = original


def fake_fit(joint="shoulder_pan", accepted=True) -> JointFit:
    nominal = JointDynamics(998.22, 2.731, 0.6, 0.052, 0.028)
    fitted = JointDynamics(700.0, 4.0, 0.84, 0.078, 0.0224)
    return JointFit(
        joint,
        nominal,
        fitted,
        0.026,
        {"kp": 0.1, "delay": 0.0003},
        ("frictionloss",),
        2e-5,
        1e-5,
        accepted,
        (2e-5,),
        1.0,
    )


def test_export_command_from_a_saved_fit(tmp_path):
    from autosysid.pipeline import save_fits

    shutil.copy(MINI_ARM_XML, tmp_path / "arm.xml")
    save_fits([fake_fit()], tmp_path / "fits.json")
    result = runner.invoke(app, ["export", "--xml", str(tmp_path / "arm.xml"), "--run-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "arm_calibrated.xml").exists() and (tmp_path / "arm_calibrated.calibration.json").exists()
    assert "exported" in result.output
    again = runner.invoke(app, ["export", "--xml", str(tmp_path / "arm.xml"), "--run-dir", str(tmp_path)])
    assert again.exit_code != 0 and "already exists" in str(again.exception)


def test_fit_and_validate_fail_cleanly_on_an_empty_run_directory(tmp_path):
    for command in ("fit", "validate"):
        result = runner.invoke(app, [command, "--xml", str(MINI_ARM_XML), "--run-dir", str(tmp_path)])
        assert result.exit_code != 0 and isinstance(result.exception, (DataError, FileNotFoundError))


def test_fit_record_round_trip_preserves_unidentified_cis():
    fit = fake_fit()
    restored = JointFit.from_dict(
        json.loads(json.dumps({**fit.to_dict(), "relative_ci": {"kp": None, "delay": 0.001}}))
    )
    assert restored.relative_ci["kp"] == float("inf") and restored.dynamics == fit.dynamics


def test_option_parsing():
    assert parse_stages("1, 3") == (1, 3)
    assert parse_joints(None, ("a", "b")) == ("a", "b") and parse_joints("b", ("a", "b")) == ("b",)
    for bad in ("", "4", "x"):
        with pytest.raises(Exception, match="stages"):
            parse_stages(bad)
    with pytest.raises(Exception, match="unknown joints"):
        parse_joints("nope", ("a", "b"))


@pytest.mark.slow
def test_demo_runs_the_whole_pipeline_and_recovers_the_known_answer(tmp_path):
    result = runner.invoke(app, ["demo", "--run-dir", str(tmp_path / "demo")])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "demo" / "mini_arm_calibrated.xml").exists()
    validation = json.loads((tmp_path / "demo" / "validation.json").read_text())["shoulder_pan"]
    assert validation["improvement"] > 3 and validation["rms_fitted_rad"] < 0.0025
    assert "Demo ground truth" in result.output


# ---- behavioral model commands ---------------------------------------------------------------------------------------


def test_behavior_sweep_shrinks_heavy_joints_and_is_recorded(tmp_path):
    shutil.copy(MINI_ARM_XML, tmp_path / "arm.xml")
    args = ["record", "--mock", "--xml", str(tmp_path / "arm.xml"), "--stages", "1", "--joints", "shoulder_lift"]
    plain = runner.invoke(app, [*args, "--run-dir", str(tmp_path / "plain")])
    swept = runner.invoke(app, [*args, "--behavior-sweep", "--run-dir", str(tmp_path / "swept")])
    assert plain.exit_code == 0 and swept.exit_code == 0, swept.output
    config = json.loads((tmp_path / "swept" / "record.json").read_text())["config"]
    assert config["joint_amplitude_factors"] == {"shoulder_lift": 0.6}
    assert json.loads((tmp_path / "plain" / "record.json").read_text())["config"]["joint_amplitude_factors"] == {}

    import numpy as np

    from autosysid.data import RawLog

    def peak(run):
        log = RawLog.load(next((tmp_path / run / "raw").glob("stage1_step*.npz")))
        return float(np.ptp(log.target_rad[:, log.column("shoulder_lift")]))

    assert peak("swept") == pytest.approx(0.6 * peak("plain"), rel=0.05)


def test_fit_behavior_writes_params_that_the_wrapper_can_load(tmp_path):
    import mujoco

    from autosysid.sim_wrapper import STS3215SimWrapper

    shutil.copy(MINI_ARM_XML, tmp_path / "arm.xml")
    run = tmp_path / "run"
    recorded = runner.invoke(
        app, ["record", "--mock", "--xml", str(tmp_path / "arm.xml"), "--run-dir", str(run), "--stages", "1,2"]
    )
    assert recorded.exit_code == 0, recorded.output
    out = tmp_path / "params.json"
    result = runner.invoke(
        app, ["fit-behavior", str(run), "--joint", "shoulder_pan", "--holdout-steps", "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert "shoulder_pan" in result.output and "held-out" in result.output
    model = mujoco.MjModel.from_xml_path(str(MINI_ARM_XML))
    wrapper = STS3215SimWrapper.from_json(model, mujoco.MjData(model), out)
    assert set(wrapper.params) == {"shoulder_pan"}


def test_fit_behavior_merges_into_existing_file_and_reports_missing_logs(tmp_path):
    out = tmp_path / "params.json"
    out.write_text(json.dumps({"wrist_roll": {"delay_s": 0.01, "tau_s": 0.06, "deadband_rad": 0.005}}))
    empty = tmp_path / "empty"
    empty.mkdir()
    result = runner.invoke(app, ["fit-behavior", str(empty), "--joint", "shoulder_pan", "--out", str(out)])
    assert result.exit_code != 0
    assert "wrist_roll" in out.read_text()  # untouched on failure


def test_fit_behavior_load_dependent_deadband_needs_gravity_xml(tmp_path):
    shutil.copy(MINI_ARM_XML, tmp_path / "arm.xml")
    run = tmp_path / "run"
    runner.invoke(app, ["record", "--mock", "--xml", str(tmp_path / "arm.xml"), "--run-dir", str(run), "--stages", "1"])
    result = runner.invoke(
        app,
        [
            "fit-behavior",
            str(run),
            "--joint",
            "shoulder_pan",
            "--load-dependent-deadband",
            "--out",
            str(tmp_path / "p.json"),
        ],
    )
    assert result.exit_code != 0
    assert not (tmp_path / "p.json").exists()


# ---- move ----------------------------------------------------------------------------------------------------------


def test_move_mock_reaches_the_tick_target_and_keeps_torque(tmp_path):
    shutil.copy(MINI_ARM_XML, tmp_path / "arm.xml")
    result = runner.invoke(
        app, ["move", "--mock", "--xml", str(tmp_path / "arm.xml"), "--to-ticks", "shoulder_lift=2300", "--yes"]
    )
    assert result.exit_code == 0, result.output
    assert "2300" in result.output and "holding" in result.output.lower()


@pytest.mark.parametrize(
    "spec", ["shoulder_lift", "shoulder_lift=abc", "shoulder_lift=9999", "wrist_roll=2000", "shoulder_lift=-5"]
)
def test_move_rejects_bad_targets_before_moving(tmp_path, spec):
    shutil.copy(MINI_ARM_XML, tmp_path / "arm.xml")
    result = runner.invoke(app, ["move", "--mock", "--xml", str(tmp_path / "arm.xml"), "--to-ticks", spec, "--yes"])
    assert result.exit_code != 0
    assert "Traceback" not in result.output


def test_move_declined_at_the_prompt_moves_nothing(tmp_path, monkeypatch):
    """The prompt is skipped under --mock, so give the command a mock bus through the real-hardware code path."""
    from autosysid import pipeline as pipeline_module

    shutil.copy(MINI_ARM_XML, tmp_path / "arm.xml")
    opened = {}

    def fake_open_bus(xml, mock, port, baudrate, calibration_path):
        bus, calibration, clock = real_open_bus(xml, True, port, baudrate, calibration_path)
        opened["bus"] = bus
        return bus, calibration, clock

    real_open_bus = pipeline_module.open_bus
    monkeypatch.setattr(pipeline_module, "open_bus", fake_open_bus)
    result = runner.invoke(
        app, ["move", "--xml", str(tmp_path / "arm.xml"), "--to-ticks", "shoulder_lift=2300"], input="n\n"
    )
    assert result.exit_code == 1
    assert not opened["bus"].torque_enabled, "declining must not energise the arm"
