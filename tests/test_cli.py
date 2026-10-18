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
