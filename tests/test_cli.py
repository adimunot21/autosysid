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
