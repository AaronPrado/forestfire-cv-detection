from click.testing import CliRunner

from src.cli import smoke


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(smoke, ["--version"])
    assert result.exit_code == 0
    assert "1.0.0" in result.output


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(smoke, ["--help"])
    assert result.exit_code == 0
    assert "Wildfire Smoke Detection" in result.output
    assert "Commands" in result.output
    assert "ingest" in result.output
    assert "process" in result.output
    assert "train" in result.output
    assert "serve" in result.output
    assert "predict" in result.output
    assert "pipeline" in result.output
