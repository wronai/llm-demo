from click.testing import CliRunner

from wronai.cli import main


def test_version():
    """Test the version command."""
    runner = CliRunner()
    result = runner.invoke(main, ["version"])
    assert result.exit_code == 0
    assert "WronAI version:" in result.output


def test_model_commands():
    """Test model commands."""
    runner = CliRunner()

    # Test model pull
    result = runner.invoke(main, ["model", "pull", "test-model"])
    assert result.exit_code == 0

    # Test model push
    result = runner.invoke(main, ["model", "push", "test-model"])
    assert result.exit_code == 0
